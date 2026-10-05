//Copyright 2023 Aberrant Behavior LLC

//Conjugate gradient for the pressure equations, beside the SOR in voxelSolveFunctions.cu: the same unknowns, seven coefficient arrays and six neighbour
//arrays, solving A*p = b with b = -divU.
//
//SOR fixes one voxel at a time from its neighbours, so a correction spreads a voxel per sweep and smooth errors take many sweeps. CG moves every unknown
//at once, downhill on the energy f(p) = p*A*p/2 - b*p. Its lowest point is the solution, and its downhill direction at p is the residual r = b - A*p.
//Each iteration:
//  1. multiplies the search direction by A (q = A*d), its one stencil pass
//  2. steps along d by alpha = r*z / d*q, exactly as far as lowers the energy most along d. The residual then moves by -alpha*q, so it needs no second
//     stencil pass
//  3. preconditions the new residual, z = M^-1*r, an approximate solve of A*z = r: how far off each unknown is rather than just how unbalanced
//  4. turns the search direction towards z, keeping beta = r*z(new) / r*z(old) of the old direction. That makes it A-conjugate to every earlier
//     direction, so minimizing along it never undoes the minimizing already done along them
//The better M approximates A, the fewer iterations it takes. With no preconditioner z is just r; Jacobi divides each residual by its unknown's own
//coefficient; multigrid runs a V-cycle (multigridFunctions.cu). The dot products are the only sums over every unknown, and they stay on the GPU, so an
//iteration never waits on the host. They're added up one of two ways (DotProductSums):
//  exact:    each node sums its own voxels in a fixed order, and the nodes' sums are added exactly, in fixed point. The result doesn't depend on how
//            the voxels are laid out or how the nodes are split between GPUs, so a run repeats exactly on any number of them. The default
//  perBlock: each thread block sums its voxels in float, and sumPartials adds the blocks' sums in a fixed order. A run repeats exactly with the same
//            layout, but splitting the nodes differently regroups the sums and changes their rounding. The original, kept to fall back on if exact
//            costs too much

#include "conjugateGradientFunctions.hu"
#include "multigridFunctions.hu"
#include "voxelSolveFunctions.hu"     //NO_VOXEL
#include <cmath>

struct DotProducts{     //doubles: float sums over a million voxels would lose the precision the step sizes need
    double rz;          //r*z of the current residual
    double dq;          //d*(A*d) of the current direction
    double rzNext;      //r*z after this iteration's step
};

static const uint SUM_THREADS = 1024;   //sumPartials' one block

//this block's share of a sum: every thread passes its value in (0 if it has none), warps sum theirs, then one warp sums the warps' and leaves the block's
//total in partials[blockIdx.x]. In float, as this GPU does doubles at 1/64 the rate
__device__ inline void addBlockSum(float value, float* partials){
    __shared__ float warpSums[32];
    for(int lanes = 16; lanes > 0; lanes >>= 1){
        value += __shfl_xor_sync(0xffffffff, value, lanes);
    }
    if(threadIdx.x % 32 == 0){
        warpSums[threadIdx.x/32] = value;
    }
    __syncthreads();
    if(threadIdx.x < 32){
        value = threadIdx.x < blockDim.x/32 ? warpSums[threadIdx.x] : 0.0f;
        for(int lanes = 16; lanes > 0; lanes >>= 1){
            value += __shfl_xor_sync(0xffffffff, value, lanes);
        }
        if(threadIdx.x == 0){
            partials[blockIdx.x] = value;
        }
    }
}

//adds up the blocks' shares of a sum, always in the same order, so it comes out the same every run (atomics would add them in whatever order the blocks
//finish). Each thread sums every SUM_THREADS-th share, then the threads' sums pair off in double
__global__ void sumPartials(uint numPartials, const float* partials, double* total){
    __shared__ double sums[SUM_THREADS];
    float sum = 0.0f;
    for(uint i = threadIdx.x; i < numPartials; i += SUM_THREADS){
        sum += partials[i];
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for(uint half = SUM_THREADS / 2; half > 0; half /= 2){
        if(threadIdx.x < half){
            sums[threadIdx.x] += sums[threadIdx.x + half];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        *total = sums[0];
    }
}

// ---- exact sums ----

static const int EXACT_DIGITS = 9;      //32-bit digits from 2^-149, the smallest float, up past the largest float's 2^128
static const uint NODE_SLOTS = 64;      //a node's interior voxels, 4^3

//an exact sum of floats in fixed point: digit k counts units of 2^(32k - 149). Each digit is a 64-bit integer holding its 32 bits plus carries not yet
//passed up, so it takes 2^31 additions before anything could overflow, and integers add up the same in any order. Summing across partitions is just
//adding these words up, one allreduce
struct ExactSum{
    long long digits[EXACT_DIGITS];
    long long nonFinite;    //how many infs or NaNs went in, so the result is NaN as a float sum's would be
};
static const uint EXACT_WORDS = sizeof(ExactSum) / sizeof(long long);

//adds a float to an exact sum: its 24-bit significand, shifted to its binary exponent, straddles at most two digits
__device__ inline void addExactly(float value, ExactSum& sum){
    uint bits = __float_as_uint(value);
    int exponent = bits >> 23 & 0xFF;
    if(value == 0.0f){
        return;
    }
    if(exponent == 0xFF){
        atomicAdd((unsigned long long*)&sum.nonFinite, 1ull);
        return;
    }
    unsigned long long significand = bits & 0x7FFFFF;
    if(exponent != 0){
        significand |= 0x800000;    //the hidden bit
    }
    else{
        exponent = 1;               //a subnormal: no hidden bit, at the smallest exponent
    }
    int position = exponent - 1;    //value = significand*2^(exponent - 150), and 2^-149 is bit 0
    significand <<= position % 32;
    long long sign = bits >> 31 ? -1 : 1;
    atomicAdd((unsigned long long*)sum.digits + position/32, (unsigned long long)(sign*(long long)(significand & 0xFFFFFFFFull)));
    atomicAdd((unsigned long long*)sum.digits + position/32 + 1, (unsigned long long)(sign*(long long)(significand >> 32)));
}

//a*b over the unknowns, added into an exact sum: a warp per node, its lanes taking interior slots lane and lane + 32, then pairing off in a fixed tree
//to the node's share, which goes in exactly. A share depends only on the node's own values, not on where its voxels are stored or which block or GPU
//takes it. Each block gathers its warps' shares, then adds them in
__global__ void addNodeDotShares(uint numNodes, const uint* interiorVoxels, const char* solveCodes, const float* a, const float* b, ExactSum* sum){
    __shared__ ExactSum blockSum;
    if(threadIdx.x < EXACT_DIGITS){
        blockSum.digits[threadIdx.x] = 0;
    }
    if(threadIdx.x == 0){
        blockSum.nonFinite = 0;
    }
    __syncthreads();
    uint lane = threadIdx.x % 32;
    uint warps = gridDim.x*blockDim.x/32;
    for(uint node = (threadIdx.x + blockIdx.x*blockDim.x)/32; node < numNodes; node += warps){    //the same for the whole warp
        float terms[2];
        #pragma unroll
        for(int half = 0; half < 2; ++half){
            uint voxel = interiorVoxels[node*NODE_SLOTS + lane + 32*half];
            terms[half] = voxel != NO_VOXEL && solveCodes[voxel] ? __fmul_rn(a[voxel], b[voxel]) : 0.0f;    //no fused multiply-adds: every build rounds alike
        }
        float share = __fadd_rn(terms[0], terms[1]);
        for(int lanes = 16; lanes > 0; lanes >>= 1){
            share = __fadd_rn(share, __shfl_xor_sync(0xffffffff, share, lanes));   //a + b == b + a exactly, so every lane ends with the same share
        }
        if(lane == 0){
            addExactly(share, blockSum);
        }
    }
    __syncthreads();
    if(threadIdx.x < EXACT_DIGITS && blockSum.digits[threadIdx.x] != 0){
        atomicAdd((unsigned long long*)sum->digits + threadIdx.x, (unsigned long long)blockSum.digits[threadIdx.x]);
    }
    if(threadIdx.x == 0 && blockSum.nonFinite){
        atomicAdd((unsigned long long*)&sum->nonFinite, (unsigned long long)blockSum.nonFinite);
    }
}

//an exact sum back to a double: pass the carries up so each digit holds just its 32 bits, then add the digits in from the top. The sign ends up in the
//top carry; a negative sum is negated first, so the conversion never subtracts two huge numbers
__global__ void finishExactSum(const ExactSum* sum, double* total){
    if(sum->nonFinite){
        *total = nan("");
        return;
    }
    unsigned long long digits[EXACT_DIGITS];
    long long carry = 0;
    for(int k = 0; k < EXACT_DIGITS; ++k){
        long long value = sum->digits[k] + carry;
        digits[k] = (unsigned long long)value & 0xFFFFFFFFull;
        carry = value >> 32;    //arithmetic shift, so the digit left behind is in [0, 2^32)
    }
    bool negative = carry < 0;
    if(negative){   //two's complement over the digits and the carry together: flip every bit, add 1
        unsigned long long add = 1;
        for(int k = 0; k < EXACT_DIGITS; ++k){
            unsigned long long value = (~digits[k] & 0xFFFFFFFFull) + add;
            digits[k] = value & 0xFFFFFFFFull;
            add = value >> 32;
        }
        carry = ~carry + (long long)add;
    }
    double magnitude = (double)carry*scalbn(1.0, 32*EXACT_DIGITS - 149);
    for(int k = EXACT_DIGITS - 1; k >= 0; --k){
        magnitude += (double)digits[k]*scalbn(1.0, 32*k - 149);
    }
    *total = negative ? -magnitude : magnitude;
}

//total = a*b over the unknowns, the same bit for bit however the nodes are stored or split up: each partition sums its own nodes' shares (layout.numNodes
//are its own), then the partitions add their digits together
static void exactDotProduct(const VoxelLayout& layout, const char* solveCodes, const float* a, const float* b, ExactSum* sum, double* total, PartitionContext& context, cudaStream_t stream){
    uint blocks = layout.numNodes*32 / BLOCKSIZE + 1;
    cudaMemsetAsync(sum, 0, sizeof(ExactSum), stream);
    addNodeDotShares<<<blocks < 1024 ? blocks : 1024, BLOCKSIZE, 0, stream>>>(layout.numNodes, layout.interiorVoxels, solveCodes, a, b, sum);
    context.sumOverPartitions((long long*)sum, EXACT_WORDS, stream);
    finishExactSum<<<1, 1, 0, stream>>>(sum, total);
}

// ---- CG ----

//start: the residual of the starting guess, r = b - A*p. Only unknowns have equations, so everything else stays 0 throughout
__global__ void startDownhill(uint numUsedVoxels, const char* solveCodes, Stencil A, const float* divU, const float* p, float* r){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        r[index] = solveCodes[index] ? -divU[index] - A.rowTimes(index, p) : 0.0f;
    }
}

//1: q = A*d; d*q sets how far to step along d. With partials (perBlock), also d*q's blocks' shares, over this partition's own voxels (the first numOwnVoxels)
__global__ void multiplyByA(uint numUsedVoxels, uint numOwnVoxels, const char* solveCodes, Stencil A, const float* d, float* q, float* partials){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    float dq = 0.0f;
    if(index < numUsedVoxels){
        float product = solveCodes[index] ? A.rowTimes(index, d) : 0.0f;
        q[index] = product;
        if(index < numOwnVoxels){
            dq = d[index]*product;
        }
    }
    if(partials != nullptr){    //the same for every thread
        addBlockSum(dq, partials);
    }
}

//2: step along d by alpha = r*z / d*q, the distance that lowers the energy most along d. The residual moves by -alpha*A*d, which is q, already known
__global__ void stepDownhill(uint numUsedVoxels, const float* d, const float* q, float* p, float* r, const DotProducts* dots){
    __shared__ float alpha;
    if(threadIdx.x == 0){
        alpha = dots->dq != 0.0 ? dots->rz / dots->dq : 0.0;
    }
    __syncthreads();
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        p[index] += alpha*d[index];
        r[index] -= alpha*q[index];
    }
}

//3, for the preconditioners that just scale each unknown: z = r / its own coefficient (Jacobi), or z = r with none (z is r itself then, so there's
//nothing to write). With partials (perBlock), also r*z's blocks' shares
__global__ void preconditionByDiagonal(uint numUsedVoxels, uint numOwnVoxels, const float* r, const float* diagonal, float* z, float* partials){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    float rz = 0.0f;
    if(index < numUsedVoxels){
        float preconditioned = diagonal != nullptr && diagonal[index] != 0.0f ? r[index] / diagonal[index] : r[index];
        if(z != r){
            z[index] = preconditioned;
        }
        if(index < numOwnVoxels){
            rz = r[index]*preconditioned;
        }
    }
    if(partials != nullptr){    //the same for every thread
        addBlockSum(rz, partials);
    }
}

//r*z's blocks' shares, once a V-cycle has made z, over this partition's own voxels
__global__ void dotResidual(uint numOwnVoxels, const float* r, const float* z, float* partials){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    addBlockSum(index < numOwnVoxels ? r[index]*z[index] : 0.0f, partials);
}

//4: the next direction is z, plus beta = r*z(new) / r*z(old) of the last one, which keeps it A-conjugate to all the earlier directions. The first time
//there's no last one (r*z(old) is 0), so it's z itself
__global__ void turnDirection(uint numUsedVoxels, const float* z, float* d, const DotProducts* dots){
    __shared__ float beta;
    if(threadIdx.x == 0){
        beta = dots->rz != 0.0 ? dots->rzNext / dots->rz : 0.0;
    }
    __syncthreads();
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        d[index] = z[index] + beta*d[index];
    }
}

//the step's r*z becomes the current one
__global__ void nextIteration(DotProducts* dots){
    dots->rz = dots->rzNext;
}

//CG with the preconditioner handed in: precondition(r, z, partials) leaves z = M^-1*r, and with partials (perBlock only; null for exact) r*z's blocks'
//shares in them. Finding the largest residual waits on the host, so like the SOR it's only checked every checkEvery iterations
//
//Split between partitions, each holds its own unknowns and copies of its neighbours' next to them (ghosts, codes 3 and 4), and every step runs in all of
//them at once. The sums and maxima cover each partition's own voxels, then every partition's; z's ghosts take their owners' values once it's made, which
//keeps d's and p's ghosts equal to their owners' too, as they're only ever combined from z's; and q = A*d reads ghost d's
template <typename Precondition>
static float conjugateGradient(const CudaVec<char>& solveCodes, Stencil A, const VoxelLayout& layout, DotProductSums sums, CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals,
                                float tolerance, uint maxIterations, uint checkEvery, uint numOwnVoxels, PartitionContext& context, cudaStream_t stream, float* z, DotProducts* dots, Precondition precondition){
    uint numUsedVoxels = p.size();
    uint blocks = numUsedVoxels / BLOCKSIZE + 1;
    float maxDivergence = (float)context.maxOverPartitions(std::abs(divU.getMax(stream, true, numOwnVoxels)));
    if(maxDivergence == 0.0f){
        return 0.0f;
    }
    float* r = residuals.devPtr();
    float* d;                   //the search direction
    float* q;                   //A*d
    float* partials = nullptr;  //perBlock: each block's share of the sum being taken
    ExactSum* exact = nullptr;  //exact: the sum being taken
    gpuErrchk(cudaMallocAsync((void**)&d, sizeof(float)*numUsedVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&q, sizeof(float)*numUsedVoxels, stream));
    if(sums == DotProductSums::perBlock){
        gpuErrchk(cudaMallocAsync((void**)&partials, sizeof(float)*blocks, stream));
    }
    else{
        gpuErrchk(cudaMallocAsync((void**)&exact, sizeof(ExactSum), stream));
    }
    //total = a*b; perBlock has its blocks' shares in partials already, from the kernel that made b, and adds the partitions' totals in partition order
    auto dotProduct = [&](const float* a, const float* b, double* total){
        if(sums == DotProductSums::perBlock){
            sumPartials<<<1, SUM_THREADS, 0, stream>>>(blocks, partials, total);
            context.sumOverPartitions(total, stream);
        }
        else{
            exactDotProduct(layout, solveCodes.devPtr(), a, b, exact, total, context, stream);
        }
    };
    cudaMemsetAsync(d, 0, sizeof(float)*numUsedVoxels, stream);
    cudaMemsetAsync(dots, 0, sizeof(DotProducts), stream);
    startDownhill<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes.devPtr(), A, divU.devPtr(), p.devPtr(), r);
    precondition(r, z, partials);
    context.fillGhosts(z, stream);
    dotProduct(r, z, &dots->rzNext);
    turnDirection<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, z, d, dots);
    nextIteration<<<1, 1, 0, stream>>>(dots);
    float maxResidual = 1.0f;
    SolveProgress progress;
    for(uint iteration = 1; iteration <= maxIterations; ++iteration){
        multiplyByA<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, numOwnVoxels, solveCodes.devPtr(), A, d, q, partials);
        dotProduct(d, q, &dots->dq);
        stepDownhill<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, d, q, p.devPtr(), r, dots);
        precondition(r, z, partials);
        context.fillGhosts(z, stream);
        dotProduct(r, z, &dots->rzNext);
        turnDirection<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, z, d, dots);
        nextIteration<<<1, 1, 0, stream>>>(dots);
        if(iteration % checkEvery == 0){    //the SOR's stopping rule: the largest residual, relative to the largest divergence
            maxResidual = (float)context.maxOverPartitions(std::abs(residuals.getMax(stream, true, numOwnVoxels))) / maxDivergence;
            if(progress.done(maxResidual, tolerance)){  //converged, or stalled
                break;
            }
        }
    }
    cudaFreeAsync(d, stream);
    cudaFreeAsync(q, stream);
    if(partials != nullptr){
        cudaFreeAsync(partials, stream);
    }
    if(exact != nullptr){
        cudaFreeAsync(exact, stream);
    }
    return maxResidual;
}

static Stencil makeStencil(const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& Anx, const CudaVec<float>& Apx, const CudaVec<float>& Any, const CudaVec<float>& Apy, const CudaVec<float>& Anz, const CudaVec<float>& Apz, const CudaVec<float>& Adiag){
    return {{neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr()},
            {Anx.devPtr(), Apx.devPtr(), Any.devPtr(), Apy.devPtr(), Anz.devPtr(), Apz.devPtr()}, Adiag.devPtr()};
}

//with no preconditioner (z = r) or Jacobi's (z = r / the unknown's own coefficient)
static float diagonallyPreconditioned(bool jacobi, const CudaVec<char>& solveCodes, Stencil A, const VoxelLayout& layout, DotProductSums sums, CudaVec<float>& divU, CudaVec<float>& p,
                                      CudaVec<float>& residuals, float tolerance, uint maxIterations, uint numOwnVoxels, PartitionContext& context, cudaStream_t stream){
    uint numUsedVoxels = p.size();
    float* z = residuals.devPtr();  //with none, z is r itself
    DotProducts* dots;
    gpuErrchk(cudaMallocAsync((void**)&dots, sizeof(DotProducts), stream));
    if(jacobi){
        gpuErrchk(cudaMallocAsync((void**)&z, sizeof(float)*numUsedVoxels, stream));
    }
    float residual = conjugateGradient(solveCodes, A, layout, sums, divU, p, residuals, tolerance, maxIterations, 16, numOwnVoxels, context, stream, z, dots, [&](const float* r, float* preconditioned, float* partials){
        if(jacobi || partials != nullptr){  //with no preconditioner, exact sums leave nothing for it to do
            preconditionByDiagonal<<<numUsedVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numUsedVoxels, numOwnVoxels, r, jacobi ? A.Adiag : nullptr, preconditioned, partials);
        }
    });
    if(jacobi){
        cudaFreeAsync(z, stream);
    }
    cudaFreeAsync(dots, stream);
    return residual;
}

float cudaConjugateGradient(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& Anx, const CudaVec<float>& Apx, const CudaVec<float>& Any, const CudaVec<float>& Apy, const CudaVec<float>& Anz, const CudaVec<float>& Apz, const CudaVec<float>& Adiag,
    CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals, float tolerance, uint maxIterations, const VoxelLayout& layout, DotProductSums sums, uint numOwnVoxels,
    PartitionContext& context, cudaStream_t stream){
    return diagonallyPreconditioned(false, solveCodes, makeStencil(neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag), layout, sums, divU, p, residuals,
                                    tolerance, maxIterations, numOwnVoxels, context, stream);
}

float cudaJacobiConjugateGradient(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& Anx, const CudaVec<float>& Apx, const CudaVec<float>& Any, const CudaVec<float>& Apy, const CudaVec<float>& Anz, const CudaVec<float>& Apz, const CudaVec<float>& Adiag,
    CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals, float tolerance, uint maxIterations, const VoxelLayout& layout, DotProductSums sums, uint numOwnVoxels,
    PartitionContext& context, cudaStream_t stream){
    return diagonallyPreconditioned(true, solveCodes, makeStencil(neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag), layout, sums, divU, p, residuals,
                                    tolerance, maxIterations, numOwnVoxels, context, stream);
}

float cudaMultigridConjugateGradient(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& Anx, const CudaVec<float>& Apx, const CudaVec<float>& Any, const CudaVec<float>& Apy, const CudaVec<float>& Anz, const CudaVec<float>& Apz, const CudaVec<float>& Adiag,
    CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals, float tolerance, uint maxIterations, const VoxelLayout& layout, DotProductSums sums, uint numOwnVoxels,
    PartitionContext& context, cudaStream_t stream){
    uint numUsedVoxels = p.size();
    Stencil A = makeStencil(neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag);
    Multigrid multigrid = buildMultigrid(A, solveCodes.devPtr(), numUsedVoxels, layout, context, stream);
    float* z;
    DotProducts* dots;
    gpuErrchk(cudaMallocAsync((void**)&z, sizeof(float)*numUsedVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&dots, sizeof(DotProducts), stream));
    //a V-cycle costs dozens of plain iterations and it should take few of them, so it checks sooner
    float residual = conjugateGradient(solveCodes, A, layout, sums, divU, p, residuals, tolerance, maxIterations, 2, numOwnVoxels, context, stream, z, dots, [&](const float* r, float* preconditioned, float* partials){
        vCycle(multigrid, r, preconditioned, stream);
        if(partials != nullptr){
            dotResidual<<<numUsedVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numOwnVoxels, r, preconditioned, partials);
        }
    });
    cudaFreeAsync(z, stream);
    cudaFreeAsync(dots, stream);
    freeMultigrid(multigrid, stream);
    return residual;
}
