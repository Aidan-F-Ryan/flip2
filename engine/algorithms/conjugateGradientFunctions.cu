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
#include <cstring>

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

// ---- kept from going under the air's pressure ----
//
//Boundaries that let go of the liquid (boundaries.cu) leave the solve an inequality. Of the unknowns they name (bounded), each ends up held, its
//equation met and its pressure over the air's, or let go, at the air's pressure, 0, with its equation left short the way that opens it (r <= 0: more
//flows out of it than in). That's the lowest point of the same energy f(p) among the pressures with p >= 0 on those unknowns, and CG finds it by
//staying on that side (MPRGP: Dostal and Schoberl 2005; Dostal, Optimal Quadratic Programming Algorithms, 2009, algorithm 5.8). The bounded unknowns
//at 0 are active, everything else free, and the residual is in two parts: the free unknowns' own, and at the active ones what they're squeezed by
//(chopped: r > 0 there, the equation asking for a pressure over the air's). At the answer both are 0. Each iteration is one of two steps:
//  - while the chopped residual is small beside the free one (PROPORTION), a step of CG over the free unknowns, preconditioned for them alone, the
//    active ones staying at 0. If it would take bounded unknowns under 0, those stop at 0, active from here on, and CG starts afresh (expansion);
//  - otherwise a step along the chopped residual, which takes hold again of the squeezed ones (proportioning), and CG starts afresh over what's
//    free then.
//The method's proof of ending at the answer wants every step to lower the energy by a known share of what a plain step down the free residual would
//(of a length nothing can overshoot by, stepBound, anything it would take under 0 stopping there; the free residual cut short where that happens is
//the one the test above weighs). A step of CG does, and a proportioning step; one of CG stopped at 0 where it crosses usually does, by far more,
//and is checked: where it doesn't, the plain step is taken in its place. The book's own expansion goes along CG's step only as far as the first
//unknown to reach 0 and then takes the plain step, which lets go of the unknowns one an iteration: on a dam break with every boundary letting go
//that's 8.1 iterations of CG a solve and up to 34, against 5.9 and up to 12 stopping them all at once.
//
//Which of the two steps comes next is decided here, from sums the GPU is waited on for, each iteration: with CG's own stopping test that's one wait
//an iteration where the plain solve has one every other. What a step of CG does at the bounds is decided on the device.

static const double PROPORTION = 0.1;   //Gamma: CG goes on while chopped*chopped <= PROPORTION^2 * free*(free cut short). Measured on that dam break: 7.0 iterations
                                        //of CG a solve at 1, 6.2 at 0.25, 5.9 at 0.1 and below, where it's only more proportioning steps

struct BoundedState{    //an iteration's numbers, on the device
    DotProducts dots;   //the CG step's
    double chopped;     //the chopped residual's square
    double freed;       //the free residual times itself cut short
    double gain;        //twice what a step of CG stopped at the bounds lowers the energy by
    float feasible;     //how far along d the first bounded unknown reaches 0
    float alpha;        //CG's step
    uint largest;       //the largest magnitude in the two parts of the residual, as a float's bits (they order as the floats do)
    uint expanded;      //whether the last step of CG reached a bound
    uint fellBack;      //and whether the plain step was taken in its place
};

static float floatOfBits(uint bits){
    float value;
    memcpy(&value, &bits, sizeof(float));
    return value;
}

//the largest magnitude among the first count values, as a float's bits, into largest (which starts at 0)
__global__ void largestOf(uint count, const float* values, uint* largest){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < count){
        uint bits = __float_as_uint(fabsf(values[index]));
        if(bits > *largest){
            atomicMax(largest, bits);
        }
    }
}

//the start: a bounded unknown under the air's pressure is at it
__global__ void raiseToBounds(uint numUsedVoxels, const char* bounded, float* p){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels && bounded[index] && p[index] < 0.0f){
        p[index] = 0.0f;
    }
}

//The residual's two parts: chopped, what an active unknown is squeezed by, and reduced, a free one's residual, cut short where a step of stepBound
//down it would take a bounded one past 0 (perStepBound is 1 over it). Each is 0 where the other isn't. largest takes the greatest magnitude in them
//before the cutting short, over this partition's own unknowns
__global__ void splitResidual(uint numUsedVoxels, uint numOwnVoxels, const char* solveCodes, const char* bounded, const float* p, const float* r, float perStepBound,
                              float* chopped, float* reduced, uint* largest){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        float squeezed = 0.0f, free = 0.0f, size = 0.0f;
        if(solveCodes[index]){
            if(bounded[index] && p[index] == 0.0f){
                squeezed = fmaxf(r[index], 0.0f);
                size = squeezed;
            }
            else{
                free = bounded[index] ? fmaxf(r[index], -p[index]*perStepBound) : r[index];
                size = fabsf(r[index]);
            }
        }
        chopped[index] = squeezed;
        reduced[index] = free;
        uint bits = __float_as_uint(size);
        if(index < numOwnVoxels && bits > *largest){
            atomicMax(largest, bits);
        }
    }
}

//proportioning steps along the chopped residual by chopped*chopped / chopped*A*chopped: stepDownhill's alpha, with this in place of r*z
__global__ void stepByChopped(BoundedState* state){
    state->dots.rz = state->chopped;
}

//the unknowns CG steps over: all of them but the active
__global__ void freeUnknowns(uint numUsedVoxels, const char* solveCodes, const char* bounded, const float* p, char* free){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        free[index] = bounded[index] && p[index] == 0.0f ? 0 : solveCodes[index];
    }
}

//preconditionByDiagonal for the free unknowns alone: z = r / each one's own coefficient (or r, with no diagonal), and 0 at the active
__global__ void preconditionFree(uint numUsedVoxels, const char* free, const float* r, const float* diagonal, float* z){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        z[index] = !free[index] ? 0.0f : diagonal != nullptr && diagonal[index] != 0.0f ? r[index] / diagonal[index] : r[index];
    }
}

__global__ void noBoundYet(BoundedState* state){
    state->feasible = INFINITY;
}

//how far along d this partition's bounded unknowns can go before the first reaches 0: the least, as a float's bits
__global__ void findFeasible(uint numOwnVoxels, const char* bounded, const float* p, const float* d, uint* feasible){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numOwnVoxels && bounded[index] && d[index] < 0.0f){
        uint bits = __float_as_uint(p[index] / -d[index]);
        if(bits < *feasible){
            atomicMin(feasible, bits);
        }
    }
}

//CG's step, r*z / d*q, and whether it takes any bounded unknown to 0 or past it
__global__ void boundStep(BoundedState* state){
    state->alpha = state->dots.dq != 0.0 ? state->dots.rz / state->dots.dq : 0.0;
    state->expanded = !(state->alpha < state->feasible);
}

//a step that reaches a bound, tried: the pressures it would leave, the bounded unknowns it takes under 0 stopping there
__global__ void tryToBounds(uint numUsedVoxels, const char* bounded, const float* p, const float* d, float* tried, const BoundedState* state){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(state->expanded && index < numUsedVoxels){
        float stepped = p[index] + state->alpha*d[index];
        tried[index] = bounded[index] && stepped < 0.0f ? 0.0f : stepped;
    }
}

//startDownhill, if the flag is set: the residual at x
__global__ void residualIf(uint numUsedVoxels, const char* solveCodes, Stencil A, const float* divU, const float* x, float* r, const uint* flag){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(*flag && index < numUsedVoxels){
        r[index] = solveCodes[index] ? -divU[index] - A.rowTimes(index, x) : 0.0f;
    }
}

//What the step tried lowers the energy by is half of (tried - p)*(r + its residual), summed over the unknowns: the two factors, in d and q, which a
//step that reaches a bound has no more use for (CG starts afresh after it)
__global__ void weighTried(uint numUsedVoxels, const float* p, const float* tried, const float* r, const float* triedResidual, float* d, float* q, const BoundedState* state){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(state->expanded && index < numUsedVoxels){
        d[index] = tried[index] - p[index];
        q[index] = r[index] + triedResidual[index];
    }
}

//the step tried stands if it lowers the energy by as much as the plain step is sure to: half of stepBound times the free residual times itself cut short
__global__ void judgeTried(BoundedState* state, float stepBound){
    state->fellBack = state->expanded && !(state->gain >= stepBound*state->freed);
}

//stepDownhill by CG's step; or, where that reaches a bound, to the pressures tried and their residual; or nowhere, where the plain step takes its place
__global__ void stepToBounds(uint numUsedVoxels, const char* bounded, const float* d, const float* q, const float* tried, const float* triedResidual, float* p, float* r,
                             const BoundedState* state){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        if(!state->expanded){
            float stepped = p[index] + state->alpha*d[index];
            p[index] = bounded[index] && stepped < 0.0f ? 0.0f : stepped;   //nothing gets here but by rounding
            r[index] -= state->alpha*q[index];
        }
        else if(!state->fellBack){
            p[index] = tried[index];
            r[index] = triedResidual[index];
        }
    }
}

//the plain step: this partition's free unknowns go stepBound down their residual, a bounded one no further than 0
__global__ void stepDownResidual(uint numOwnVoxels, const char* solveCodes, const char* bounded, const float* r, float* p, float stepBound, const BoundedState* state){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(state->fellBack && index < numOwnVoxels && solveCodes[index] && !(bounded[index] && p[index] == 0.0f)){
        float stepped = p[index] + stepBound*r[index];
        p[index] = bounded[index] ? fmaxf(stepped, 0.0f) : stepped;
    }
}

//From the pressures the solve with every unknown held came to, and with the same stopping rule, on what's left of the residual's two parts: it has
//stalled once it has gone as many iterations without beating its best as the plain solve does (checkEvery of them a check).
//precondition(r, z) leaves z = M^-1*r over the free unknowns, which free names (0 for the active), and 0 at the active; refree() is called once
//free has changed, before the next of them. The sums are the exact ones whatever the plain solve's are
template <typename Precondition, typename Refree>
static float boundedConjugateGradient(const CudaVec<char>& solveCodes, Stencil A, const VoxelLayout& layout, CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals, const char* bounded,
                                       char* free, float tolerance, uint maxIterations, uint checkEvery, uint numOwnVoxels, PartitionContext& context, cudaStream_t stream, float* z,
                                       Precondition precondition, Refree refree){
    uint numUsedVoxels = p.size();
    uint blocks = numUsedVoxels / BLOCKSIZE + 1;
    float* r = residuals.devPtr();
    float* d;           //the search direction
    float* q;           //A*d; and the chopped residual
    float* tried;       //the pressures a step that reaches a bound would leave
    BoundedState* state;
    ExactSum* exact;
    gpuErrchk(cudaMallocAsync((void**)&d, sizeof(float)*numUsedVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&q, sizeof(float)*numUsedVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&tried, sizeof(float)*numUsedVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&state, sizeof(BoundedState), stream));
    gpuErrchk(cudaMallocAsync((void**)&exact, sizeof(ExactSum), stream));
    cudaMemsetAsync(state, 0, sizeof(BoundedState), stream);
    BoundedState seen;
    //the largest divergence, as the plain solve has it, and the largest of the unknowns' own coefficients (in state's two integers, for the one
    //wait): the matrix's norm is no more than twice that, as a row's other coefficients come to no more than its own, and a step down the residual
    //of 1 over the norm lowers the energy by half its length times the residual times itself cut short, at the least
    largestOf<<<numOwnVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numOwnVoxels, divU.devPtr(), &state->largest);
    largestOf<<<numOwnVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numOwnVoxels, A.Adiag, &state->expanded);
    gpuErrchk(cudaMemcpyAsync(&seen, state, sizeof(BoundedState), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    float maxDivergence = (float)context.maxOverPartitions(floatOfBits(seen.largest));
    float maxDiagonal = (float)context.maxOverPartitions(floatOfBits(seen.expanded));
    float maxResidual = 0.0f;
    if(maxDivergence != 0.0f && maxDiagonal != 0.0f){
        float stepBound = 0.5f / maxDiagonal;
        auto dotProduct = [&](const float* a, const float* b, double* total){
            exactDotProduct(layout, solveCodes.devPtr(), a, b, exact, total, context, stream);
        };
        cudaMemsetAsync(state, 0, sizeof(BoundedState), stream);
        cudaMemsetAsync(d, 0, sizeof(float)*numUsedVoxels, stream);
        raiseToBounds<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, bounded, p.devPtr());
        startDownhill<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes.devPtr(), A, divU.devPtr(), p.devPtr(), r);
        bool changed = true;    //which unknowns are free, since refree last knew
        bool afresh = true;     //CG's next direction has no last one to keep any of
        float best = INFINITY;
        uint sinceBest = 0;
        maxResidual = 1.0f;
        for(uint iteration = 1; iteration <= maxIterations; ++iteration){
            cudaMemsetAsync(&state->largest, 0, sizeof(uint), stream);
            splitResidual<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, numOwnVoxels, solveCodes.devPtr(), bounded, p.devPtr(), r, 1.0f / stepBound, q, z, &state->largest);
            dotProduct(q, q, &state->chopped);
            dotProduct(r, z, &state->freed);
            gpuErrchk(cudaMemcpyAsync(&seen, state, sizeof(BoundedState), cudaMemcpyDeviceToHost, stream));
            gpuErrchk(cudaStreamSynchronize(stream));
            maxResidual = (float)context.maxOverPartitions(floatOfBits(seen.largest)) / maxDivergence;
            sinceBest = maxResidual < best ? 0 : sinceBest + 1;
            best = fminf(best, maxResidual);
            if(maxResidual < tolerance || sinceBest >= STALLED_CHECKS*checkEvery){   //converged, or stalled
                break;
            }
            if(seen.expanded){
                changed = afresh = true;
            }
            if(seen.chopped > PROPORTION*PROPORTION*seen.freed){    //proportioning: q is the chopped residual, and z takes A times it
                context.fillGhosts(q, stream);
                multiplyByA<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, numOwnVoxels, solveCodes.devPtr(), A, q, z, nullptr);
                dotProduct(q, z, &state->dots.dq);
                stepByChopped<<<1, 1, 0, stream>>>(state);
                stepDownhill<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, q, z, p.devPtr(), r, &state->dots);
                cudaMemsetAsync(&state->expanded, 0, sizeof(uint), stream);
                changed = afresh = true;
                continue;
            }
            if(changed){
                freeUnknowns<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes.devPtr(), bounded, p.devPtr(), free);
                refree();
                changed = false;
            }
            precondition(r, z);
            context.fillGhosts(z, stream);
            dotProduct(r, z, &state->dots.rzNext);
            if(afresh){
                cudaMemsetAsync(&state->dots.rz, 0, sizeof(double), stream);    //turnDirection keeps none of the last direction
                afresh = false;
            }
            turnDirection<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, z, d, &state->dots);
            nextIteration<<<1, 1, 0, stream>>>(&state->dots);
            multiplyByA<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, numOwnVoxels, solveCodes.devPtr(), A, d, q, nullptr);
            dotProduct(d, q, &state->dots.dq);
            noBoundYet<<<1, 1, 0, stream>>>(state);
            findFeasible<<<numOwnVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numOwnVoxels, bounded, p.devPtr(), d, (uint*)&state->feasible);
            context.leastOverPartitions(&state->feasible, stream);
            boundStep<<<1, 1, 0, stream>>>(state);
            tryToBounds<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, bounded, p.devPtr(), d, tried, state);
            residualIf<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes.devPtr(), A, divU.devPtr(), tried, z, &state->expanded);
            weighTried<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, p.devPtr(), tried, r, z, d, q, state);
            dotProduct(d, q, &state->gain);     //d*q again, and nothing reads it, after a step that reached no bound
            judgeTried<<<1, 1, 0, stream>>>(state, stepBound);
            stepToBounds<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, bounded, d, q, tried, z, p.devPtr(), r, state);
            stepDownResidual<<<numOwnVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numOwnVoxels, solveCodes.devPtr(), bounded, r, p.devPtr(), stepBound, state);
            context.fillGhosts(p.devPtr(), stream);
            residualIf<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes.devPtr(), A, divU.devPtr(), p.devPtr(), r, &state->fellBack);
        }
    }
    cudaFreeAsync(d, stream);
    cudaFreeAsync(q, stream);
    cudaFreeAsync(tried, stream);
    cudaFreeAsync(state, stream);
    cudaFreeAsync(exact, stream);
    return maxResidual;
}

static Stencil makeStencil(const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& Anx, const CudaVec<float>& Apx, const CudaVec<float>& Any, const CudaVec<float>& Apy, const CudaVec<float>& Anz, const CudaVec<float>& Apz, const CudaVec<float>& Adiag){
    return {{neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr()},
            {Anx.devPtr(), Apx.devPtr(), Any.devPtr(), Apy.devPtr(), Anz.devPtr(), Apz.devPtr()}, Adiag.devPtr()};
}

//with no preconditioner (z = r) or Jacobi's (z = r / the unknown's own coefficient)
static float diagonallyPreconditioned(bool jacobi, const CudaVec<char>& solveCodes, Stencil A, const VoxelLayout& layout, DotProductSums sums, CudaVec<float>& divU, CudaVec<float>& p,
                                      CudaVec<float>& residuals, float tolerance, uint maxIterations, uint numOwnVoxels, PartitionContext& context, cudaStream_t stream, const char* bounded){
    uint numUsedVoxels = p.size();
    if(bounded != nullptr){
        float* preconditioned;
        char* free;
        gpuErrchk(cudaMallocAsync((void**)&preconditioned, sizeof(float)*numUsedVoxels, stream));
        gpuErrchk(cudaMallocAsync((void**)&free, numUsedVoxels, stream));
        float residual = boundedConjugateGradient(solveCodes, A, layout, divU, p, residuals, bounded, free, tolerance, maxIterations, 16, numOwnVoxels, context, stream, preconditioned,
            [&](const float* r, float* z){
                preconditionFree<<<numUsedVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numUsedVoxels, free, r, jacobi ? A.Adiag : nullptr, z);
            }, [](){});
        cudaFreeAsync(preconditioned, stream);
        cudaFreeAsync(free, stream);
        return residual;
    }
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
    PartitionContext& context, cudaStream_t stream, const char* bounded){
    return diagonallyPreconditioned(false, solveCodes, makeStencil(neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag), layout, sums, divU, p, residuals,
                                    tolerance, maxIterations, numOwnVoxels, context, stream, bounded);
}

float cudaJacobiConjugateGradient(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& Anx, const CudaVec<float>& Apx, const CudaVec<float>& Any, const CudaVec<float>& Apy, const CudaVec<float>& Anz, const CudaVec<float>& Apz, const CudaVec<float>& Adiag,
    CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals, float tolerance, uint maxIterations, const VoxelLayout& layout, DotProductSums sums, uint numOwnVoxels,
    PartitionContext& context, cudaStream_t stream, const char* bounded){
    return diagonallyPreconditioned(true, solveCodes, makeStencil(neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag), layout, sums, divU, p, residuals,
                                    tolerance, maxIterations, numOwnVoxels, context, stream, bounded);
}

float cudaMultigridConjugateGradient(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& Anx, const CudaVec<float>& Apx, const CudaVec<float>& Any, const CudaVec<float>& Apy, const CudaVec<float>& Anz, const CudaVec<float>& Apz, const CudaVec<float>& Adiag,
    CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals, float tolerance, uint maxIterations, const VoxelLayout& layout, DotProductSums sums, uint numOwnVoxels,
    PartitionContext& context, cudaStream_t stream, const char* bounded){
    uint numUsedVoxels = p.size();
    Stencil A = makeStencil(neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag);
    float* z;
    gpuErrchk(cudaMallocAsync((void**)&z, sizeof(float)*numUsedVoxels, stream));
    if(bounded != nullptr){     //the V-cycle's grids are the free unknowns', and their equations are made again whenever which those are changes
        char* free;
        gpuErrchk(cudaMallocAsync((void**)&free, numUsedVoxels, stream));
        Multigrid multigrid = {};
        float residual = boundedConjugateGradient(solveCodes, A, layout, divU, p, residuals, bounded, free, tolerance, maxIterations, 2, numOwnVoxels, context, stream, z,
            [&](const float* r, float* preconditioned){
                vCycle(multigrid, r, preconditioned, stream);
            }, [&](){
                if(multigrid.numLevels == 0){
                    multigrid = buildMultigrid(A, free, numUsedVoxels, layout, context, stream, true);
                }
                else{
                    coarsenMultigrid(multigrid, stream);
                }
            });
        if(multigrid.numLevels != 0){
            freeMultigrid(multigrid, stream);
        }
        cudaFreeAsync(z, stream);
        cudaFreeAsync(free, stream);
        return residual;
    }
    Multigrid multigrid = buildMultigrid(A, solveCodes.devPtr(), numUsedVoxels, layout, context, stream);
    DotProducts* dots;
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
