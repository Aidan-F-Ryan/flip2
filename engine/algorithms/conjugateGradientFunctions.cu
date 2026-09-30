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
//coefficient; multigrid runs a V-cycle (multigridFunctions.cu). The dot products are the only sums over every unknown: each kernel leaves its blocks'
//shares, sumPartials adds those up in a fixed order so a run repeats exactly, and they stay on the GPU, so an iteration never waits on the host

#include "conjugateGradientFunctions.hu"
#include "multigridFunctions.hu"
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

//start: the residual of the starting guess, r = b - A*p. Only unknowns have equations, so everything else stays 0 throughout
__global__ void startDownhill(uint numUsedVoxels, const char* solveCodes, Stencil A, const float* divU, const float* p, float* r){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        r[index] = solveCodes[index] ? -divU[index] - A.rowTimes(index, p) : 0.0f;
    }
}

//1: q = A*d, and d*q's blocks' shares, which set how far to step along d
__global__ void multiplyByA(uint numUsedVoxels, const char* solveCodes, Stencil A, const float* d, float* q, float* partials){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    float dq = 0.0f;
    if(index < numUsedVoxels){
        float product = solveCodes[index] ? A.rowTimes(index, d) : 0.0f;
        q[index] = product;
        dq = d[index]*product;
    }
    addBlockSum(dq, partials);
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
//nothing to write). Also r*z's blocks' shares
__global__ void preconditionByDiagonal(uint numUsedVoxels, const float* r, const float* diagonal, float* z, float* partials){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    float rz = 0.0f;
    if(index < numUsedVoxels){
        float preconditioned = diagonal != nullptr && diagonal[index] != 0.0f ? r[index] / diagonal[index] : r[index];
        if(z != r){
            z[index] = preconditioned;
        }
        rz = r[index]*preconditioned;
    }
    addBlockSum(rz, partials);
}

//r*z's blocks' shares, once a V-cycle has made z
__global__ void dotResidual(uint numUsedVoxels, const float* r, const float* z, float* partials){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    addBlockSum(index < numUsedVoxels ? r[index]*z[index] : 0.0f, partials);
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

//CG with the preconditioner handed in: precondition(r, z, partials) leaves z = M^-1*r and r*z's blocks' shares in partials. Finding the largest residual
//waits on the host, so like the SOR it's only checked every checkEvery iterations
template <typename Precondition>
static float conjugateGradient(const CudaVec<char>& solveCodes, Stencil A, CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals, float tolerance, uint maxIterations,
                                uint checkEvery, cudaStream_t stream, float* z, DotProducts* dots, Precondition precondition){
    uint numUsedVoxels = p.size();
    uint blocks = numUsedVoxels / BLOCKSIZE + 1;
    float maxDivergence = std::abs(divU.getMax(stream, true));
    if(maxDivergence == 0.0f){
        return 0.0f;
    }
    float* r = residuals.devPtr();
    float* d;           //the search direction
    float* q;           //A*d
    float* partials;    //each block's share of the sum being taken
    gpuErrchk(cudaMallocAsync((void**)&d, sizeof(float)*numUsedVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&q, sizeof(float)*numUsedVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&partials, sizeof(float)*blocks, stream));
    cudaMemsetAsync(d, 0, sizeof(float)*numUsedVoxels, stream);
    cudaMemsetAsync(dots, 0, sizeof(DotProducts), stream);
    startDownhill<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes.devPtr(), A, divU.devPtr(), p.devPtr(), r);
    precondition(r, z, partials);
    sumPartials<<<1, SUM_THREADS, 0, stream>>>(blocks, partials, &dots->rzNext);
    turnDirection<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, z, d, dots);
    nextIteration<<<1, 1, 0, stream>>>(dots);
    float maxResidual = 1.0f;
    for(uint iteration = 1; iteration <= maxIterations; ++iteration){
        multiplyByA<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes.devPtr(), A, d, q, partials);
        sumPartials<<<1, SUM_THREADS, 0, stream>>>(blocks, partials, &dots->dq);
        stepDownhill<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, d, q, p.devPtr(), r, dots);
        precondition(r, z, partials);
        sumPartials<<<1, SUM_THREADS, 0, stream>>>(blocks, partials, &dots->rzNext);
        turnDirection<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, z, d, dots);
        nextIteration<<<1, 1, 0, stream>>>(dots);
        if(iteration % checkEvery == 0){    //the SOR's stopping rule: the largest residual, relative to the largest divergence
            float previousResidual = maxResidual;
            maxResidual = std::abs(residuals.getMax(stream, true)) / maxDivergence;
            if(maxResidual < tolerance || std::abs(previousResidual - maxResidual) < tolerance / 1000){  //converged, or stalled
                break;
            }
        }
    }
    cudaFreeAsync(d, stream);
    cudaFreeAsync(q, stream);
    cudaFreeAsync(partials, stream);
    return maxResidual;
}

static Stencil makeStencil(const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& Anx, const CudaVec<float>& Apx, const CudaVec<float>& Any, const CudaVec<float>& Apy, const CudaVec<float>& Anz, const CudaVec<float>& Apz, const CudaVec<float>& Adiag){
    return {{neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr()},
            {Anx.devPtr(), Apx.devPtr(), Any.devPtr(), Apy.devPtr(), Anz.devPtr(), Apz.devPtr()}, Adiag.devPtr()};
}

//with no preconditioner (z = r) or Jacobi's (z = r / the unknown's own coefficient)
static float diagonallyPreconditioned(bool jacobi, const CudaVec<char>& solveCodes, Stencil A, CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals, float tolerance, uint maxIterations, cudaStream_t stream){
    uint numUsedVoxels = p.size();
    float* z = residuals.devPtr();  //with none, z is r itself
    DotProducts* dots;
    gpuErrchk(cudaMallocAsync((void**)&dots, sizeof(DotProducts), stream));
    if(jacobi){
        gpuErrchk(cudaMallocAsync((void**)&z, sizeof(float)*numUsedVoxels, stream));
    }
    float residual = conjugateGradient(solveCodes, A, divU, p, residuals, tolerance, maxIterations, 16, stream, z, dots, [&](const float* r, float* preconditioned, float* partials){
        preconditionByDiagonal<<<numUsedVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numUsedVoxels, r, jacobi ? A.Adiag : nullptr, preconditioned, partials);
    });
    if(jacobi){
        cudaFreeAsync(z, stream);
    }
    cudaFreeAsync(dots, stream);
    return residual;
}

float cudaConjugateGradient(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& Anx, const CudaVec<float>& Apx, const CudaVec<float>& Any, const CudaVec<float>& Apy, const CudaVec<float>& Anz, const CudaVec<float>& Apz, const CudaVec<float>& Adiag,
    CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals, float tolerance, uint maxIterations, cudaStream_t stream){
    return diagonallyPreconditioned(false, solveCodes, makeStencil(neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag), divU, p, residuals, tolerance, maxIterations, stream);
}

float cudaJacobiConjugateGradient(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& Anx, const CudaVec<float>& Apx, const CudaVec<float>& Any, const CudaVec<float>& Apy, const CudaVec<float>& Anz, const CudaVec<float>& Apz, const CudaVec<float>& Adiag,
    CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals, float tolerance, uint maxIterations, cudaStream_t stream){
    return diagonallyPreconditioned(true, solveCodes, makeStencil(neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag), divU, p, residuals, tolerance, maxIterations, stream);
}

float cudaMultigridConjugateGradient(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& Anx, const CudaVec<float>& Apx, const CudaVec<float>& Any, const CudaVec<float>& Apy, const CudaVec<float>& Anz, const CudaVec<float>& Apz, const CudaVec<float>& Adiag,
    CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals, float tolerance, uint maxIterations, const VoxelLayout& layout, float scale, cudaStream_t stream){
    uint numUsedVoxels = p.size();
    Stencil A = makeStencil(neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag);
    Multigrid multigrid = buildMultigrid(A, solveCodes.devPtr(), numUsedVoxels, layout, scale, stream);
    float* z;
    DotProducts* dots;
    gpuErrchk(cudaMallocAsync((void**)&z, sizeof(float)*numUsedVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&dots, sizeof(DotProducts), stream));
    //a V-cycle costs dozens of plain iterations and it should take few of them, so it checks sooner
    float residual = conjugateGradient(solveCodes, A, divU, p, residuals, tolerance, maxIterations, 2, stream, z, dots, [&](const float* r, float* preconditioned, float* partials){
        vCycle(multigrid, r, preconditioned, stream);
        dotResidual<<<numUsedVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numUsedVoxels, r, preconditioned, partials);
    });
    cudaFreeAsync(z, stream);
    cudaFreeAsync(dots, stream);
    freeMultigrid(multigrid, stream);
    return residual;
}
