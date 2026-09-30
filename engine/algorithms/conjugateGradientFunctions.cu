//Copyright 2023 Aberrant Behavior LLC

//Conjugate gradient for the pressure equations, beside the SOR in voxelSolveFunctions.cu: the same unknowns, seven coefficient arrays and six neighbour
//arrays, solving A*p = b with b = -divU.
//
//SOR fixes one voxel at a time from its neighbours, so a correction spreads a voxel per sweep and smooth errors take many sweeps. CG moves every unknown
//at once, downhill on the energy f(p) = p*A*p/2 - b*p. Its lowest point is the solution, and its downhill direction at p is the residual r = b - A*p.
//Each iteration:
//  1. multiplies the search direction by A (q = A*d), its one stencil pass
//  2. steps along d by alpha = r*r / d*q, exactly as far as lowers the energy most along d. The residual then moves by -alpha*q, so it needs no second
//     stencil pass
//  3. turns the search direction towards the new residual, keeping beta = r*r(new) / r*r(old) of the old direction. That makes it A-conjugate to every
//     earlier direction, so minimizing along it never undoes the minimizing already done along them
//The dot products in alpha and beta are the only sums over every unknown. Each kernel adds its block's share into them as it goes, and they stay on the
//GPU, so an iteration never waits on the host

#include "conjugateGradientFunctions.hu"
#include <cmath>

struct DotProducts{     //doubles: float sums over a million voxels would lose the precision the step sizes need
    double rr;          //r*r of the current residual
    double dq;          //d*(A*d) of the current direction
    double rrNext;      //r*r of the residual after this iteration's step
};

struct Stencil{         //an unknown's row of A: its own coefficient and its six neighbours'. They're 0 across walls and air, so only unknowns count
    const uint* neighbors[6];
    const float* A[6];
    const float* Adiag;

    __device__ float rowTimes(uint index, const float* x) const{   //(A*x) at an unknown, the same row the SOR relaxes
        float sum = Adiag[index]*x[index];
        #pragma unroll
        for(int face = 0; face < 6; ++face){
            if(A[face][index] != 0.0f){
                sum += A[face][index]*x[neighbors[face][index]];
            }
        }
        return sum;
    }
};

//adds this block's share of a sum into total: every thread passes its value in (0 if it has none), warps sum theirs, then one warp sums the warps'
__device__ inline void addBlockSum(double value, double* total){
    __shared__ double warpSums[32];
    for(int lanes = 16; lanes > 0; lanes >>= 1){
        value += __shfl_xor_sync(0xffffffff, value, lanes);
    }
    if(threadIdx.x % 32 == 0){
        warpSums[threadIdx.x/32] = value;
    }
    __syncthreads();
    if(threadIdx.x < 32){
        value = threadIdx.x < blockDim.x/32 ? warpSums[threadIdx.x] : 0.0;
        for(int lanes = 16; lanes > 0; lanes >>= 1){
            value += __shfl_xor_sync(0xffffffff, value, lanes);
        }
        if(threadIdx.x == 0){
            atomicAdd(total, value);
        }
    }
}

//start: the residual of the starting guess, r = b - A*p, and the first search direction is straight downhill, d = r. Only unknowns have equations,
//so everything else stays 0 throughout
__global__ void startDownhill(uint numUsedVoxels, const char* solveCodes, Stencil A, const float* divU, const float* p, float* r, float* d, DotProducts* dots){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    float residual = 0.0f;
    if(index < numUsedVoxels){
        residual = solveCodes[index] ? -divU[index] - A.rowTimes(index, p) : 0.0f;
        r[index] = residual;
        d[index] = residual;
    }
    addBlockSum(residual*(double)residual, &dots->rr);
}

//1: q = A*d, and d*q, which sets how far to step along d
__global__ void multiplyByA(uint numUsedVoxels, const char* solveCodes, Stencil A, const float* d, float* q, DotProducts* dots){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    double dq = 0.0;
    if(index < numUsedVoxels){
        float product = solveCodes[index] ? A.rowTimes(index, d) : 0.0f;
        q[index] = product;
        dq = d[index]*(double)product;
    }
    addBlockSum(dq, &dots->dq);
}

//2: step along d by alpha = r*r / d*q, the distance that lowers the energy most along d. The residual moves by -alpha*A*d, which is q, already known
__global__ void stepDownhill(uint numUsedVoxels, const float* d, const float* q, float* p, float* r, DotProducts* dots){
    __shared__ float alpha;
    if(threadIdx.x == 0){
        alpha = dots->dq != 0.0 ? dots->rr / dots->dq : 0.0;
    }
    __syncthreads();
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    float residual = 0.0f;
    if(index < numUsedVoxels){
        p[index] += alpha*d[index];
        residual = r[index] - alpha*q[index];
        r[index] = residual;
    }
    addBlockSum(residual*(double)residual, &dots->rrNext);
}

//3: the next direction is downhill from here, plus beta = r*r(new) / r*r(old) of the last one, which keeps it A-conjugate to all the earlier directions
__global__ void turnDirection(uint numUsedVoxels, const float* r, float* d, const DotProducts* dots){
    __shared__ float beta;
    if(threadIdx.x == 0){
        beta = dots->rr != 0.0 ? dots->rrNext / dots->rr : 0.0;
    }
    __syncthreads();
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        d[index] = r[index] + beta*d[index];
    }
}

//the step's residual becomes the current one, and the sums start over
__global__ void nextIteration(DotProducts* dots){
    dots->rr = dots->rrNext;
    dots->dq = 0.0;
    dots->rrNext = 0.0;
}

float cudaConjugateGradient(const CudaVec<char>& solveCodes, const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz, const CudaVec<uint>& neighborPz,
    const CudaVec<float>& Anx, const CudaVec<float>& Apx, const CudaVec<float>& Any, const CudaVec<float>& Apy, const CudaVec<float>& Anz, const CudaVec<float>& Apz, const CudaVec<float>& Adiag,
    CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals, float tolerance, uint maxIterations, cudaStream_t stream){
    uint numUsedVoxels = p.size();
    uint blocks = numUsedVoxels / BLOCKSIZE + 1;
    uint checkEvery = 16;   //finding the largest residual waits on the host, so like the SOR, only every so often
    float maxDivergence = std::abs(divU.getMax(stream, true));
    if(maxDivergence == 0.0f){
        return 0.0f;
    }
    Stencil A = {{neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr()},
                 {Anx.devPtr(), Apx.devPtr(), Any.devPtr(), Apy.devPtr(), Anz.devPtr(), Apz.devPtr()}, Adiag.devPtr()};
    float* r = residuals.devPtr();
    float* d;   //the search direction
    float* q;   //A*d
    DotProducts* dots;
    gpuErrchk(cudaMallocAsync((void**)&d, sizeof(float)*numUsedVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&q, sizeof(float)*numUsedVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&dots, sizeof(DotProducts), stream));
    cudaMemsetAsync(dots, 0, sizeof(DotProducts), stream);
    startDownhill<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes.devPtr(), A, divU.devPtr(), p.devPtr(), r, d, dots);
    float maxResidual = 1.0f;
    for(uint iteration = 1; iteration <= maxIterations; ++iteration){
        multiplyByA<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes.devPtr(), A, d, q, dots);
        stepDownhill<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, d, q, p.devPtr(), r, dots);
        turnDirection<<<blocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, r, d, dots);
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
    cudaFreeAsync(dots, stream);
    return maxResidual;
}
