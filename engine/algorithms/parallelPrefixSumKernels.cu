//Copyright 2023 Aberrant Behavior LLC

#include "parallelPrefixSumKernels.hu"

__global__ void parallelPrefix(uint numElements, uint* array, uint* blockSums){ // BLOCKSIZE threads operating on WORKSIZE (2x BLOCKSIZE) elements
    uint index = threadIdx.x + blockIdx.x*WORKSIZE;
    __shared__ uint shared[SM_ADDRESS(WORKSIZE)];
    __shared__ uint blockSum;
    if(index < numElements){
        shared[SM_ADDRESS(threadIdx.x)] = array[index];
    }
    else{
        shared[SM_ADDRESS(threadIdx.x)] = 0;
    }
    if(index + blockDim.x < numElements){
        shared[SM_ADDRESS(threadIdx.x + blockDim.x)] = array[index + blockDim.x];
    }
    else{
        shared[SM_ADDRESS(threadIdx.x + blockDim.x)] = 0;
    }
    blockWiseExclusivePrefixSum(shared, WORKSIZE, blockSum);
    blockSums[blockIdx.x] = blockSum;
    if(index < numElements){    //copy first of the two values handled by each thread
        array[index] = shared[SM_ADDRESS(threadIdx.x + 1)];
    }
    if(index + blockDim.x < numElements){
        if(threadIdx.x == blockDim.x-1){
            array[index + blockDim.x] = blockSum;
        }
        else {
            array[index + blockDim.x] = shared[SM_ADDRESS(threadIdx.x + blockDim.x + 1)];
        }

    }
}

__global__ void parallelPrefixApplyPreviousBlockSum(uint numElements, uint* array, uint* blockSums){ //designed for WORKSIZE threads per block, 2x BLOCKSIZE
    uint index = threadIdx.x + (blockIdx.x)*blockDim.x;
    if(blockIdx.x > 0 && index < numElements){    //STORE
        array[index] += blockSums[blockIdx.x - 1];
    }
}

/**
 * @brief Wrapper for executing CUDA parallel prefix sum
 * 
 * @tparam T 
 * @param numElements 
 * @param array 
 * @param blockSums 
 * @param stream 
 */

template <typename T>
void cudaParallelPrefixSum_internal(uint numElements, T* array, T* blockSums, cudaStream_t stream){
    parallelPrefix<<<numElements/WORKSIZE + 1, BLOCKSIZE, 0, stream>>>(numElements, array, blockSums);
    gpuErrchk(cudaPeekAtLastError());
    if(numElements / WORKSIZE > 0){
        cudaParallelPrefixSum_internal((numElements/WORKSIZE + 1), blockSums, blockSums + numElements / WORKSIZE + 1, stream);
    }
    parallelPrefixApplyPreviousBlockSum<<<numElements/WORKSIZE + 1, WORKSIZE, 0, stream>>>(numElements, array, blockSums);
    gpuErrchk(cudaPeekAtLastError());
}

template <typename T>
void cudaParallelPrefixSum(uint numElements, T* array, cudaStream_t stream){
    uint totalBlockSumsSize = 0;
    T* blockSums;
    for(uint i = numElements / WORKSIZE + 1; i >= 1; i = i / WORKSIZE + 1){
        totalBlockSumsSize += i;
        if(i == 1)
            break;
    }
    gpuErrchk(cudaMallocAsync((void**)&blockSums, sizeof(T) * totalBlockSumsSize, stream));
    cudaParallelPrefixSum_internal(numElements, array, blockSums, stream);
    gpuErrchk(cudaFreeAsync(blockSums, stream));
}

template void cudaParallelPrefixSum(uint numElements, uint* array, cudaStream_t stream);