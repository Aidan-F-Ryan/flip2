#include "parallelPrefixSumKernels.hu"

__global__ void parallelPrefix(uint numElements, uint* array, uint* blockSums){ // BLOCKSIZE threads operating on WORKSIZE (2x BLOCKSIZE) elements
    uint index = threadIdx.x + blockIdx.x*WORKSIZE;
    __shared__ uint shared[WORKSIZE];
    __shared__ uint blockSum;
    if(index < numElements){
        shared[threadIdx.x] = array[index];
    }
    else{
        shared[threadIdx.x] = 0;
    }
    if(index + blockDim.x < numElements){
        shared[threadIdx.x + blockDim.x] = array[index + blockDim.x];
    }
    else{
        shared[threadIdx.x + blockDim.x] = 0;
    }
    for(uint i = 0; 1<<(i+1)-1 < WORKSIZE; ++i){
        __syncthreads();
        if(((threadIdx.x + 1)<<(i+1)) - 1 < WORKSIZE){
            shared[((threadIdx.x + 1)<<(i+1)) - 1] += shared[((threadIdx.x + 1)<<(i+1)) - 1 - (1<<i)];
        }
    }
    __syncthreads();

    if(threadIdx.x == 0){
        blockSum = shared[WORKSIZE - 1];
        blockSums[blockIdx.x] = blockSum;
        shared[WORKSIZE-1] = 0;
    }
    __syncthreads();
    
    for(int i = sizeof(uint)*8 - __clz(WORKSIZE>>1) - 1; i >= 0; --i){
        __syncthreads();
        if(((threadIdx.x + 1)<<(i+1)) - 1 < WORKSIZE){
            uint temp = shared[((threadIdx.x + 1)<<(i+1)) - 1];
            shared[((threadIdx.x + 1)<<(i+1)) - 1] += shared[((threadIdx.x + 1)<<(i+1)) - 1 - (1<<i)];
            shared[((threadIdx.x + 1)<<(i+1)) - 1 - (1<<i)] = temp;
        }
    }
    __syncthreads();
    if(index < numElements){    //copy first of the two values handled by each thread
        array[index] = shared[threadIdx.x + 1];
    }
    if(index + blockDim.x < numElements){
        if(threadIdx.x == blockDim.x-1){
            array[index + blockDim.x] = blockSum;
        }
        else {
            array[index + blockDim.x] = shared[threadIdx.x + blockDim.x + 1];
        }

    }
}

__global__ void parallelPrefixApplyPreviousBlockSum(uint numElements, uint* array, uint* blockSums){ //designed for WORKSIZE threads per block, 2x BLOCKSIZE
    uint index = threadIdx.x + (blockIdx.x)*blockDim.x;
    if(blockIdx.x > 0 && index < numElements){    //STORE
        array[index] += blockSums[blockIdx.x - 1];
    }
}