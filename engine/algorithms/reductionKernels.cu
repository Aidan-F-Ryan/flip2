//Copyright Aberrant Behavior LLC 2023

#include "reductionKernels.hu"

__global__ void parallelReduction(uint numElements, uint* array, uint* outArray)
{
    __shared__ uint shared[WORKSIZE];
    shared[threadIdx.x] = 0;
    shared[threadIdx.x + blockDim.x] = 0;

    uint index = threadIdx.x + blockDim.x*blockIdx.x;
    if(index < numElements){
        shared[threadIdx.x] = array[index];
        if(index + blockDim.x < numElements){
            shared[threadIdx.x + blockDim.x] = array[index + blockDim.x];
        }

        for(uint i = blockDim.x; i > 0; i>>=1){
            __syncthreads();
            if(threadIdx.x < i){
                shared[threadIdx.x] += shared[threadIdx.x + i];
            }
        }

        if(threadIdx.x == 0){
            outArray[blockIdx.x] = shared[threadIdx.x];
        }
    }
}