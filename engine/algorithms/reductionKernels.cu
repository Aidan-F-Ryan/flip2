//Copyright 2023 Aberrant Behavior LLC

#include "parallelPrefixSumKernels.hu"
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

template <typename T>
__global__ void parallelMaximum(uint numElements, T* array){
    __shared__ T shared[WORKSIZE];
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numElements){
        shared[threadIdx.x] = array[index];
    }
    blockwiseMaximum(shared, WORKSIZE);
    __syncthreads();
    if(threadIdx.x == 0){
        array[index] = shared[0];
    }
}

template<typename T>
__global__ void coalesceCoalesceCOALESCE(uint numElements, T* array, uint stride){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    uint stridedIndex = index*stride;
    if(stridedIndex < numElements){
        array[index] = array[stridedIndex];
    }
}

template <typename T>
void cudaParallelMaximum(uint numElements, T* array, cudaStream_t stream){
    for(uint elementsToProcess = numElements; elementsToProcess > 1; elementsToProcess /= WORKSIZE){
        parallelMaximum<<<elementsToProcess / WORKSIZE + 1, WORKSIZE, 0, stream>>>(elementsToProcess, array);
        cudaStreamSynchronize(stream);
        coalesceCoalesceCOALESCE<<<elementsToProcess / WORKSIZE / WORKSIZE + 1, WORKSIZE, 0, stream>>>(elementsToProcess, array, WORKSIZE);
    }
}

template void cudaParallelMaximum(uint, float*, cudaStream_t);
template void cudaParallelMaximum(uint, uint*, cudaStream_t);
template void cudaParallelMaximum(uint, int*, cudaStream_t);
template void cudaParallelMaximum(uint, char*, cudaStream_t);
