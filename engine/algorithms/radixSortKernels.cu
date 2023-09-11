#include "radixSortKernels.hu"

__global__ void radixBinUintByBitIndex(uint numElements, uint* inArray, uint bitIndex, uint* front, uint* back){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numElements){
        if((inArray[index] & (1<<bitIndex)) == 0){
            front[index] = 1;
            back[index] = 0;
        }
        else{
            front[index] = 0;
            back[index] = 1;
        }
    }
    
}


__global__ void coalesceFrontBack(uint numElements, uint* outArray, uint* front, uint* back){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    __shared__ uint maxFrontIndex;
    __shared__ uint thisBlockFront[BLOCKSIZE+1];
    __shared__ uint thisBlockBack[BLOCKSIZE+1];
    if(threadIdx.x == 0 && index < numElements){
        maxFrontIndex = front[numElements-1];
        if((int)index - 1 >= 0){
            thisBlockFront[0] = front[index - 1];
            thisBlockBack[0] = back[index - 1];
        }
        else{
            thisBlockFront[0] = 0;
            thisBlockBack[0] = 0;
        }
    }
    __syncthreads();
    if(index < numElements){
        thisBlockFront[threadIdx.x + 1] = front[index];
        thisBlockBack[threadIdx.x + 1] = back[index];
    }
    __syncthreads();
    if(index < numElements){
        if(thisBlockFront[threadIdx.x] != thisBlockFront[threadIdx.x+1]){
            outArray[thisBlockFront[threadIdx.x]] = index;
        }
        if(thisBlockBack[threadIdx.x] != thisBlockBack[threadIdx.x+1]){
            outArray[thisBlockBack[threadIdx.x] + maxFrontIndex] = index;
        }
    }
}


template <typename T>
__global__ void reorderGridIndices(uint numElements, uint* sortedIndices, T* inArray, T* outArray){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numElements){
        outArray[index] = inArray[sortedIndices[index]];
    }
}

template __global__ void reorderGridIndices(uint numElements, uint *sortedIndices, uint* inArray, uint* outArray);
template __global__ void reorderGridIndices(uint numElements, uint *sortedIndices, float* inArray, float* outArray);