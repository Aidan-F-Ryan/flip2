//Copyright 2023 Aberrant Behavior LLC

#include "radixSortKernels.hu"
#include "parallelPrefixSumKernels.hu"

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
        if(index > 0){
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
        // if(thisBlockBack[threadIdx.x] != thisBlockBack[threadIdx.x+1]){
        else{
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

/**
 * @brief Wrapper for performing CUDA radix inclusive sort on uint array
 * 
 * @param numElements 
 * @param inArray 
 * @param outArray 
 * @param sortedIndices 
 * @param front 
 * @param back 
 * @param blockSumsFront 
 * @param blockSumsBack 
 * @param frontStream 
 * @param backStream 
 */

void cudaRadixSortUint(uint numElements, uint* inArray, uint* outArray, uint* sortedIndices, uint* front, uint* back, cudaStream_t frontStream, cudaStream_t backStream, uint*& reorderedIndicesRelativeToOriginal){
    uint* tReordered;

    cudaMallocAsync((void**)&reorderedIndicesRelativeToOriginal, sizeof(uint) * numElements, backStream); //reordered indices relative to original position, for shuffling positions
    cudaMallocAsync((void**)&tReordered, sizeof(uint) * numElements, backStream); //reordered indices relative to original position, for shuffling positions

    for(uint i = 0; i < sizeof(uint)*8; ++i){
        radixBinUintByBitIndex<<<numElements/BLOCKSIZE + 1, BLOCKSIZE, 0, frontStream>>>(numElements, inArray, i, front, back);
        
        cudaStreamSynchronize(frontStream);
        cudaParallelPrefixSum<uint>(numElements, front, frontStream);
        cudaStreamSynchronize(frontStream);
        cudaParallelPrefixSum<uint>(numElements, back, backStream);

        cudaStreamSynchronize(backStream);
        
        coalesceFrontBack<<<numElements/BLOCKSIZE + 1, BLOCKSIZE, 0, frontStream>>>(numElements, sortedIndices, front, back);
        reorderGridIndices<<<numElements/BLOCKSIZE + 1, BLOCKSIZE, 0, frontStream>>>(numElements, sortedIndices, inArray, outArray);

        cudaStreamSynchronize(frontStream);
        
        if(i == 0){
            cudaMemcpyAsync(reorderedIndicesRelativeToOriginal, sortedIndices, sizeof(uint)*numElements, cudaMemcpyDeviceToDevice, frontStream);
        }
        else{
            reorderGridIndices<<<numElements/BLOCKSIZE + 1, BLOCKSIZE, 0, frontStream>>>(numElements, sortedIndices, reorderedIndicesRelativeToOriginal, tReordered);
            
            uint* temp = tReordered;
            tReordered = reorderedIndicesRelativeToOriginal;
            reorderedIndicesRelativeToOriginal = temp;
        }

        uint* tempGP = inArray;
        inArray = outArray;
        outArray = tempGP;
        cudaStreamSynchronize(frontStream);
    }
    cudaFreeAsync(tReordered, backStream);
}