#include "kernels.hu"
#include "parallelPrefixSumKernels.hu"

#include <stdio.h>

__global__ void rootCell(float* px, float* py, float* pz, uint numParticles, Grid grid, uint* gridPosition){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        uint x = floorf((px[index] - grid.negX) / grid.cellSize);
        uint y = floorf((py[index] - grid.negY) / grid.cellSize);
        uint z = floorf((pz[index] - grid.negZ) / grid.cellSize);
        gridPosition[index] = x + y*grid.sizeX + z*grid.sizeX*grid.sizeY;
    }
}

__global__ void subCell(float* px, float* py, float* pz, uint numParticles, Grid grid, uint* gridPosition, uint* subCellPositionX, uint* subCellPositionY, uint* subCellPositionZ, uint refinementLevel, uint xySize){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        uint moduloWRTxySize = gridPosition[index] % xySize;
        uint z = gridPosition[index] / (xySize);
        uint y = (moduloWRTxySize) / grid.sizeX;
        uint x = (moduloWRTxySize) % grid.sizeX;
        
        float gridCellPositionX = (px[index] - grid.negX - x*grid.cellSize);
        float gridCellPositionY = (py[index] - grid.negY - y*grid.cellSize);
        float gridCellPositionZ = (pz[index] - grid.negZ - z*grid.cellSize);
        float curSubCellSize = grid.cellSize/(2.0f*(1<<refinementLevel));

        subCellPositionX[index] = floorf(gridCellPositionX/curSubCellSize);
        subCellPositionY[index] = floorf(gridCellPositionY/curSubCellSize);
        subCellPositionZ[index] = floorf(gridCellPositionZ/curSubCellSize);
        
    }
}

__global__ void radixBinParticlesByGridPositionBitIndex(uint numParticles, uint* gridPosition, uint bitIndex, uint* front, uint* back){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        if((gridPosition[index] & (1<<bitIndex)) == 0){
            front[index] = 1;
            back[index] = 0;
        }
        else{
            front[index] = 0;
            back[index] = 1;
        }
    }
    
}


__global__ void coalesceFrontBack(uint numParticles, uint* sortedParticleIndices, uint* front, uint* back){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    __shared__ uint maxFrontIndex;
    __shared__ uint thisBlockFront[BLOCKSIZE+1];
    __shared__ uint thisBlockBack[BLOCKSIZE+1];
    if(threadIdx.x == 0 && index < numParticles){
        maxFrontIndex = front[numParticles-1];
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
    if(index < numParticles){
        thisBlockFront[threadIdx.x + 1] = front[index];
        thisBlockBack[threadIdx.x + 1] = back[index];
    }
    __syncthreads();
    if(index < numParticles){
        if(thisBlockFront[threadIdx.x] != thisBlockFront[threadIdx.x+1]){
            sortedParticleIndices[thisBlockFront[threadIdx.x]] = index;
        }
        if(thisBlockBack[threadIdx.x] != thisBlockBack[threadIdx.x+1]){
            sortedParticleIndices[thisBlockBack[threadIdx.x] + maxFrontIndex] = index;
        }
    }
}

__global__ void reorderGridIndices(uint numParticles, uint* sortedParticleIndices, uint* gridPosition, uint* sortedGridPosition){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        sortedGridPosition[index] = gridPosition[sortedParticleIndices[index]];
    }
}

void kernels::cudaFindGridCell(float* px, float* py, float* pz, uint numParticles, Grid grid, uint* gridPosition, cudaStream_t stream){
    rootCell<<<numParticles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(px, py, pz, numParticles, grid, gridPosition);
}

void kernels::cudaFindSubCell(float* px, float* py, float* pz, uint numParticles, Grid grid, uint* gridPosition, uint* subCellPositionX, uint* subCellPositionY, uint* subCellPositionZ, uint numRefinementLevels, cudaStream_t stream){
    subCell<<<numParticles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(px, py, pz, numParticles, grid, gridPosition, subCellPositionX, subCellPositionY, subCellPositionZ, numRefinementLevels, grid.sizeX*grid.sizeY);
}

template <typename T>
void kernels::cudaParallelPrefixSum(uint numElements, T* array, T* blockSums, cudaStream_t stream){
    parallelPrefix<<<numElements/BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numElements, array, blockSums);
    parallelPrefixApplyPreviousBlockSum<<<numElements/BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numElements, array, blockSums);
}

void kernels::cudaSortParticles(uint numParticles, uint*& gridPosition, cudaStream_t stream){
    uint* ogGridPosition = gridPosition;

    uint* sortedGridPosition;
    uint* sortedParticleIndices;
    uint* front;
    uint* back;
    uint* blockSumsFront;
    uint* blockSumsBack;

    cudaMalloc((void**)&sortedGridPosition, sizeof(uint)*numParticles);
    cudaMalloc((void**)&sortedParticleIndices, sizeof(uint)*numParticles);
    cudaMalloc((void**)&front, sizeof(uint)*numParticles);
    cudaMalloc((void**)&back, sizeof(uint)*numParticles);
    cudaMalloc((void**)&blockSumsFront, sizeof(uint)*numParticles/BLOCKSIZE + 1);
    cudaMalloc((void**)&blockSumsBack, sizeof(uint)*numParticles/BLOCKSIZE + 1);

    cudaStream_t backStream;
    cudaStreamCreate(&backStream);
    for(uint i = 0; i < sizeof(uint)*8; ++i){
        radixBinParticlesByGridPositionBitIndex<<<numParticles/BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticles, gridPosition, i, front, back);
        cudaStreamSynchronize(stream);
        kernels::cudaParallelPrefixSum<uint>(numParticles, front, blockSumsFront, stream);
        kernels::cudaParallelPrefixSum<uint>(numParticles, back, blockSumsBack, backStream);
        cudaStreamSynchronize(backStream);
        coalesceFrontBack<<<numParticles/BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticles, sortedParticleIndices, front, back);
        reorderGridIndices<<<numParticles/BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticles, sortedParticleIndices, gridPosition, sortedGridPosition);
        cudaStreamSynchronize(stream);
        uint* tempGP = gridPosition;
        gridPosition = sortedGridPosition;
        sortedGridPosition = tempGP;
    }
    if(ogGridPosition != sortedGridPosition){
        cudaFree(sortedGridPosition);
    }

    cudaFree(sortedParticleIndices);
    cudaFree(front);
    cudaFree(back);
    cudaFree(blockSumsFront);
    cudaFree(blockSumsBack);
}