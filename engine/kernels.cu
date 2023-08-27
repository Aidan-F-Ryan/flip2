#include "kernels.hu"

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

__global__ void subCell(float* px, float* py, float* pz, uint numParticles, Grid grid, uint* gridPosition, char* subCellPosition, uint numRefinementLevels, uint xySize){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        uint moduloWRTxySize = gridPosition[index] % xySize;
        uint z = gridPosition[index] / (xySize);
        uint y = (moduloWRTxySize) / grid.sizeX;
        uint x = (moduloWRTxySize) % grid.sizeX;
        float gridCellPositionX = px[index] - grid.negX - x*grid.cellSize;
        float gridCellPositionY = py[index] - grid.negY - y*grid.cellSize;
        float gridCellPositionZ = pz[index] - grid.negZ - z*grid.cellSize;
        float curCellHalfSize = grid.cellSize/2.0f;
        for(uint i = 0; i < numRefinementLevels; ++i){
            char position = 0;
            position |= gridCellPositionX > curCellHalfSize ? 1 : 0;
            position |= gridCellPositionY > curCellHalfSize ? 2 : 0;
            position |= gridCellPositionZ > curCellHalfSize ? 4 : 0;
            subCellPosition[index*numRefinementLevels+i] = position;
            if(position & 1 != 0){
                gridCellPositionX -= curCellHalfSize;
            }
            if(position & 2 != 0){
                gridCellPositionY -= curCellHalfSize;
            }
            if(position & 4 != 0){
                gridCellPositionZ -= curCellHalfSize;
            }
            curCellHalfSize /= 2.0f;
        }
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

__global__ void parallelPrefix(uint numElements, uint* array, uint* blockSums){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    __shared__ uint shared[BLOCKSIZE];
    __shared__ uint blockSum;
    if(index < numElements){
        shared[threadIdx.x] = array[index];
    }
    else{
        shared[threadIdx.x] = 0;
    }
    for(uint i = 0; 1<<(i+1)-1 < BLOCKSIZE; ++i){
        __syncthreads();
        if(((threadIdx.x + 1)<<(i+1)) - 1 < BLOCKSIZE){
            shared[((threadIdx.x + 1)<<(i+1)) - 1] += shared[((threadIdx.x + 1)<<(i+1)) - 1 - (1<<i)];
        }
    }
    __syncthreads();

    if(threadIdx.x == 0){
        blockSum = shared[BLOCKSIZE - 1];
        blockSums[blockIdx.x] = blockSum;
        shared[BLOCKSIZE-1] = 0;
    }
    __syncthreads();
    
    for(int i = sizeof(uint)*8 - __clz(BLOCKSIZE>>1) - 1; i >= 0; --i){
        __syncthreads();
        if(((threadIdx.x + 1)<<(i+1)) - 1 < BLOCKSIZE){
            uint temp = shared[((threadIdx.x + 1)<<(i+1)) - 1];
            shared[((threadIdx.x + 1)<<(i+1)) - 1] += shared[((threadIdx.x + 1)<<(i+1)) - 1 - (1<<i)];
            shared[((threadIdx.x + 1)<<(i+1)) - 1 - (1<<i)] = temp;
        }
    }
    __syncthreads();
    if(index < numElements){
        if(threadIdx.x < blockDim.x-1){
            if(threadIdx.x == numElements-1){
                array[index] = blockSums[blockIdx.x];
            }
            else{
                array[index] = shared[threadIdx.x+1];
            }
        }
        else if(threadIdx.x == blockDim.x-1){
            array[index] = blockSum;
        }

    }
}

__global__ void parallelPrefixApplyPreviousBlockSum(uint numElements, uint* array, uint* blockSums){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    __shared__ uint prevBlockSum;
    if(threadIdx.x == 0){
        prevBlockSum = 0;
        for(uint i = 0; i < blockIdx.x; ++i){
            prevBlockSum += blockSums[i];
        }
    }
    __syncthreads();
    if(index < numElements){
        array[index] += prevBlockSum;
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

__global__ void reorderGridIndices(uint numParticles, uint* sortedParticleIndices, uint* gridPosition, uint* sortedGridPosition, char* subCellPosition, char* sortedSubCellPosition, uint numRefinementLevels){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        sortedGridPosition[index] = gridPosition[sortedParticleIndices[index]];
        for(uint i = 0; i < numRefinementLevels; ++i){
            sortedSubCellPosition[index*numRefinementLevels + i] = subCellPosition[sortedParticleIndices[index]*numRefinementLevels + i];
        }
    }
}

void kernels::cudaFindGridCell(float* px, float* py, float* pz, uint numParticles, Grid grid, uint* gridPosition, cudaStream_t stream){
    rootCell<<<numParticles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(px, py, pz, numParticles, grid, gridPosition);
}

void kernels::cudaFindSubCell(float* px, float* py, float* pz, uint numParticles, Grid grid, uint* gridPosition, char* subCellPosition, uint numRefinementLevels, cudaStream_t stream){
    subCell<<<numParticles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(px, py, pz, numParticles, grid, gridPosition, subCellPosition, numRefinementLevels, grid.sizeX*grid.sizeY);
}

void kernels::cudaSortParticles(uint numParticles, uint*& gridPosition, char*& subCellPosition, uint numRefinementLevels, cudaStream_t stream){
    uint* ogGridPosition = gridPosition;
    char* ogSubCellPosition = subCellPosition;

    uint* sortedGridPosition;
    uint* sortedParticleIndices;
    char* sortedSubCellPosition;
    uint* front;
    uint* back;
    uint* blockSumsFront;
    uint* blockSumsBack;

    cudaMalloc((void**)&sortedGridPosition, sizeof(uint)*numParticles);
    cudaMalloc((void**)&sortedParticleIndices, sizeof(uint)*numParticles);
    cudaMalloc((void**)&front, sizeof(uint)*numParticles);
    cudaMalloc((void**)&back, sizeof(uint)*numParticles);
    cudaMalloc((void**)&sortedSubCellPosition, sizeof(char)*numParticles*numRefinementLevels);
    cudaMalloc((void**)&blockSumsFront, sizeof(uint)*numParticles/BLOCKSIZE + 1);
    cudaMalloc((void**)&blockSumsBack, sizeof(uint)*numParticles/BLOCKSIZE + 1);

    cudaStream_t backStream;
    cudaStreamCreate(&backStream);
    for(uint i = 0; i < sizeof(uint)*8; ++i){
        radixBinParticlesByGridPositionBitIndex<<<numParticles/BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticles, gridPosition, i, front, back);
        cudaStreamSynchronize(stream);
        parallelPrefix<<<numParticles/BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticles, front, blockSumsFront);
        parallelPrefix<<<numParticles/BLOCKSIZE + 1, BLOCKSIZE, 0, backStream>>>(numParticles, back, blockSumsBack);
        parallelPrefixApplyPreviousBlockSum<<<numParticles/BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticles, front, blockSumsFront);
        parallelPrefixApplyPreviousBlockSum<<<numParticles/BLOCKSIZE + 1, BLOCKSIZE, 0, backStream>>>(numParticles, back, blockSumsBack);
        cudaStreamSynchronize(backStream);
        coalesceFrontBack<<<numParticles/BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticles, sortedParticleIndices, front, back);
        reorderGridIndices<<<numParticles/BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticles, sortedParticleIndices, gridPosition, sortedGridPosition, subCellPosition, sortedSubCellPosition, numRefinementLevels);
        cudaStreamSynchronize(stream);
        uint* tempGP = gridPosition;
        gridPosition = sortedGridPosition;
        sortedGridPosition = tempGP;

        char* tempSCP = subCellPosition;
        subCellPosition = sortedSubCellPosition;
        sortedSubCellPosition = tempSCP;
    }
    if(ogGridPosition != sortedGridPosition){
        cudaFree(sortedGridPosition);
    }
    if(ogSubCellPosition != sortedSubCellPosition){
        cudaFree(sortedSubCellPosition);
    }

    cudaFree(sortedParticleIndices);
    cudaFree(front);
    cudaFree(back);
    cudaFree(blockSumsFront);
    cudaFree(blockSumsBack);
}