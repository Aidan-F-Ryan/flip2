#include "kernels.hu"
#include "parallelPrefixSumKernels.hu"
#include "radixSortKernels.hu"
#include "reductionKernels.hu"
#include "../typedefs.h"


/**
 * @brief find root node containing each particle in domain
 * 
 * @param px 
 * @param py 
 * @param pz 
 * @param numParticles 
 * @param grid 
 * @param gridPosition 
 * @return __global__ 
 */

__global__ void rootCell(float* px, float* py, float* pz, uint numParticles, Grid grid, uint* gridPosition){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        uint x = floorf((px[index] - grid.negX) / grid.cellSize);
        uint y = floorf((py[index] - grid.negY) / grid.cellSize);
        uint z = floorf((pz[index] - grid.negZ) / grid.cellSize);
        gridPosition[index] = x + y*grid.sizeX + z*grid.sizeX*grid.sizeY;
    }
}

template <typename T>
__device__ T square(T in){
    return in*in;
}

/**
 * @brief Find subcell containing each particle inside its containing grid node
 * 
 * @param px 
 * @param py 
 * @param pz 
 * @param numParticles 
 * @param grid 
 * @param gridPosition 
 * @param subCellPositionX 
 * @param subCellPositionY 
 * @param subCellPositionZ 
 * @param refinementLevel 
 * @param xySize 
 * @return __global__ 
 */

__global__ void subCellCreateNumSubCellsTouchedEachDimension(float* px, float* py, float* pz, uint numParticles, Grid grid, uint* gridPosition, uint* subCellsTouchedX, uint* subCellsTouchedY, uint* subCellsTouchedZ, uint refinementLevel, float radius, uint xySize){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        uint moduloWRTxySize = gridPosition[index] % xySize;
        uint gridIDz = gridPosition[index] / (xySize);
        uint gridIDy = (moduloWRTxySize) / grid.sizeX;
        uint gridIDx = (moduloWRTxySize) % grid.sizeX;
        
        float pxInGridCell = (px[index] - grid.negX - gridIDx*grid.cellSize);
        float pyInGridCell = (py[index] - grid.negY - gridIDy*grid.cellSize);
        float pzInGridCell = (pz[index] - grid.negZ - gridIDz*grid.cellSize);
        float subCellWidth = grid.cellSize/(2.0f*(1<<refinementLevel));

        uint xTouched = 0;
        uint yTouched = 0;
        uint zTouched = 0;

        uint subCellPositionX = floorf(pxInGridCell/subCellWidth);
        uint subCellPositionY = floorf(pyInGridCell/subCellWidth);
        uint subCellPositionZ = floorf(pzInGridCell/subCellWidth);
        
        uint apronCells = floorf(radius);

        subCellPositionX += apronCells;
        subCellPositionY += apronCells;
        subCellPositionZ += apronCells;

        float halfSubCellWidth = subCellWidth / 2.0f;
        float radiusSCW_squared = square(radius*subCellWidth);

        for(uint x = subCellPositionX - apronCells; x < subCellPositionX + apronCells; ++x){
            for(uint y = subCellPositionY - apronCells; y < subCellPositionY + apronCells; ++y){
                for(uint z = subCellPositionZ - apronCells; z < subCellPositionZ + apronCells; ++z){
                    float subCellBaseX = x * subCellWidth;
                    float subCellBaseY = y * subCellWidth;
                    float subCellBaseZ = z * subCellWidth;

                    if(square(pxInGridCell - subCellBaseX)
                     + square(pyInGridCell - subCellBaseY + halfSubCellWidth)
                     + square(pzInGridCell - subCellBaseZ + halfSubCellWidth) < radiusSCW_squared){
                        ++xTouched;
                    }
                    if(square(pxInGridCell - subCellBaseX + halfSubCellWidth)
                     + square(pyInGridCell - subCellBaseY)
                     + square(pzInGridCell - subCellBaseZ + halfSubCellWidth) < radiusSCW_squared){
                        ++yTouched;
                    }
                    if(square(pxInGridCell - subCellBaseX + halfSubCellWidth)
                     + square(pyInGridCell - subCellBaseY + halfSubCellWidth)
                     + square(pzInGridCell - subCellBaseZ) < radiusSCW_squared){
                        ++zTouched;
                    }
                }
            }
        }

        subCellsTouchedX[index] = xTouched;
        subCellsTouchedY[index] = yTouched;
        subCellsTouchedZ[index] = zTouched;
    }
}

__global__ void subCellCreateLists(float* px, float* py, float* pz, uint numParticles, Grid grid, uint* gridPosition,
        uint* numSubCellsTouchedX, uint* numSubCellsTouchedY, uint* numSubCellsTouchedZ, uint* subCellsX, uint* subCellsY,
        uint* subCellsZ, uint refinementLevel, float radius, uint xySize){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        uint moduloWRTxySize = gridPosition[index] % xySize;
        uint gridIDz = gridPosition[index] / (xySize);
        uint gridIDy = (moduloWRTxySize) / grid.sizeX;
        uint gridIDx = (moduloWRTxySize) % grid.sizeX;
        
        float pxInGridCell = (px[index] - grid.negX - gridIDx*grid.cellSize);
        float pyInGridCell = (py[index] - grid.negY - gridIDy*grid.cellSize);
        float pzInGridCell = (pz[index] - grid.negZ - gridIDz*grid.cellSize);
        float subCellWidth = grid.cellSize/(2.0f*(1<<refinementLevel));

        uint subCellPositionX = floorf(pxInGridCell/subCellWidth);
        uint subCellPositionY = floorf(pyInGridCell/subCellWidth);
        uint subCellPositionZ = floorf(pzInGridCell/subCellWidth);
        
        uint apronCells = floorf(radius);

        subCellPositionX += apronCells;
        subCellPositionY += apronCells;
        subCellPositionZ += apronCells;

        float halfSubCellWidth = subCellWidth / 2.0f;
        float radiusSCW_squared = square(radius*subCellWidth);

        uint xWritten = 0;
        uint yWritten = 0;
        uint zWritten = 0;

        uint numVoxelsInNodeDimension = 2*apronCells + (2<<refinementLevel);

        uint subCellsTouchedStartX;
        uint subCellsTouchedStartY;
        uint subCellsTouchedStartZ;
        if(index == 0){
            subCellsTouchedStartX = 0;
            subCellsTouchedStartY = 0;
            subCellsTouchedStartZ = 0;
        }
        else{
            subCellsTouchedStartX = numSubCellsTouchedX[index - 1];
            subCellsTouchedStartY = numSubCellsTouchedY[index - 1];
            subCellsTouchedStartZ = numSubCellsTouchedZ[index - 1];
        }

        for(uint x = subCellPositionX - apronCells; x < subCellPositionX + apronCells; ++x){
            for(uint y = subCellPositionY - apronCells; y < subCellPositionY + apronCells; ++y){
                for(uint z = subCellPositionZ - apronCells; z < subCellPositionZ + apronCells; ++z){
                    float subCellBaseX = x * subCellWidth;
                    float subCellBaseY = y * subCellWidth;
                    float subCellBaseZ = z * subCellWidth;

                    if(square(pxInGridCell - subCellBaseX)
                     + square(pyInGridCell - subCellBaseY + halfSubCellWidth)
                     + square(pzInGridCell - subCellBaseZ + halfSubCellWidth) < radiusSCW_squared){
                        subCellsX[subCellsTouchedStartX + xWritten++] = x + y*numVoxelsInNodeDimension + z*numVoxelsInNodeDimension*numVoxelsInNodeDimension;

                    }
                    if(square(pxInGridCell - subCellBaseX + halfSubCellWidth)
                     + square(pyInGridCell - subCellBaseY)
                     + square(pzInGridCell - subCellBaseZ + halfSubCellWidth) < radiusSCW_squared){
                        subCellsY[subCellsTouchedStartY + yWritten++] = y + x*numVoxelsInNodeDimension + z * numVoxelsInNodeDimension * numVoxelsInNodeDimension;
                    }
                    if(square(pxInGridCell - subCellBaseX + halfSubCellWidth)
                     + square(pyInGridCell - subCellBaseY + halfSubCellWidth)
                     + square(pzInGridCell - subCellBaseZ) < radiusSCW_squared){
                        subCellsZ[subCellsTouchedStartZ + zWritten++] = z + x*numVoxelsInNodeDimension + y * numVoxelsInNodeDimension * numVoxelsInNodeDimension;
                    }
                }
            }
        }
    }
}

/**
 * @brief wrapper for rootCell
 * 
 * @param px 
 * @param py 
 * @param pz 
 * @param numParticles 
 * @param grid 
 * @param gridPosition 
 * @param stream 
 */


void kernels::cudaFindGridCell(float* px, float* py, float* pz, uint numParticles, Grid grid, uint* gridPosition, cudaStream_t stream){
    rootCell<<<numParticles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(px, py, pz, numParticles, grid, gridPosition);
}

void kernels::cudaFindSubCell(float* px, float* py, float* pz,
    uint numParticles, Grid grid, uint* gridPosition,
    uint* subCellsTouchedX, uint* subCellsTouchedY,
    uint* subCellsTouchedZ, CudaVec<uint>& subCellPositionX, 
    CudaVec<uint>& subCellPositionY, CudaVec<uint>& subCellPositionZ, 
    uint numRefinementLevels, float radius, cudaStream_t stream)
{
    cudaStream_t prefixSumStream;
    cudaStreamCreate(&prefixSumStream);

    uint* blockSumsSubCells = nullptr;

    cudaMallocAsync((void**)&blockSumsSubCells, sizeof(uint)*(numParticles / WORKSIZE + 1), prefixSumStream);

    subCellCreateNumSubCellsTouchedEachDimension<<<numParticles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>
        (px, py, pz, numParticles, grid, gridPosition, subCellsTouchedX, subCellsTouchedY, subCellsTouchedZ,
        numRefinementLevels, radius, grid.sizeX*grid.sizeY);
    

    cudaParallelPrefixSum(numParticles, subCellsTouchedX, blockSumsSubCells, prefixSumStream);
    cudaParallelPrefixSum(numParticles, subCellsTouchedY, blockSumsSubCells, prefixSumStream);
    cudaParallelPrefixSum(numParticles, subCellsTouchedZ, blockSumsSubCells, prefixSumStream);

    cudaFreeAsync(blockSumsSubCells, prefixSumStream);
    uint subCellListSizeX[1];
    uint subCellListSizeY[1];
    uint subCellListSizeZ[1];
    cudaMemcpyAsync(subCellListSizeX, subCellsTouchedX + numParticles - 1, sizeof(uint), cudaMemcpyDeviceToHost, prefixSumStream);
    cudaMemcpyAsync(subCellListSizeY, subCellsTouchedY + numParticles - 1, sizeof(uint), cudaMemcpyDeviceToHost, prefixSumStream);
    cudaMemcpyAsync(subCellListSizeZ, subCellsTouchedZ + numParticles - 1, sizeof(uint), cudaMemcpyDeviceToHost, prefixSumStream);
    cudaStreamSynchronize(prefixSumStream);

    subCellPositionX.resizeAsync(*subCellListSizeX, stream);
    subCellPositionY.resizeAsync(*subCellListSizeY, stream);
    subCellPositionZ.resizeAsync(*subCellListSizeZ, stream);

    subCellCreateLists<<<numParticles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(px, py, pz, numParticles, grid, gridPosition,
        subCellsTouchedX, subCellsTouchedY, subCellsTouchedZ, subCellPositionX.devPtr(), subCellPositionY.devPtr(), subCellPositionZ.devPtr(),
        numRefinementLevels, radius, grid.sizeX*grid.sizeY);

    cudaStreamDestroy(prefixSumStream);
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
void kernels::cudaParallelPrefixSum(uint numElements, T* array, T* blockSums, cudaStream_t stream){
    parallelPrefix<<<numElements/WORKSIZE + 1, BLOCKSIZE, 0, stream>>>(numElements, array, blockSums);
    if(numElements / WORKSIZE > 0){
        T* tempBlockSums;
        cudaMallocAsync((void**)&tempBlockSums, sizeof(T) * ((numElements/WORKSIZE + 1) / WORKSIZE + 1), stream);
        cudaParallelPrefixSum((numElements/WORKSIZE + 1), blockSums,  tempBlockSums, stream);
        cudaFreeAsync(tempBlockSums, stream);
    }
    parallelPrefixApplyPreviousBlockSum<<<numElements/WORKSIZE + 1, WORKSIZE, 0, stream>>>(numElements, array, blockSums);
}

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

void kernels::cudaRadixSortUint(uint numElements, uint* inArray, uint* outArray, uint* sortedIndices, uint* front, uint* back, cudaStream_t frontStream, cudaStream_t backStream, uint*& reorderedIndicesRelativeToOriginal){
    uint* tReordered;

    cudaMallocAsync((void**)&reorderedIndicesRelativeToOriginal, sizeof(uint) * numElements, backStream); //reordered indices relative to original position, for shuffling positions
    cudaMallocAsync((void**)&tReordered, sizeof(uint) * numElements, backStream); //reordered indices relative to original position, for shuffling positions

    for(uint i = 0; i < sizeof(uint)*8; ++i){
        radixBinUintByBitIndex<<<numElements/BLOCKSIZE + 1, BLOCKSIZE, 0, frontStream>>>(numElements, inArray, i, front, back);
        
        uint* blockSumsFront;
        uint* blockSumsBack;
        cudaMallocAsync((void**)&blockSumsFront, sizeof(uint)*numElements/WORKSIZE + 1, frontStream);
        cudaMallocAsync((void**)&blockSumsBack, sizeof(uint)*numElements/WORKSIZE + 1, backStream);

        cudaStreamSynchronize(frontStream);
        kernels::cudaParallelPrefixSum<uint>(numElements, front, blockSumsFront, frontStream);
        kernels::cudaParallelPrefixSum<uint>(numElements, back, blockSumsBack, backStream);

        cudaFreeAsync(blockSumsFront, frontStream);
        cudaFreeAsync(blockSumsBack, backStream);

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
    }
    cudaFreeAsync(tReordered, backStream);
}

/**
 * @brief Sort particles globally by containing root nodes
 * 
 * @param numParticles 
 * @param gridPosition 
 * @param stream 
 */

void kernels::cudaSortParticlesByGridNode(uint numParticles, uint*& gridPosition, uint*& reorderedIndicesRelativeToOriginal, cudaStream_t stream){
    uint* ogGridPosition = gridPosition;
    uint* ogReordered = reorderedIndicesRelativeToOriginal;

    uint* sortedGridPosition;
    uint* sortedParticleIndices;
    uint* front;
    uint* back;

    cudaMallocAsync((void**)&sortedGridPosition, sizeof(uint)*numParticles, stream);
    cudaMallocAsync((void**)&sortedParticleIndices, sizeof(uint)*numParticles, stream);
    cudaMallocAsync((void**)&front, sizeof(uint)*numParticles, stream);
    cudaMallocAsync((void**)&back, sizeof(uint)*numParticles, stream);

    cudaStream_t backStream;
    cudaStreamCreate(&backStream);
    
    cudaRadixSortUint(numParticles, gridPosition, sortedGridPosition, sortedParticleIndices, front, back, stream, backStream, reorderedIndicesRelativeToOriginal);

    if(ogGridPosition != sortedGridPosition){
        cudaFreeAsync(sortedGridPosition, stream);
    }

    // if(ogReordered != reorderedIndicesRelativeToOriginal){
        // cudaFree(reorderedIndicesRelativeToOriginal);
    // }

    cudaFreeAsync(sortedParticleIndices, stream);
    cudaFreeAsync(front, stream);
    cudaFreeAsync(back, stream);
}

__global__ void markUniqueGridCells(uint numElements, uint* gridCells, uint* uniqueGridNodes){
    uint index = threadIdx.x + blockDim.x * blockIdx.x;
    if(index < numElements){
        if(index == 0){
            uniqueGridNodes[index] = 1;
        }
        else if(gridCells[index] != gridCells[index - 1]){
            uniqueGridNodes[index] = 1;
        }
        else{
            uniqueGridNodes[index] = 0;
        }
    }
}

uint kernels::cudaMarkUniqueGridCellsAndCount(uint numParticles, uint* gridCells, uint* uniqueGridNodes, cudaStream_t stream){
    markUniqueGridCells<<<numParticles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticles, gridCells, uniqueGridNodes);
    
    uint* uniqueBlockSums;
    uint numGridNodes[1];
    
    cudaMallocAsync((void**)&uniqueBlockSums, sizeof(uint) * (numParticles / WORKSIZE + 1), stream);
    cudaParallelPrefixSum(numParticles,uniqueGridNodes, uniqueBlockSums, stream);
    cudaMemcpyAsync(numGridNodes, uniqueGridNodes + numParticles - 1, sizeof(uint), cudaMemcpyDeviceToHost, stream);

    cudaFreeAsync(uniqueBlockSums, stream);

    return numGridNodes[0];
}

__global__ void mapNodeIndicesToParticles(uint numParticles, uint* uniqueGridNodes, uint* gridNodeIndicesToFirstParticleIndex){
    uint index = threadIdx.x + blockDim.x*blockIdx.x;
    if(index < numParticles){
        if(index == 0){
            gridNodeIndicesToFirstParticleIndex[0] = 0;
        }
        else{
            if(uniqueGridNodes[index] != uniqueGridNodes[index - 1]){
                gridNodeIndicesToFirstParticleIndex[uniqueGridNodes[index] - 1] = index;
            }
        }
    }
}

void kernels::cudaMapNodeIndicesToParticles(uint numParticles, uint* uniqueGridNodes, uint* gridNodeIndicesToFirstParticleIndex, cudaStream_t stream){
    mapNodeIndicesToParticles<<<numParticles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticles, uniqueGridNodes, gridNodeIndicesToFirstParticleIndex);
}

__global__ void sumParticlesPerNode(uint numGridNodes, uint numParticles, uint* gridNodeIndicesToFirstParticleIndex, uint* subCellsTouchedPerParticle,
        uint* subCellsDim, uint* numNonZeroVoxels, uint* numParticlesInVoxelLists, uint numVoxelsPerNode){
    uint index = threadIdx.x + blockDim.x*blockIdx.x;

    extern __shared__ uint voxelCount[];
    __shared__ uint sums[BLOCKSIZE];
    __shared__ uint maxParticleNum;

    if(threadIdx.x == 0){
        if(blockIdx.x < numGridNodes - 1){
            maxParticleNum = gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1];
        }
        else{
            maxParticleNum = numParticles;
        }
    }
    for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        voxelCount[i] = 0;
    }

    __syncthreads();

    for(uint i = gridNodeIndicesToFirstParticleIndex[blockIdx.x] + threadIdx.x; i < maxParticleNum; i += blockDim.x){
        for(uint j = subCellsTouchedPerParticle[i]; j < subCellsTouchedPerParticle[i+1]; ++j){
            atomicAdd(voxelCount + subCellsDim[j], 1);
        }
    }
}

void kernels::cudaSumParticlesPerNodeAndWriteNumUsedVoxels(uint numGridNodes, uint numParticles, uint* gridNodeIndicesToFirstParticleIndex, uint* subCellsTouchedPerParticle, uint* subCellsDim, uint* numNonZeroVoxels,
    uint* numParticlesInVoxelLists, uint numVoxelsPerNode, cudaStream_t stream){
    sumParticlesPerNode<<<numGridNodes, BLOCKSIZE, sizeof(uint) * numVoxelsPerNode, stream>>>(
        numGridNodes, numParticles, gridNodeIndicesToFirstParticleIndex, subCellsTouchedPerParticle, subCellsDim, numNonZeroVoxels,
        numParticlesInVoxelLists, numVoxelsPerNode);
    gpuErrchk(cudaPeekAtLastError());
        
    }