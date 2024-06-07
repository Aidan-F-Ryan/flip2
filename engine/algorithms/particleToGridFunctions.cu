//Copyright 2023 Aberrant Behavior LLC

#include "particleToGridFunctions.hu"
#include "parallelPrefixSumKernels.hu"
#include "radixSortKernels.hu"

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

__global__ void rootCell(double* px, double* py, double* pz, uint numParticles, Grid grid, uint* gridPosition){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        if(px[index] < grid.negX){
            px[index] = grid.negX;
        }
        else if(px[index] > grid.negX + grid.sizeX*grid.cellSize){
            px[index] = grid.negX + grid.sizeX*grid.cellSize;
        }
        if(py[index] < grid.negY){
            py[index] = grid.negY;
        }
        else if(py[index] > grid.negY + grid.sizeY*grid.cellSize){
            py[index] = grid.negY + grid.sizeY*grid.cellSize;
        }
        if(pz[index] < grid.negZ){
            pz[index] = grid.negZ;
        }
        else if(pz[index] > grid.negZ + grid.sizeZ*grid.cellSize){
            pz[index] = grid.negZ + grid.sizeZ*grid.cellSize;
        }
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

__global__ void subCellCreateNumSubCellsTouchedEachDimension(double* px, double* py, double* pz, uint numParticles, Grid grid, uint* gridPosition, uint* subCellsTouchedX, uint* subCellsTouchedY, uint* subCellsTouchedZ, uint refinementLevel, double radius, uint xySize){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        uint moduloWRTxySize = gridPosition[index] % xySize;
        uint gridIDz = gridPosition[index] / (xySize);
        uint gridIDy = moduloWRTxySize / grid.sizeX;
        uint gridIDx = moduloWRTxySize % grid.sizeX;
        
        uint apronCells = floorf(radius);
        double subCellWidth = grid.cellSize/(2.0f*(1<<refinementLevel));
        double pxInGridCell = (px[index] - grid.negX - gridIDx*grid.cellSize + apronCells*subCellWidth);
        double pyInGridCell = (py[index] - grid.negY - gridIDy*grid.cellSize + apronCells*subCellWidth);
        double pzInGridCell = (pz[index] - grid.negZ - gridIDz*grid.cellSize + apronCells*subCellWidth);

        uint xTouched = 0;
        uint yTouched = 0;
        uint zTouched = 0;

        uint subCellPositionX = floorf(pxInGridCell/subCellWidth);
        uint subCellPositionY = floorf(pyInGridCell/subCellWidth);
        uint subCellPositionZ = floorf(pzInGridCell/subCellWidth);
        

        double halfSubCellWidth = subCellWidth / 2.0f;
        double radiusSCW = radius*subCellWidth;

        for(uint x = subCellPositionX - apronCells; x < subCellPositionX + apronCells; ++x){
            for(uint y = subCellPositionY - apronCells; y < subCellPositionY + apronCells; ++y){
                for(uint z = subCellPositionZ - apronCells; z < subCellPositionZ + apronCells; ++z){
                    double subCellBaseX = x * subCellWidth;
                    double subCellBaseY = y * subCellWidth;
                    double subCellBaseZ = z * subCellWidth;

                    double dpx = pxInGridCell - subCellBaseX;
                    double dpy = pyInGridCell - subCellBaseY;
                    double dpz = pzInGridCell - subCellBaseZ;

                    double xCheck = sqrtf(square(dpx) + square(dpy + halfSubCellWidth) + square(dpz + halfSubCellWidth));
                    double yCheck = sqrtf(square(dpx + halfSubCellWidth) + square(dpy) + square(dpz + halfSubCellWidth));
                    double zCheck = sqrtf(square(dpx + halfSubCellWidth) + square(dpy + halfSubCellWidth) + square(dpz));

                    if(xCheck < radiusSCW){
                        ++xTouched;
                    }
                    if(yCheck < radiusSCW){
                        ++yTouched;
                    }
                    if(zCheck < radiusSCW){
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

__global__ void subCellCreateLists(double* px, double* py, double* pz, uint numParticles, Grid grid, uint* gridPosition,
        uint* numSubCellsTouchedX, uint* numSubCellsTouchedY, uint* numSubCellsTouchedZ, uint* subCellsX, uint* subCellsY,
        uint* subCellsZ, uint refinementLevel, double radius, uint xySize){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        uint moduloWRTxySize = gridPosition[index] % xySize;
        uint gridIDz = gridPosition[index] / (xySize);
        uint gridIDy = moduloWRTxySize / grid.sizeX;
        uint gridIDx = moduloWRTxySize % grid.sizeX;
        
        uint apronCells = floorf(radius);
        double subCellWidth = grid.cellSize/(2.0f*(1<<refinementLevel));
        double pxInGridCell = (px[index] - grid.negX - gridIDx*grid.cellSize + apronCells*subCellWidth);    //including apron cell width offset
        double pyInGridCell = (py[index] - grid.negY - gridIDy*grid.cellSize + apronCells*subCellWidth);
        double pzInGridCell = (pz[index] - grid.negZ - gridIDz*grid.cellSize + apronCells*subCellWidth);

        uint subCellPositionX = floorf(pxInGridCell/subCellWidth);
        uint subCellPositionY = floorf(pyInGridCell/subCellWidth);
        uint subCellPositionZ = floorf(pzInGridCell/subCellWidth);
        

        double halfSubCellWidth = subCellWidth / 2.0f;
        double radiusSCW = radius*subCellWidth;

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
                    double subCellBaseX = x * subCellWidth;
                    double subCellBaseY = y * subCellWidth;
                    double subCellBaseZ = z * subCellWidth;
                    
                    double dpx = pxInGridCell - subCellBaseX;
                    double dpy = pyInGridCell - subCellBaseY;
                    double dpz = pzInGridCell - subCellBaseZ;

                    double xCheck = sqrtf(square(dpx) + square(dpy + halfSubCellWidth) + square(dpz + halfSubCellWidth));
                    double yCheck = sqrtf(square(dpx + halfSubCellWidth) + square(dpy) + square(dpz + halfSubCellWidth));
                    double zCheck = sqrtf(square(dpx + halfSubCellWidth) + square(dpy + halfSubCellWidth) + square(dpz));

                    if(xCheck < radiusSCW){
                        subCellsX[subCellsTouchedStartX + xWritten++] = x + y*numVoxelsInNodeDimension + z*numVoxelsInNodeDimension*numVoxelsInNodeDimension; //x is x, y is y, z is z
                    }
                    if(yCheck < radiusSCW){
                        subCellsY[subCellsTouchedStartY + yWritten++] = x + y*numVoxelsInNodeDimension + z * numVoxelsInNodeDimension * numVoxelsInNodeDimension; //y is x, x is y, z is z
                    }
                    if(zCheck < radiusSCW){
                        subCellsZ[subCellsTouchedStartZ + zWritten++] = x + y*numVoxelsInNodeDimension + z * numVoxelsInNodeDimension * numVoxelsInNodeDimension; //z is x, x is y, y is z
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


void cudaFindGridCell(double* px, double* py, double* pz, uint numParticles, Grid grid, uint* gridPosition, cudaStream_t stream){
    rootCell<<<numParticles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(px, py, pz, numParticles, grid, gridPosition);
    cudaStreamSynchronize(stream);
}

void cudaFindSubCell(double* px, double* py, double* pz,
    uint numParticles, Grid grid, uint* gridPosition,
    uint* subCellsTouchedX, uint* subCellsTouchedY,
    uint* subCellsTouchedZ, CudaVec<uint>& subCellPositionX, 
    CudaVec<uint>& subCellPositionY, CudaVec<uint>& subCellPositionZ, 
    uint numRefinementLevels, double radius, cudaStream_t stream)
{
    cudaStream_t prefixSumStream;
    cudaStreamCreate(&prefixSumStream);

    subCellCreateNumSubCellsTouchedEachDimension<<<numParticles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>
        (px, py, pz, numParticles, grid, gridPosition, subCellsTouchedX, subCellsTouchedY, subCellsTouchedZ,
        numRefinementLevels, radius, grid.sizeX*grid.sizeY);
    cudaStreamSynchronize(stream);
    

    cudaParallelPrefixSum(numParticles, subCellsTouchedX, prefixSumStream);
    cudaParallelPrefixSum(numParticles, subCellsTouchedY, prefixSumStream);
    cudaParallelPrefixSum(numParticles, subCellsTouchedZ, prefixSumStream);
    cudaStreamSynchronize(prefixSumStream);
    
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
    cudaStreamSynchronize(stream);

    subCellCreateLists<<<numParticles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(px, py, pz, numParticles, grid, gridPosition,
        subCellsTouchedX, subCellsTouchedY, subCellsTouchedZ, subCellPositionX.devPtr(), subCellPositionY.devPtr(), subCellPositionZ.devPtr(),
        numRefinementLevels, radius, grid.sizeX*grid.sizeY);

    cudaStreamDestroy(prefixSumStream);
}

/**
 * @brief Sort particles globally by containing root nodes
 * 
 * @param numParticles 
 * @param gridPosition 
 * @param stream 
 */

void cudaSortParticlesByGridNode(uint numParticles, uint*& gridPosition, uint*& reorderedIndicesRelativeToOriginal, cudaStream_t stream){
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
    // cudaStreamSynchronize(stream);

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

uint cudaMarkUniqueGridCellsAndCount(uint numParticles, uint* gridCells, uint* uniqueGridNodes, cudaStream_t stream){
    markUniqueGridCells<<<numParticles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticles, gridCells, uniqueGridNodes);
    
    uint numGridNodes;
    
    cudaParallelPrefixSum(numParticles,uniqueGridNodes, stream);
    cudaMemcpyAsync(&numGridNodes, uniqueGridNodes + numParticles - 1, sizeof(uint), cudaMemcpyDeviceToHost, stream);

    return numGridNodes;
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

void cudaMapNodeIndicesToParticles(uint numParticles, uint* uniqueGridNodes, uint* gridNodeIndicesToFirstParticleIndex, cudaStream_t stream){
    mapNodeIndicesToParticles<<<numParticles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticles, uniqueGridNodes, gridNodeIndicesToFirstParticleIndex);
}

__global__ void zeroYDimArray(Grid grid, uint numUniqueGridNodes, uint* yDimFirstNodeIndex){
    uint index = threadIdx.x + blockDim.x*blockIdx.x;
    if(index < grid.sizeX*grid.sizeY){
        yDimFirstNodeIndex[index] = numUniqueGridNodes;
    }
}

//find first node index of each used y-dimension row in memory, allows for y dimension-scan)
__global__ void getYDimPositionFirstInRow(uint numUniqueGridNodes, uint* gridNodeIndicesToFirstParticleIndex, uint* gridNodeIDs, uint* yDimFirstNodeIndex, Grid grid){
    uint index = threadIdx.x + blockDim.x*blockIdx.x;
    if(index < numUniqueGridNodes){
        uint prevGridNodeUniqueIndex;
        if(index != 0){
            prevGridNodeUniqueIndex = gridNodeIDs[gridNodeIndicesToFirstParticleIndex[index - 1]];
        }
        else{
            prevGridNodeUniqueIndex = numUniqueGridNodes;
        }
        uint gridNodeUniqueIndex = gridNodeIDs[gridNodeIndicesToFirstParticleIndex[index]];
        uint prevYRow = prevGridNodeUniqueIndex / grid.sizeX;
        uint yRow = gridNodeUniqueIndex / grid.sizeX;
        if(prevYRow != yRow){
            yDimFirstNodeIndex[yRow] = gridNodeUniqueIndex;
        }
    }
}

void cudaGetFirstNodeInYRows(uint numUniqueGridNodes, uint* gridNodeIndicesToFirstParticleIndex, uint* gridNodeIDs, uint* yDimFirstNodeIndex, const Grid& grid, cudaStream_t stream){
    zeroYDimArray<<<grid.sizeX*grid.sizeY / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(grid, numUniqueGridNodes, yDimFirstNodeIndex);
    cudaStreamSynchronize(stream);
    getYDimPositionFirstInRow<<<numUniqueGridNodes / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numUniqueGridNodes, gridNodeIndicesToFirstParticleIndex, gridNodeIDs, yDimFirstNodeIndex, grid);
}