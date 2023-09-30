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
        uint gridIDy = moduloWRTxySize / grid.sizeX;
        uint gridIDx = moduloWRTxySize % grid.sizeX;
        
        uint apronCells = floorf(radius);
        float subCellWidth = grid.cellSize/(2.0f*(1<<refinementLevel));
        float pxInGridCell = (px[index] - grid.negX - gridIDx*grid.cellSize + apronCells*subCellWidth);
        float pyInGridCell = (py[index] - grid.negY - gridIDy*grid.cellSize + apronCells*subCellWidth);
        float pzInGridCell = (pz[index] - grid.negZ - gridIDz*grid.cellSize + apronCells*subCellWidth);

        uint xTouched = 0;
        uint yTouched = 0;
        uint zTouched = 0;

        uint subCellPositionX = floorf(pxInGridCell/subCellWidth);
        uint subCellPositionY = floorf(pyInGridCell/subCellWidth);
        uint subCellPositionZ = floorf(pzInGridCell/subCellWidth);
        

        float halfSubCellWidth = subCellWidth / 2.0f;
        float radiusSCW = radius*subCellWidth;

        for(uint x = subCellPositionX - apronCells; x < subCellPositionX + apronCells; ++x){
            for(uint y = subCellPositionY - apronCells; y < subCellPositionY + apronCells; ++y){
                for(uint z = subCellPositionZ - apronCells; z < subCellPositionZ + apronCells; ++z){
                    float subCellBaseX = x * subCellWidth;
                    float subCellBaseY = y * subCellWidth;
                    float subCellBaseZ = z * subCellWidth;

                    float dpx = pxInGridCell - subCellBaseX;
                    float dpy = pyInGridCell - subCellBaseY;
                    float dpz = pzInGridCell - subCellBaseZ;

                    float xCheck = sqrtf(square(dpx) + square(dpy + halfSubCellWidth) + square(dpz + halfSubCellWidth));
                    float yCheck = sqrtf(square(dpx + halfSubCellWidth) + square(dpy) + square(dpz + halfSubCellWidth));
                    float zCheck = sqrtf(square(dpx + halfSubCellWidth) + square(dpy + halfSubCellWidth) + square(dpz));

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

__global__ void subCellCreateLists(float* px, float* py, float* pz, uint numParticles, Grid grid, uint* gridPosition,
        uint* numSubCellsTouchedX, uint* numSubCellsTouchedY, uint* numSubCellsTouchedZ, uint* subCellsX, uint* subCellsY,
        uint* subCellsZ, uint refinementLevel, float radius, uint xySize){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        uint moduloWRTxySize = gridPosition[index] % xySize;
        uint gridIDz = gridPosition[index] / (xySize);
        uint gridIDy = moduloWRTxySize / grid.sizeX;
        uint gridIDx = moduloWRTxySize % grid.sizeX;
        
        uint apronCells = floorf(radius);
        float subCellWidth = grid.cellSize/(2.0f*(1<<refinementLevel));
        float pxInGridCell = (px[index] - grid.negX - gridIDx*grid.cellSize + apronCells*subCellWidth);    //including apron cell width offset
        float pyInGridCell = (py[index] - grid.negY - gridIDy*grid.cellSize + apronCells*subCellWidth);
        float pzInGridCell = (pz[index] - grid.negZ - gridIDz*grid.cellSize + apronCells*subCellWidth);

        uint subCellPositionX = floorf(pxInGridCell/subCellWidth);
        uint subCellPositionY = floorf(pyInGridCell/subCellWidth);
        uint subCellPositionZ = floorf(pzInGridCell/subCellWidth);
        

        float halfSubCellWidth = subCellWidth / 2.0f;
        float radiusSCW = radius*subCellWidth;

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
                    
                    float dpx = pxInGridCell - subCellBaseX;
                    float dpy = pyInGridCell - subCellBaseY;
                    float dpz = pzInGridCell - subCellBaseZ;

                    float xCheck = sqrtf(square(dpx) + square(dpy + halfSubCellWidth) + square(dpz + halfSubCellWidth));
                    float yCheck = sqrtf(square(dpx + halfSubCellWidth) + square(dpy) + square(dpz + halfSubCellWidth));
                    float zCheck = sqrtf(square(dpx + halfSubCellWidth) + square(dpy + halfSubCellWidth) + square(dpz));

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
 * @brief Wrapper for executing CUDA parallel prefix sum
 * 
 * @tparam T 
 * @param numElements 
 * @param array 
 * @param blockSums 
 * @param stream 
 */

template <typename T>
void kernels::cudaParallelPrefixSum_internal(uint numElements, T* array, T* blockSums, cudaStream_t stream){
    parallelPrefix<<<numElements/WORKSIZE + 1, BLOCKSIZE, 0, stream>>>(numElements, array, blockSums);
    gpuErrchk(cudaPeekAtLastError());
    if(numElements / WORKSIZE > 0){
        cudaParallelPrefixSum_internal((numElements/WORKSIZE + 1), blockSums, blockSums + numElements / WORKSIZE + 1, stream);
    }
    parallelPrefixApplyPreviousBlockSum<<<numElements/WORKSIZE + 1, WORKSIZE, 0, stream>>>(numElements, array, blockSums);
    gpuErrchk(cudaPeekAtLastError());
}

template <typename T>
void kernels::cudaParallelPrefixSum(uint numElements, T* array, cudaStream_t stream){
    uint totalBlockSumsSize = 0;
    T* blockSums;
    for(uint i = numElements / WORKSIZE; i > 0; i = i / WORKSIZE ){
        totalBlockSumsSize += i + 1;
    }
    gpuErrchk(cudaMallocAsync((void**)&blockSums, sizeof(T) * totalBlockSumsSize, stream));
    kernels::cudaParallelPrefixSum_internal(numElements, array, blockSums, stream);
    gpuErrchk(cudaFreeAsync(blockSums, stream));
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
        
        cudaStreamSynchronize(frontStream);
        kernels::cudaParallelPrefixSum<uint>(numElements, front, frontStream);
        kernels::cudaParallelPrefixSum<uint>(numElements, back, backStream);

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
template <typename T>
__global__ void zeroArray(uint size, T* array){
    uint index = threadIdx.x + blockDim.x*blockIdx.x;
    if(index < size){
        array[index] = 0;
    }
}


uint kernels::cudaMarkUniqueGridCellsAndCount(uint numParticles, uint* gridCells, uint* uniqueGridNodes, cudaStream_t stream){
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

void kernels::cudaMapNodeIndicesToParticles(uint numParticles, uint* uniqueGridNodes, uint* gridNodeIndicesToFirstParticleIndex, cudaStream_t stream){
    mapNodeIndicesToParticles<<<numParticles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticles, uniqueGridNodes, gridNodeIndicesToFirstParticleIndex);
}

__global__ void getYDimPositionFirstInRow(uint numUniqueGridNodes, uint* gridNodeIndicesToFirstParticleIndex, uint* gridNodeIDs, uint* yDimFirstNodeIndex, Grid grid){
    uint index = threadIdx.x + blockDim.x*blockIdx.x;
    if(index < numUniqueGridNodes){
        uint prevGridNodeUniqueIndex;
        if(index != 0){
            prevGridNodeUniqueIndex = gridNodeIDs[gridNodeIndicesToFirstParticleIndex[index - 1]];
        }
        else{
            prevGridNodeUniqueIndex = 0;
        }
        uint gridNodeUniqueIndex = gridNodeIDs[gridNodeIndicesToFirstParticleIndex[index]];
        uint prevYRow = prevGridNodeUniqueIndex / grid.sizeX;
        uint yRow = gridNodeUniqueIndex / grid.sizeX;
        if(prevYRow != yRow){
            yDimFirstNodeIndex[yRow] = gridNodeUniqueIndex;
        }
    }
}

void kernels::cudaGetFirstNodeInYRows(uint numUniqueGridNodes, uint* gridNodeIndicesToFirstParticleIndex, uint* gridNodeIDs, uint* yDimFirstNodeIndex, Grid grid, cudaStream_t stream){
    getYDimPositionFirstInRow<<<numUniqueGridNodes / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numUniqueGridNodes, gridNodeIndicesToFirstParticleIndex, gridNodeIDs, yDimFirstNodeIndex, grid);
}

__global__ void sumParticlesPerNode(uint numGridNodes, uint numParticles, uint* gridNodeIndicesToFirstParticleIndex, uint* subCellsTouchedPerParticle,
        uint* subCellsDim, uint* numNonZeroVoxels, uint* numParticlesInVoxelLists, uint numVoxelsPerNode){
    extern __shared__ uint voxelCount[];
    __shared__ uint sums[WORKSIZE];
    __shared__ uint nonZeroVoxels[WORKSIZE];
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
        uint firstSubCellsTouched;
        if(i == 0){
            firstSubCellsTouched = 0;
        }
        else{
            firstSubCellsTouched = subCellsTouchedPerParticle[i-1];
        }
        for(uint j = firstSubCellsTouched; j < subCellsTouchedPerParticle[i]; ++j){
            atomicAdd(voxelCount + subCellsDim[j], 1);
        }
    }

    sums[threadIdx.x] = 0;
    sums[threadIdx.x + blockDim.x] = 0;
    nonZeroVoxels[threadIdx.x] = 0;
    nonZeroVoxels[threadIdx.x + blockDim.x] = 0;
    __syncthreads();

    for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sums[i % WORKSIZE] += voxelCount[i];
        if(voxelCount[i] != 0){
            ++nonZeroVoxels[i % WORKSIZE];
        }
    }

    for(uint i = blockDim.x; i > 0; i>>=1){
        __syncthreads();
        if(threadIdx.x < i){
            sums[threadIdx.x] += sums[threadIdx.x + i];
            nonZeroVoxels[threadIdx.x] += nonZeroVoxels[threadIdx.x + i];
        }
    }

    if(threadIdx.x == 0){
        numParticlesInVoxelLists[blockIdx.x] = sums[0];
        numNonZeroVoxels[blockIdx.x] = nonZeroVoxels[0];
    }
}

void kernels::cudaSumParticlesPerNodeAndWriteNumUsedVoxels(uint numGridNodes, uint numParticles, uint* gridNodeIndicesToFirstParticleIndex, uint* subCellsTouchedPerParticle, uint* subCellsDim, uint* numNonZeroVoxels,
    uint* numParticlesInVoxelLists, uint numVoxelsPerNode, cudaStream_t stream){
    sumParticlesPerNode<<<numGridNodes, BLOCKSIZE, sizeof(uint) * numVoxelsPerNode, stream>>>(
        numGridNodes, numParticles, gridNodeIndicesToFirstParticleIndex, subCellsTouchedPerParticle, subCellsDim, numNonZeroVoxels,
        numParticlesInVoxelLists, numVoxelsPerNode);

    gpuErrchk(cudaPeekAtLastError());
    cudaStream_t particleListStream;
    cudaStreamCreate(&particleListStream);

    cudaStreamSynchronize(stream);
    cudaParallelPrefixSum(numGridNodes, numNonZeroVoxels, stream);
    cudaParallelPrefixSum(numGridNodes, numParticlesInVoxelLists, particleListStream);
    
    gpuErrchk( cudaStreamSynchronize(particleListStream) );
    gpuErrchk( cudaStreamDestroy(particleListStream) );
}


__global__ void createParticleListStartIndices(uint totalNumberVoxelsDimension, uint numGridNodes, uint numParticles, uint numVoxelsPerNode, uint* voxelIDs,
    uint* perVoxelParticleListStartIndices, uint* numUsedVoxelsPerNode, uint* subCellsTouchedPerParticle, uint* subCellsDim,
    uint* firstParticleInNodeIndex, uint* particleLists, uint* firstParticleListIndexPerNode)   //need to revisit how particle list creation is done
{

    //firstParticleInNodeIndex used to index node->subCellsTouchedPerParticle
    //subCellsTouchedPerParticle used to index particle->subCellsDim
    extern __shared__ uint voxelUsedIndex[]; //sized to 2*numVoxelsPerNode
    __shared__ uint maxParticleNum;
    uint* voxelCount = voxelUsedIndex + numVoxelsPerNode;
    __shared__ uint numUsedVoxelsThisNode;

    __shared__ uint firstParticleListIndex;
    __shared__ uint voxelUsedMax;
    __shared__ uint voxelCountMax;

    if(threadIdx.x == 0){
        if(blockIdx.x < numGridNodes - 1){
            maxParticleNum = firstParticleInNodeIndex[blockIdx.x + 1];
        }
        else{
            maxParticleNum = numParticles;
        }
        if(blockIdx.x == 0){
            numUsedVoxelsThisNode = 0;
            firstParticleListIndex = 0;
        }
        else{
            numUsedVoxelsThisNode = numUsedVoxelsPerNode[blockIdx.x - 1];
            firstParticleListIndex = firstParticleListIndexPerNode[blockIdx.x-1];
        }
    }
    for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        voxelUsedIndex[i] = 0;
        voxelCount[i] = 0;
    }

    __syncthreads();

    for(uint i = firstParticleInNodeIndex[blockIdx.x] + threadIdx.x; i < maxParticleNum; i += blockDim.x){
        uint firstSubCellsTouched;
        if(i == 0){
            firstSubCellsTouched = 0;
        }
        else{
            firstSubCellsTouched = subCellsTouchedPerParticle[i-1];
        }
        for(uint j = firstSubCellsTouched; j < subCellsTouchedPerParticle[i]; ++j){
            atomicMax(voxelUsedIndex + subCellsDim[j], 1);  //marking used voxels
            atomicAdd(voxelCount + subCellsDim[j], 1);
        }
    }

    __syncthreads();
    //now have number of particles per voxel, need blockWise prefix sum on voxelCount to get offset of each used voxel's particle list
    blockWiseExclusivePrefixSum(voxelUsedIndex, numVoxelsPerNode, voxelUsedMax);
    blockWiseExclusivePrefixSum(voxelCount, numVoxelsPerNode, voxelCountMax);

    for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        uint nextVal = voxelUsedMax;
        uint nextCount = voxelCountMax;
        if(i != numVoxelsPerNode-1){
            nextVal = voxelUsedIndex[i+1];
            nextCount = voxelCount[i+1];
        }
        if(voxelUsedIndex[i] != nextVal){
            voxelIDs[numUsedVoxelsThisNode + voxelUsedIndex[i]] = i;   //store in-node voxel ID to voxelIDs array
            perVoxelParticleListStartIndices[numUsedVoxelsThisNode + voxelUsedIndex[i]] = voxelCount[i] + firstParticleListIndex;

            //need to reframe from per-particle thread to per-voxel thread, each thread iterates over all particles in node to generate per-voxel list of particles
            uint numParticlesWrittenToCurrentVoxel = 0;
            for(uint particleIndex = firstParticleInNodeIndex[blockIdx.x]; particleIndex < maxParticleNum; ++particleIndex){    //whole loop needs to be its own kernel, iterate only over used nodes and write accordingly
                int firstSubCellsTouched;
                if(particleIndex == 0){
                    firstSubCellsTouched = 0;
                }
                else{
                    firstSubCellsTouched = subCellsTouchedPerParticle[particleIndex-1];
                }
                for(uint subCellTouchedPerParticleIndex = firstSubCellsTouched;                            //current version wastes threads and unnecessarily prolongs runtime of each thread
                        subCellTouchedPerParticleIndex < subCellsTouchedPerParticle[particleIndex];
                        ++subCellTouchedPerParticleIndex)
                {
                    if(subCellsDim[subCellTouchedPerParticleIndex] == i){   //need to convert this to shared mem array 
                        particleLists[firstParticleListIndex + voxelCount[i] + numParticlesWrittenToCurrentVoxel++] = particleIndex;
                    }
                }
            }
        }
    }
}
    
void kernels::cudaParticleListCreate(uint totalNumberVoxelsDimension, uint numGridNodes, uint numParticles, uint numVoxelsPerNode,
    uint* voxelIDs, uint* perVoxelParticleListStartIndices, uint* numUsedVoxelsPerNode, uint* firstParticleListIndexPerNode,
    uint* subCellsTouchedPerParticle, uint* subCellsDim, uint* firstParticleInNodeIndex, uint* particleLists,
    cudaStream_t stream)
{
    createParticleListStartIndices<<<numGridNodes, BLOCKSIZE, 2*numVoxelsPerNode*sizeof(uint), stream>>>(
        totalNumberVoxelsDimension, numGridNodes, numParticles, numVoxelsPerNode, voxelIDs, perVoxelParticleListStartIndices,
        numUsedVoxelsPerNode, subCellsTouchedPerParticle, subCellsDim, firstParticleInNodeIndex, particleLists, firstParticleListIndexPerNode);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk( cudaStreamSynchronize(stream) );
}

__device__ float weightFromDistance(float in, float radius){
    if(in > radius){
        return 0.0f;
    }
    else{
        return 1.0f - in/radius;
    }
}

__global__ void voxelUGather(uint numVoxelsPerNode, uint numUsedVoxelsInGrid, uint numParticlesInParticleLists, uint* gridPosition, float* particleVs,
    float* px, float* py, float* pz, uint* nodeIndexToFirstVoxelIndex, uint* voxelIDs, uint* perVoxelParticleListStartIndices, uint* particleLists,
    float* voxelUs, Grid grid, uint xySize, uint refinementLevel, float radius, VelocityGatherDimension solveDimension, uint numVoxels1D)
{
    extern __shared__ float thisNodeU[];
    __shared__ uint maxVoxelIndex;
    __shared__ uint voxelStartIndex;
    __shared__ uint thisNodeParticleListStopIndex;
    __shared__ uint thisNodeParticleListStartIndex;

    for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        thisNodeU[i] = 0.0f;
    }
    if(threadIdx.x == 0){
        maxVoxelIndex = nodeIndexToFirstVoxelIndex[blockIdx.x];
        if(maxVoxelIndex < numUsedVoxelsInGrid - 1){
            thisNodeParticleListStopIndex = perVoxelParticleListStartIndices[maxVoxelIndex];
        }
        else{
            thisNodeParticleListStopIndex = numParticlesInParticleLists;
        }
        if(blockIdx.x == 0){
            voxelStartIndex = 0;
        }
        else{
            voxelStartIndex = nodeIndexToFirstVoxelIndex[blockIdx.x - 1];
        }
        thisNodeParticleListStartIndex = perVoxelParticleListStartIndices[voxelStartIndex];
    }
    __syncthreads();
    float subCellWidth = grid.cellSize/(2.0f*(1<<refinementLevel));
    uint apronCells = floorf(radius);
    uint numVoxels2D = numVoxels1D*numVoxels1D;
    for(uint curParticleInNode = thisNodeParticleListStartIndex + threadIdx.x; curParticleInNode < thisNodeParticleListStopIndex; curParticleInNode += blockDim.x){
        uint curThreadParticleVoxelID;
        // uint curThreadParticleVoxelListIndex;
        uint particleIndex = particleLists[curParticleInNode];
        for(uint curVoxelIndex = voxelStartIndex; curVoxelIndex < maxVoxelIndex; ++curVoxelIndex){
            uint voxelParticleListStart = perVoxelParticleListStartIndices[curVoxelIndex];
            uint voxelParticleListStop;
            if(curVoxelIndex < numUsedVoxelsInGrid - 1){
                voxelParticleListStop = perVoxelParticleListStartIndices[curVoxelIndex+1];
            }
            else{
                voxelParticleListStop = numParticlesInParticleLists;
            }
            if(curParticleInNode >= voxelParticleListStart && curParticleInNode < voxelParticleListStop){
                curThreadParticleVoxelID = voxelIDs[curVoxelIndex];
            }
        }
        uint voxelIDx = curThreadParticleVoxelID % numVoxels1D;
        uint voxelIDy = (curThreadParticleVoxelID % numVoxels2D) / numVoxels1D;
        uint voxelIDz = curThreadParticleVoxelID / numVoxels2D;

        float voxelPx = voxelIDx * subCellWidth;
        float voxelPy = voxelIDy * subCellWidth;
        float voxelPz = voxelIDz * subCellWidth;
        
        if(solveDimension == VelocityGatherDimension::X){
            voxelPy += 0.5*subCellWidth;
            voxelPz += 0.5*subCellWidth;
        }
        else if(solveDimension == VelocityGatherDimension::X){
            voxelPx += 0.5*subCellWidth;
            voxelPz += 0.5*subCellWidth;
        }
        else{
            voxelPx += 0.5*subCellWidth;
            voxelPy += 0.5*subCellWidth;
        }

        uint moduloWRTxySize = gridPosition[particleIndex] % xySize;
        uint gridIDz = gridPosition[particleIndex] / (xySize);
        uint gridIDy = (moduloWRTxySize) / grid.sizeX;
        uint gridIDx = (moduloWRTxySize) % grid.sizeX;
        
        float pxInGridCell = (px[particleIndex] - grid.negX - gridIDx*grid.cellSize + apronCells*subCellWidth);
        float pyInGridCell = (py[particleIndex] - grid.negY - gridIDy*grid.cellSize + apronCells*subCellWidth);
        float pzInGridCell = (pz[particleIndex] - grid.negZ - gridIDz*grid.cellSize + apronCells*subCellWidth);

        float dpx = pxInGridCell - voxelPx;
        float dpy = pyInGridCell - voxelPy;
        float dpz = pzInGridCell - voxelPz;
        float dpx_2 = dpx*dpx;
        float dpy_2 = dpy*dpy;
        float dpz_2 = dpz*dpz;

        float particleToNodeDistance = sqrtf(dpx_2 + dpy_2 + dpz_2) / subCellWidth;

        atomicAdd(thisNodeU + curThreadParticleVoxelID, weightFromDistance(particleToNodeDistance, radius) * particleVs[particleIndex]);
    }
    __syncthreads();
    for(uint usedVoxelIndex = voxelStartIndex + threadIdx.x; usedVoxelIndex < maxVoxelIndex; usedVoxelIndex += blockDim.x){
        voxelUs[usedVoxelIndex] = thisNodeU[voxelIDs[usedVoxelIndex]];
    }
}

void kernels::cudaVoxelUGather(uint numUsedVoxelsGrid, uint numGridNodes, uint numParticles, uint numVoxelsPerNode, uint numParticlesInParticleLists,
    uint* gridPosition, float* particleVs, float* px, float* py, float* pz, uint* nodeIndexToFirstVoxelIndex, uint* voxelIDs,
    uint* perVoxelParticleListStartIndices, uint* particleLists, float* voxelUs, Grid grid, uint xySize, uint refinementLevel, float radius,
    uint numVoxels1D, VelocityGatherDimension solveDimension, cudaStream_t stream)
{
    voxelUGather<<<numGridNodes, BLOCKSIZE, sizeof(float) * numVoxelsPerNode, stream>>>(numVoxelsPerNode, numUsedVoxelsGrid, numParticlesInParticleLists,
        gridPosition, particleVs, px, py, pz, nodeIndexToFirstVoxelIndex, voxelIDs, perVoxelParticleListStartIndices, particleLists, voxelUs, grid, xySize,
        refinementLevel, radius, solveDimension, numVoxels1D);
}