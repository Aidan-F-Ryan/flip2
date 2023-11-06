//Copyright 2023 Aberrant Behavior LLC

#include "voxelSolveFunctions.hu"

__device__ bool isIndexApronCell(uint voxelIndex, const uint& numVoxels1D, const uint& refinementLevel, const float& radius){
    uint numApronCellsAtBorder = floorf(radius);
    uint rightBoundApronCells = (1<<refinementLevel) + numApronCellsAtBorder;
    uint xWiseIndex = voxelIndex % (numVoxels1D);
    uint yWiseIndex = (voxelIndex / numVoxels1D) % numVoxels1D;
    uint zWiseIndex = voxelIndex / (numVoxels1D*numVoxels1D);
    return xWiseIndex < numApronCellsAtBorder
        || xWiseIndex > rightBoundApronCells
        || yWiseIndex < numApronCellsAtBorder
        || yWiseIndex > rightBoundApronCells
        || zWiseIndex < numApronCellsAtBorder
        || zWiseIndex > rightBoundApronCells;
}

__device__ void addVoxelDataForUsedNode(const uint thisThreadNodeIndexToHandle, float* sharedBlockVoxelStorage, const uint* nodeIndexToFirstVoxelIndex,
        const uint* voxelIDs, const float* voxelData)
{
    uint startVoxelID;
    if(thisThreadNodeIndexToHandle == 0){
        startVoxelID = 0;
    }
    else{
        startVoxelID = nodeIndexToFirstVoxelIndex[thisThreadNodeIndexToHandle-1];
    }
    for(uint currentUsedVoxelIndex = startVoxelID + threadIdx.x;
            currentUsedVoxelIndex < nodeIndexToFirstVoxelIndex[thisThreadNodeIndexToHandle];
            currentUsedVoxelIndex += blockDim.x){
        sharedBlockVoxelStorage[voxelIDs[currentUsedVoxelIndex]] += voxelData[currentUsedVoxelIndex];
    }

}

__device__ void getNeighboringApronCellData(const uint thisThreadNodeIndexToHandle, const uint numVoxels1D, float* sharedBlockVoxelStorage, const uint* nodeIndexToFirstVoxelIndex,
        const uint* voxelIDs, const float* voxelData, const int xOffset, const int yOffset, const int zOffset)
{
    uint numVoxels2D = numVoxels1D*numVoxels1D;
    uint startVoxelID;
    if(thisThreadNodeIndexToHandle == 0){
        startVoxelID = 0;
    }
    else{
        startVoxelID = nodeIndexToFirstVoxelIndex[thisThreadNodeIndexToHandle-1];
    }
    for(uint currentUsedVoxelIndex = startVoxelID + threadIdx.x;
            currentUsedVoxelIndex < nodeIndexToFirstVoxelIndex[thisThreadNodeIndexToHandle];
            currentUsedVoxelIndex += blockDim.x){
        uint usedVoxelIndex = voxelIDs[currentUsedVoxelIndex];
        uint sourceIDx = usedVoxelIndex % numVoxels1D;
        uint sourceIDy = (usedVoxelIndex % numVoxels2D) / numVoxels1D;
        uint sourceIDz = usedVoxelIndex / numVoxels2D;
        int targetIDx = sourceIDx;
        int targetIDy = sourceIDy;
        int targetIDz = sourceIDz;
        if(xOffset < 0){    //right of source mapped to left of target
            if(sourceIDx >= numVoxels1D + xOffset){
                targetIDx = sourceIDx - numVoxels1D - xOffset;
                sharedBlockVoxelStorage[targetIDx + targetIDy*numVoxels1D + targetIDz*numVoxels2D] += voxelData[currentUsedVoxelIndex];
            }
        }
        else if(xOffset > 0){   //left of source mapped to right of target
            if(sourceIDx < xOffset){
                targetIDx = numVoxels1D - xOffset + sourceIDx;
                sharedBlockVoxelStorage[targetIDx + targetIDy*numVoxels1D + targetIDz*numVoxels2D] += voxelData[currentUsedVoxelIndex];
            }
        }
        else if(yOffset < 0){
            if(sourceIDy >= numVoxels1D + yOffset){
                targetIDy = sourceIDy - numVoxels1D - yOffset;
                sharedBlockVoxelStorage[targetIDx + targetIDy*numVoxels1D + targetIDz*numVoxels2D] += voxelData[currentUsedVoxelIndex];
            }
        }
        else if(yOffset > 0){
            if(sourceIDy < yOffset){
                targetIDy = numVoxels1D - yOffset + sourceIDy;
                sharedBlockVoxelStorage[targetIDx + targetIDy*numVoxels1D + targetIDz*numVoxels2D] += voxelData[currentUsedVoxelIndex];
            }
        }
        else if(zOffset < 0){
            if(sourceIDz >= numVoxels1D + zOffset){
                targetIDz = sourceIDz - numVoxels1D - zOffset;
                sharedBlockVoxelStorage[targetIDz + targetIDy*numVoxels1D + targetIDz*numVoxels2D] += voxelData[currentUsedVoxelIndex];
            }
        }
        else if(zOffset > 0){
            if(sourceIDz < zOffset){
                targetIDz = numVoxels1D - zOffset + sourceIDz;
                sharedBlockVoxelStorage[targetIDx + targetIDy*numVoxels1D + targetIDz*numVoxels2D] += voxelData[currentUsedVoxelIndex];
            }
        }
    }

}

__device__ void loadVoxelDataForThisBlock(const uint numVoxelsPerNode, const uint numVoxels1D, const uint numUniqueGridNodes, float* sharedBlockStorage,
    const uint* nodeIndexToFirstVoxelIndex, const uint* voxelIDs, const float* voxelData, const uint numApronCells,
    const Grid& grid, const uint* yDimToFirstNodeIndex, const uint* gridNodeIndicesToFirstParticleIndex, const uint* gridNodeIDs)
{
    for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sharedBlockStorage[i] = 0.0f;
    }
    __syncthreads();
    addVoxelDataForUsedNode(blockIdx.x, sharedBlockStorage,
        nodeIndexToFirstVoxelIndex, voxelIDs, voxelData);
    
    uint curGridNodeID = gridNodeIDs[gridNodeIndicesToFirstParticleIndex[blockIdx.x]];
    uint curX = (curGridNodeID) % grid.sizeX;
    uint curY = (curGridNodeID % (grid.sizeX*grid.sizeY)) / grid.sizeX;
    uint curZ = curGridNodeID / (grid.sizeX*grid.sizeY);

    int xNeighborLeft = curX - 1;
    int xNeighborRight = curX + 1;
    int yNeighborLeft = curY - 1;
    int yNeighborRight = curY + 1;
    int zNeighborLeft = curZ - 1;
    int zNeighborRight = curZ + 1;

    __syncthreads();
    uint neighborUniqueID;
    int totalNumApronCells = 2 * numApronCells;
    if(xNeighborLeft >= 0 && blockIdx.x > 0){
        if((gridNodeIDs[gridNodeIndicesToFirstParticleIndex[blockIdx.x-1]]) == curGridNodeID - 1){
            neighborUniqueID = blockIdx.x - 1;
            getNeighboringApronCellData(neighborUniqueID, numVoxels1D, sharedBlockStorage,
                nodeIndexToFirstVoxelIndex, voxelIDs, voxelData, -totalNumApronCells, 0, 0);
        }
    }
    __syncthreads();
    if(xNeighborRight < grid.sizeX && blockIdx.x < blockDim.x-1){
        if((gridNodeIDs[gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1]]) == curGridNodeID + 1){
            neighborUniqueID = blockIdx.x + 1;
            getNeighboringApronCellData(neighborUniqueID, numVoxels1D, sharedBlockStorage,
                nodeIndexToFirstVoxelIndex, voxelIDs, voxelData, totalNumApronCells, 0, 0);
        }
    }
    if(yNeighborLeft >= 0 && (neighborUniqueID = yDimToFirstNodeIndex[yNeighborLeft + curZ*grid.sizeY]) < numUniqueGridNodes){
        getNeighboringApronCellData(neighborUniqueID, numVoxels1D, sharedBlockStorage,
            nodeIndexToFirstVoxelIndex, voxelIDs, voxelData, 0, -totalNumApronCells, 0);
    }
    __syncthreads();
    if(yNeighborRight < grid.sizeY && (neighborUniqueID = yDimToFirstNodeIndex[yNeighborRight + curZ*grid.sizeY]) < numUniqueGridNodes){
        getNeighboringApronCellData(neighborUniqueID, numVoxels1D, sharedBlockStorage,
            nodeIndexToFirstVoxelIndex, voxelIDs, voxelData, 0, totalNumApronCells, 0);
    }
    __syncthreads();
    if(zNeighborRight >= 0 && (neighborUniqueID = yDimToFirstNodeIndex[zNeighborRight*grid.sizeY + curY]) < numUniqueGridNodes){
        getNeighboringApronCellData(neighborUniqueID, numVoxels1D, sharedBlockStorage,
            nodeIndexToFirstVoxelIndex, voxelIDs, voxelData, 0, 0, -totalNumApronCells);
    }
    __syncthreads();
    if(zNeighborRight < grid.sizeZ && (neighborUniqueID = yDimToFirstNodeIndex[zNeighborLeft*grid.sizeY + curY]) < numUniqueGridNodes){
        getNeighboringApronCellData(neighborUniqueID, numVoxels1D, sharedBlockStorage,
            nodeIndexToFirstVoxelIndex, voxelIDs, voxelData, 0, 0, totalNumApronCells);
    }
    __syncthreads();
}

__global__ void calculateDivU(uint numVoxelsPerNode, uint numVoxels1D, uint numUsedVoxelsInGrid, const uint* nodeIndexToFirstVoxelIndex,
                                const uint* voxelIDs, const float* voxelUs, float radius, uint refinementLevel, Grid grid,
                                const uint* yDimFirstNodeIndex, const uint* gridNodeIndicesToFirstParticleIndex, const uint* gridNodes,
                                float* divU, VelocityGatherDimension dimension)
{
    extern __shared__ float sharedVoxels[];
    __shared__ float* sharedDivU;
    __shared__ float dx;
    loadVoxelDataForThisBlock(numVoxelsPerNode, numVoxels1D, numUsedVoxelsInGrid, sharedVoxels, nodeIndexToFirstVoxelIndex, voxelIDs,
        voxelUs, radius, grid, yDimFirstNodeIndex, gridNodeIndicesToFirstParticleIndex, gridNodes);
    if(threadIdx.x == 0){
        sharedDivU = sharedVoxels + numVoxelsPerNode;
        dx = grid.cellSize / (1<<refinementLevel);
    }
    __syncthreads();
    
    for(uint internalBlockID = threadIdx.x; internalBlockID < numVoxelsPerNode; internalBlockID += blockDim.x){
        sharedDivU[internalBlockID] = 0.0f;
        if(dimension == VelocityGatherDimension::X){
            uint blockIDx = internalBlockID % numVoxels1D;
            if (blockIDx < numVoxels1D - 1){
                float vX = sharedVoxels[internalBlockID + 1];
                sharedDivU[internalBlockID] = vX - sharedVoxels[internalBlockID];
            }
        }
        else if(dimension == VelocityGatherDimension::Y){
            uint blockIDy = internalBlockID % (numVoxels1D * numVoxels1D) / numVoxels1D;
            if(blockIDy < numVoxels1D - 1){
                float vY = sharedVoxels[internalBlockID + numVoxels1D];
                sharedDivU[internalBlockID] = vY - sharedVoxels[internalBlockID];
            }
        }
        else{
            uint blockIDz = internalBlockID / (numVoxels1D * numVoxels1D);
            if(blockIDz < numVoxels1D - 1){
                float vZ = sharedVoxels[internalBlockID + numVoxels1D*numVoxels1D];
                sharedDivU[internalBlockID] = vZ - sharedVoxels[internalBlockID];
            }
        }
        sharedDivU[internalBlockID] /= dx;
    }
    __shared__ uint voxelIndexStart;
    if(threadIdx.x == 0){
        if(blockIdx.x == 0){
            voxelIndexStart = 0;
        }
        else{
            voxelIndexStart = nodeIndexToFirstVoxelIndex[blockIdx.x - 1];
        }
    }
    __syncthreads();
    for(uint voxelIndex = voxelIndexStart + threadIdx.x; voxelIndex < nodeIndexToFirstVoxelIndex[blockIdx.x]; voxelIndex += blockDim.x){
        divU[voxelIndex] += sharedDivU[voxelIDs[voxelIndex]];
    }

}

void cudaCalcDivU(const CudaVec<uint>& nodeIndexToFirstVoxelIndex, const CudaVec<uint>& voxelIDs, 
                        const CudaVec<float>& voxelsUx, const CudaVec<float>& voxelsUy, const CudaVec<float>& voxelsUz,
                        float radius, uint refinementLevel, const Grid& grid, const CudaVec<uint>& yDimFirstNodeIndex,
                        const CudaVec<uint>& gridNodeIndicesToFirstParticleIndex, const CudaVec<uint>& gridNodes, uint numVoxelsPerNode,
                        uint numVoxels1D, CudaVec<float>& divU, CudaVec<float>& Ax, CudaVec<float>& Ay, CudaVec<float>& Az,
                        CudaVec<float>& Adiag, cudaStream_t stream)
{
    divU.zeroDeviceAsync(stream);
    calculateDivU<<<nodeIndexToFirstVoxelIndex.size(), BLOCKSIZE, 2*sizeof(float)*numVoxelsPerNode, stream>>>
        (numVoxelsPerNode, numVoxels1D, nodeIndexToFirstVoxelIndex.size(), nodeIndexToFirstVoxelIndex.devPtr(),
        voxelIDs.devPtr(), voxelsUx.devPtr(), radius, refinementLevel, grid, yDimFirstNodeIndex.devPtr(),
        gridNodeIndicesToFirstParticleIndex.devPtr(), gridNodes.devPtr(), divU.devPtr(), VelocityGatherDimension::X);
    gpuErrchk(cudaPeekAtLastError());
    calculateDivU<<<nodeIndexToFirstVoxelIndex.size(), BLOCKSIZE, 2*sizeof(float)*numVoxelsPerNode, stream>>>
        (numVoxelsPerNode, numVoxels1D, nodeIndexToFirstVoxelIndex.size(), nodeIndexToFirstVoxelIndex.devPtr(),
        voxelIDs.devPtr(), voxelsUy.devPtr(), radius, refinementLevel, grid, yDimFirstNodeIndex.devPtr(),
        gridNodeIndicesToFirstParticleIndex.devPtr(), gridNodes.devPtr(), divU.devPtr(), VelocityGatherDimension::Y);
    gpuErrchk(cudaPeekAtLastError());
    calculateDivU<<<nodeIndexToFirstVoxelIndex.size(), BLOCKSIZE, 2*sizeof(float)*numVoxelsPerNode, stream>>>
        (numVoxelsPerNode, numVoxels1D, nodeIndexToFirstVoxelIndex.size(), nodeIndexToFirstVoxelIndex.devPtr(),
        voxelIDs.devPtr(), voxelsUz.devPtr(), radius, refinementLevel, grid, yDimFirstNodeIndex.devPtr(),
        gridNodeIndicesToFirstParticleIndex.devPtr(), gridNodes.devPtr(), divU.devPtr(), VelocityGatherDimension::Z);
    gpuErrchk(cudaPeekAtLastError());
    cudaStreamSynchronize((stream));
}

__global__ void GSiteration(uint numVoxelsPerNode, uint numVoxels1D, float radius, uint* nodeIndexUsedVoxels, uint* voxelIDs, bool* solids, float* divU, float* p, float* pOld){
    uint linThreadIdx = threadIdx.x + threadIdx.y*numVoxels1D + threadIdx.z*numVoxels1D*numVoxels1D;
    uint linBlockDim = blockDim.x * blockDim.y * blockDim.z;
    extern __shared__ bool sharedSolids[];
    __shared__ uint voxelIndexStart;
    if(threadIdx.x == 0){
        if(blockIdx.x == 0){
            voxelIndexStart = 0;
        }
        else{
            voxelIndexStart = nodeIndexUsedVoxels[blockIdx.x - 1]; 
        }
    }
    
    for(uint i = linThreadIdx; i < numVoxelsPerNode; i += linBlockDim){
        sharedSolids[i] = false;
    }
    __syncthreads();
    
}

void cudaGSiteration()