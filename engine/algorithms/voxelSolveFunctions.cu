//Copyright 2023 Aberrant Behavior LLC

#include "voxelSolveFunctions.hu"

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
        else if(solveDimension == VelocityGatherDimension::Y){
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

void cudaVoxelUGather(uint numUsedVoxelsGrid, uint numGridNodes, uint numParticles, uint numVoxelsPerNode, uint numParticlesInParticleLists,
    uint* gridPosition, float* particleVs, float* px, float* py, float* pz, uint* nodeIndexToFirstVoxelIndex, uint* voxelIDs,
    uint* perVoxelParticleListStartIndices, uint* particleLists, float* voxelUs, Grid grid, uint xySize, uint refinementLevel, float radius,
    uint numVoxels1D, VelocityGatherDimension solveDimension, cudaStream_t stream)
{
    voxelUGather<<<numGridNodes, BLOCKSIZE, sizeof(float) * numVoxelsPerNode, stream>>>(numVoxelsPerNode, numUsedVoxelsGrid, numParticlesInParticleLists,
        gridPosition, particleVs, px, py, pz, nodeIndexToFirstVoxelIndex, voxelIDs, perVoxelParticleListStartIndices, particleLists, voxelUs, grid, xySize,
        refinementLevel, radius, solveDimension, numVoxels1D);
}

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
    loadVoxelDataForThisBlock(numVoxelsPerNode, numVoxels1D, numUsedVoxelsInGrid, sharedVoxels, nodeIndexToFirstVoxelIndex, voxelIDs,
        voxelUs, radius, grid, yDimFirstNodeIndex, gridNodeIndicesToFirstParticleIndex, gridNodes);
    if(threadIdx.x == 0){
        sharedDivU = sharedVoxels + numVoxelsPerNode;
    }
    __syncthreads();
    for(uint internalBlockID = threadIdx.x; internalBlockID < numVoxelsPerNode; internalBlockID += blockDim.x){
        sharedDivU[internalBlockID] = 0.0f;
        if(dimension == VelocityGatherDimension::X){
            uint blockIDx = internalBlockID % numVoxels1D;
            if (blockIDx < numVoxels1D - 1){
                sharedDivU[internalBlockID] = sharedVoxels[internalBlockID+1] - sharedVoxels[internalBlockID]; 
            }
        }
        else if(dimension == VelocityGatherDimension::Y){
            uint blockIDy = internalBlockID % (numVoxels1D * numVoxels1D) / numVoxels1D;
            if(blockIDy < numVoxels1D - 1){
                sharedDivU[internalBlockID] = sharedVoxels[internalBlockID + numVoxels1D] - sharedVoxels[internalBlockID];
            }
        }
        else{
            uint blockIDz = internalBlockID / (numVoxels1D * numVoxels1D);
            if(blockIDz < numVoxels1D - 1){
                sharedDivU[internalBlockID] = sharedVoxels[internalBlockID + (numVoxels1D * numVoxels1D)] - sharedVoxels[internalBlockID];
            }
        }
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
    for(uint voxelIndex = voxelIndexStart + threadIdx.x; voxelIndex < nodeIndexToFirstVoxelIndex[blockIdx.x]; voxelIndex += blockIdx.x){
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