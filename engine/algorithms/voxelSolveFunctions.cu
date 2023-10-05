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
    for(uint currentUsedVoxelIndex = startVoxelID + threadIdx.x; currentUsedVoxelIndex < nodeIndexToFirstVoxelIndex[thisThreadNodeIndexToHandle]; currentUsedVoxelIndex += blockDim.x){
        sharedBlockVoxelStorage[voxelIDs[currentUsedVoxelIndex]] += voxelData[voxelIDs[currentUsedVoxelIndex]];
    }

}

__device__ void loadVoxelDataForThisBlock(const uint numVoxelsPerNode, const uint numUniqueGridNodes, float* sharedBlockStorage,
    const uint* nodeIndexToFirstVoxelIndex, const uint* voxelIDs, const float* voxelData, const float radius, const uint refinementLevel,
    const Grid& grid, const uint* yDimToFirstNodeIndex, const uint* gridNodeIndicesToFirstParticleIndex, const uint* gridNodeIDs)
{
    for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sharedBlockStorage[i] = 0.0f;
    }
    __syncthreads();
    addVoxelDataForUsedNode(threadIdx.x + blockIdx.x*blockDim.x, sharedBlockStorage,
        nodeIndexToFirstVoxelIndex, voxelIDs, voxelData);
    
    uint curGridNodeID = gridNodeIDs[gridNodeIndicesToFirstParticleIndex[blockIdx.x]];
    uint curY = (curGridNodeID / grid.sizeX) % grid.sizeY;
    uint curZ = curGridNodeID / (grid.sizeX*grid.sizeY);

    uint yNeighborLeft = curY - 1;
    uint yNeighborRight = curY + 1;
    uint zNeighborLeft = curZ - 1;
    uint zNeighborRight = curZ + 1;

    __syncthreads();
    uint neighborUniqueID;
    if((neighborUniqueID = yDimToFirstNodeIndex[yNeighborLeft + curZ*grid.sizeY]) < numUniqueGridNodes){
        addVoxelDataForUsedNode(neighborUniqueID, sharedBlockStorage,
            nodeIndexToFirstVoxelIndex, voxelIDs, voxelData);
    }
    __syncthreads();
    if((neighborUniqueID = yDimToFirstNodeIndex[yNeighborRight + curZ*grid.sizeY]) < numUniqueGridNodes){
        addVoxelDataForUsedNode(neighborUniqueID, sharedBlockStorage,
            nodeIndexToFirstVoxelIndex, voxelIDs, voxelData);
    }
    __syncthreads();
    if((neighborUniqueID = yDimToFirstNodeIndex[zNeighborRight*grid.sizeY + curY]) < numUniqueGridNodes){
        addVoxelDataForUsedNode(neighborUniqueID, sharedBlockStorage,
            nodeIndexToFirstVoxelIndex, voxelIDs, voxelData);
    }
    __syncthreads();
    if((neighborUniqueID = yDimToFirstNodeIndex[zNeighborLeft*grid.sizeY + curY]) < numUniqueGridNodes){
        addVoxelDataForUsedNode(neighborUniqueID, sharedBlockStorage,
            nodeIndexToFirstVoxelIndex, voxelIDs, voxelData);
    }
    __syncthreads();
}

__global__ void calculateDivU(uint numVoxelsPerNode, uint numVoxels1D, uint numUsedVoxelsInGrid, const uint* nodeIndexToFirstVoxelIndex, const uint* voxelIDs,
                                const float* voxelUs, float radius, uint refinementLevel, Grid grid, const uint* yDimFirstNodeIndex,
                                const uint* gridNodeIndicesToFirstParticleIndex, const uint* gridNodes, float* divU)
{
    extern __shared__ float sharedVoxels[];
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxelsInGrid){
        loadVoxelDataForThisBlock(numVoxelsPerNode, numUsedVoxelsInGrid, sharedVoxels,
                nodeIndexToFirstVoxelIndex, voxelIDs, voxelUs, radius, refinementLevel,
                grid, yDimFirstNodeIndex, gridNodeIndicesToFirstParticleIndex,
                gridNodes);
    }
}

void cudaPressureSolve(const CudaVec<uint>& nodeIndexToFirstVoxelIndex, const CudaVec<uint>& voxelIDs, const CudaVec<float>& voxelUs,
                        float radius, uint refinementLevel, const Grid& grid, const CudaVec<uint>& yDimFirstNodeIndex,
                        const CudaVec<uint>& gridNodeIndicesToFirstParticleIndex, const CudaVec<uint>& gridNodes, uint numVoxelsPerNode,
                        uint numVoxels1D,
                        CudaVec<float> divU, CudaVec<float> Ax, CudaVec<float> Ay, CudaVec<float> Az, CudaVec<float> Adiag, cudaStream_t stream)
{
    calculateDivU<<<nodeIndexToFirstVoxelIndex.size(), BLOCKSIZE, sizeof(float)*numVoxelsPerNode, stream>>>
        (numVoxelsPerNode, numVoxels1D, nodeIndexToFirstVoxelIndex.size(), nodeIndexToFirstVoxelIndex.devPtr(),
        voxelIDs.devPtr(), voxelUs.devPtr(), radius, refinementLevel, grid, yDimFirstNodeIndex.devPtr(),
        gridNodeIndicesToFirstParticleIndex.devPtr(), gridNodes.devPtr(), divU.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}