//Copyright 2023 Aberrant Behavior LLC

#include "voxelSolveFunctions.hu"
#include <cmath>

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
    for(int currentUsedVoxelIndex = startVoxelID + threadIdx.x;
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
    for(int currentUsedVoxelIndex = startVoxelID + threadIdx.x;
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
    for(int i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
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

//velocities are along negative faces of each
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
    
    for(int internalBlockID = threadIdx.x; internalBlockID < numVoxelsPerNode; internalBlockID += blockDim.x){
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
    for(int voxelIndex = voxelIndexStart + threadIdx.x; voxelIndex < nodeIndexToFirstVoxelIndex[blockIdx.x]; voxelIndex += blockDim.x){
        divU[voxelIndex] += sharedDivU[voxelIDs[voxelIndex]];
    }

}

void cudaCalcDivU(const CudaVec<uint>& nodeIndexToFirstVoxelIndex, const CudaVec<uint>& voxelIDs, 
                        const CudaVec<float>& voxelsUx, const CudaVec<float>& voxelsUy, const CudaVec<float>& voxelsUz,
                        float radius, uint refinementLevel, const Grid& grid, const CudaVec<uint>& yDimFirstNodeIndex,
                        const CudaVec<uint>& gridNodeIndicesToFirstParticleIndex, const CudaVec<uint>& gridNodes, uint numVoxelsPerNode,
                        uint numVoxels1D, CudaVec<float>& divU,
                        cudaStream_t stream)
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
}

#include "reductionKernels.hu"

__global__ void GSiteration(uint numVoxelsPerNode, uint numVoxels1D, float radius, const uint* nodeIndexUsedVoxels, const uint* voxelIDs, const char* solids, const float* divU, float* p, float* residuals, float density, float dt, Grid grid){
    extern __shared__ float sharedDivU[];
    __shared__ float* sharedP;
    __shared__ float* sharedRes;
    __shared__ uint* processedVoxels;
    __shared__ char* sharedSolids;
    __shared__ uint voxelIndexStart;
    __shared__ float scale;
    if(threadIdx.x == 0){
        sharedP = sharedDivU + numVoxelsPerNode;
        sharedRes = sharedP + numVoxelsPerNode;
        processedVoxels = (uint*)(sharedRes + numVoxelsPerNode);
        sharedSolids = (char*)(processedVoxels + numVoxelsPerNode);

        float voxelSize = grid.cellSize / (numVoxels1D - 2*floorf(radius));
        scale = dt / (density*voxelSize*voxelSize);
        if(blockIdx.x == 0){
            voxelIndexStart = 0;
        }
        else{
            voxelIndexStart = nodeIndexUsedVoxels[blockIdx.x - 1]; 
        }
    }
    __syncthreads();
    
    for(int i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sharedDivU[i] = 0.0f;
        sharedRes[i] = 0.0f;
        sharedP[i] = 0.0f;
        sharedSolids[i] = false;
        processedVoxels[i] = numVoxelsPerNode;
    }
    __syncthreads();
    for(int i = threadIdx.x; voxelIndexStart + i < nodeIndexUsedVoxels[blockIdx.x]; i += blockDim.x){
        processedVoxels[i] = voxelIDs[voxelIndexStart + i];
        sharedDivU[processedVoxels[i]] = divU[voxelIndexStart + i];
        sharedP[processedVoxels[i]] = p[voxelIndexStart + i];
        sharedSolids[processedVoxels[i]] = solids[voxelIndexStart + i];
    }
    __syncthreads();

    for(int i = threadIdx.x; processedVoxels[i] != numVoxelsPerNode; i += blockDim.x){
        if(!solids[processedVoxels[i]]){
            uint voxelIDx = processedVoxels[i] % numVoxels1D;
            uint voxelIDy = (processedVoxels[i] % (numVoxels1D*numVoxels1D)) / numVoxels1D;
            uint voxelIDz = processedVoxels[i] / (numVoxels1D*numVoxels1D);
            uint apronPad = floorf(radius);
            if(voxelIDx >= apronPad && voxelIDx < numVoxels1D - apronPad){
                if(voxelIDy >= apronPad && voxelIDy < numVoxels1D - apronPad){
                    if(voxelIDz >= apronPad && voxelIDz < numVoxels1D - apronPad){
                        uint voxelID = processedVoxels[i];
                        float Adiag = 0.0f;
                        float Apx = 0.0f;
                        float Apy = 0.0f;
                        float Apz = 0.0f;
                        float Anx = 0.0f;
                        float Any = 0.0f;
                        float Anz = 0.0f;

                        if(!sharedSolids[voxelID-1]){
                            Adiag += scale;
                            Anx = -scale;
                        }
                        if(!sharedSolids[voxelID+1]){
                            Apx = -scale;
                            Adiag += scale;
                        }
                        if(!sharedSolids[voxelID - numVoxels1D]){
                            Any = -scale;
                            Adiag += scale;
                        }
                        if(!sharedSolids[voxelID + numVoxels1D]){
                            Apy = -scale;
                            Adiag += scale;
                        }
                        if(!sharedSolids[voxelID - numVoxels1D*numVoxels1D]){
                            Anz = -scale;
                            Adiag += scale;
                        }
                        if(!sharedSolids[voxelID + numVoxels1D*numVoxels1D]){
                            Apz = -scale;
                            Adiag += scale;
                        }
                        sharedP[voxelID] = (-sharedDivU[voxelID] + (Anx*sharedP[voxelID-1] + Apx*sharedP[voxelID + 1] + Any*sharedP[voxelID-numVoxels1D] + Apy*sharedP[voxelID+numVoxels1D] + Anz*sharedP[voxelID-numVoxels1D*numVoxels1D] + Apz*sharedP[voxelID+numVoxels1D*numVoxels1D])) / (Adiag + 0.00001f);
                    }
                }

            }
        }
    }
    __syncthreads();
    for(int i = threadIdx.x; processedVoxels[i] != numVoxelsPerNode; i += blockDim.x){
        if(!solids[processedVoxels[i]]){
            uint voxelIDx = processedVoxels[i] % numVoxels1D;
            uint voxelIDy = (processedVoxels[i] % (numVoxels1D*numVoxels1D)) / numVoxels1D;
            uint voxelIDz = processedVoxels[i] / (numVoxels1D*numVoxels1D);
            uint apronPad = floorf(radius);
            if(voxelIDx >= apronPad && voxelIDx < numVoxels1D - apronPad){
                if(voxelIDy >= apronPad && voxelIDy < numVoxels1D - apronPad){
                    if(voxelIDz >= apronPad && voxelIDz < numVoxels1D - apronPad){
                        uint voxelID = processedVoxels[i];
                        float Adiag = 0.0f;
                        float Apx = 0.0f;
                        float Apy = 0.0f;
                        float Apz = 0.0f;
                        float Anx = 0.0f;
                        float Any = 0.0f;
                        float Anz = 0.0f;

                        if(!sharedSolids[voxelID-1]){
                            Adiag += scale;
                            Anx = -scale;
                        }
                        if(!sharedSolids[voxelID+1]){
                            Apx = -scale;
                            Adiag += scale;
                        }
                        if(!sharedSolids[voxelID - numVoxels1D]){
                            Any = -scale;
                            Adiag += scale;
                        }
                        if(!sharedSolids[voxelID + numVoxels1D]){
                            Apy = -scale;
                            Adiag += scale;
                        }
                        if(!sharedSolids[voxelID - numVoxels1D*numVoxels1D]){
                            Anz = -scale;
                            Adiag += scale;
                        }
                        if(!sharedSolids[voxelID + numVoxels1D*numVoxels1D]){
                            Apz = -scale;
                            Adiag += scale;
                        }
                        sharedRes[voxelID] = (sharedDivU[voxelID] + (Adiag + 0.00001f)*sharedP[voxelID] - (Anx*sharedP[voxelID-1] + Apx*sharedP[voxelID + 1] + Any*sharedP[voxelID-numVoxels1D] + Apy*sharedP[voxelID+numVoxels1D] + Anz*sharedP[voxelID-numVoxels1D*numVoxels1D] + Apz*sharedP[voxelID+numVoxels1D*numVoxels1D]));
                    }
                }

            }
        }
    }
    // blockwiseMaximum(sharedRes, numVoxelsPerNode);
    __syncthreads();
    for(int i = threadIdx.x; processedVoxels[i] != numVoxelsPerNode; i += blockDim.x){
        p[voxelIndexStart + i] = sharedP[processedVoxels[i]];
        residuals[voxelIndexStart + i] = sharedRes[processedVoxels[i]];
    }
    // __syncthreads();
    // if(threadIdx.x == 0){
    //     residuals[blockIdx.x] = sharedRes[0];
    // }
}

float cudaGSiteration(const uint& numVoxelsPerNode, const uint& numVoxels1D, const CudaVec<uint>& nodeIndexUsedVoxels, const CudaVec<uint>& voxelIDs, const CudaVec<char>& solids, const CudaVec<float>& divU, CudaVec<float>& p, CudaVec<float>& residuals, const float& radius, const float& density, const float& dt, const Grid& grid, const float& threshold, const uint& maxIterations, cudaStream_t stream){
    float maxResidual = 10.0f;
    float prevMaxResidual = 10.0f;
    uint iterations = 0;
    while(maxResidual > threshold && iterations < maxIterations){
        cudaStreamSynchronize(stream);
        GSiteration<<<nodeIndexUsedVoxels.size(), 32, (3*sizeof(float) + sizeof(uint) + sizeof(char))*numVoxelsPerNode, stream >>>(numVoxelsPerNode, numVoxels1D, radius, nodeIndexUsedVoxels.devPtr(), voxelIDs.devPtr(), solids.devPtr(), divU.devPtr(), p.devPtr(), residuals.devPtr(), density, dt, grid);
        cudaStreamSynchronize(stream);

        cudaParallelMaximum(residuals.size(), residuals.devPtr(), stream);
        prevMaxResidual = maxResidual;
        cudaStreamSynchronize(stream);
        cudaMemcpyAsync(&maxResidual, residuals.devPtr(), sizeof(float), cudaMemcpyDeviceToHost, stream);
        ++iterations;
        cudaStreamSynchronize(stream);
        std::cout<<"Residual: "<<maxResidual<<"\n\tdRes: "<<std::abs(maxResidual - prevMaxResidual)<<"\n\tdt: "<<dt<<"\n\tIteration: "<<iterations<<std::endl;
        if(iterations == maxIterations || (std::abs(prevMaxResidual - maxResidual) < 0.000001f && maxResidual > threshold)){    //if not converging
            return maxResidual;
        }
    }
    return maxResidual;
}

//velocities are along negative faces of voxels
__global__ void pressureToAcceleration(uint numVoxelsPerNode, uint numVoxels1D, float dt, float radius, float density, uint* nodeIndexUsedVoxels, uint* voxelIDs, uint* solids, float* p, float* voxelUs, Grid grid, VelocityGatherDimension dim){
    extern __shared__ float sharedP[];
    __shared__ float* sharedU;
    __shared__ uint* processedVoxels;
    __shared__ char* sharedSolids;
    __shared__ uint voxelIndexStart;
    __shared__ float scale;
    if(threadIdx.x == 0){
        sharedU = sharedP + numVoxelsPerNode;
        processedVoxels = (uint*)(sharedU + numVoxelsPerNode);
        sharedSolids = (char*)(processedVoxels + numVoxelsPerNode);

        float voxelSize = grid.cellSize / (numVoxels1D - 2*floorf(radius));
        scale = dt / (density*voxelSize*voxelSize);
        if(blockIdx.x == 0){
            voxelIndexStart = 0;
        }
        else{
            voxelIndexStart = nodeIndexUsedVoxels[blockIdx.x - 1]; 
        }
    }
    __syncthreads();
    
    for(int i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sharedU[i] = 0.0f;
        sharedP[i] = 0.0f;
        sharedSolids[i] = false;
        processedVoxels[i] = numVoxelsPerNode;
    }
    __syncthreads();
    for(int i = threadIdx.x; voxelIndexStart + i < nodeIndexUsedVoxels[blockIdx.x]; i += blockDim.x){
        processedVoxels[i] = voxelIDs[voxelIndexStart + i];
        sharedU[processedVoxels[i]] = voxelUs[voxelIndexStart + i];
        sharedP[processedVoxels[i]] = p[voxelIndexStart + i];
        sharedSolids[processedVoxels[i]] = solids[voxelIndexStart + i];
    }
    __syncthreads();

    for(int i = threadIdx.x; processedVoxels[i] != numVoxelsPerNode; i += blockDim.x){
        if(!solids[processedVoxels[i]]){
            uint voxelIDx = processedVoxels[i] % numVoxels1D;
            uint voxelIDy = (processedVoxels[i] % (numVoxels1D*numVoxels1D)) / numVoxels1D;
            uint voxelIDz = processedVoxels[i] / (numVoxels1D*numVoxels1D);
            uint apronPad = floorf(radius);
            if(voxelIDx >= apronPad && voxelIDx < numVoxels1D - apronPad){
                if(voxelIDy >= apronPad && voxelIDy < numVoxels1D - apronPad){
                    if(voxelIDz >= apronPad && voxelIDz < numVoxels1D - apronPad){
                        uint voxelID = processedVoxels[i];
                        float dpx = 0.0f;
                        if(dim == VelocityGatherDimension::X){
                            if(!sharedSolids[voxelID-1]){
                                dpx = sharedP[voxelID] - sharedP[voxelID - 1];
                            }
                        }
                        else if(dim == VelocityGatherDimension::Y){
                            if(!sharedSolids[voxelID - numVoxels1D]){
                                dpx = sharedP[voxelID] - sharedP[voxelID - numVoxels1D];
                            }
                        }
                        else{
                            if(!sharedSolids[voxelID - numVoxels1D*numVoxels1D]){
                                dpx = sharedP[voxelID] - sharedP[voxelID - numVoxels1D*numVoxels1D];
                            }
                        }
                        sharedU[voxelID] = sharedU[voxelID] - dpx * scale;
                    }
                }

            }
        }
    }
    __syncthreads();
    for(int i = threadIdx.x; processedVoxels[i] != numVoxelsPerNode; i += blockDim.x){
        voxelUs[voxelIndexStart + i] = sharedU[processedVoxels[i]];
    }
}

void cudaVelocityUpdate(uint numVoxelsPerNode, uint numVoxels1D, float dt, float radius, float density,  CudaVec<uint>& nodeIndexUsedVoxels,  CudaVec<uint>& voxelIDs,  CudaVec<uint>& solids,  CudaVec<float>& p,  CudaVec<float>& voxelsUx, CudaVec<float>& voxelsUy, CudaVec<float>& voxelsUz, Grid grid, cudaStream_t stream){
    pressureToAcceleration<<<nodeIndexUsedVoxels.size(), 32, 4*4*numVoxelsPerNode, stream>>>(numVoxelsPerNode, numVoxels1D, dt, radius, density, nodeIndexUsedVoxels.devPtr(), voxelIDs.devPtr(), solids.devPtr(), p.devPtr(), voxelsUx.devPtr(), grid, VelocityGatherDimension::X);
    pressureToAcceleration<<<nodeIndexUsedVoxels.size(), 32, 4*4*numVoxelsPerNode, stream>>>(numVoxelsPerNode, numVoxels1D, dt, radius, density, nodeIndexUsedVoxels.devPtr(), voxelIDs.devPtr(), solids.devPtr(), p.devPtr(), voxelsUy.devPtr(), grid, VelocityGatherDimension::Y);
    pressureToAcceleration<<<nodeIndexUsedVoxels.size(), 32, 4*4*numVoxelsPerNode, stream>>>(numVoxelsPerNode, numVoxels1D, dt, radius, density, nodeIndexUsedVoxels.devPtr(), voxelIDs.devPtr(), solids.devPtr(), p.devPtr(), voxelsUz.devPtr(), grid, VelocityGatherDimension::Z);
    cudaStreamSynchronize(stream);
}