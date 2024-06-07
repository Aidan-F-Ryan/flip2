//Copyright 2023 Aberrant Behavior LLC

#include "voxelSolveFunctions.hu"
#include <cmath>

__device__ bool isIndexApronCell(uint voxelIndex, const uint& numVoxels1D, const uint& refinementLevel, const double& radius){
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

__device__ void addVoxelDataForUsedNode(const uint thisThreadNodeIndexToHandle, double* sharedBlockVoxelStorage, const uint* nodeIndexToFirstVoxelIndex,
        const uint* voxelIDs, const double* voxelData)
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

__device__ void getNeighboringApronCellData(const uint thisThreadNodeIndexToHandle, const uint numVoxels1D, double* sharedBlockVoxelStorage, const uint* nodeIndexToFirstVoxelIndex,
        const uint* voxelIDs, const double* voxelData, const int xOffset, const int yOffset, const int zOffset)
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

__device__ void loadVoxelDataForThisBlock(const uint numVoxelsPerNode, const uint numVoxels1D, const uint numUniqueGridNodes, double* sharedBlockStorage,
    const uint* nodeIndexToFirstVoxelIndex, const uint* voxelIDs, const double* voxelData, const uint numApronCells,
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
    if(xNeighborRight < grid.sizeX && blockIdx.x < numUniqueGridNodes-1){
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
    if(zNeighborLeft >= 0 && (neighborUniqueID = yDimToFirstNodeIndex[zNeighborLeft*grid.sizeY + curY]) < numUniqueGridNodes){
        getNeighboringApronCellData(neighborUniqueID, numVoxels1D, sharedBlockStorage,
            nodeIndexToFirstVoxelIndex, voxelIDs, voxelData, 0, 0, -totalNumApronCells);
    }
    __syncthreads();
    if(zNeighborRight < grid.sizeZ && (neighborUniqueID = yDimToFirstNodeIndex[zNeighborRight*grid.sizeY + curY]) < numUniqueGridNodes){
        getNeighboringApronCellData(neighborUniqueID, numVoxels1D, sharedBlockStorage,
            nodeIndexToFirstVoxelIndex, voxelIDs, voxelData, 0, 0, totalNumApronCells);
    }
    __syncthreads();
}

__global__ void applyGravityKernel(uint numUsedVoxelsInGrid, const double dt, const char* solids, double* voxelsUz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxelsInGrid){
        if(!solids[index]){
            voxelsUz[index] += -9.8f*dt;
        }
    }
}
__global__ void removeGravityKernel(uint numUsedVoxelsInGrid, const double dt, const char* solids, double* voxelsUz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxelsInGrid){
        if(!solids[index]){
            voxelsUz[index] -= -9.8f*dt;
        }
    }
}

void applyGravity(const CudaVec<char>& solids, CudaVec<double>& voxelsUy, double dt, cudaStream_t stream){
    applyGravityKernel<<<voxelsUy.size() / WORKSIZE + 1, WORKSIZE, 0, stream>>>(voxelsUy.size(), dt, solids.devPtr(), voxelsUy.devPtr());
    cudaStreamSynchronize(stream);
}

void removeGravity(const CudaVec<char>& solids, CudaVec<double>& voxelsUy, double dt, cudaStream_t stream){
    removeGravityKernel<<<voxelsUy.size() / WORKSIZE + 1, WORKSIZE, 0, stream>>>(voxelsUy.size(), dt, solids.devPtr(), voxelsUy.devPtr());
    cudaStreamSynchronize(stream);
}

//velocities are along negative faces of each
__global__ void calculateDivU(uint numVoxelsPerNode, uint numVoxels1D, uint numUsedVoxelsInGrid, const uint* nodeIndexToFirstVoxelIndex,
                                const uint* voxelIDs, const double* voxelUs, double radius, uint refinementLevel, Grid grid,
                                const uint* yDimFirstNodeIndex, const uint* gridNodeIndicesToFirstParticleIndex, const uint* gridNodes,
                                double* divU, VelocityGatherDimension dimension)
{
    extern __shared__ double sharedVoxels[];
    __shared__ double* sharedDivU;
    // __shared__ double dx;
    loadVoxelDataForThisBlock(numVoxelsPerNode, numVoxels1D, numUsedVoxelsInGrid, sharedVoxels, nodeIndexToFirstVoxelIndex, voxelIDs,
        voxelUs, radius, grid, yDimFirstNodeIndex, gridNodeIndicesToFirstParticleIndex, gridNodes);
    if(threadIdx.x == 0){
        sharedDivU = sharedVoxels + numVoxelsPerNode;
        // dx = calcSubCellWidth(refinementLevel, grid);
    }
    __syncthreads();
    
    for(int internalBlockID = threadIdx.x; internalBlockID < numVoxelsPerNode; internalBlockID += blockDim.x){
        sharedDivU[internalBlockID] = 0.0f;
        if(dimension == VelocityGatherDimension::X){
            uint blockIDx = internalBlockID % numVoxels1D;
            if (blockIDx < numVoxels1D - 1){
                double vX = sharedVoxels[internalBlockID + 1];
                sharedDivU[internalBlockID] = vX - sharedVoxels[internalBlockID];
            }
        }
        else if(dimension == VelocityGatherDimension::Y){
            uint blockIDy = internalBlockID % (numVoxels1D * numVoxels1D) / numVoxels1D;
            if(blockIDy < numVoxels1D - 1){
                double vY = sharedVoxels[internalBlockID + numVoxels1D];
                sharedDivU[internalBlockID] = vY - sharedVoxels[internalBlockID];
            }
        }
        else{
            uint blockIDz = internalBlockID / (numVoxels1D * numVoxels1D);
            if(blockIDz < numVoxels1D - 1){
                double vZ = sharedVoxels[internalBlockID + numVoxels1D*numVoxels1D];
                sharedDivU[internalBlockID] = vZ - sharedVoxels[internalBlockID];
            }
        }
        sharedDivU[internalBlockID];
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
                        const CudaVec<double>& voxelsUx, const CudaVec<double>& voxelsUy, const CudaVec<double>& voxelsUz,
                        double radius, uint refinementLevel, const Grid& grid, const CudaVec<uint>& yDimFirstNodeIndex,
                        const CudaVec<uint>& gridNodeIndicesToFirstParticleIndex, const CudaVec<uint>& gridNodes, uint numVoxelsPerNode,
                        uint numVoxels1D, CudaVec<double>& divU,
                        cudaStream_t stream)
{
    divU.zeroDeviceAsync(stream);
    cudaStreamSynchronize(stream);
    calculateDivU<<<nodeIndexToFirstVoxelIndex.size(), BLOCKSIZE, 2*sizeof(double)*numVoxelsPerNode, stream>>>
        (numVoxelsPerNode, numVoxels1D, nodeIndexToFirstVoxelIndex.size(), nodeIndexToFirstVoxelIndex.devPtr(),
        voxelIDs.devPtr(), voxelsUx.devPtr(), radius, refinementLevel, grid, yDimFirstNodeIndex.devPtr(),
        gridNodeIndicesToFirstParticleIndex.devPtr(), gridNodes.devPtr(), divU.devPtr(), VelocityGatherDimension::X);
    gpuErrchk(cudaPeekAtLastError());
    calculateDivU<<<nodeIndexToFirstVoxelIndex.size(), BLOCKSIZE, 2*sizeof(double)*numVoxelsPerNode, stream>>>
        (numVoxelsPerNode, numVoxels1D, nodeIndexToFirstVoxelIndex.size(), nodeIndexToFirstVoxelIndex.devPtr(),
        voxelIDs.devPtr(), voxelsUy.devPtr(), radius, refinementLevel, grid, yDimFirstNodeIndex.devPtr(),
        gridNodeIndicesToFirstParticleIndex.devPtr(), gridNodes.devPtr(), divU.devPtr(), VelocityGatherDimension::Y);
    gpuErrchk(cudaPeekAtLastError());
    calculateDivU<<<nodeIndexToFirstVoxelIndex.size(), BLOCKSIZE, 2*sizeof(double)*numVoxelsPerNode, stream>>>
        (numVoxelsPerNode, numVoxels1D, nodeIndexToFirstVoxelIndex.size(), nodeIndexToFirstVoxelIndex.devPtr(),
        voxelIDs.devPtr(), voxelsUz.devPtr(), radius, refinementLevel, grid, yDimFirstNodeIndex.devPtr(),
        gridNodeIndicesToFirstParticleIndex.devPtr(), gridNodes.devPtr(), divU.devPtr(), VelocityGatherDimension::Z);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaPeekAtLastError());
}

#include "reductionKernels.hu"

__device__ bool isApronCell(const double& radius, const uint& numVoxels1D, const uint& voxelID){
    uint x = voxelID % numVoxels1D;
    uint y = (voxelID % numVoxels1D*numVoxels1D) / numVoxels1D;
    uint z = voxelID / (numVoxels1D*numVoxels1D);
    // uint numApronCells = floor(radius);
    uint numApronCells = 1;
    uint rightbound = numVoxels1D - numApronCells;
    return x < numApronCells || x >= rightbound || y < numApronCells || y >= rightbound || z < numApronCells || z >= rightbound;
    // return x >= rightbound || y >= rightbound || z >= rightbound;
}

__device__ inline bool isRed(const uint& voxelID, const uint& numVoxels1D){
    uint x = voxelID % numVoxels1D;
    uint y = (voxelID % (numVoxels1D*numVoxels1D)) / numVoxels1D;
    uint z = voxelID / (numVoxels1D*numVoxels1D);
    return !((x + y + z) & 1);
}

__global__ void GSiteration(uint numVoxelsPerNode, uint numVoxels1D, uint refinementLevel, double radius, const uint* nodeIndexUsedVoxels, const uint* voxelIDs, const char* solids, const double* divU, double* p, double* residuals, double density, double dt, Grid grid, bool red, double w){
    extern __shared__ double sharedDivU[];
    __shared__ double* sharedP;
    __shared__ double* sharedRes;
    __shared__ uint* processedVoxels;
    __shared__ char* sharedSolids;
    __shared__ uint voxelIndexStart;
    __shared__ double scale;
    if(threadIdx.x == 0){
        sharedP = sharedDivU + numVoxelsPerNode;
        sharedRes = sharedP + numVoxelsPerNode;
        processedVoxels = (uint*)(sharedRes + numVoxelsPerNode);
        sharedSolids = (char*)(processedVoxels + numVoxelsPerNode);

        double voxelSize = calcSubCellWidth(refinementLevel, grid);
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

    for(int i = threadIdx.x; i < numVoxelsPerNode && processedVoxels[i] != numVoxelsPerNode ; i += blockDim.x){
        if((red == isRed(processedVoxels[i], numVoxels1D)) && !sharedSolids[processedVoxels[i]] && !isApronCell(radius, numVoxels1D, processedVoxels[i])){
            uint voxelID = processedVoxels[i];
            double Adiag = 0.0;
            double Apx = 0.0;
            double Apy = 0.0;
            double Apz = 0.0;
            double Anx = 0.0;
            double Any = 0.0;
            double Anz = 0.0;

            if(!sharedSolids[voxelID-1]){
                if(sharedDivU[voxelID-1] != 0.0){
                    Anx = -scale;
                }
                Adiag += scale;
            }
            if(!sharedSolids[voxelID+1]){
                if(sharedDivU[voxelID+1] != 0.0){
                    Apx = -scale;
                }
                Adiag += scale;
            }
            if(!sharedSolids[voxelID - numVoxels1D]){
                if(sharedDivU[voxelID - numVoxels1D] != 0.0){
                    Any = -scale;
                }
                Adiag += scale;
            }
            if(!sharedSolids[voxelID + numVoxels1D]){
                if(sharedDivU[voxelID + numVoxels1D] != 0.0){
                    Apy = -scale;
                }
                Adiag += scale;
            }
            if(!sharedSolids[voxelID - numVoxels1D*numVoxels1D]){
                if(sharedDivU[voxelID - numVoxels1D*numVoxels1D] != 0.0){
                    Anz = -scale;
                }
                Adiag += scale;
            }
            if(!sharedSolids[voxelID + numVoxels1D*numVoxels1D]){
                if(sharedDivU[voxelID + numVoxels1D*numVoxels1D] != 0.0){
                    Apz = -scale;
                }
                Adiag += scale;
            }
            sharedP[voxelID] = sharedP[voxelID] + w*((-sharedDivU[voxelID] - (Anx*sharedP[voxelID-1] + Apx*sharedP[voxelID + 1] + Any*sharedP[voxelID-numVoxels1D] + Apy*sharedP[voxelID+numVoxels1D] + Anz*sharedP[voxelID-numVoxels1D*numVoxels1D] + Apz*sharedP[voxelID+numVoxels1D*numVoxels1D])) / (Adiag + 0.000000001) - sharedP[voxelID]);
            sharedRes[voxelID] = (-sharedDivU[voxelID] - (Adiag + 0.000000001)*sharedP[voxelID] + (Anx*sharedP[voxelID-1] + Apx*sharedP[voxelID + 1] + Any*sharedP[voxelID-numVoxels1D] + Apy*sharedP[voxelID+numVoxels1D] + Anz*sharedP[voxelID-numVoxels1D*numVoxels1D] + Apz*sharedP[voxelID+numVoxels1D*numVoxels1D]));
        }
    }
    __syncthreads();
    for(int i = threadIdx.x; i < numVoxelsPerNode && processedVoxels[i] != numVoxelsPerNode; i += blockDim.x){
        p[voxelIndexStart + i] = sharedP[processedVoxels[i]];
        residuals[voxelIndexStart + i] = sharedRes[processedVoxels[i]];
    }
}

double cudaGSiteration(const uint& numVoxelsPerNode, const uint& numVoxels1D, const uint& refinementLevel, const CudaVec<uint>& nodeIndexUsedVoxels, const CudaVec<uint>& voxelIDs, const CudaVec<char>& solids, const CudaVec<double>& divU, CudaVec<double>& p, CudaVec<double>& residuals, const double& radius, const double& density, const double& dt, const Grid& grid, const double& threshold, const uint& maxIterations, cudaStream_t stream){
    double maxResidual = 10.0;
    double prevMaxResidual = 100.0;
    uint iterations = 0;
    uint batchCheckEvery = 1024;
    double w = 1.9;
    while(maxResidual > threshold && iterations < maxIterations){
        GSiteration<<<nodeIndexUsedVoxels.size(), 256, (3*sizeof(double) + sizeof(uint) + sizeof(char))*numVoxelsPerNode, stream >>>(numVoxelsPerNode, numVoxels1D, refinementLevel, radius, nodeIndexUsedVoxels.devPtr(), voxelIDs.devPtr(), solids.devPtr(), divU.devPtr(), p.devPtr(), residuals.devPtr(), density, dt, grid, true, w);
        GSiteration<<<nodeIndexUsedVoxels.size(), 256, (3*sizeof(double) + sizeof(uint) + sizeof(char))*numVoxelsPerNode, stream >>>(numVoxelsPerNode, numVoxels1D, refinementLevel, radius, nodeIndexUsedVoxels.devPtr(), voxelIDs.devPtr(), solids.devPtr(), divU.devPtr(), p.devPtr(), residuals.devPtr(), density, dt, grid, false, w);
        if(iterations % batchCheckEvery == 0){
            prevMaxResidual = maxResidual;
            cudaStreamSynchronize(stream);
            maxResidual = residuals.getMax(stream, true);
        }
        ++iterations;
        if(iterations == maxIterations || (std::abs(prevMaxResidual - maxResidual) < threshold / 1000 && maxResidual > threshold)){    //if not converging
            // std::cout<<"Breaking because not converging\n";
            return maxResidual;
        }
    }
    return maxResidual;
}

//velocities are along negative faces of voxels
__global__ void pressureToAcceleration(uint numVoxelsPerNode, uint numVoxels1D, double dt, double radius, double density, const uint* nodeIndexUsedVoxels, const uint* voxelIDs, const char* solids, const double* p, double* voxelUs, Grid grid, uint refinementLevel, VelocityGatherDimension dim){
    extern __shared__ double sharedP[];
    __shared__ double* sharedU;
    __shared__ uint* processedVoxels;
    __shared__ char* sharedSolids;
    __shared__ uint voxelIndexStart;
    __shared__ double scale;
    if(threadIdx.x == 0){
        sharedU = sharedP + numVoxelsPerNode;
        processedVoxels = (uint*)(sharedU + numVoxelsPerNode);
        sharedSolids = (char*)(processedVoxels + numVoxelsPerNode);

        double voxelSize = calcSubCellWidth(refinementLevel, grid);
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

    for(int i = threadIdx.x; i < numVoxelsPerNode && processedVoxels[i] != numVoxelsPerNode; i += blockDim.x){
        if(!sharedSolids[processedVoxels[i]] && !isApronCell(radius, numVoxels1D, processedVoxels[i])){
            uint voxelID = processedVoxels[i];
            double dpx = 0.0f;
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
    __syncthreads();
    for(int i = threadIdx.x; i < numVoxelsPerNode && processedVoxels[i] != numVoxelsPerNode; i += blockDim.x){
        voxelUs[voxelIndexStart + i] = sharedU[processedVoxels[i]];
    }
}

void cudaVelocityUpdate(uint numVoxelsPerNode, uint numVoxels1D, double dt, double radius, double density,  const CudaVec<uint>& nodeIndexUsedVoxels,  const CudaVec<uint>& voxelIDs,  const CudaVec<char>& solids,  const CudaVec<double>& p,  CudaVec<double>& voxelsUx, CudaVec<double>& voxelsUy, CudaVec<double>& voxelsUz, uint refinementLevel, Grid grid, cudaStream_t stream){
    pressureToAcceleration<<<nodeIndexUsedVoxels.size(), 32, 3*sizeof(double)*numVoxelsPerNode + sizeof(char)*numVoxelsPerNode, stream>>>(numVoxelsPerNode, numVoxels1D, dt, radius, density, nodeIndexUsedVoxels.devPtr(), voxelIDs.devPtr(), solids.devPtr(), p.devPtr(), voxelsUx.devPtr(), grid, refinementLevel, VelocityGatherDimension::X);
    pressureToAcceleration<<<nodeIndexUsedVoxels.size(), 32, 3*sizeof(double)*numVoxelsPerNode + sizeof(char)*numVoxelsPerNode, stream>>>(numVoxelsPerNode, numVoxels1D, dt, radius, density, nodeIndexUsedVoxels.devPtr(), voxelIDs.devPtr(), solids.devPtr(), p.devPtr(), voxelsUy.devPtr(), grid, refinementLevel, VelocityGatherDimension::Y);
    pressureToAcceleration<<<nodeIndexUsedVoxels.size(), 32, 3*sizeof(double)*numVoxelsPerNode + sizeof(char)*numVoxelsPerNode, stream>>>(numVoxelsPerNode, numVoxels1D, dt, radius, density, nodeIndexUsedVoxels.devPtr(), voxelIDs.devPtr(), solids.devPtr(), p.devPtr(), voxelsUz.devPtr(), grid, refinementLevel, VelocityGatherDimension::Z);
    cudaStreamSynchronize(stream);
}