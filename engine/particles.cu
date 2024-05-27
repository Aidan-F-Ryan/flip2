//Copyright 2023 Aberrant Behavior LLC

#include "particles.hu"

#include "algorithms/radixSortKernels.hu"
#include "algorithms/particleToGridFunctions.hu"
#include "algorithms/perVoxelParticleListFunctions.hu"
#include "algorithms/voxelSolveFunctions.hu"
#include "algorithms/parallelPrefixSumKernels.hu"

#include "typedefs.h"
#include <cmath>
#include <iostream>

Particles::Particles(uint size)
: size(size)
, radius(2)
, refinementLevel(1)
{
    setupCudaDevices();

    cudaStreamCreate(&stream);
    px.resize(size);
    py.resize(size);
    pz.resize(size);
    
    vx.resize(size);
    vy.resize(size);
    vz.resize(size);

    gridCell.resize(size);

    reorderedGridIndices.resize(size);

    uniqueGridNodeIndices.resize(size);

    numVoxels1D = 2*(uint)(std::floor(radius)) + (2<<refinementLevel);
    numVoxelsPerNode = numVoxels1D*numVoxels1D*numVoxels1D;
    frameDt = 1.0f/24.0f;
}

void Particles::setDomain(float nx, float ny, float nz, uint x, uint y, uint z, float cellSize){
    grid.setNegativeCorner(nx, ny, nz);
    grid.setSize(x, y, z, cellSize);
    yDimNumUsedGridNodes.resizeAsync(y*z, stream);
    }

void Particles::alignParticlesToGrid(){
    cudaFindGridCell(px.devPtr(), py.devPtr(), pz.devPtr(), size, grid, gridCell.devPtr(), stream);
}

void Particles::sortParticles(){
    float *d_px, *d_py, *d_pz;
    gpuErrchk(cudaMallocAsync((void**)&d_px, sizeof(float)*size, stream));
    gpuErrchk(cudaMallocAsync((void**)&d_py, sizeof(float)*size, stream));
    gpuErrchk(cudaMallocAsync((void**)&d_pz, sizeof(float)*size, stream));


    uint* tempGridCell = gridCell.devPtr();
    uint* tempSortedIndices = reorderedGridIndices.devPtr();
    cudaSortParticlesByGridNode(size, tempGridCell, tempSortedIndices, stream);
    // cudaStreamSynchronize(stream);

    //@TODO: need to create per-CudaVec stream for allocation/dealloc to get around this overlap issue when freeing on constant stream, leave compute streams in place for all else, CudaVec_stream sync on devPtr call
    gridCell.swapDevicePtrAsync(tempGridCell, stream);
    reorderedGridIndices.swapDevicePtrAsync(tempSortedIndices, stream);

    reorderGridIndices<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, reorderedGridIndices.devPtr(), px.devPtr(), d_px);
    reorderGridIndices<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, reorderedGridIndices.devPtr(), py.devPtr(), d_py);
    reorderGridIndices<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, reorderedGridIndices.devPtr(), pz.devPtr(), d_pz);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaStreamSynchronize(stream));
    px.swapDevicePtrAsync(d_px, stream);
    py.swapDevicePtrAsync(d_py, stream);
    pz.swapDevicePtrAsync(d_pz, stream);
}

__device__ float square(const float& x){
    return x*x;
}

__device__ float distance(const float& dx, const float& dy, const float& dz){
    return sqrtf(fmaf(dx, dx, fmaf(dy, dy, dz*dz)));
}

__global__ void getNumberVoxelsUsedPerNode(uint numUsedGridNodes, uint numParticles, uint numVoxelsPerNode, uint numVoxels1D, uint* gridNodeIndicesToFirstParticleIndex, uint* gridPosition, float* px, float* py, float* pz, uint* numVoxelsEachNode, float radius, Grid grid, uint refinementLevel){
    extern __shared__ uint sharedVoxelCount[];
    __shared__ uint lastParticleIndex;
    __shared__ uint xySize;
    __shared__ uint gridIDx;
    __shared__ uint gridIDy;
    __shared__ uint gridIDz;

    if(threadIdx.x == 0){
        xySize = grid.sizeX*grid.sizeY;
        uint gridID = gridPosition[blockIdx.x];
        gridIDx = gridID % grid.sizeX;
        gridIDy = (gridID % xySize) / grid.sizeX;
        gridIDz = gridID / xySize;
        if(blockIdx.x == numUsedGridNodes - 1){
            lastParticleIndex = numParticles;
        }
        else{
            lastParticleIndex = gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1];
        }
    }
    for(int i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sharedVoxelCount[SM_ADDRESS(i)] = 0;
    }
    __syncthreads();

    for(int i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        uint voxelIDx = i % numVoxels1D;
        uint voxelIDy = (i % numVoxels1D*numVoxels1D) / numVoxels1D;
        uint voxelIDz = i / (numVoxels1D*numVoxels1D);
        uint globalVoxelIDx = gridIDx * numVoxels1D + voxelIDx;
        uint globalVoxelIDy = gridIDy * numVoxels1D + voxelIDy;
        uint globalVoxelIDz = gridIDz * numVoxels1D + voxelIDz;

        if(globalVoxelIDx < floorf(radius) || globalVoxelIDy < floorf(radius) || globalVoxelIDz < floorf(radius)){
            sharedVoxelCount[SM_ADDRESS(i)] = 1;
        }
        else if(globalVoxelIDx > grid.sizeX*numVoxels1D + floorf(radius) || globalVoxelIDy > grid.sizeY*numVoxels1D + floorf(radius) || globalVoxelIDz > grid.sizeZ*numVoxels1D + floorf(radius)){
            sharedVoxelCount[SM_ADDRESS(i)] = 1;
        }
    }

    for(int index = gridNodeIndicesToFirstParticleIndex[blockIdx.x] + threadIdx.x; index < lastParticleIndex; index += blockDim.x){
        uint moduloWRTxySize = gridPosition[index] % xySize;
        uint gridIDz = gridPosition[index] / (xySize);
        uint gridIDy = moduloWRTxySize / grid.sizeX;
        uint gridIDx = moduloWRTxySize % grid.sizeX;
        
        uint apronCells = floorf(radius);
        float subCellWidth = grid.cellSize/(2.0f*(1<<refinementLevel));
        float pxInGridCell = (px[index] - grid.negX - gridIDx*grid.cellSize + apronCells*subCellWidth);
        float pyInGridCell = (py[index] - grid.negY - gridIDy*grid.cellSize + apronCells*subCellWidth);
        float pzInGridCell = (pz[index] - grid.negZ - gridIDz*grid.cellSize + apronCells*subCellWidth);

        uint subCellPositionX = floorf(pxInGridCell/subCellWidth);
        uint subCellPositionY = floorf(pyInGridCell/subCellWidth);
        uint subCellPositionZ = floorf(pzInGridCell/subCellWidth);
        

        float halfSubCellWidth = subCellWidth / 2.0f;
        float radiusSCW = radius*subCellWidth;

        for(int x = subCellPositionX - apronCells; x < subCellPositionX + apronCells; ++x){
            for(int y = subCellPositionY - apronCells; y < subCellPositionY + apronCells; ++y){
                for(int z = subCellPositionZ - apronCells; z < subCellPositionZ + apronCells; ++z){
                    float subCellBaseX = x * subCellWidth;
                    float subCellBaseY = y * subCellWidth;
                    float subCellBaseZ = z * subCellWidth;

                    float dpx = pxInGridCell - subCellBaseX;
                    float dpy = pyInGridCell - subCellBaseY;
                    float dpz = pzInGridCell - subCellBaseZ;

                    float xCheck = distance(dpx, dpy + halfSubCellWidth, dpz + halfSubCellWidth);
                    float yCheck = distance(dpx + halfSubCellWidth, dpy, dpz + halfSubCellWidth);
                    float zCheck = distance(dpx + halfSubCellWidth, dpy + halfSubCellWidth, dpz);

                    if(xCheck < radiusSCW || yCheck < radiusSCW || zCheck < radiusSCW){
                        sharedVoxelCount[SM_ADDRESS(x + y*numVoxels1D + z * numVoxels1D * numVoxels1D)] = 1;
                    }
                }
            }
        }
    }
    __shared__ uint totalNodeUsedVoxels;

    __syncthreads();

    blockWiseExclusivePrefixSum(sharedVoxelCount, numVoxelsPerNode, totalNodeUsedVoxels);

    __syncthreads();

    if(threadIdx.x == 0){
        numVoxelsEachNode[blockIdx.x] = totalNodeUsedVoxels;
    }
}

__global__ void generateVoxelIDs(uint numUsedGridNodes, uint numParticles, uint numVoxelsPerNode, uint numVoxels1D, uint* gridNodeIndicesToFirstParticleIndex, uint* gridPosition, float* px, float* py, float* pz, uint* numVoxelsEachNode, uint* voxelIDs, char* solids, float radius, Grid grid, uint refinementLevel){
    extern __shared__ uint sharedVoxelCount[];
    __shared__ uint* sharedVoxelIndices;
    __shared__ char* sharedSolids;
    __shared__ uint lastParticleIndex;
    __shared__ uint xySize;
    __shared__ uint startVoxelIndex;
    __shared__ uint gridIDx;
    __shared__ uint gridIDy;
    __shared__ uint gridIDz;

    if(threadIdx.x == 0){
        xySize = grid.sizeX*grid.sizeY;
        uint gridID = gridPosition[blockIdx.x];
        gridIDx = gridID % grid.sizeX;
        gridIDy = (gridID % xySize) / grid.sizeX;
        gridIDz = gridID / xySize;
        if(blockIdx.x == numUsedGridNodes - 1){
            lastParticleIndex = numParticles;
        }
        else{
            lastParticleIndex = gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1];
        }
        if(blockIdx.x == 0){
            startVoxelIndex = 0;
        }
        else{
            startVoxelIndex = numVoxelsEachNode[blockIdx.x - 1];
        }
        sharedVoxelIndices = sharedVoxelCount + SM_ADDRESS(numVoxelsPerNode);
        sharedSolids = (char*)(sharedVoxelIndices + numVoxelsPerNode);
    }
    __syncthreads();
    for(int i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sharedVoxelCount[SM_ADDRESS(i)] = 0;
        sharedSolids[i] = false;
    }
    __syncthreads();

    for(int i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        uint voxelIDx = i % numVoxels1D;
        uint voxelIDy = (i % numVoxels1D*numVoxels1D) / numVoxels1D;
        uint voxelIDz = i / (numVoxels1D*numVoxels1D);
        uint globalVoxelIDx = gridIDx * numVoxels1D + voxelIDx;
        uint globalVoxelIDy = gridIDy * numVoxels1D + voxelIDy;
        uint globalVoxelIDz = gridIDz * numVoxels1D + voxelIDz;

        if(globalVoxelIDx < floorf(radius) || globalVoxelIDy < floorf(radius) || globalVoxelIDz < floorf(radius)){
            sharedVoxelCount[SM_ADDRESS(i)] = 1;
            sharedSolids[i] = true;
        }
        else if(globalVoxelIDx > grid.sizeX*numVoxels1D + floorf(radius) || globalVoxelIDy > grid.sizeY*numVoxels1D + floorf(radius) || globalVoxelIDz > grid.sizeZ*numVoxels1D + floorf(radius)){
            sharedVoxelCount[SM_ADDRESS(i)] = 1;
            sharedSolids[i] = true;
        }
    }

    for(int index = gridNodeIndicesToFirstParticleIndex[blockIdx.x] + threadIdx.x; index < lastParticleIndex; index += blockDim.x){
        uint moduloWRTxySize = gridPosition[index] % xySize;
        uint gridIDz = gridPosition[index] / (xySize);
        uint gridIDy = moduloWRTxySize / grid.sizeX;
        uint gridIDx = moduloWRTxySize % grid.sizeX;
        
        uint apronCells = floorf(radius);
        float subCellWidth = grid.cellSize/(2.0f*(1<<refinementLevel));
        float pxInGridCell = (px[index] - grid.negX - gridIDx*grid.cellSize + apronCells*subCellWidth);
        float pyInGridCell = (py[index] - grid.negY - gridIDy*grid.cellSize + apronCells*subCellWidth);
        float pzInGridCell = (pz[index] - grid.negZ - gridIDz*grid.cellSize + apronCells*subCellWidth);

        uint subCellPositionX = floorf(pxInGridCell/subCellWidth);
        uint subCellPositionY = floorf(pyInGridCell/subCellWidth);
        uint subCellPositionZ = floorf(pzInGridCell/subCellWidth);
        

        float halfSubCellWidth = subCellWidth / 2.0f;
        float radiusSCW = radius*subCellWidth;

        for(int x = subCellPositionX - apronCells; x < subCellPositionX + apronCells; ++x){
            for(int y = subCellPositionY - apronCells; y < subCellPositionY + apronCells; ++y){
                for(int z = subCellPositionZ - apronCells; z < subCellPositionZ + apronCells; ++z){
                    float subCellBaseX = x * subCellWidth;
                    float subCellBaseY = y * subCellWidth;
                    float subCellBaseZ = z * subCellWidth;

                    float dpx = pxInGridCell - subCellBaseX;
                    float dpy = pyInGridCell - subCellBaseY;
                    float dpz = pzInGridCell - subCellBaseZ;
                    
                    float xCheck = distance(dpx, dpy + halfSubCellWidth, dpz + halfSubCellWidth);
                    float yCheck = distance(dpx + halfSubCellWidth, dpy, dpz + halfSubCellWidth);
                    float zCheck = distance(dpx + halfSubCellWidth, dpy + halfSubCellWidth, dpz);


                    if(xCheck < radiusSCW || yCheck < radiusSCW || zCheck < radiusSCW){
                        sharedVoxelCount[SM_ADDRESS(x + y*numVoxels1D + z * numVoxels1D * numVoxels1D)] = 1;
                    }
                }
            }
        }
    }
    __shared__ uint totalNodeUsedVoxels;

    __syncthreads();

    blockWiseExclusivePrefixSum(sharedVoxelCount, numVoxelsPerNode, totalNodeUsedVoxels);

    __syncthreads();
    for(int i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        uint compare;
        if(i != numVoxelsPerNode - 1){
            compare = sharedVoxelCount[SM_ADDRESS(i + 1)];
        }
        else{
            compare = totalNodeUsedVoxels;
        }
        if(sharedVoxelCount[SM_ADDRESS(i)] != compare){
            sharedVoxelIndices[sharedVoxelCount[SM_ADDRESS(i)]] = i;//coalesce the shared memory
            // voxelIDs[startVoxelIndex + sharedVoxelCount[i]] = i;
        }
    }
    __syncthreads();
    for(int i = threadIdx.x; i < totalNodeUsedVoxels; i += blockDim.x){
        voxelIDs[startVoxelIndex + i] = sharedVoxelIndices[i];
        solids[startVoxelIndex + i] = sharedSolids[sharedVoxelIndices[i]];
    }
}

void Particles::generateVoxels(){
    numUsedGridNodes = cudaMarkUniqueGridCellsAndCount(size, gridCell.devPtr(), uniqueGridNodeIndices.devPtr(), stream);
    
    gridNodeIndicesToFirstParticleIndex.resizeAsync(numUsedGridNodes, stream);

    cudaMapNodeIndicesToParticles(size, uniqueGridNodeIndices.devPtr(), gridNodeIndicesToFirstParticleIndex.devPtr(), stream);

    nodeIndexUsedVoxels.resizeAsync(numUsedGridNodes, stream);
    cudaStreamSynchronize(stream);
    cudaGetFirstNodeInYRows(numUsedGridNodes, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), yDimNumUsedGridNodes.devPtr(), grid, stream);

    getNumberVoxelsUsedPerNode<<<numUsedGridNodes, 32, SM_ADDRESS(numVoxelsPerNode) * sizeof(uint), stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), nodeIndexUsedVoxels.devPtr(), radius, grid, refinementLevel);
    cudaStreamSynchronize(stream);

    cudaParallelPrefixSum(numUsedGridNodes, nodeIndexUsedVoxels.devPtr(), stream);
    cudaStreamSynchronize(stream);

    uint numUsedVoxels;
    cudaMemcpyAsync(&numUsedVoxels, nodeIndexUsedVoxels.devPtr() + numUsedGridNodes - 1, sizeof(uint), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    voxelIDsUsed.resizeAsync(numUsedVoxels, stream);
    voxelsUx.resizeAsync(numUsedVoxels, stream);
    voxelsUy.resizeAsync(numUsedVoxels, stream);
    voxelsUz.resizeAsync(numUsedVoxels, stream);
    solids.resizeAsync(numUsedVoxels, stream);
    divU.resizeAsync(numUsedVoxels, stream);
    p.resizeAsync(numUsedVoxels, stream);
    residuals.resizeAsync(numUsedVoxels, stream);
    p.zeroDeviceAsync(stream);
    cudaStreamSynchronize(stream);

    generateVoxelIDs<<<numUsedGridNodes, 32, (SM_ADDRESS(numVoxelsPerNode) + numVoxelsPerNode)*sizeof(uint) + numVoxelsPerNode, stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), solids.devPtr(), radius, grid, refinementLevel);
    cudaStreamSynchronize(stream);
    
    // Adiag.resizeAsync(numUsedVoxels, stream);
    // Ax.resizeAsync(numUsedVoxels, stream);
    // Ay.resizeAsync(numUsedVoxels, stream);
    // Az.resizeAsync(numUsedVoxels, stream);
    // Adiag.zeroDeviceAsync(stream);
    // Ax.zeroDeviceAsync(stream);
    // Ay.zeroDeviceAsync(stream);
    // Az.zeroDeviceAsync(stream);

    std::cout<<"Using "<<(CudaVec<uint>::GPU_MEMORY_ALLOCATED + CudaVec<float>::GPU_MEMORY_ALLOCATED + CudaVec<char>::GPU_MEMORY_ALLOCATED) / (1<<20)<<" MB on GPU\n";
}

__device__ float weightFromDistance(float in, float radius){
    if(in > radius){
        return 0.0f;
    }
    else{
        return 1.0f - in/radius;
    }
}

__global__ void gatherParticleVelsToVoxels(uint numUsedGridNodes, uint numParticles, uint numVoxelsPerNode, uint numVoxels1D, uint* gridNodeIndicesToFirstParticleIndex, uint* gridPosition, float* px, float* py, float* pz, float* vDim, uint* numVoxelsEachNode, uint* voxelIDs, float* voxelUs, char* solids, float radius, Grid grid, uint refinementLevel, VelocityGatherDimension dimension){
    extern __shared__ float sharedVoxelU[];
    __shared__ float* sharedWeightSums;
    __shared__ char* sharedSolids;
    __shared__ uint lastParticleIndex;
    __shared__ uint xySize;
    __shared__ uint startVoxelIndex;
    // __shared__ uint gridIDx;
    // __shared__ uint gridIDy;
    // __shared__ uint gridIDz;

    if(threadIdx.x == 0){
        xySize = grid.sizeX*grid.sizeY;
        // uint gridID = gridPosition[blockIdx.x];
        // gridIDx = gridID % grid.sizeX;
        // gridIDy = (gridID % xySize) / grid.sizeX;
        // gridIDz = gridID / xySize;
        if(blockIdx.x == numUsedGridNodes - 1){
            lastParticleIndex = numParticles;
        }
        else{
            lastParticleIndex = gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1];
        }
        if(blockIdx.x == 0){
            startVoxelIndex = 0;
        }
        else{
            startVoxelIndex = numVoxelsEachNode[blockIdx.x - 1];
        }
        sharedWeightSums = sharedVoxelU + numVoxelsPerNode;
        sharedSolids = (char*)(sharedWeightSums + numVoxelsPerNode);
    }
    __syncthreads();
    for(int i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sharedVoxelU[i] = 0.0;
        sharedSolids[i] = false;
    }
    __syncthreads();

    for(int i = startVoxelIndex + threadIdx.x; i < numVoxelsEachNode[blockIdx.x]; i += blockDim.x){
        sharedSolids[voxelIDs[i]] = solids[i];
    }
    
    for(int index = gridNodeIndicesToFirstParticleIndex[blockIdx.x] + threadIdx.x; index < lastParticleIndex; index += blockDim.x){
        uint moduloWRTxySize = gridPosition[index] % xySize;
        uint gridIDz = gridPosition[index] / (xySize);
        uint gridIDy = moduloWRTxySize / grid.sizeX;
        uint gridIDx = moduloWRTxySize % grid.sizeX;
        
        uint apronCells = floorf(radius);
        float subCellWidth = grid.cellSize/(2.0f*(1<<refinementLevel));
        float pxInGridCell = (px[index] - grid.negX - gridIDx*grid.cellSize + apronCells*subCellWidth);
        float pyInGridCell = (py[index] - grid.negY - gridIDy*grid.cellSize + apronCells*subCellWidth);
        float pzInGridCell = (pz[index] - grid.negZ - gridIDz*grid.cellSize + apronCells*subCellWidth);

        uint subCellPositionX = floorf(pxInGridCell/subCellWidth);
        uint subCellPositionY = floorf(pyInGridCell/subCellWidth);
        uint subCellPositionZ = floorf(pzInGridCell/subCellWidth);
        

        float halfSubCellWidth = subCellWidth / 2.0f;
        float radiusSCW = radius*subCellWidth;

        for(int x = subCellPositionX - apronCells; x < subCellPositionX + apronCells; ++x){
            for(int y = subCellPositionY - apronCells; y < subCellPositionY + apronCells; ++y){
                for(int z = subCellPositionZ - apronCells; z < subCellPositionZ + apronCells; ++z){
                    float subCellBaseX = x * subCellWidth;
                    float subCellBaseY = y * subCellWidth;
                    float subCellBaseZ = z * subCellWidth;

                    if(dimension == VelocityGatherDimension::X){
                        subCellBaseY += halfSubCellWidth;
                        subCellBaseZ += halfSubCellWidth;
                    }
                    else if(dimension == VelocityGatherDimension::Y){
                        subCellBaseX += halfSubCellWidth;
                        subCellBaseZ += halfSubCellWidth;
                    }
                    else{
                        subCellBaseY += halfSubCellWidth;
                        subCellBaseX += halfSubCellWidth;
                    }

                    float dpx = pxInGridCell - subCellBaseX;
                    float dpy = pyInGridCell - subCellBaseY;
                    float dpz = pzInGridCell - subCellBaseZ;

                    float weight = weightFromDistance(distance(dpx, dpy, dpz), radiusSCW);

                    atomicAdd(sharedWeightSums + x + y*numVoxels1D + z * numVoxels1D*numVoxels1D, weight);
                    atomicAdd(sharedVoxelU + x + y*numVoxels1D + z * numVoxels1D*numVoxels1D, weight*vDim[index]);
                }
            }
        }
    }
    __syncthreads();
    for(int i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){//wall boundary conditions only, need to rework for reading in solid vel vector
        if(sharedSolids[i]){
            sharedVoxelU[i] = 0.0f;
        }
    }
    __syncthreads();
    for(int i = startVoxelIndex + threadIdx.x; i < numVoxelsEachNode[blockIdx.x]; i += blockDim.x){
        voxelUs[i] = sharedVoxelU[voxelIDs[i]] / (sharedWeightSums[voxelIDs[i]] + 0.0000001);
    }
}



void Particles::particleVelToVoxels(){
    gatherParticleVelsToVoxels<<<numUsedGridNodes, 32, 2*sizeof(float)*numVoxelsPerNode + numVoxelsPerNode, stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelsUx.devPtr(), solids.devPtr(), radius, grid, refinementLevel, VelocityGatherDimension::X);
    gatherParticleVelsToVoxels<<<numUsedGridNodes, 32, 2*sizeof(float)*numVoxelsPerNode + numVoxelsPerNode, stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vy.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelsUy.devPtr(), solids.devPtr(), radius, grid, refinementLevel, VelocityGatherDimension::Y);
    gatherParticleVelsToVoxels<<<numUsedGridNodes, 32, 2*sizeof(float)*numVoxelsPerNode + numVoxelsPerNode, stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vz.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelsUz.devPtr(), solids.devPtr(), radius, grid, refinementLevel, VelocityGatherDimension::Z);
    cudaStreamSynchronize(stream);
}

__global__ void gridVelUpdate(uint numUsedGridNodes, uint* nodeIndexUsedVoxels, uint numVoxelsPerNode, uint numVoxels1D, uint* voxelIDsUsed, float* p, float* voxelUs, VelocityGatherDimension dimension, float dt, float density){
    extern __shared__ float sharedP[];
    __shared__ float* sharedU;
    __shared__ uint voxelIndexStart;
    if(threadIdx.x == 0){
        sharedU = sharedP + numVoxelsPerNode;
        if(blockIdx.x == 0){
            voxelIndexStart = 0;
        }
        else{
            voxelIndexStart = nodeIndexUsedVoxels[blockIdx.x - 1];
        }
    }
    
    for(int i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sharedP[i] = 0.0f;
    }
    __syncthreads();

    for(int voxelID = voxelIndexStart + threadIdx.x; voxelID < nodeIndexUsedVoxels[blockIdx.x]; voxelID += blockDim.x){
        sharedP[voxelIDsUsed[voxelID]] = p[voxelID];
        sharedU[voxelIDsUsed[voxelID]] = voxelUs[voxelID];
    }
    __syncthreads();
    for(int i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        
    }

}

void Particles::pressureSolve(){
    cudaCalcDivU(nodeIndexUsedVoxels, voxelIDsUsed, voxelsUx, voxelsUy, voxelsUz,
            radius, refinementLevel, grid, yDimNumUsedGridNodes, gridNodeIndicesToFirstParticleIndex,
            gridCell, numVoxelsPerNode, numVoxels1D, divU, stream);
    gpuErrchk(cudaPeekAtLastError());
    float density = 0.014;
    float threshold = 0.0001f;
    uint maxIterations = 1024;
    float previousTerminatingResidual = 100;
    float terminatingResidual;
    dt = frameDt / 10.0f;
    //@TODO: need to use courant number for dt from max voxel u and voxel dimensions
    cudaStreamSynchronize(stream);
    while(previousTerminatingResidual - (terminatingResidual = cudaGSiteration(numVoxelsPerNode, numVoxels1D, nodeIndexUsedVoxels, voxelIDsUsed, solids, divU, p, residuals, radius, density, dt, grid, threshold, maxIterations, stream)) > 0){    //while residual getting smaller
        dt /= 2.0f;
        p.zeroDeviceAsync(stream);
        previousTerminatingResidual = terminatingResidual;
        // residuals.zeroDeviceAsync(stream);
        cudaStreamSynchronize(stream);
    }
    std::cout<<"Terminating Residual: "<<terminatingResidual<<"\nThreshold: "<<threshold<<"\n";
    cudaStreamSynchronize(stream);
}

void Particles::updateVoxelVelocities(){

}


void Particles::advectParticles(){
    
}


void Particles::setupCudaDevices(){
    int numDevices;
    gpuErrchk(cudaGetDeviceCount(&numDevices));
    deviceProp.resize(numDevices);
    for(int i = 0; i < numDevices; ++i){
        gpuErrchk( cudaGetDeviceProperties(deviceProp.data() + i, i) );
    }

    for(int i = 0; i < deviceProp.size(); ++i){
        std::cout<<"Device "<<i<<": "<<deviceProp[i].name<<" with "<<deviceProp[i].totalGlobalMem / (1<<20)<<"MB VRAM available"<<std::endl;
    }
}

Particles::~Particles(){
    gpuErrchk( cudaStreamDestroy(stream) );
}