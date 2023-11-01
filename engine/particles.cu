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
    // subCellsTouchedX.resize(size);
    // subCellsTouchedY.resize(size);
    // subCellsTouchedZ.resize(size);

    reorderedGridIndices.resize(size);

    uniqueGridNodeIndices.resize(size);

    numVoxels1D = 2*(uint)(std::floor(radius)) + (2<<refinementLevel);
    numVoxelsPerNode = numVoxels1D*numVoxels1D*numVoxels1D;
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
    cudaMalloc((void**)&d_px, sizeof(float)*size);    
    cudaMalloc((void**)&d_py, sizeof(float)*size);    
    cudaMalloc((void**)&d_pz, sizeof(float)*size);    


    uint* tempGridCell = gridCell.devPtr();
    uint* tempSortedIndices = reorderedGridIndices.devPtr();
    cudaSortParticlesByGridNode(size, tempGridCell, tempSortedIndices, stream);
    gridCell.swapDevicePtr(tempGridCell);
    reorderedGridIndices.swapDevicePtr(tempSortedIndices);

    reorderGridIndices<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, reorderedGridIndices.devPtr(), px.devPtr(), d_px);
    reorderGridIndices<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, reorderedGridIndices.devPtr(), py.devPtr(), d_py);
    reorderGridIndices<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, reorderedGridIndices.devPtr(), pz.devPtr(), d_pz);
    cudaStreamSynchronize(stream);
    px.swapDevicePtr(d_px);
    py.swapDevicePtr(d_py);
    pz.swapDevicePtr(d_pz);
}

// void Particles::alignParticlesToSubCells(){
//     cudaFindSubCell(px.devPtr(), py.devPtr(), pz.devPtr(), size, grid, 
//         gridCell.devPtr(), subCellsTouchedX.devPtr(), subCellsTouchedY.devPtr(),
//         subCellsTouchedZ.devPtr(), subCellsX, subCellsY, subCellsZ, refinementLevel, radius, stream);

// }

__device__ float square(const float& x){
    return x*x;
}

__global__ void getNumberVoxelsUsedPerNode(uint numUsedGridNodes, uint numParticles, uint numVoxelsPerNode, uint numVoxels1D, uint* gridNodeIndicesToFirstParticleIndex, uint* gridPosition, float* px, float* py, float* pz, uint* numVoxelsEachNode, float radius, Grid grid, uint refinementLevel){
    extern __shared__ uint sharedVoxelCount[];
    __shared__ uint lastParticleIndex;
    __shared__ uint xySize;
    if(threadIdx.x == 0){
        xySize = grid.sizeX*grid.sizeY;
        if(blockIdx.x == numUsedGridNodes - 1){
            lastParticleIndex = numParticles;
        }
        else{
            lastParticleIndex = gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1];
        }
    }
    for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sharedVoxelCount[i] = 0;
    }
    __syncthreads();

    for(uint index = gridNodeIndicesToFirstParticleIndex[blockIdx.x] + threadIdx.x; index < lastParticleIndex; index += blockDim.x){
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

                    if(xCheck < radiusSCW || yCheck < radiusSCW || zCheck < radiusSCW){
                        sharedVoxelCount[x + y*numVoxels1D + z * numVoxels1D * numVoxels1D] = 1;
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

__global__ void generateVoxelIDs(uint numUsedGridNodes, uint numParticles, uint numVoxelsPerNode, uint numVoxels1D, uint* gridNodeIndicesToFirstParticleIndex, uint* gridPosition, float* px, float* py, float* pz, uint* numVoxelsEachNode, uint* voxelIDs, float radius, Grid grid, uint refinementLevel){
    extern __shared__ uint sharedVoxelCount[];
    __shared__ uint lastParticleIndex;
    __shared__ uint xySize;
    __shared__ uint startVoxelIndex;
    if(threadIdx.x == 0){
        xySize = grid.sizeX*grid.sizeY;
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
    }
    for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sharedVoxelCount[i] = 0;
    }
    __syncthreads();

    for(uint index = gridNodeIndicesToFirstParticleIndex[blockIdx.x] + threadIdx.x; index < lastParticleIndex; index += blockDim.x){
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

                    if(xCheck < radiusSCW || yCheck < radiusSCW || zCheck < radiusSCW){
                        sharedVoxelCount[x + y*numVoxels1D + z * numVoxels1D * numVoxels1D] = 1;
                    }
                }
            }
        }
    }
    __shared__ uint totalNodeUsedVoxels;

    __syncthreads();

    blockWiseExclusivePrefixSum(sharedVoxelCount, numVoxelsPerNode, totalNodeUsedVoxels);

    __syncthreads();
    for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        uint compare;
        if(i != numVoxelsPerNode - 1){
            compare = sharedVoxelCount[i + 1];
        }
        else{
            compare = totalNodeUsedVoxels;
        }
        if(sharedVoxelCount[i] != compare){
            voxelIDs[startVoxelIndex + sharedVoxelCount[i]] = i;
        }
    }
}

void Particles::generateVoxels(){
    numUsedGridNodes = cudaMarkUniqueGridCellsAndCount(size, gridCell.devPtr(), uniqueGridNodeIndices.devPtr(), stream);
    
    gridNodeIndicesToFirstParticleIndex.resizeAsync(numUsedGridNodes, stream);
    CudaVec<uint> yDimIndices;
    yDimIndices.resizeAsync(grid.sizeY*grid.sizeZ, stream);

    cudaMapNodeIndicesToParticles(size, uniqueGridNodeIndices.devPtr(), gridNodeIndicesToFirstParticleIndex.devPtr(), stream);

    nodeIndexUsedVoxels.resizeAsync(numUsedGridNodes, stream);
    cudaStreamSynchronize(stream);

    getNumberVoxelsUsedPerNode<<<numUsedGridNodes, BLOCKSIZE, numVoxelsPerNode * sizeof(uint), stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), nodeIndexUsedVoxels.devPtr(), radius, grid, refinementLevel);
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
    divU.resizeAsync(numUsedVoxels, stream);
    cudaStreamSynchronize(stream);

    generateVoxelIDs<<<numUsedGridNodes, BLOCKSIZE, numVoxelsPerNode*sizeof(uint), stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), radius, grid, refinementLevel);
    cudaStreamSynchronize(stream);

    std::cout<<"Using "<<(CudaVec<uint>::GPU_MEMORY_ALLOCATED + CudaVec<float>::GPU_MEMORY_ALLOCATED + CudaVec<char>::GPU_MEMORY_ALLOCATED) / (1<<20)<<" MB on GPU\n";
}

// void Particles::particleVelToVoxels(){
//     cudaVoxelUGather(numUsedVoxelsX, numUsedGridNodes, size, numVoxelsPerNode, totalNumParticlesInPerVoxelListsX, gridCell.devPtr(), vx.devPtr(), px.devPtr(),
//         py.devPtr(), pz.devPtr(), nodeIndexToFirstUsedVoxelX.devPtr(), voxelIDsX.devPtr(), voxelParticleListStartX.devPtr(), particleListsX.devPtr(), voxelsUx.devPtr(),
//         grid, grid.sizeX*grid.sizeY, refinementLevel, radius, numVoxels1D, VelocityGatherDimension::X, stream);
//     cudaVoxelUGather(numUsedVoxelsY, numUsedGridNodes, size, numVoxelsPerNode, totalNumParticlesInPerVoxelListsY, gridCell.devPtr(), vy.devPtr(), px.devPtr(),
//         py.devPtr(), pz.devPtr(), nodeIndexToFirstUsedVoxelY.devPtr(), voxelIDsY.devPtr(), voxelParticleListStartY.devPtr(), particleListsY.devPtr(), voxelsUy.devPtr(),
//         grid, grid.sizeX*grid.sizeY, refinementLevel, radius, numVoxels1D, VelocityGatherDimension::Y, stream);
//     cudaVoxelUGather(numUsedVoxelsZ, numUsedGridNodes, size, numVoxelsPerNode, totalNumParticlesInPerVoxelListsZ, gridCell.devPtr(), vz.devPtr(), px.devPtr(),
//         py.devPtr(), pz.devPtr(), nodeIndexToFirstUsedVoxelZ.devPtr(), voxelIDsZ.devPtr(), voxelParticleListStartZ.devPtr(), particleListsZ.devPtr(), voxelsUz.devPtr(),
//         grid, grid.sizeX*grid.sizeY, refinementLevel, radius, numVoxels1D, VelocityGatherDimension::Z, stream);
//     gpuErrchk( cudaStreamSynchronize(stream) );
// }

__device__ float weightFromDistance(float in, float radius){
    if(in > radius){
        return 0.0f;
    }
    else{
        return 1.0f - in/radius;
    }
}

__global__ void gatherParticleVelsToVoxels(uint numUsedGridNodes, uint numParticles, uint numVoxelsPerNode, uint numVoxels1D, uint* gridNodeIndicesToFirstParticleIndex, uint* gridPosition, float* px, float* py, float* pz, float* vDim, uint* numVoxelsEachNode, uint* voxelIDs, float* voxelUs, float radius, Grid grid, uint refinementLevel, VelocityGatherDimension dimension){
    extern __shared__ float sharedVoxelU[];
    __shared__ uint lastParticleIndex;
    __shared__ uint xySize;
    __shared__ uint startVoxelIndex;
    if(threadIdx.x == 0){
        xySize = grid.sizeX*grid.sizeY;
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
    }
    for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sharedVoxelU[i] = 0.0;
    }
    __syncthreads();

    for(uint index = gridNodeIndicesToFirstParticleIndex[blockIdx.x] + threadIdx.x; index < lastParticleIndex; index += blockDim.x){
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

        for(uint x = subCellPositionX - apronCells; x < subCellPositionX + apronCells; ++x){
            for(uint y = subCellPositionY - apronCells; y < subCellPositionY + apronCells; ++y){
                for(uint z = subCellPositionZ - apronCells; z < subCellPositionZ + apronCells; ++z){
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

                    float weightedVx = vDim[index] * weightFromDistance(sqrtf(fmaf(dpx, dpx, fma(dpy, dpy, dpz*dpz))), radius);

                    atomicAdd(sharedVoxelU + x + y*numVoxels1D + z * numVoxels1D*numVoxels1D, weightedVx);
                }
            }
        }
    }
    __syncthreads();
    for(uint i = startVoxelIndex + threadIdx.x; i < numVoxelsEachNode[blockIdx.x]; i += blockDim.x){
        voxelUs[i] = sharedVoxelU[voxelIDs[i]];
    }
}



void Particles::particleVelToVoxels(){
    gatherParticleVelsToVoxels<<<numUsedGridNodes, BLOCKSIZE, sizeof(float)*numVoxelsPerNode, stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelsUx.devPtr(), radius, grid, refinementLevel, VelocityGatherDimension::X);
    gatherParticleVelsToVoxels<<<numUsedGridNodes, BLOCKSIZE, sizeof(float)*numVoxelsPerNode, stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vy.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelsUy.devPtr(), radius, grid, refinementLevel, VelocityGatherDimension::Y);
    gatherParticleVelsToVoxels<<<numUsedGridNodes, BLOCKSIZE, sizeof(float)*numVoxelsPerNode, stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vz.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelsUz.devPtr(), radius, grid, refinementLevel, VelocityGatherDimension::Z);
    cudaStreamSynchronize(stream);
}

// __global__ void cudaTotalNumVoxelsAllDims(uint numVoxelsPerNode, uint* nodeLastVoxelX, uint* voxelIDsX, uint* nodeLastVoxelY, uint* voxelIDsY, uint* nodeLastVoxelZ, uint* voxelIDsZ, uint* totalUsedVoxelsPerNode){
//     extern __shared__ uint voxelUsed[];
//     for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
//         voxelUsed[i] = 0;
//     }
//     __shared__ uint startVoxelIndex;
//     if(threadIdx.x == 0){
//         if(blockIdx.x == 0){
//             startVoxelIndex = 0;
//         }
//         else{
//             startVoxelIndex = nodeLastVoxelX[blockIdx.x - 1];
//         }
//     }
//     __syncthreads();
//     for(uint i = startVoxelIndex + threadIdx.x; i < nodeLastVoxelX[blockIdx.x]; i += blockDim.x){
//         voxelUsed[voxelIDsX[i]] = 1;
//     }
//     __syncthreads();

//     if(threadIdx.x == 0){
//         if(blockIdx.x == 0){
//             startVoxelIndex = 0;
//         }
//         else{
//             startVoxelIndex = nodeLastVoxelY[blockIdx.x - 1];
//         }
//     }
//     __syncthreads();
//     for(uint i = startVoxelIndex + threadIdx.x; i < nodeLastVoxelY[blockIdx.x]; i += blockDim.x){
//         voxelUsed[voxelIDsY[i]] = 1;
//     }
//     __syncthreads();
    
//     if(threadIdx.x == 0){
//         if(blockIdx.x == 0){
//             startVoxelIndex = 0;
//         }
//         else{
//             startVoxelIndex = nodeLastVoxelZ[blockIdx.x - 1];
//         }
//     }
//     __syncthreads();
//     for(uint i = startVoxelIndex + threadIdx.x; i < nodeLastVoxelZ[blockIdx.x]; i += blockDim.x){
//         voxelUsed[voxelIDsZ[i]] = 1;
//     }
//     __syncthreads();
//     __shared__ uint blockSum;
//     blockWiseExclusivePrefixSum(voxelUsed, numVoxelsPerNode, blockSum);
//     __syncthreads();
//     if(threadIdx.x == 0){
//         totalUsedVoxelsPerNode[blockIdx.x] = blockSum;
//     }
// }

// __global__ void cudaGenerateUniversalIndices(uint numVoxelsPerNode, uint* nodeLastVoxelX, uint* voxelIDsX, uint* nodeLastVoxelY, uint* voxelIDsY, uint* nodeLastVoxelZ, uint* voxelIDsZ, uint* totalUsedVoxelsPerNode, uint* voxelIDsUsed){
//     extern __shared__ uint voxelUsed[];
//     for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
//         voxelUsed[i] = 0;
//     }
//     __shared__ uint startVoxelIndex;
//     if(threadIdx.x == 0){
//         if(blockIdx.x == 0){
//             startVoxelIndex = 0;
//         }
//         else{
//             startVoxelIndex = nodeLastVoxelX[blockIdx.x - 1];
//         }
//     }
//     __syncthreads();
//     for(uint i = startVoxelIndex + threadIdx.x; i < nodeLastVoxelX[blockIdx.x]; i += blockDim.x){
//         voxelUsed[voxelIDsX[i]] = 1;
//     }
//     __syncthreads();

//     if(threadIdx.x == 0){
//         if(blockIdx.x == 0){
//             startVoxelIndex = 0;
//         }
//         else{
//             startVoxelIndex = nodeLastVoxelY[blockIdx.x - 1];
//         }
//     }
//     __syncthreads();
//     for(uint i = startVoxelIndex + threadIdx.x; i < nodeLastVoxelY[blockIdx.x]; i += blockDim.x){
//         voxelUsed[voxelIDsY[i]] = 1;
//     }
//     __syncthreads();
    
//     if(threadIdx.x == 0){
//         if(blockIdx.x == 0){
//             startVoxelIndex = 0;
//         }
//         else{
//             startVoxelIndex = nodeLastVoxelZ[blockIdx.x - 1];
//         }
//     }
//     __syncthreads();
//     for(uint i = startVoxelIndex + threadIdx.x; i < nodeLastVoxelZ[blockIdx.x]; i += blockDim.x){
//         voxelUsed[voxelIDsZ[i]] = 1;
//     }
//     __syncthreads();
//     __shared__ uint blockSum;
//     blockWiseExclusivePrefixSum(voxelUsed, numVoxelsPerNode, blockSum);
//     __syncthreads();
//     if(threadIdx.x == 0){
//         if(blockIdx.x == 0){
//             startVoxelIndex = 0;
//         }
//         else{
//             startVoxelIndex = totalUsedVoxelsPerNode[blockIdx.x - 1];
//         }
//     }
//     __syncthreads();
//     for(uint i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){   //should invert
//         uint compare;
//         if(i == numVoxelsPerNode-1){
//             compare = blockSum;
//         }
//         else{
//             compare = voxelUsed[i+1];
//         }
//         if(voxelUsed[i] != compare){
//             voxelIDsUsed[startVoxelIndex + voxelUsed[i]] = i;
//         }
//     }
// }

// void Particles::universalIndexCreation(){   //@TODO: need to switch particle lists/voxel gather to universal indexing
//     CudaVec<uint> totalUsedVoxelsPerNode;
//     totalUsedVoxelsPerNode.resizeAsync(nodeIndexToFirstUsedVoxelX.size(), stream);
//     cudaTotalNumVoxelsAllDims<<<totalUsedVoxelsPerNode.size(), BLOCKSIZE, numVoxelsPerNode * sizeof(uint), stream>>>(numVoxelsPerNode, nodeIndexToFirstUsedVoxelX.devPtr(), voxelIDsX.devPtr(), nodeIndexToFirstUsedVoxelY.devPtr(), voxelIDsY.devPtr(), nodeIndexToFirstUsedVoxelZ.devPtr(), voxelIDsZ.devPtr(), totalUsedVoxelsPerNode.devPtr());

//     cudaParallelPrefixSum(totalUsedVoxelsPerNode.size(), totalUsedVoxelsPerNode.devPtr(), stream);

//     cudaStreamSynchronize(stream);

//     uint totalNumUsedVoxels;
//     cudaMemcpy(&totalNumUsedVoxels, totalUsedVoxelsPerNode.devPtr() + totalUsedVoxelsPerNode.size() - 1, sizeof(uint), cudaMemcpyDeviceToHost);

//     voxelIDsUsed.resizeAsync(totalNumUsedVoxels, stream);
//     divU.resizeAsync(totalNumUsedVoxels, stream);
//     cudaStreamSynchronize(stream);
//     divU.zeroDeviceAsync(stream);

//     //now that the voxel lookup is sized we need to get voxelID filled

//     cudaGenerateUniversalIndices<<<totalUsedVoxelsPerNode.size(), BLOCKSIZE, numVoxelsPerNode * sizeof(uint), stream>>>(numVoxelsPerNode, nodeIndexToFirstUsedVoxelX.devPtr(), voxelIDsX.devPtr(), nodeIndexToFirstUsedVoxelY.devPtr(), voxelIDsY.devPtr(), nodeIndexToFirstUsedVoxelZ.devPtr(), voxelIDsZ.devPtr(), totalUsedVoxelsPerNode.devPtr(), voxelIDsUsed.devPtr());
// }


void Particles::pressureSolve(){
    cudaCalcDivU(nodeIndexUsedVoxels, voxelIDsUsed, voxelsUx, voxelsUy, voxelsUz,
            radius, refinementLevel, grid, yDimNumUsedGridNodes, gridNodeIndicesToFirstParticleIndex,
            gridCell, numVoxelsPerNode, numVoxels1D, divU, Ax, Ay, Az, Adiag, stream);
    gpuErrchk(cudaPeekAtLastError());
    cudaStreamSynchronize(stream);
}


void Particles::setupCudaDevices(){
    int numDevices;
    gpuErrchk(cudaGetDeviceCount(&numDevices));
    deviceProp.resize(numDevices);
    for(uint i = 0; i < numDevices; ++i){
        gpuErrchk( cudaGetDeviceProperties(deviceProp.data() + i, i) );
    }

    for(uint i = 0; i < deviceProp.size(); ++i){
        std::cout<<"Device "<<i<<": "<<deviceProp[i].name<<" with "<<deviceProp[i].totalGlobalMem / (1<<20)<<"MB VRAM available"<<std::endl;
    }
}

Particles::~Particles(){
    gpuErrchk( cudaStreamDestroy(stream) );
}