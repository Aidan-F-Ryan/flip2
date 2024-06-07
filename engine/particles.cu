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

void Particles::setDomain(double nx, double ny, double nz, uint x, uint y, uint z, double cellSize){
    grid.setNegativeCorner(nx, ny, nz);
    grid.setSize(x, y, z, cellSize);
    yDimNumUsedGridNodes.resizeAsync(y*z, stream);
    }

void Particles::alignParticlesToGrid(){
    cudaFindGridCell(px.devPtr(), py.devPtr(), pz.devPtr(), size, grid, gridCell.devPtr(), stream);
}

void Particles::sortParticles(){
    double *d_px, *d_py, *d_pz;
    gpuErrchk(cudaMallocAsync((void**)&d_px, sizeof(double)*size, stream));
    gpuErrchk(cudaMallocAsync((void**)&d_py, sizeof(double)*size, stream));
    gpuErrchk(cudaMallocAsync((void**)&d_pz, sizeof(double)*size, stream));


    uint* tempGridCell = gridCell.devPtr();
    uint* tempSortedIndices = reorderedGridIndices.devPtr();
    cudaStreamSynchronize(stream);
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
    cudaStreamSynchronize(stream);
}

__device__ double square(const double& x){
    return x*x;
}

__device__ double distance(const double& dx, const double& dy, const double& dz){
    return sqrt(fma(dx, dx, fma(dy, dy, dz*dz)));
}

__global__ void getNumberVoxelsUsedPerNode(uint numUsedGridNodes, uint numParticles, uint numVoxelsPerNode, uint numVoxels1D, uint* gridNodeIndicesToFirstParticleIndex, uint* gridPosition, double* px, double* py, double* pz, uint* numVoxelsEachNode, double radius, Grid grid, uint refinementLevel){
    extern __shared__ uint sharedVoxelCount[];
    __shared__ char* sharedSolids;
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
        sharedSolids = (char*)(sharedVoxelCount + numVoxelsPerNode);
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

        if(globalVoxelIDx < floor(radius) || globalVoxelIDy < floor(radius) || globalVoxelIDz < floor(radius)){
            sharedVoxelCount[SM_ADDRESS(i)] = 1;
        }
        else if(globalVoxelIDx > grid.sizeX*numVoxels1D + floor(radius) || globalVoxelIDy > grid.sizeY*numVoxels1D + floor(radius) || globalVoxelIDz > grid.sizeZ*numVoxels1D + floor(radius)){
            sharedVoxelCount[SM_ADDRESS(i)] = 1;
        }
    }

    for(int index = gridNodeIndicesToFirstParticleIndex[blockIdx.x] + threadIdx.x; index < lastParticleIndex; index += blockDim.x){
        uint moduloWRTxySize = gridPosition[index] % xySize;
        uint gridIDz = gridPosition[index] / (xySize);
        uint gridIDy = moduloWRTxySize / grid.sizeX;
        uint gridIDx = moduloWRTxySize % grid.sizeX;
        
        uint apronCells = floor(radius);
        double subCellWidth = calcSubCellWidth(refinementLevel, grid);
        double pxInGridCell = (px[index] - grid.negX - gridIDx*grid.cellSize + apronCells*subCellWidth);
        double pyInGridCell = (py[index] - grid.negY - gridIDy*grid.cellSize + apronCells*subCellWidth);
        double pzInGridCell = (pz[index] - grid.negZ - gridIDz*grid.cellSize + apronCells*subCellWidth);

        uint subCellPositionX = floor(pxInGridCell/subCellWidth);
        uint subCellPositionY = floor(pyInGridCell/subCellWidth);
        uint subCellPositionZ = floor(pzInGridCell/subCellWidth);

        double halfSubCellWidth = subCellWidth / 2.0;
        double radiusSCW = radius*subCellWidth;

        for(int x = subCellPositionX - apronCells; x < subCellPositionX + apronCells; ++x){
            for(int y = subCellPositionY - apronCells; y < subCellPositionY + apronCells; ++y){
                for(int z = subCellPositionZ - apronCells; z < subCellPositionZ + apronCells; ++z){
                    double subCellBaseX = x * subCellWidth;
                    double subCellBaseY = y * subCellWidth;
                    double subCellBaseZ = z * subCellWidth;

                    double dpx = pxInGridCell - subCellBaseX;
                    double dpy = pyInGridCell - subCellBaseY;
                    double dpz = pzInGridCell - subCellBaseZ;

                    double xCheck = distance(dpx, dpy + halfSubCellWidth, dpz + halfSubCellWidth);
                    double yCheck = distance(dpx + halfSubCellWidth, dpy, dpz + halfSubCellWidth);
                    double zCheck = distance(dpx + halfSubCellWidth, dpy + halfSubCellWidth, dpz);

                    if(xCheck < radiusSCW || yCheck < radiusSCW || zCheck < radiusSCW)
                    {
                        if(x < numVoxels1D && y < numVoxels1D && z < numVoxels1D){
                            sharedVoxelCount[SM_ADDRESS(x + y*numVoxels1D + z * numVoxels1D * numVoxels1D)] = 1;
                        }
                        else{
                            printf("Error with particle numVoxelsUsed %f %f %f\n", px[index], py[index], pz[index]);
                        }
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

__global__ void generateVoxelIDs(uint numUsedGridNodes, uint numParticles, uint numVoxelsPerNode, uint numVoxels1D, uint* gridNodeIndicesToFirstParticleIndex, uint* gridPosition, double* px, double* py, double* pz, uint* numVoxelsEachNode, uint* voxelIDs, char* solids, double radius, Grid grid, uint refinementLevel){
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

        if(globalVoxelIDx < floor(radius) || globalVoxelIDy < floor(radius) || globalVoxelIDz < floor(radius)){
            sharedVoxelCount[SM_ADDRESS(i)] = 1;
            sharedSolids[i] = true;
        }
        else if(globalVoxelIDx > grid.sizeX*numVoxels1D + floor(radius) || globalVoxelIDy > grid.sizeY*numVoxels1D + floor(radius) || globalVoxelIDz > grid.sizeZ*numVoxels1D + floor(radius)){
            sharedVoxelCount[SM_ADDRESS(i)] = 1;
            sharedSolids[i] = true;
        }
    }

    for(int index = gridNodeIndicesToFirstParticleIndex[blockIdx.x] + threadIdx.x; index < lastParticleIndex; index += blockDim.x){
        uint moduloWRTxySize = gridPosition[index] % xySize;
        uint gridIDz = gridPosition[index] / (xySize);
        uint gridIDy = moduloWRTxySize / grid.sizeX;
        uint gridIDx = moduloWRTxySize % grid.sizeX;
        
        uint apronCells = floor(radius);
        double subCellWidth = calcSubCellWidth(refinementLevel, grid);
        double pxInGridCell = (px[index] - grid.negX - gridIDx*grid.cellSize + apronCells*subCellWidth);
        double pyInGridCell = (py[index] - grid.negY - gridIDy*grid.cellSize + apronCells*subCellWidth);
        double pzInGridCell = (pz[index] - grid.negZ - gridIDz*grid.cellSize + apronCells*subCellWidth);

        uint subCellPositionX = floor(pxInGridCell/subCellWidth);
        uint subCellPositionY = floor(pyInGridCell/subCellWidth);
        uint subCellPositionZ = floor(pzInGridCell/subCellWidth);
        

        double halfSubCellWidth = subCellWidth / 2.0f;
        double radiusSCW = radius*subCellWidth;

        for(int x = subCellPositionX - apronCells; x < subCellPositionX + apronCells; ++x){
            for(int y = subCellPositionY - apronCells; y < subCellPositionY + apronCells; ++y){
                for(int z = subCellPositionZ - apronCells; z < subCellPositionZ + apronCells; ++z){
                    double subCellBaseX = x * subCellWidth;
                    double subCellBaseY = y * subCellWidth;
                    double subCellBaseZ = z * subCellWidth;

                    double dpx = pxInGridCell - subCellBaseX;
                    double dpy = pyInGridCell - subCellBaseY;
                    double dpz = pzInGridCell - subCellBaseZ;
                    
                    double xCheck = distance(dpx, dpy + halfSubCellWidth, dpz + halfSubCellWidth);
                    double yCheck = distance(dpx + halfSubCellWidth, dpy, dpz + halfSubCellWidth);
                    double zCheck = distance(dpx + halfSubCellWidth, dpy + halfSubCellWidth, dpz);


                    if(xCheck < radiusSCW || yCheck < radiusSCW || zCheck < radiusSCW){
                        if(x < numVoxels1D && y < numVoxels1D && z < numVoxels1D)
                            sharedVoxelCount[SM_ADDRESS(x + y*numVoxels1D + z * numVoxels1D * numVoxels1D)] = 1;
                        else{
                            printf("Error with particle voxelIDs %f %f %f\n", px[index], py[index], pz[index]);
                        }
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

    std::cout<<"Using "<<(CudaVec<uint>::GPU_MEMORY_ALLOCATED + CudaVec<double>::GPU_MEMORY_ALLOCATED + CudaVec<char>::GPU_MEMORY_ALLOCATED) / (1<<20)<<" MB on GPU\n";
}

__device__ double weightFromDistance(double in, double radius){
    if(in > radius){
        return 0.0f;
    }
    else{
        return 1.0f - in/radius;
    }
}

__global__ void gatherParticleVelsToVoxels(uint numUsedGridNodes, uint numParticles, uint numVoxelsPerNode, uint numVoxels1D, uint* gridNodeIndicesToFirstParticleIndex, uint* gridPosition, double* px, double* py, double* pz, double* vDim, uint* numVoxelsEachNode, uint* voxelIDs, double* voxelUs, char* solids, double radius, Grid grid, uint refinementLevel, VelocityGatherDimension dimension){
    extern __shared__ double sharedVoxelU[];
    __shared__ double* sharedWeightSums;
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
        
        uint apronCells = floor(radius);
        double subCellWidth = calcSubCellWidth(refinementLevel, grid);
        double pxInGridCell = (px[index] - grid.negX - gridIDx*grid.cellSize + apronCells*subCellWidth);
        double pyInGridCell = (py[index] - grid.negY - gridIDy*grid.cellSize + apronCells*subCellWidth);
        double pzInGridCell = (pz[index] - grid.negZ - gridIDz*grid.cellSize + apronCells*subCellWidth);

        uint subCellPositionX = floor(pxInGridCell/subCellWidth);
        uint subCellPositionY = floor(pyInGridCell/subCellWidth);
        uint subCellPositionZ = floor(pzInGridCell/subCellWidth);
        

        double halfSubCellWidth = subCellWidth / 2.0f;
        double radiusSCW = radius*subCellWidth;

        for(int x = subCellPositionX - apronCells; x < subCellPositionX + apronCells; ++x){
            for(int y = subCellPositionY - apronCells; y < subCellPositionY + apronCells; ++y){
                for(int z = subCellPositionZ - apronCells; z < subCellPositionZ + apronCells; ++z){
                    double subCellBaseX = x * subCellWidth;
                    double subCellBaseY = y * subCellWidth;
                    double subCellBaseZ = z * subCellWidth;

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

                    double dpx = pxInGridCell - subCellBaseX;
                    double dpy = pyInGridCell - subCellBaseY;
                    double dpz = pzInGridCell - subCellBaseZ;

                    double weight = weightFromDistance(distance(dpx, dpy, dpz), radiusSCW);
                    if(x < numVoxels1D && y < numVoxels1D && z < numVoxels1D){
                        atomicAdd(sharedWeightSums + x + y*numVoxels1D + z * numVoxels1D*numVoxels1D, weight);
                        atomicAdd(sharedVoxelU + x + y*numVoxels1D + z * numVoxels1D*numVoxels1D, weight*vDim[index]);
                    }
                    else{
                        printf("Error with particle pV->vU %f %f %f\n", px[index], py[index], pz[index]);
                    }
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

#include "algorithms/reductionKernels.hu"
#include <cmath>

double Particles::getCourantDt(){
    double maxVel = max(abs(vx.getMax(stream, true)), max(abs(vy.getMax(stream, true)), abs(vz.getMax(stream, true))));
    double voxelSize = (grid.cellSize / (numVoxels1D - 2*std::floor(radius)));
    std::cout<<"maxVel: "<<maxVel<<" voxelSize: "<<voxelSize<<"\n";
    return 0.7 * (voxelSize / maxVel);
}

void Particles::particleVelToVoxels(){
    gatherParticleVelsToVoxels<<<numUsedGridNodes, 32, 2*sizeof(double)*numVoxelsPerNode + numVoxelsPerNode, stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelsUx.devPtr(), solids.devPtr(), radius, grid, refinementLevel, VelocityGatherDimension::X);
    gatherParticleVelsToVoxels<<<numUsedGridNodes, 32, 2*sizeof(double)*numVoxelsPerNode + numVoxelsPerNode, stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vy.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelsUy.devPtr(), solids.devPtr(), radius, grid, refinementLevel, VelocityGatherDimension::Y);
    gatherParticleVelsToVoxels<<<numUsedGridNodes, 32, 2*sizeof(double)*numVoxelsPerNode + numVoxelsPerNode, stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vz.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelsUz.devPtr(), solids.devPtr(), radius, grid, refinementLevel, VelocityGatherDimension::Z);
    cudaStreamSynchronize(stream);
}

void Particles::pressureSolve(){
    double density = 0.014;
    double threshold = 0.01;
    uint maxIterations = 16384;
    double previousTerminatingResidual = 100;
    double terminatingResidual;
    //@TODO: need to use courant number for dt from max voxel u and voxel dimensions
    dt = getCourantDt();
    std::cout<<"initial dt: "<<dt<<std::endl;
    cudaStreamSynchronize(stream);
    if(frameDt - elapsedTimeThisFrame < dt){
        dt = frameDt - elapsedTimeThisFrame;
    }
    applyGravity(solids, voxelsUy, dt, stream);
    cudaStreamSynchronize(stream);
    gpuErrchk(cudaPeekAtLastError());
    cudaCalcDivU(nodeIndexUsedVoxels, voxelIDsUsed, voxelsUx, voxelsUy, voxelsUz,
            radius, refinementLevel, grid, yDimNumUsedGridNodes, gridNodeIndicesToFirstParticleIndex,
            gridCell, numVoxelsPerNode, numVoxels1D, divU, stream);
    cudaStreamSynchronize(stream);
    gpuErrchk(cudaPeekAtLastError());
    while(previousTerminatingResidual - (terminatingResidual = cudaGSiteration(numVoxelsPerNode, numVoxels1D, refinementLevel, nodeIndexUsedVoxels, voxelIDsUsed, solids, divU, p, residuals, radius, density, dt, grid, threshold, maxIterations, stream)) > 0){    //while residual getting smaller
        gpuErrchk(cudaPeekAtLastError());
        if(terminatingResidual < threshold){
            break;
        }
        removeGravity(solids, voxelsUy, dt, stream);
        gpuErrchk(cudaPeekAtLastError());
        dt /= 2.0f;
        p.zeroDeviceAsync(stream);
        gpuErrchk(cudaPeekAtLastError());
        applyGravity(solids, voxelsUy, dt, stream);
        gpuErrchk(cudaPeekAtLastError());
        previousTerminatingResidual = terminatingResidual;
        // residuals.zeroDeviceAsync(stream);
        cudaStreamSynchronize(stream);
        cudaCalcDivU(nodeIndexUsedVoxels, voxelIDsUsed, voxelsUx, voxelsUy, voxelsUz,
                radius, refinementLevel, grid, yDimNumUsedGridNodes, gridNodeIndicesToFirstParticleIndex,
                gridCell, numVoxelsPerNode, numVoxels1D, divU, stream);
        gpuErrchk(cudaPeekAtLastError());
        cudaStreamSynchronize(stream);
        gpuErrchk(cudaPeekAtLastError());
    }
    elapsedTime += dt;
    elapsedTimeThisFrame += dt;
    std::cout<<"dt: "<<dt<<" ElapsedTime: "<<elapsedTime<<" elapsedTimeThisFrame: "<<elapsedTimeThisFrame<<"\nTerminating Residual: "<<terminatingResidual<<"\nThreshold: "<<threshold<<"\n";
    cudaStreamSynchronize(stream);
    gpuErrchk(cudaPeekAtLastError());
}

void Particles::updateVoxelVelocities(){
    cudaVelocityUpdate(numVoxelsPerNode, numVoxels1D, dt, radius, 0.014, nodeIndexUsedVoxels, voxelIDsUsed, solids, p, voxelsUx, voxelsUy, voxelsUz, refinementLevel, grid, stream);
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void gatherVoxelVelsToParticles(uint numUsedGridNodes, uint numParticles, uint numVoxelsPerNode, uint numVoxels1D, uint* gridNodeIndicesToFirstParticleIndex, uint* gridPosition, double* px, double* py, double* pz, double* vDim, uint* numVoxelsEachNode, uint* voxelIDs, double* voxelUs, double radius, Grid grid, uint refinementLevel, VelocityGatherDimension dimension){
    extern __shared__ double sharedVoxelU[];
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
    }
    __syncthreads();
    for(int i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sharedVoxelU[i] = 0.0;
    }
    __syncthreads();

    for(int i = startVoxelIndex + threadIdx.x; i < numVoxelsEachNode[blockIdx.x]; i += blockDim.x){
        sharedVoxelU[voxelIDs[i]] = voxelUs[i];
    }
    
    for(int index = gridNodeIndicesToFirstParticleIndex[blockIdx.x] + threadIdx.x; index < lastParticleIndex; index += blockDim.x){
        uint moduloWRTxySize = gridPosition[index] % xySize;
        uint gridIDz = gridPosition[index] / (xySize);
        uint gridIDy = moduloWRTxySize / grid.sizeX;
        uint gridIDx = moduloWRTxySize % grid.sizeX;
        
        uint apronCells = floor(radius);
        double subCellWidth = calcSubCellWidth(refinementLevel, grid);
        double pxInGridCell = (px[index] - grid.negX - gridIDx*grid.cellSize + apronCells*subCellWidth);
        double pyInGridCell = (py[index] - grid.negY - gridIDy*grid.cellSize + apronCells*subCellWidth);
        double pzInGridCell = (pz[index] - grid.negZ - gridIDz*grid.cellSize + apronCells*subCellWidth);

        uint subCellPositionX = floor(pxInGridCell/subCellWidth);
        uint subCellPositionY = floor(pyInGridCell/subCellWidth);
        uint subCellPositionZ = floor(pzInGridCell/subCellWidth);
        

        double halfSubCellWidth = subCellWidth / 2.0f;
        double radiusSCW = radius*subCellWidth;

        double particleVel = 0.0f;
        double weightSum = 0.0f;

        for(int x = subCellPositionX - apronCells; x < numVoxels1D && x < subCellPositionX + apronCells; ++x){
            for(int y = subCellPositionY - apronCells; y < numVoxels1D && y < subCellPositionY + apronCells; ++y){
                for(int z = subCellPositionZ - apronCells; z < numVoxels1D && z < subCellPositionZ + apronCells; ++z){
                    double subCellBaseX = x * subCellWidth;
                    double subCellBaseY = y * subCellWidth;
                    double subCellBaseZ = z * subCellWidth;

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

                    double dpx = pxInGridCell - subCellBaseX;
                    double dpy = pyInGridCell - subCellBaseY;
                    double dpz = pzInGridCell - subCellBaseZ;

                    double weight = weightFromDistance(distance(dpx, dpy, dpz), radiusSCW);
                    if(x < numVoxels1D && y < numVoxels1D && z < numVoxels1D){
                        if(sharedVoxelU[x + y*numVoxels1D + z*numVoxels1D*numVoxels1D] != 0.0){
                            weightSum += weight;
                            particleVel += weight*sharedVoxelU[x + y*numVoxels1D + z*numVoxels1D*numVoxels1D];
                        }
                    }
                    else{
                        printf("Error handling particle vU->pV %f %f %f %d %d %d\n", px[index], py[index], pz[index], x, y, z);
                    }
                }
            }
        }
        vDim[index] = particleVel / (weightSum + 0.00000001);
    }
}

__global__ void advectParticlePositions(uint numParticles, double dt, double* position, double* v, Grid grid){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        position[index] += dt*v[index];
    }
}

#include <curand.h>
#include <curand_kernel.h>

__global__ void moveSolids(uint numUsedGridNodes, uint numParticles, uint numVoxelsPerNode, uint numVoxels1D, uint* gridNodeIndicesToFirstParticleIndex, uint* gridPosition, double* px, double* py, double* pz, uint* numVoxelsEachNode, uint* voxelIDs, char* solids, double radius, Grid grid, uint refinementLevel){
    extern __shared__ char sharedSolids[];
    __shared__ uint* processedVoxels;
    __shared__ uint lastParticleIndex;
    __shared__ uint xySize;
    __shared__ uint startVoxelIndex;
    curandState state;
    curand_init(1291293829, threadIdx.x + blockIdx.x*blockDim.x, 0, &state);
    if(threadIdx.x == 0){
        xySize = grid.sizeX*grid.sizeY;
        processedVoxels = (uint*)(sharedSolids + numVoxelsPerNode);
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
    __syncthreads();
    for(int i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sharedSolids[i] = 0;
        processedVoxels[i] = numVoxelsPerNode;
    }
    __syncthreads();

    for(int i = startVoxelIndex + threadIdx.x; i < numVoxelsEachNode[blockIdx.x]; i += blockDim.x){
        sharedSolids[voxelIDs[i]] = solids[i];
        processedVoxels[i - startVoxelIndex] = voxelIDs[i];
    }
    __syncthreads();
    for(int index = gridNodeIndicesToFirstParticleIndex[blockIdx.x] + threadIdx.x; index < lastParticleIndex; index += blockDim.x){
        uint moduloWRTxySize = gridPosition[index] % xySize;
        uint gridIDz = gridPosition[index] / (xySize);
        uint gridIDy = moduloWRTxySize / grid.sizeX;
        uint gridIDx = moduloWRTxySize % grid.sizeX;
        
        int apronCells = floor(radius);
        double subCellWidth = calcSubCellWidth(refinementLevel, grid);
        double pxInGridCell = (px[index] - grid.negX - gridIDx*grid.cellSize + apronCells*subCellWidth);
        double pyInGridCell = (py[index] - grid.negY - gridIDy*grid.cellSize + apronCells*subCellWidth);
        double pzInGridCell = (pz[index] - grid.negZ - gridIDz*grid.cellSize + apronCells*subCellWidth);

        uint subCellPositionX = floor(pxInGridCell/subCellWidth);
        uint subCellPositionY = floor(pyInGridCell/subCellWidth);
        uint subCellPositionZ = floor(pzInGridCell/subCellWidth);
        
        if((subCellPositionX >= numVoxels1D || subCellPositionY >= numVoxels1D || subCellPositionZ >= numVoxels1D) || sharedSolids[subCellPositionX + subCellPositionY*numVoxels1D + subCellPositionZ*numVoxels1D*numVoxels1D]){
            //if particle position is out of voxel (shouldn't happen thanks to courant) or is in a solid voxel, need to move it to a fluid voxel
            int i = 0;
            for(; i < numVoxelsEachNode[blockIdx.x] && !sharedSolids[processedVoxels[i]]; ++i){
            }
            int newVoxelX = processedVoxels[i] % numVoxels1D;
            int newVoxelY = (processedVoxels[i] % (numVoxels1D*numVoxels1D)) / numVoxels1D;
            int newVoxelZ = processedVoxels[i] / (numVoxels1D*numVoxels1D);

            px[index] = grid.negX + gridIDx*grid.cellSize + (newVoxelX - apronCells)*subCellWidth + curand_uniform(&state)*subCellWidth;
            py[index] = grid.negY + gridIDy*grid.cellSize + (newVoxelY - apronCells)*subCellWidth + curand_uniform(&state)*subCellWidth;
            pz[index] = grid.negZ + gridIDz*grid.cellSize + (newVoxelZ - apronCells)*subCellWidth + curand_uniform(&state)*subCellWidth;
        }
    }
}

void Particles::voxelVelsToParticles(){
    gatherVoxelVelsToParticles<<<numUsedGridNodes, 32, 2*sizeof(double)*numVoxelsPerNode + numVoxelsPerNode, stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelsUx.devPtr(), radius, grid, refinementLevel, VelocityGatherDimension::X);
    gpuErrchk(cudaPeekAtLastError());
    gatherVoxelVelsToParticles<<<numUsedGridNodes, 32, 2*sizeof(double)*numVoxelsPerNode + numVoxelsPerNode, stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vy.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelsUy.devPtr(), radius, grid, refinementLevel, VelocityGatherDimension::Y);
    gpuErrchk(cudaPeekAtLastError());
    gatherVoxelVelsToParticles<<<numUsedGridNodes, 32, 2*sizeof(double)*numVoxelsPerNode + numVoxelsPerNode, stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vz.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelsUz.devPtr(), radius, grid, refinementLevel, VelocityGatherDimension::Z);
    gpuErrchk(cudaPeekAtLastError());
}

void Particles::advectParticles(){
    advectParticlePositions<<<size / WORKSIZE + 1, WORKSIZE, 0, stream>>>(size, dt, px.devPtr(), vx.devPtr(), grid);
    gpuErrchk(cudaPeekAtLastError());
    advectParticlePositions<<<size / WORKSIZE + 1, WORKSIZE, 0, stream>>>(size, dt, py.devPtr(), vy.devPtr(), grid);
    gpuErrchk(cudaPeekAtLastError());
    advectParticlePositions<<<size / WORKSIZE + 1, WORKSIZE, 0, stream>>>(size, dt, pz.devPtr(), vz.devPtr(), grid);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaPeekAtLastError());
}

void Particles::moveSolidParticles(){
    //move particles that move to solid cell/out of bounds back to a fluid voxel in that grid
    moveSolids<<<numUsedGridNodes, 32, sizeof(char)*numVoxelsPerNode + sizeof(uint)*numVoxelsPerNode, stream>>>(numUsedGridNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), solids.devPtr(), radius, grid, refinementLevel);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaStreamSynchronize(stream));
    //reinterpolate velocity for all particles
    voxelVelsToParticles();
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

void Particles::solveFrame(double fps){
    frameDt = 1.0f/fps;
    dt = frameDt / 10.0f;
    elapsedTimeThisFrame = 0.0f;
    while(elapsedTimeThisFrame < frameDt){
        particleVelToVoxels();
        cudaStreamSynchronize(stream);
        pressureSolve();
        cudaStreamSynchronize(stream);
        updateVoxelVelocities();
        cudaStreamSynchronize(stream);
        voxelVelsToParticles();
        cudaStreamSynchronize(stream);
        advectParticles();
        cudaStreamSynchronize(stream);
        moveSolidParticles();
        cudaStreamSynchronize(stream);
        initialize();
    }
}

void Particles::initialize(){
        alignParticlesToGrid();
        cudaStreamSynchronize(stream);
        sortParticles();
        cudaStreamSynchronize(stream);
        generateVoxels();
        cudaStreamSynchronize(stream);
}

#include <iostream>
#include <fstream>

void Particles::writePositionsToFile(const std::string& fileName){
    std::ofstream file;
    px.download(stream);
    py.download(stream);
    pz.download(stream);
    file.open(fileName);
    cudaStreamSynchronize(stream);
    for(int i = 0; i < size; ++i){
        file<<px[i]<<", "<<py[i]<<", "<<pz[i]<<"\n";
    }
    file.close();
}

Particles::~Particles(){
    gpuErrchk( cudaStreamDestroy(stream) );
}