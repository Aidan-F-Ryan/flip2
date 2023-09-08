#include "algorithms/kernels.hu"
#include "particles.hu"
#include "grid.hpp"
#include "algorithms/radixSortKernels.hu"
#include "typedefs.h"
#include <cmath>
#include <iostream>

Particles::Particles(uint size)
: size(size)
, radius(1.5)
, refinementLevel(0)
{
    cudaStreamCreate(&stream);
    px.resize(size);
    py.resize(size);
    pz.resize(size);
    
    vx.resize(size);
    vy.resize(size);
    vz.resize(size);

    gridCell.resize(size);
    subCellsTouchedX.resize(size);
    subCellsTouchedY.resize(size);
    subCellsTouchedZ.resize(size);

    reorderedGridIndices.resize(size);

    uniqueGridNodeIndices.resize(size);

    numVoxels1D = 2*(uint)(std::floor(radius)) + (2<<refinementLevel);
    numVoxelsPerNode = numVoxels1D*numVoxels1D*numVoxels1D;
}

void Particles::setDomain(float nx, float ny, float nz, uint x, uint y, uint z, float cellSize){
    grid.setNegativeCorner(nx, ny, nz);
    grid.setSize(x, y, z, cellSize);
}

void Particles::alignParticlesToGrid(){
    kernels::cudaFindGridCell(px.devPtr(), py.devPtr(), pz.devPtr(), size, grid, gridCell.devPtr(), stream);
}

void Particles::sortParticles(){
    float *d_px, *d_py, *d_pz;
    cudaMalloc((void**)&d_px, sizeof(float)*size);    
    cudaMalloc((void**)&d_py, sizeof(float)*size);    
    cudaMalloc((void**)&d_pz, sizeof(float)*size);    


    uint* tempGridCell = gridCell.devPtr();
    uint* tempSortedIndices = reorderedGridIndices.devPtr();
    kernels::cudaSortParticlesByGridNode(size, tempGridCell, tempSortedIndices, stream);
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

void Particles::alignParticlesToSubCells(){
    kernels::cudaFindSubCell(px.devPtr(), py.devPtr(), pz.devPtr(), size, grid, 
        gridCell.devPtr(), subCellsTouchedX.devPtr(), subCellsTouchedY.devPtr(),
        subCellsTouchedZ.devPtr(), subCellsX, subCellsY, subCellsZ, refinementLevel, radius, stream);

}

void Particles::generateVoxels(){
    uint numGridNodes = kernels::cudaMarkUniqueGridCellsAndCount(size, gridCell.devPtr(), uniqueGridNodeIndices.devPtr(), stream);
    
    gridNodeIndicesToFirstParticleIndex.resizeAsync(numGridNodes, stream);

    kernels::cudaMapNodeIndicesToParticles(size, uniqueGridNodeIndices.devPtr(), gridNodeIndicesToFirstParticleIndex.devPtr(), stream);
    
    uint* numUsedVoxelsPerNodeX;
    uint* numUsedVoxelsPerNodeY;
    uint* numUsedVoxelsPerNodeZ;
    uint* numParticlesInVoxelListsX;
    uint* numParticlesInVoxelListsY;
    uint* numParticlesInVoxelListsZ;
    gpuErrchk( cudaMallocAsync((void**)&numUsedVoxelsPerNodeX, sizeof(uint)*numGridNodes, stream) );
    gpuErrchk( cudaMallocAsync((void**)&numParticlesInVoxelListsX, sizeof(uint)*numGridNodes, stream) );
    gpuErrchk( cudaMallocAsync((void**)&numUsedVoxelsPerNodeY, sizeof(uint)*numGridNodes, stream) );
    gpuErrchk( cudaMallocAsync((void**)&numParticlesInVoxelListsY, sizeof(uint)*numGridNodes, stream) );
    gpuErrchk( cudaMallocAsync((void**)&numUsedVoxelsPerNodeZ, sizeof(uint)*numGridNodes, stream) );
    gpuErrchk( cudaMallocAsync((void**)&numParticlesInVoxelListsZ, sizeof(uint)*numGridNodes, stream) );

    kernels::cudaSumParticlesPerNodeAndWriteNumUsedVoxels(numGridNodes, size, gridNodeIndicesToFirstParticleIndex.devPtr(), subCellsTouchedX.devPtr(), subCellsX.devPtr(), numUsedVoxelsPerNodeX, numParticlesInVoxelListsX, numVoxelsPerNode, stream);
    kernels::cudaSumParticlesPerNodeAndWriteNumUsedVoxels(numGridNodes, size, gridNodeIndicesToFirstParticleIndex.devPtr(), subCellsTouchedY.devPtr(), subCellsY.devPtr(), numUsedVoxelsPerNodeY, numParticlesInVoxelListsY, numVoxelsPerNode, stream);
    kernels::cudaSumParticlesPerNodeAndWriteNumUsedVoxels(numGridNodes, size, gridNodeIndicesToFirstParticleIndex.devPtr(), subCellsTouchedZ.devPtr(), subCellsZ.devPtr(), numUsedVoxelsPerNodeZ, numParticlesInVoxelListsZ, numVoxelsPerNode, stream);

    gpuErrchk( cudaFreeAsync(numUsedVoxelsPerNodeX, stream) );
    gpuErrchk( cudaFreeAsync(numParticlesInVoxelListsX, stream) );
    gpuErrchk( cudaFreeAsync(numUsedVoxelsPerNodeY, stream) );
    gpuErrchk( cudaFreeAsync(numParticlesInVoxelListsY, stream) );
    gpuErrchk( cudaFreeAsync(numUsedVoxelsPerNodeZ, stream) );
    gpuErrchk( cudaFreeAsync(numParticlesInVoxelListsZ, stream) );
}

void Particles::particleVelToVoxels(){
    
}

Particles::~Particles(){
    cudaStreamDestroy(stream);
}