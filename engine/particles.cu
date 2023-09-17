#include "algorithms/kernels.hu"
#include "particles.hu"
#include "grid.hpp"
#include "algorithms/radixSortKernels.hu"
#include "typedefs.h"
#include <cmath>
#include <iostream>

Particles::Particles(uint size)
: size(size)
, radius(2)
, refinementLevel(1)
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
    numUsedGridNodes = kernels::cudaMarkUniqueGridCellsAndCount(size, gridCell.devPtr(), uniqueGridNodeIndices.devPtr(), stream);
    
    gridNodeIndicesToFirstParticleIndex.resizeAsync(numUsedGridNodes, stream);

    kernels::cudaMapNodeIndicesToParticles(size, uniqueGridNodeIndices.devPtr(), gridNodeIndicesToFirstParticleIndex.devPtr(), stream);
    nodeIndexToFirstUsedVoxelX.resizeAsync(numUsedGridNodes, stream);
    nodeIndexToFirstUsedVoxelY.resizeAsync(numUsedGridNodes, stream);
    nodeIndexToFirstUsedVoxelZ.resizeAsync(numUsedGridNodes, stream);
    uint* numParticlesInVoxelListsX;
    uint* numParticlesInVoxelListsY;
    uint* numParticlesInVoxelListsZ;
    gpuErrchk( cudaMallocAsync((void**)&numParticlesInVoxelListsX, sizeof(uint)*numUsedGridNodes, stream) );
    gpuErrchk( cudaMallocAsync((void**)&numParticlesInVoxelListsY, sizeof(uint)*numUsedGridNodes, stream) );
    gpuErrchk( cudaMallocAsync((void**)&numParticlesInVoxelListsZ, sizeof(uint)*numUsedGridNodes, stream) );

    //get sizes for number of voxels used and number of particles including duplicates in particle lists for used voxels in each node
    kernels::cudaSumParticlesPerNodeAndWriteNumUsedVoxels(numUsedGridNodes, size, gridNodeIndicesToFirstParticleIndex.devPtr(),
        subCellsTouchedX.devPtr(), subCellsX.devPtr(), nodeIndexToFirstUsedVoxelX.devPtr(), numParticlesInVoxelListsX, numVoxelsPerNode, stream);
    kernels::cudaSumParticlesPerNodeAndWriteNumUsedVoxels(numUsedGridNodes, size, gridNodeIndicesToFirstParticleIndex.devPtr(),
        subCellsTouchedY.devPtr(), subCellsY.devPtr(), nodeIndexToFirstUsedVoxelY.devPtr(), numParticlesInVoxelListsY, numVoxelsPerNode, stream);
    kernels::cudaSumParticlesPerNodeAndWriteNumUsedVoxels(numUsedGridNodes, size, gridNodeIndicesToFirstParticleIndex.devPtr(),
        subCellsTouchedZ.devPtr(), subCellsZ.devPtr(), nodeIndexToFirstUsedVoxelZ.devPtr(), numParticlesInVoxelListsZ, numVoxelsPerNode, stream);

    gpuErrchk( cudaMemcpyAsync(&numUsedVoxelsX, nodeIndexToFirstUsedVoxelX.devPtr() + numUsedGridNodes -1, sizeof(uint), cudaMemcpyDeviceToHost,
        stream) );
    gpuErrchk( cudaMemcpyAsync(&totalNumParticlesInPerVoxelListsX, numParticlesInVoxelListsX + numUsedGridNodes -1, sizeof(uint),
        cudaMemcpyDeviceToHost, stream) );

    voxelIDsX.resizeAsync(numUsedVoxelsX, stream); //size voxel id array to total number of voxels used
    voxelParticleListStartX.resizeAsync(numUsedVoxelsX, stream);   //particle list start index for each voxel using total number of voxels
    particleListsX.resizeAsync(totalNumParticlesInPerVoxelListsX, stream);  //size overall particle list array, position for each node start known from numParticlesInVoxelLists
    
    gpuErrchk( cudaMemcpyAsync(&numUsedVoxelsY, nodeIndexToFirstUsedVoxelY.devPtr() + numUsedGridNodes -1, sizeof(uint), cudaMemcpyDeviceToHost,
        stream) );
    gpuErrchk( cudaMemcpyAsync(&totalNumParticlesInPerVoxelListsY, numParticlesInVoxelListsY + numUsedGridNodes -1, sizeof(uint),
        cudaMemcpyDeviceToHost, stream) );

    voxelIDsY.resizeAsync(numUsedVoxelsY, stream); //size voxel id array to total number of voxels used
    voxelParticleListStartY.resizeAsync(numUsedVoxelsY, stream);   //particle list start index for each voxel using total number of voxels
    particleListsY.resizeAsync(totalNumParticlesInPerVoxelListsY, stream);  //size overall particle list array, position for each node start known from numParticlesInVoxelLists

    gpuErrchk( cudaMemcpyAsync(&numUsedVoxelsZ, nodeIndexToFirstUsedVoxelZ.devPtr() + numUsedGridNodes -1, sizeof(uint), cudaMemcpyDeviceToHost,
        stream) );
    gpuErrchk( cudaMemcpyAsync(&totalNumParticlesInPerVoxelListsZ, numParticlesInVoxelListsZ + numUsedGridNodes -1, sizeof(uint),
        cudaMemcpyDeviceToHost, stream) );

    voxelIDsZ.resizeAsync(numUsedVoxelsZ, stream); //size voxel id array to total number of voxels used
    voxelParticleListStartZ.resizeAsync(numUsedVoxelsZ, stream);   //particle list start index for each voxel using total number of voxels
    particleListsZ.resizeAsync(totalNumParticlesInPerVoxelListsZ, stream);  //size overall particle list array, position for each node start known from numParticlesInVoxelLists

    //  create particle lists.
    //  voxelParticleListsStart = parallel prefix sum over number of particles in each voxel list. Each node starts at numUsedVoxelsPerNode[nodeNum]
    //  particleList = particles in each voxel, starting at voxelParticleListsStart[voxelNum] -> note this is sparse
    //  voxelIDs = voxel position within node, sparse
    kernels::cudaParticleListCreate(numUsedVoxelsX, numUsedGridNodes, size, numVoxelsPerNode, voxelIDsX.devPtr(),
        voxelParticleListStartX.devPtr(), nodeIndexToFirstUsedVoxelX.devPtr(), numParticlesInVoxelListsX, subCellsTouchedX.devPtr(), subCellsX.devPtr(),
        gridNodeIndicesToFirstParticleIndex.devPtr(), particleListsX.devPtr(), stream);

    kernels::cudaParticleListCreate(numUsedVoxelsY, numUsedGridNodes, size, numVoxelsPerNode, voxelIDsY.devPtr(),
        voxelParticleListStartY.devPtr(), nodeIndexToFirstUsedVoxelY.devPtr(), numParticlesInVoxelListsY, subCellsTouchedY.devPtr(), subCellsY.devPtr(),
        gridNodeIndicesToFirstParticleIndex.devPtr(), particleListsY.devPtr(), stream);
    
    kernels::cudaParticleListCreate(numUsedVoxelsZ, numUsedGridNodes, size, numVoxelsPerNode, voxelIDsZ.devPtr(),
        voxelParticleListStartZ.devPtr(), nodeIndexToFirstUsedVoxelZ.devPtr(), numParticlesInVoxelListsZ, subCellsTouchedZ.devPtr(), subCellsZ.devPtr(),
        gridNodeIndicesToFirstParticleIndex.devPtr(), particleListsZ.devPtr(), stream);

    voxelsUx.resizeAsync(numUsedVoxelsX, stream);
    voxelsUy.resizeAsync(numUsedVoxelsY, stream);
    voxelsUz.resizeAsync(numUsedVoxelsZ, stream);

    gpuErrchk( cudaFreeAsync(numParticlesInVoxelListsX, stream) );
    gpuErrchk( cudaFreeAsync(numParticlesInVoxelListsY, stream) );
    gpuErrchk( cudaFreeAsync(numParticlesInVoxelListsZ, stream) );
}

void Particles::particleVelToVoxels(){
    // kernels::cudaVoxelUGather(numUsedVoxelsX, numUsedGridNodes, size, numVoxelsPerNode, vx.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(),
        // nodeIndexToFirstUsedVoxelX.devPtr(), voxelIDsX.devPtr(), voxelParticleListStartX.devPtr(), particleListsX.devPtr(), voxelsUx.devPtr(), stream);

    kernels::cudaVoxelUGather(numUsedVoxelsX, numUsedGridNodes, size, numVoxelsPerNode, totalNumParticlesInPerVoxelListsX, gridCell.devPtr(), vx.devPtr(), px.devPtr(),
        py.devPtr(), pz.devPtr(), nodeIndexToFirstUsedVoxelX.devPtr(), voxelIDsX.devPtr(), voxelParticleListStartX.devPtr(), particleListsX.devPtr(), voxelsUx.devPtr(),
        grid, grid.sizeX*grid.sizeY, refinementLevel, radius, numVoxels1D, stream);
    cudaStreamSynchronize(stream);
}

Particles::~Particles(){
    cudaStreamDestroy(stream);
}