#include "particles.hu"
#include "grid.hpp"
#include "algorithms/radixSortKernels.hu"

Particles::Particles(uint size)
: size(size)
, radius(1.5)
{
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
    cudaStreamCreate(&stream);
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
        subCellsTouchedZ.devPtr(), subCellsX, subCellsY, subCellsZ, 2, radius, stream);
}

Particles::~Particles(){
    cudaStreamDestroy(stream);
}