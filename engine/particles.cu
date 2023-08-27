#include "particles.hpp"
#include "grid.hpp"

Particles::Particles(uint size)
: size(size)
{
    px.resize(size);
    py.resize(size);
    pz.resize(size);
    
    vx.resize(size);
    vy.resize(size);
    vz.resize(size);

    gridCell.resize(size);
    subCellX.resize(size);
    subCellY.resize(size);
    subCellZ.resize(size);
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
    uint* tempGridCell = gridCell.devPtr();
    kernels::cudaSortParticlesByGridNode(size, tempGridCell, stream);
    gridCell.swapDevicePtr(tempGridCell);
}

void Particles::alignParticlesToSubCells(){
    kernels::cudaFindSubCell(px.devPtr(), py.devPtr(), pz.devPtr(), size, grid, gridCell.devPtr(), subCellX.devPtr(), subCellY.devPtr(), subCellZ.devPtr(), 2, stream);
}

Particles::~Particles(){
    cudaStreamDestroy(stream);
}