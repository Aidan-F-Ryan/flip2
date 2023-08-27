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
    subCell.resize(size*REFINEMENTLEVELS);    //starting with 10 levels of refinement
    cudaStreamCreate(&stream);
}

void Particles::setDomain(float nx, float ny, float nz, uint x, uint y, uint z, float cellSize){
    grid.setNegativeCorner(nx, ny, nz);
    grid.setSize(x, y, z, cellSize);
}

void Particles::alignParticlesToGrid(){
    kernels::cudaFindGridCell(px.devPtr(), py.devPtr(), pz.devPtr(), size, grid, gridCell.devPtr(), stream);
    kernels::cudaFindSubCell(px.devPtr(), py.devPtr(), pz.devPtr(), size, grid, gridCell.devPtr(), subCell.devPtr(), REFINEMENTLEVELS, stream);
}

void Particles::sortParticles(){
    uint* tempGridCell = gridCell.devPtr();
    char* tempSubCell = subCell.devPtr();
    kernels::cudaSortParticles(size, tempGridCell, tempSubCell, REFINEMENTLEVELS, stream);
    gridCell.swapDevicePtr(tempGridCell);
    subCell.swapDevicePtr(tempSubCell);
}


Particles::~Particles(){
    cudaStreamDestroy(stream);
}