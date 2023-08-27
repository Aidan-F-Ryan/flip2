#ifndef PARTICLES_H
#define PARTICLES_H

#include "typedefs.h"
#include "cudaVec.hu"
#include "grid.hpp"
#include "kernels.hu"

#define REFINEMENTLEVELS 10

class Particles{
public:
    Particles(uint size)
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

    void setDomain(float nx, float ny, float nz, uint x, uint y, uint z, float cellSize){
        grid.setNegativeCorner(nx, ny, nz);
        grid.setSize(x, y, z, cellSize);
    }

    void alignParticlesToGrid(){
        kernels::cudaFindGridCell(px.devPtr(), py.devPtr(), pz.devPtr(), size, grid, gridCell.devPtr(), stream);
        kernels::cudaFindSubCell(px.devPtr(), py.devPtr(), pz.devPtr(), size, grid, gridCell.devPtr(), subCell.devPtr(), REFINEMENTLEVELS, stream);
    }

    void sortParticles(){
        uint* tempGridCell = gridCell.devPtr();
        char* tempSubCell = subCell.devPtr();
        kernels::cudaSortParticles(size, tempGridCell, tempSubCell, REFINEMENTLEVELS, stream);
        gridCell.swapDevicePtr(tempGridCell);
        subCell.swapDevicePtr(tempSubCell);
    }


    ~Particles(){
        cudaStreamDestroy(stream);
    }

private:
    CudaVec<float> px, py, pz;
    CudaVec<float> vx, vy, vz;
    CudaVec<uint> gridCell;
    CudaVec<char> subCell;
    uint size;
    Grid grid;
    cudaStream_t stream;

    friend class ParticleSystemTester;
};

#endif