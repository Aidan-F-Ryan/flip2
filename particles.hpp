#ifndef PARTICLES_H
#define PARTICLES_H

#include "typedefs.h"
#include "cudaVec.hu"
#include "grid.hpp"
#include "kernels.hu"
#include <random>

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

    void randomizeParticlePositions(){
        std::random_device rd;
        std::default_random_engine e2(rd());
        std::uniform_real_distribution<> distX(grid.negX, grid.negX + grid.sizeX*grid.cellSize);
        std::uniform_real_distribution<> distY(grid.negY, grid.negY + grid.sizeY*grid.cellSize);
        std::uniform_real_distribution<> distZ(grid.negZ, grid.negZ + grid.sizeZ*grid.cellSize);

        for(uint i = 0; i < size; ++i){
            px[i] = distX(e2);
            py[i] = distY(e2);
            pz[i] = distZ(e2);
        }

        px.upload();
        py.upload();
        pz.upload();
    }

    void alignParticlesToGrid(){
        kernels::cudaFindGridCell(px.devPtr(), py.devPtr(), pz.devPtr(), size, grid, gridCell.devPtr(), stream);
        kernels::cudaFindSubCell(px.devPtr(), py.devPtr(), pz.devPtr(), size, grid, gridCell.devPtr(), subCell.devPtr(), REFINEMENTLEVELS, stream);
        cudaStreamSynchronize(stream);
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
};

#endif