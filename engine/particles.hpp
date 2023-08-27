#ifndef PARTICLES_H
#define PARTICLES_H

#include "typedefs.h"
#include "cudaVec.hu"
#include "kernels.hu"

#define REFINEMENTLEVELS 10

class Particles{
public:
    Particles(uint size);

    void setDomain(float nx, float ny, float nz, uint x, uint y, uint z, float cellSize);

    void alignParticlesToGrid();
    void alignParticlesToSubCells();

    void sortParticles();

    ~Particles();
private:
    CudaVec<float> px, py, pz;
    CudaVec<float> vx, vy, vz;
    CudaVec<uint> gridCell;
    CudaVec<uint> subCellX, subCellY, subCellZ;
    uint size;
    Grid grid;
    cudaStream_t stream;

    friend class ParticleSystemTester;
};

#endif