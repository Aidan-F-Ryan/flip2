//Copyright 2023 Aberrant Behavior LLC

//Emitters, sinks and open faces (see sources.hu), run as initialize re-bins the particles each substep:
//  markRemovedParticles before rootCell reflects anything back off the walls; killRemovedParticles after it, giving them the cell past the last, so the
//  sort puts them at the end; dropRemovedParticles after the sort, cutting them off; and, once particles that crossed into another partition have moved
//  there, emitParticles, which re-bins and sorts again if it added any

#include "particles.hu"
#include <cmath>
#include <cub/cub.cuh>

//every per-particle array, resized to newSize, keeping the first keep particles. Also the sort's and the node finder's per-particle working space
void Particles::resizeParticleArrays(uint newSize, uint keep){
    auto resize = [&](auto& array){
        using T = std::remove_reference_t<decltype(*array.devPtr())>;
        if(newSize <= array.size()){    //shrinking: the array keeps its memory, and just holds fewer
            array.adoptAsync(array.devPtr(), newSize, stream);
            return;
        }
        T* fresh;
        gpuErrchk(cudaMallocAsync((void**)&fresh, sizeof(T)*newSize, stream));
        if(keep > 0){
            gpuErrchk(cudaMemcpyAsync(fresh, array.devPtr(), sizeof(T)*keep, cudaMemcpyDeviceToDevice, stream));
        }
        array.adoptAsync(fresh, newSize, stream);
    };
    resize(px);
    resize(py);
    resize(pz);
    resize(vx);
    resize(vy);
    resize(vz);
    resize(gridCell);
    if(reorderedGridIndices.size() < newSize){
        reorderedGridIndices.resizeAsync(newSize, stream);
    }
    if(uniqueGridNodeIndices.size() < newSize){
        uniqueGridNodeIndices.resizeAsync(newSize, stream);
    }
    size = newSize;
}

//particles past an open face or within a voxel of it, and particles inside a sink: flagged, while they're still where they went
__global__ void markRemoved(uint numParticles, const double* px, const double* py, const double* pz, Sources sources, Grid grid, double voxelSize, char* removed){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        double position[3] = {px[index], py[index], pz[index]};
        double low[3] = {grid.negX, grid.negY, grid.negZ};
        double high[3] = {grid.negX + (double)grid.sizeX*grid.cellSize, grid.negY + (double)grid.sizeY*grid.cellSize, grid.negZ + (double)grid.sizeZ*grid.cellSize};
        bool gone = false;
        for(int axis = 0; axis < 3; ++axis){
            gone = gone || (sources.openFaces >> 2*axis & 1 && position[axis] < low[axis] + voxelSize) || (sources.openFaces >> (2*axis + 1) & 1 && position[axis] >= high[axis] - voxelSize);
        }
        for(int sink = 0; sink < sources.numSinks && !gone; ++sink){
            gone = sources.sinks[sink].contains(position[0], position[1], position[2]);
        }
        removed[index] = gone;
    }
}

__global__ void killRemoved(uint numParticles, const char* removed, uint deadCell, uint* gridPosition){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles && removed[index]){
        gridPosition[index] = deadCell;
    }
}

void Particles::markRemovedParticles(){
    if(!sources.removes() || size == 0){
        return;
    }
    removedFlags.resizeAsync(size, stream);
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    markRemoved<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, px.devPtr(), py.devPtr(), pz.devPtr(), sources, grid, voxelSize, removedFlags.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}

void Particles::killRemovedParticles(){
    if(!sources.removes() || size == 0){
        return;
    }
    killRemoved<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, removedFlags.devPtr(), grid.sizeX*grid.sizeY*grid.sizeZ, gridCell.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void countCellsBelow(const uint* sortedCells, uint count, uint cell, uint* result){   //how many of the sorted cells are under cell
    uint low = 0;
    uint high = count;
    while(low < high){
        uint middle = low + (high - low) / 2;
        if(sortedCells[middle] < cell){
            low = middle + 1;
        }
        else{
            high = middle;
        }
    }
    *result = low;
}

void Particles::dropRemovedParticles(){
    if(!sources.removes() || size == 0){
        return;
    }
    uint* found;
    uint live;
    gpuErrchk(cudaMallocAsync((void**)&found, sizeof(uint), stream));
    countCellsBelow<<<1, 1, 0, stream>>>(gridCell.devPtr(), size, grid.sizeX*grid.sizeY*grid.sizeZ, found);
    gpuErrchk(cudaMemcpyAsync(&live, found, sizeof(uint), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaFreeAsync(found, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    if(live < size){
        resizeParticleArrays(live, live);
    }
}

__device__ inline unsigned long long mixBits64(unsigned long long x){  //splitmix64
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30))*0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27))*0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

//An emitter as a conveyor: a jittered seeding lattice (particlesPerVoxel per voxel) sliding along at the emitter's velocity, whose points become particles
//as they enter the emitter, or sweep right through it, during a substep; all of them at the start. Inside the emitter and a voxel around it, the grid's
//faces are pinned to its velocity (pinEmitterFaces), so the particles it made keep moving with the lattice and nothing doubles up. That delivers exactly
//particlesPerVoxel per voxel of volume swept, A*v per second, at any CFL; filling empty voxels instead loses a voxel's worth whenever fluid moves more than a
//voxel in a substep. Like any true source it adds fluid even where some already is
struct EmitterLattice{
    long long first[3];         //the lattice points that can emit: their indices, and how many along each axis
    long long extent[3];
    double origin[3];           //the lattice's corner, at time 0: the domain's
    double spacing;
    double shift[3];            //how far the lattice has slid by now: velocity times time
    double step[3];             //and in this substep: velocity times dt
    bool everything;            //the first fill: every point inside emits
    double low[3];              //where this partition's particles can go: the domain, and along z its node planes
    double high[3];

    __host__ __device__ unsigned long long count() const{
        return (unsigned long long)extent[0]*extent[1]*extent[2];
    }

    __device__ void point(unsigned long long index, long long lattice[3]) const{
        lattice[0] = first[0] + (long long)(index % extent[0]);
        lattice[1] = first[1] + (long long)(index / extent[0] % extent[1]);
        lattice[2] = first[2] + (long long)(index / (extent[0]*extent[1]));
    }
};

//a number in [0, 1) that depends only on the seed, the lattice point and which coordinate it's for, so each point keeps its jitter as it slides
__device__ inline double latticeJitter(unsigned long long seed, const long long lattice[3], int coordinate){
    unsigned long long key = mixBits64(mixBits64(mixBits64(seed ^ (unsigned long long)lattice[0]) ^ (unsigned long long)lattice[1]) ^ (unsigned long long)lattice[2]);
    return (mixBits64(key + coordinate) >> 11)*(1.0 / 9007199254740992.0);
}

//whether a lattice point emits a particle this substep, and where it is now
__device__ inline bool emits(const EmitterLattice& lattice, const FluidShape& emitter, unsigned long long seed, unsigned long long index, double now[3]){
    long long point[3];
    lattice.point(index, point);
    double before[3];
    for(int axis = 0; axis < 3; ++axis){
        now[axis] = lattice.origin[axis] + (point[axis] + latticeJitter(seed, point, axis))*lattice.spacing + lattice.shift[axis];
        before[axis] = now[axis] - lattice.step[axis];
        if(now[axis] < lattice.low[axis] || now[axis] >= lattice.high[axis]){
            return false;   //outside the domain, or in another partition's planes
        }
    }
    if(lattice.everything){
        return emitter.contains(now[0], now[1], now[2]);
    }
    return !emitter.contains(before[0], before[1], before[2]) && emitter.crossedBy(before, now);
}

//the fluid inside the emitter moves at its velocity
__global__ void constrainEmitterVelocities(uint numParticles, const double* px, const double* py, const double* pz, float* vx, float* vy, float* vz, FluidShape emitter){
    for(uint index = threadIdx.x + blockIdx.x*blockDim.x; index < numParticles; index += blockDim.x*gridDim.x){
        if(emitter.contains(px[index], py[index], pz[index])){
            vx[index] = emitter.velocity[0];
            vy[index] = emitter.velocity[1];
            vz[index] = emitter.velocity[2];
        }
    }
}

__global__ void findEmissions(EmitterLattice lattice, FluidShape emitter, unsigned long long seed, uint* emitting){
    unsigned long long index = threadIdx.x + (unsigned long long)blockIdx.x*blockDim.x;
    if(index < lattice.count()){
        double now[3];
        emitting[index] = emits(lattice, emitter, seed, index, now);
    }
}

//the new particles, after the first particles; ends holds each lattice point's running count of emissions
__global__ void emitFromLattice(EmitterLattice lattice, FluidShape emitter, unsigned long long seed, const uint* emitting, const uint* ends, uint first,
                                double* px, double* py, double* pz, float* vx, float* vy, float* vz){
    unsigned long long index = threadIdx.x + (unsigned long long)blockIdx.x*blockDim.x;
    if(index < lattice.count() && emitting[index]){
        double now[3];
        emits(lattice, emitter, seed, index, now);
        uint particle = first + ends[index] - 1;
        px[particle] = now[0];
        py[particle] = now[1];
        pz[particle] = now[2];
        vx[particle] = emitter.velocity[0];
        vy[particle] = emitter.velocity[1];
        vz[particle] = emitter.velocity[2];
    }
}

void Particles::emitParticles(){
    if(sources.numEmitters == 0){
        return;
    }
    int interiorWidth = 2<<refinementLevel;
    double voxelSize = grid.cellSize / interiorWidth;
    double origin[3] = {grid.negX, grid.negY, grid.negZ};
    double extent[3] = {(double)grid.sizeX*grid.cellSize, (double)grid.sizeY*grid.cellSize, (double)grid.sizeZ*grid.cellSize};
    EmitterLattice lattice;
    lattice.spacing = voxelSize / sources.latticePerSide;
    lattice.everything = substepIndex == 0;     //initialize, before the first substep
    for(int axis = 0; axis < 3; ++axis){
        lattice.origin[axis] = origin[axis];
        lattice.low[axis] = origin[axis];
        lattice.high[axis] = origin[axis] + extent[axis];
    }
    lattice.low[2] = origin[2] + (double)boxLo*grid.cellSize;
    lattice.high[2] = origin[2] + (double)boxHi*grid.cellSize;
    uint added = 0;
    for(int emitter = 0; emitter < sources.numEmitters; ++emitter){
        const FluidShape& shape = sources.emitters[emitter];
        if(size > 0){
            constrainEmitterVelocities<<<std::min(size / BLOCKSIZE + 1, 4096u), BLOCKSIZE, 0, stream>>>(size, px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr(), shape);
        }
        //the lattice points that can be inside the emitter now, or have swept through it this substep
        bool none = false;
        for(int axis = 0; axis < 3; ++axis){
            lattice.shift[axis] = shape.velocity[axis]*elapsedTime;
            lattice.step[axis] = lattice.everything ? 0.0 : shape.velocity[axis]*dt;
            double low = std::max(std::min(shape.low[axis], shape.low[axis] + lattice.step[axis]), lattice.low[axis]);
            double high = std::min(std::max(shape.high[axis], shape.high[axis] + lattice.step[axis]), lattice.high[axis]);
            lattice.first[axis] = (long long)std::floor((low - lattice.shift[axis] - origin[axis]) / lattice.spacing) - 1;
            lattice.extent[axis] = (long long)std::ceil((high - lattice.shift[axis] - origin[axis]) / lattice.spacing) + 1 - lattice.first[axis];
            none = none || high <= low || lattice.extent[axis] <= 0;
        }
        if(none){
            continue;
        }
        unsigned long long count = lattice.count();
        if(count >= (1ull << 31)){
            std::cerr<<"Particles: an emitter covers "<<count<<" lattice points, more than it can emit at once\n";
            exit(1);
        }
        uint* emitting;
        uint* ends;
        void* scratch = nullptr;
        size_t scratchBytes = 0;
        gpuErrchk(cudaMallocAsync((void**)&emitting, sizeof(uint)*count, stream));
        gpuErrchk(cudaMallocAsync((void**)&ends, sizeof(uint)*count, stream));
        findEmissions<<<(uint)(count / BLOCKSIZE + 1), BLOCKSIZE, 0, stream>>>(lattice, shape, sources.seed + emitter, emitting);
        cub::DeviceScan::InclusiveSum(scratch, scratchBytes, emitting, ends, (int)count, stream);
        gpuErrchk(cudaMallocAsync(&scratch, scratchBytes, stream));
        cub::DeviceScan::InclusiveSum(scratch, scratchBytes, emitting, ends, (int)count, stream);
        uint newParticles;
        gpuErrchk(cudaMemcpyAsync(&newParticles, ends + count - 1, sizeof(uint), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        if(newParticles > 0){
            uint first = size;
            resizeParticleArrays(size + newParticles, size);
            emitFromLattice<<<(uint)(count / BLOCKSIZE + 1), BLOCKSIZE, 0, stream>>>(lattice, shape, sources.seed + emitter, emitting, ends, first,
                px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr());
            added += newParticles;
        }
        gpuErrchk(cudaPeekAtLastError());
        cudaFreeAsync(emitting, stream);
        cudaFreeAsync(ends, stream);
        cudaFreeAsync(scratch, stream);
    }
    if(added > 0){  //the new particles need their cells, and to be sorted in with the rest; rootCell leaves the others as they are
        alignParticlesToGrid();
        sortParticles();
    }
}

//a block per node storing voxels: each face of its voxels inside an emitter, or within a voxel of it, takes the emitter's velocity, so the fluid there
//leaves at exactly that speed whatever gravity and the pressure solve did to it. The voxel's margin matters: advection blends the faces around a particle,
//and without it the faces just past the emitter would speed up the particles still inside. The first emitter holding a face sets it
__global__ void pinEmitterFaces(Sources sources, ForceVoxels voxels){
    uint node = blockIdx.x;
    uint first = node == 0 ? 0 : voxels.nodeVoxelEnds[node - 1];
    uint last = voxels.nodeVoxelEnds[node];
    const Grid& grid = voxels.grid;
    int interiorWidth = 2<<voxels.refinementLevel;
    int voxels1D = interiorWidth + 2*voxels.apronCells;
    double voxelSize = (double)grid.cellSize / interiorWidth;
    uint cell = voxels.nodeCells[node];
    long long origin[3] = {(long long)(cell % grid.sizeX)*interiorWidth - voxels.apronCells, (long long)(cell / grid.sizeX % grid.sizeY)*interiorWidth - voxels.apronCells,
                           (long long)(cell / (grid.sizeX*grid.sizeY))*interiorWidth - voxels.apronCells};
    double corner[3] = {grid.negX, grid.negY, grid.negZ};
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        if(voxels.solids[index]){
            continue;
        }
        int slot = voxels.voxelSlots[index];
        long long voxel[3] = {origin[0] + slot % voxels1D, origin[1] + slot / voxels1D % voxels1D, origin[2] + slot / (voxels1D*voxels1D)};
        for(int dim = 0; dim < 3; ++dim){   //a dim face sits on the voxel's lower boundary along dim, and mid-voxel along the others
            double face[3];
            for(int axis = 0; axis < 3; ++axis){
                face[axis] = corner[axis] + (voxel[axis] + (axis == dim ? 0.0 : 0.5))*voxelSize;
            }
            for(int emitter = 0; emitter < sources.numEmitters; ++emitter){
                if(sources.emitters[emitter].containsWithin(face[0], face[1], face[2], voxelSize)){
                    voxels.velocities[dim][index] = sources.emitters[emitter].velocity[dim];
                    break;
                }
            }
        }
    }
}

void Particles::pinEmitterVelocities(){
    if(sources.numEmitters == 0 || numStoredNodes == 0){
        return;
    }
    pinEmitterFaces<<<numStoredNodes, 128, 0, stream>>>(sources, forceVoxels());
    gpuErrchk(cudaPeekAtLastError());
}
