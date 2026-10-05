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
    for(CudaVec<float>* data : particleFloats()){
        resize(*data);
    }
    resize(gridCell);
    resize(particleIds);
    resize(particleBirths);
    if(apic && newSize > keep){     //new particles start with no velocity gradient
        for(CudaVec<float>& gradient : affine){
            gpuErrchk(cudaMemsetAsync(gradient.devPtr() + keep, 0, sizeof(float)*(newSize - keep), stream));
        }
    }
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

bool Particles::removing() const{
    return sources.removes() || (obstacles.count() > 0 && substepIndex == 0) || airBand() || twoPhase.escaping();
}

//escaped air (twoPhase.escaping()): bubbles too small for the grid, which G2P marked, go
__global__ void markBubbles(uint numParticles, const uint* ids, char* removed){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles && (ids[index] & (AIR_PARTICLE | ESCAPED_PARTICLE)) == (AIR_PARTICLE | ESCAPED_PARTICLE)){
        removed[index] = 1;
    }
}

void Particles::markRemovedParticles(){
    if(!removing() || size == 0){
        return;
    }
    removedFlags.resizeAsync(size, stream);
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    markRemoved<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, px.devPtr(), py.devPtr(), pz.devPtr(), sources, grid, voxelSize, removedFlags.devPtr());
    if(twoPhase.escaping()){
        markBubbles<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, particleIds.devPtr(), removedFlags.devPtr());
    }
    gpuErrchk(cudaPeekAtLastError());
    markParticlesInsideObstacles();
}

void Particles::killRemovedParticles(){
    if(!removing() || size == 0){
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
    if(!removing() || size == 0){
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
    double low[3];              //the domain
    double high[3];
    double ownLow, ownHigh;     //and along z, this partition's node planes: where its own particles can go

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

//whether a lattice point emits a particle this substep, in whichever partition's planes, and where it is now
__device__ inline bool emits(const EmitterLattice& lattice, const FluidShape& emitter, unsigned long long seed, unsigned long long index, double now[3]){
    long long point[3];
    lattice.point(index, point);
    double before[3];
    for(int axis = 0; axis < 3; ++axis){
        now[axis] = lattice.origin[axis] + (point[axis] + latticeJitter(seed, point, axis))*lattice.spacing + lattice.shift[axis];
        before[axis] = now[axis] - lattice.step[axis];
        if(now[axis] < lattice.low[axis] || now[axis] >= lattice.high[axis]){
            return false;   //outside the domain
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

//whether a point that emits does so in this partition's planes: the particle is this partition's to make
__device__ inline bool ownPoint(const EmitterLattice& lattice, const double now[3]){
    return now[2] >= lattice.ownLow && now[2] < lattice.ownHigh;
}

//per lattice point: whether it emits (emitting), and whether it does in this partition's planes (own; the same array as emitting in a lone partition)
__global__ void findEmissions(EmitterLattice lattice, FluidShape emitter, unsigned long long seed, uint* emitting, uint* own){
    unsigned long long index = threadIdx.x + (unsigned long long)blockIdx.x*blockDim.x;
    if(index < lattice.count()){
        double now[3];
        bool anywhere = emits(lattice, emitter, seed, index, now);
        emitting[index] = anywhere;
        own[index] = anywhere && ownPoint(lattice, now);
    }
}

//whether a lattice point seeds fluid to start with, for fluid which, a mesh: inside it, and not inside a box or sphere fluid (the host seeded those),
//an earlier mesh fluid, or an obstacle (fluid there would only be removed or pushed out)
__device__ inline bool fills(const EmitterLattice& lattice, const FluidShapes& fluids, int which, const Obstacles& obstacles, unsigned long long seed, unsigned long long index, double now[3]){
    if(!emits(lattice, fluids.shapes[which], seed, index, now)){
        return false;
    }
    for(int other = 0; other < fluids.count; ++other){
        if(other != which && (fluids.shapes[other].kind != FluidShape::MESH || other < which) && fluids.shapes[other].contains(now[0], now[1], now[2])){
            return false;
        }
    }
    float distance;
    float3 normal;
    return !(nearestObstacle(obstacles, make_float3((float)now[0], (float)now[1], (float)now[2]), distance, normal) >= 0 && distance < 0.0f);
}

__global__ void findFills(EmitterLattice lattice, FluidShapes fluids, int which, Obstacles obstacles, unsigned long long seed, uint* emitting, uint* own){
    unsigned long long index = threadIdx.x + (unsigned long long)blockIdx.x*blockDim.x;
    if(index < lattice.count()){
        double now[3];
        bool anywhere = fills(lattice, fluids, which, obstacles, seed, index, now);
        emitting[index] = anywhere;
        own[index] = anywhere && ownPoint(lattice, now);
    }
}

//the new particles' ids and birth times. A liquid particle's id is firstId plus its lattice point's place among every point emitting now, in any
//partition's planes (ranks: each point's running count of those), so it doesn't depend on which partition makes it; it keeps to the bits that number
//a particle (numbers: Particles::idNumbers). Air's particles are AIR_PARTICLE and no number
__global__ void identifyEmitted(unsigned long long count, const uint* own, const uint* ends, const uint* ranks, uint first, uint firstId, uint numbers, bool air,
                                float birth, uint* ids, float* births){
    unsigned long long index = threadIdx.x + (unsigned long long)blockIdx.x*blockDim.x;
    if(index < count && own[index]){
        uint particle = first + ends[index] - 1;
        ids[particle] = air ? AIR_PARTICLE : (firstId + ranks[index] - 1) & numbers;
        births[particle] = birth;
    }
}

__global__ void fillFromLattice(EmitterLattice lattice, FluidShapes fluids, int which, Obstacles obstacles, unsigned long long seed, const uint* emitting, const uint* ends, uint first,
                                double* px, double* py, double* pz, float* vx, float* vy, float* vz){
    unsigned long long index = threadIdx.x + (unsigned long long)blockIdx.x*blockDim.x;
    if(index < lattice.count() && emitting[index]){
        double now[3];
        fills(lattice, fluids, which, obstacles, seed, index, now);
        uint particle = first + ends[index] - 1;
        px[particle] = now[0];
        py[particle] = now[1];
        pz[particle] = now[2];
        vx[particle] = fluids.shapes[which].velocity[0];
        vy[particle] = fluids.shapes[which].velocity[1];
        vz[particle] = fluids.shapes[which].velocity[2];
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

//The air band's fill (twophase.cu): what a lattice point needs to know to become air
struct BandFill{
    const char* rings;      //per node cell of the domain: its ring (Particles::markBeyondBand)
    const uint* occupied;   //a bit per voxel of the domain: whether any particle is in it
    char last;              //the band's last ring
    bool everywhere;        //the first fill: the empty voxels of the nodes holding liquid too. After it those are left empty, as they are with no air:
                            //a gap that opens inside the liquid would otherwise fill with air that has to find its way out
    int perSide;            //lattice points per voxel along an axis
    int nodeWidth;          //voxels per node along an axis
    uint cells[3];          //node cells along each axis
};

//a bit per voxel of the domain holding any particle
__global__ void markOccupiedVoxels(uint numParticles, const double* px, const double* py, const double* pz, EmitterLattice lattice, BandFill band, uint* occupied){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        double position[3] = {px[index], py[index], pz[index]};
        unsigned long long voxel[3];
        for(int axis = 0; axis < 3; ++axis){
            long long along = (long long)floor((position[axis] - lattice.origin[axis]) / (lattice.spacing*band.perSide));
            long long most = (long long)band.cells[axis]*band.nodeWidth - 1;
            voxel[axis] = (unsigned long long)(along < 0 ? 0 : along > most ? most : along);
        }
        unsigned long long bit = voxel[0] + (unsigned long long)band.cells[0]*band.nodeWidth*(voxel[1] + (unsigned long long)band.cells[1]*band.nodeWidth*voxel[2]);
        atomicOr(occupied + (bit >> 5), 1u << (bit & 31));
    }
}

//whether a lattice point becomes air: its node is in the band, its voxel holds no particle, and it isn't inside an obstacle; and where it is
__device__ inline bool fillsBand(const EmitterLattice& lattice, const BandFill& band, const Obstacles& obstacles, unsigned long long seed, unsigned long long index, double now[3]){
    long long point[3];
    lattice.point(index, point);
    unsigned long long voxel[3];
    for(int axis = 0; axis < 3; ++axis){
        if(point[axis] < 0 || point[axis] >= (long long)band.cells[axis]*band.nodeWidth*band.perSide){
            return false;   //outside the domain
        }
        voxel[axis] = (unsigned long long)(point[axis] / band.perSide);
        now[axis] = lattice.origin[axis] + (point[axis] + latticeJitter(seed, point, axis))*lattice.spacing;
    }
    char ring = band.rings[voxel[0]/band.nodeWidth + band.cells[0]*(voxel[1]/band.nodeWidth + band.cells[1]*(voxel[2]/band.nodeWidth))];
    if(ring > band.last || (ring == 0 && !band.everywhere)){
        return false;
    }
    unsigned long long bit = voxel[0] + (unsigned long long)band.cells[0]*band.nodeWidth*(voxel[1] + (unsigned long long)band.cells[1]*band.nodeWidth*voxel[2]);
    if(band.occupied[bit >> 5] >> (bit & 31) & 1){
        return false;
    }
    float distance;
    float3 normal;
    return !(nearestObstacle(obstacles, make_float3((float)now[0], (float)now[1], (float)now[2]), distance, normal) >= 0 && distance < 0.0f);
}

__global__ void findBandFills(EmitterLattice lattice, BandFill band, Obstacles obstacles, unsigned long long seed, uint* emitting, uint* own){
    unsigned long long index = threadIdx.x + (unsigned long long)blockIdx.x*blockDim.x;
    if(index < lattice.count()){
        double now[3];
        bool anywhere = fillsBand(lattice, band, obstacles, seed, index, now);
        emitting[index] = anywhere;
        own[index] = anywhere && ownPoint(lattice, now);
    }
}

__global__ void fillBandFromLattice(EmitterLattice lattice, BandFill band, Obstacles obstacles, unsigned long long seed, const uint* emitting, const uint* ends, uint first,
                                    double* px, double* py, double* pz, float* vx, float* vy, float* vz){
    unsigned long long index = threadIdx.x + (unsigned long long)blockIdx.x*blockDim.x;
    if(index < lattice.count() && emitting[index]){
        double now[3];
        fillsBand(lattice, band, obstacles, seed, index, now);
        uint particle = first + ends[index] - 1;
        px[particle] = now[0];
        py[particle] = now[1];
        pz[particle] = now[2];
        vx[particle] = 0.0f;
        vy[particle] = 0.0f;
        vz[particle] = 0.0f;
    }
}

void Particles::emitParticles(){
    bool filling = false;   //the start, with fluid in meshes to seed
    for(int fluid = 0; fluid < fluids.count && substepIndex == 0; ++fluid){
        filling = filling || fluids.shapes[fluid].kind == FluidShape::MESH;
    }
    if(sources.numEmitters == 0 && !filling && !airBand()){
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
    lattice.ownLow = origin[2] + (double)boxLo*grid.cellSize;
    lattice.ownHigh = origin[2] + (double)boxHi*grid.cellSize;
    uint added = 0;
    //Of the lattice points that can be inside the shape now, or have swept through it this substep, the ones find marks become particles, made by place.
    //Every partition looks at all of them, not only those in its own planes: each new particle's id is its point's place among all that emit, and the
    //next id to hand out moves on by how many did, so the partitions agree on both without a word between them
    auto emitFrom = [&](const FluidShape& shape, bool air, auto find, auto place){
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
            return;
        }
        unsigned long long count = lattice.count();
        if(count >= (1ull << 31)){
            std::cerr<<"Particles: an emitter covers "<<count<<" lattice points, more than it can emit at once\n";
            exit(1);
        }
        bool lone = numRanks == 1;      //then every point that emits is its own
        uint* emitting;     //per point: whether it emits, and ranks, the running count of those
        uint* ranks;
        uint* own;          //and whether it does in this partition's planes, and ends, the running count of those
        uint* ends;
        void* scratch = nullptr;
        size_t scratchBytes = 0;
        gpuErrchk(cudaMallocAsync((void**)&emitting, sizeof(uint)*count, stream));
        gpuErrchk(cudaMallocAsync((void**)&ranks, sizeof(uint)*count, stream));
        own = emitting;
        ends = ranks;
        if(!lone){
            gpuErrchk(cudaMallocAsync((void**)&own, sizeof(uint)*count, stream));
            gpuErrchk(cudaMallocAsync((void**)&ends, sizeof(uint)*count, stream));
        }
        find(count, emitting, own);
        cub::DeviceScan::InclusiveSum(scratch, scratchBytes, emitting, ranks, (int)count, stream);
        gpuErrchk(cudaMallocAsync(&scratch, scratchBytes, stream));
        cub::DeviceScan::InclusiveSum(scratch, scratchBytes, emitting, ranks, (int)count, stream);
        if(!lone){
            cub::DeviceScan::InclusiveSum(scratch, scratchBytes, own, ends, (int)count, stream);
        }
        uint counts[2];     //the particles this partition makes, and every partition together
        gpuErrchk(cudaMemcpyAsync(counts, ends + count - 1, sizeof(uint), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaMemcpyAsync(counts + 1, ranks + count - 1, sizeof(uint), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        uint newParticles = counts[0];
        if(newParticles > 0){
            uint first = size;
            resizeParticleArrays(size + newParticles, size);
            place(count, own, ends, first);
            identifyEmitted<<<(uint)(count / BLOCKSIZE + 1), BLOCKSIZE, 0, stream>>>(count, own, ends, ranks, first, (uint)nextParticleId, idNumbers(), air, (float)elapsedTime,
                particleIds.devPtr(), particleBirths.devPtr());
            added += newParticles;
        }
        if(!air){   //air takes no numbers
            if(nextParticleId <= idNumbers() && nextParticleId + counts[1] > idNumbers() && verbose()){
                std::cerr<<"Particles: more than "<<(unsigned long long)idNumbers() + 1<<" particles have been made, more than their ids can number: from here on new particles repeat old ids\n";
            }
            nextParticleId += counts[1];
        }
        gpuErrchk(cudaPeekAtLastError());
        cudaFreeAsync(emitting, stream);
        cudaFreeAsync(ranks, stream);
        if(!lone){
            cudaFreeAsync(own, stream);
            cudaFreeAsync(ends, stream);
        }
        cudaFreeAsync(scratch, stream);
    };
    for(int emitter = 0; emitter < sources.numEmitters; ++emitter){
        const FluidShape& shape = sources.emitters[emitter];
        if(size > 0){
            constrainEmitterVelocities<<<std::min(size / BLOCKSIZE + 1, 4096u), BLOCKSIZE, 0, stream>>>(size, px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr(), shape);
        }
        unsigned long long seed = sources.seed + emitter;
        emitFrom(shape, false, [&](unsigned long long count, uint* emitting, uint* own){
            findEmissions<<<(uint)(count / BLOCKSIZE + 1), BLOCKSIZE, 0, stream>>>(lattice, shape, seed, emitting, own);
        }, [&](unsigned long long count, const uint* emitting, const uint* ends, uint first){
            emitFromLattice<<<(uint)(count / BLOCKSIZE + 1), BLOCKSIZE, 0, stream>>>(lattice, shape, seed, emitting, ends, first,
                px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr());
        });
    }
    for(int fluid = 0; fluid < fluids.count && filling; ++fluid){
        if(fluids.shapes[fluid].kind != FluidShape::MESH){
            continue;
        }
        unsigned long long seed = sources.seed + MAX_SOURCE_SHAPES + fluid;  //not an emitter's
        emitFrom(fluids.shapes[fluid], false, [&](unsigned long long count, uint* emitting, uint* own){
            findFills<<<(uint)(count / BLOCKSIZE + 1), BLOCKSIZE, 0, stream>>>(lattice, fluids, fluid, obstacles.state(), seed, emitting, own);
        }, [&](unsigned long long count, const uint* emitting, const uint* ends, uint first){
            fillFromLattice<<<(uint)(count / BLOCKSIZE + 1), BLOCKSIZE, 0, stream>>>(lattice, fluids, fluid, obstacles.state(), seed, emitting, ends, first,
                px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr());
        });
    }
    if(airBand()){  //air, at rest, in the band's voxels that hold nothing (twophase.cu): every lattice point of this partition's planes is asked, which a
                    //band of nodes listed first would save
        if(added > 0){  //the rings count the liquid just made too: its particles need their cells
            alignParticlesToGrid();
        }
        findBandRings();
        BandFill band;
        band.rings = bandRings.devPtr();
        band.last = (char)bandRingCount();
        band.everywhere = substepIndex == 0;
        band.perSide = sources.latticePerSide;
        band.nodeWidth = interiorWidth;
        band.cells[0] = grid.sizeX;
        band.cells[1] = grid.sizeY;
        band.cells[2] = grid.sizeZ;
        size_t words = (size_t)((unsigned long long)grid.sizeX*grid.sizeY*grid.sizeZ*interiorWidth*interiorWidth*interiorWidth / 32 + 1);
        uint* occupied;
        gpuErrchk(cudaMallocAsync((void**)&occupied, sizeof(uint)*words, stream));
        gpuErrchk(cudaMemsetAsync(occupied, 0, sizeof(uint)*words, stream));
        band.occupied = occupied;
        if(size > 0){
            markOccupiedVoxels<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, px.devPtr(), py.devPtr(), pz.devPtr(), lattice, band, occupied);
        }
        FluidShape everywhere;      //a box: this partition's planes of the domain
        for(int axis = 0; axis < 3; ++axis){
            everywhere.low[axis] = lattice.low[axis];
            everywhere.high[axis] = lattice.high[axis];
        }
        everywhere.low[2] = lattice.ownLow;
        everywhere.high[2] = lattice.ownHigh;
        //not an emitter's seed or a fluid's, and another every substep: a voxel emptied twice doesn't fill the same way twice
        unsigned long long seed = sources.seed + 2*MAX_SOURCE_SHAPES + 0x9e3779b97f4a7c15ull*(substepIndex + 1);
        emitFrom(everywhere, true, [&](unsigned long long count, uint* emitting, uint* own){
            findBandFills<<<(uint)(count / BLOCKSIZE + 1), BLOCKSIZE, 0, stream>>>(lattice, band, obstacles.state(), seed, emitting, own);
        }, [&](unsigned long long count, const uint* emitting, const uint* ends, uint first){
            fillBandFromLattice<<<(uint)(count / BLOCKSIZE + 1), BLOCKSIZE, 0, stream>>>(lattice, band, obstacles.state(), seed, emitting, ends, first,
                px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr());
        });
        gpuErrchk(cudaFreeAsync(occupied, stream));
    }
    if(added > 0){  //the new particles need their cells, and to be sorted in with the rest; rootCell leaves the others as they are
        alignParticlesToGrid();
        sortParticles();
    }
}

void Particles::setSourceMeshes(const std::vector<SceneObstacle>& meshes, const std::vector<FluidShape>& startingFluids){
    sourceMeshes.build(meshes, grid.cellSize / (2<<refinementLevel), stream);
    fluids.count = 0;
    for(const FluidShape& fluid : startingFluids){
        if(fluids.count < MAX_SOURCE_SHAPES){
            fluids.shapes[fluids.count++] = fluid;
        }
    }
    placeSourceMeshes();
}

//every mesh emitter, sink and fluid takes its field where it is now, and the box around it; one left out of the build holds nothing
void Particles::placeSourceMeshes(){
    if(sourceMeshes.count() == 0){
        return;
    }
    sourceMeshes.update(elapsedTime);
    auto place = [&](FluidShape& shape){
        if(shape.kind != FluidShape::MESH){
            return;
        }
        int index = sourceMeshes.builtFrom(shape.meshIndex);
        if(index < 0){
            shape.kind = FluidShape::BOX;
            for(int axis = 0; axis < 3; ++axis){
                shape.low[axis] = INFINITY;
                shape.high[axis] = -INFINITY;
            }
            return;
        }
        shape.mesh = sourceMeshes.state().items[index];
        sourceMeshes.bounds(index, shape.low, shape.high);
    };
    for(int emitter = 0; emitter < sources.numEmitters; ++emitter){
        place(sources.emitters[emitter]);
    }
    for(int sink = 0; sink < sources.numSinks; ++sink){
        place(sources.sinks[sink]);
    }
    for(int fluid = 0; fluid < fluids.count; ++fluid){
        place(fluids.shapes[fluid]);
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
