//Copyright 2023 Aberrant Behavior LLC

//The numbers a regression test compares frame by frame (see diagnostics.hu), in two passes: one over the particles for their sums, the box around them,
//the fastest and the hash, and one over the nodes holding particles for how full their voxels are. Each partition runs both over its own particles and
//nodes, then the partitions combine their parts: integers added exactly, maxima exactly, and the double sums in partition order

#include "diagnostics.hu"
#include "particles.hu"
#include "algorithms/voxelSolveFunctions.hu"    //calcSubCellWidth
#include <cmath>
#include <cstring>
#include <vector>

//the particle sums, each a double, in this order
enum ParticleSum{
    KINETIC_ENERGY,
    MOMENTUM_X, MOMENTUM_Y, MOMENTUM_Z,
    ANGULAR_MOMENTUM_X, ANGULAR_MOMENTUM_Y, ANGULAR_MOMENTUM_Z,
    POSITION_X, POSITION_Y, POSITION_Z,
    NUM_PARTICLE_SUMS
};
static const uint PARTIAL_WIDTH = NUM_PARTICLE_SUMS + 6;  //per block: the sums, then the lowest and highest x, y, z

//the particle pass always runs this many blocks of this many threads, so the same particles in the same order are always summed in the same groups
static const uint SUM_BLOCKS = 256;
static const uint SUM_THREADS = 256;

//the integer totals, in this order, each summed over the partitions
enum IntegerTotal{
    PARTICLE_COUNT,
    HASH_HIGH,
    HASH_LOW,
    OCCUPIED_VOXELS,
    PER_VOXEL,                          //the histogram's buckets start here
    CORE = PER_VOXEL + DIAGNOSTIC_BUCKETS,
    NUM_INTEGER_TOTALS = CORE + DIAGNOSTIC_BUCKETS
};

__device__ inline unsigned long long mixBits(unsigned long long x){   //splitmix64's finalizer: each input bit flips about half the output bits
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    return x ^ x >> 31;
}

//a 64-bit hash of a particle's exact state: the bits of its position and velocity
__device__ inline unsigned long long particleHash(double x, double y, double z, float u, float v, float w){
    unsigned long long hash = mixBits((unsigned long long)__double_as_longlong(x) + 0x9e3779b97f4a7c15ull);
    hash = mixBits(hash ^ (unsigned long long)__double_as_longlong(y));
    hash = mixBits(hash ^ (unsigned long long)__double_as_longlong(z));
    hash = mixBits(hash ^ ((unsigned long long)__float_as_uint(v) << 32 | __float_as_uint(u)));
    return mixBits(hash ^ __float_as_uint(w));
}

struct Add{
    __device__ double operator()(double a, double b) const{
        return a + b;
    }
};

struct Lowest{
    __device__ double operator()(double a, double b) const{
        return fmin(a, b);
    }
};

struct Highest{
    __device__ double operator()(double a, double b) const{
        return fmax(a, b);
    }
};

//value combined over the block, the same bits in every thread: a butterfly within each warp (combine(a, b) == combine(b, a), so every lane ends with the
//same bits), then the warps' results in warp order
template <typename Combine>
__device__ double acrossBlock(double value, double* warpValues, Combine combine){
    for(int lanes = 16; lanes > 0; lanes >>= 1){
        value = combine(value, __shfl_xor_sync(0xffffffff, value, lanes));
    }
    __syncthreads();    //the last call's readers are done with warpValues
    if(threadIdx.x % 32 == 0){
        warpValues[threadIdx.x / 32] = value;
    }
    __syncthreads();
    double total = warpValues[0];
    for(uint warp = 1; warp < blockDim.x / 32; ++warp){
        total = combine(total, warpValues[warp]);
    }
    return total;
}

//each block's share of the particle sums and of the box around the particles, into partials; the fastest speed and the hash into their totals, which as
//integers come out the same in any order
__global__ void sumParticles(uint numParticles, const double* px, const double* py, const double* pz, const float* vx, const float* vy, const float* vz,
                             double3 centre, double* partials, unsigned long long* integers, unsigned int* fastestBits){
    __shared__ double warpValues[SUM_THREADS / 32];
    double sums[NUM_PARTICLE_SUMS] = {};
    double lowest[3] = {INFINITY, INFINITY, INFINITY};
    double highest[3] = {-INFINITY, -INFINITY, -INFINITY};
    float fastest = 0.0f;
    unsigned long long hashHigh = 0;
    unsigned long long hashLow = 0;
    for(uint index = threadIdx.x + blockIdx.x*blockDim.x; index < numParticles; index += blockDim.x*gridDim.x){
        double position[3] = {px[index], py[index], pz[index]};
        float velocity[3] = {vx[index], vy[index], vz[index]};
        double r[3] = {position[0] - centre.x, position[1] - centre.y, position[2] - centre.z};
        sums[KINETIC_ENERGY] += 0.5*((double)velocity[0]*velocity[0] + (double)velocity[1]*velocity[1] + (double)velocity[2]*velocity[2]);
        sums[ANGULAR_MOMENTUM_X] += r[1]*velocity[2] - r[2]*velocity[1];
        sums[ANGULAR_MOMENTUM_Y] += r[2]*velocity[0] - r[0]*velocity[2];
        sums[ANGULAR_MOMENTUM_Z] += r[0]*velocity[1] - r[1]*velocity[0];
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            sums[MOMENTUM_X + axis] += velocity[axis];
            sums[POSITION_X + axis] += position[axis];
            lowest[axis] = fmin(lowest[axis], position[axis]);
            highest[axis] = fmax(highest[axis], position[axis]);
        }
        fastest = fmaxf(fastest, sqrtf(velocity[0]*velocity[0] + velocity[1]*velocity[1] + velocity[2]*velocity[2]));
        unsigned long long hash = particleHash(position[0], position[1], position[2], velocity[0], velocity[1], velocity[2]);
        hashHigh += hash >> 32;
        hashLow += hash & 0xffffffffull;
    }
    double* mine = partials + blockIdx.x*PARTIAL_WIDTH;
    for(int sum = 0; sum < NUM_PARTICLE_SUMS; ++sum){
        double total = acrossBlock(sums[sum], warpValues, Add());
        if(threadIdx.x == 0){
            mine[sum] = total;
        }
    }
    for(int axis = 0; axis < 3; ++axis){
        double low = acrossBlock(lowest[axis], warpValues, Lowest());
        double high = acrossBlock(highest[axis], warpValues, Highest());
        if(threadIdx.x == 0){
            mine[NUM_PARTICLE_SUMS + axis] = low;
            mine[NUM_PARTICLE_SUMS + 3 + axis] = high;
        }
    }
    atomicMax(fastestBits, __float_as_uint(fastest));  //a non-negative float's bits order like the float
    atomicAdd(integers + HASH_HIGH, hashHigh);
    atomicAdd(integers + HASH_LOW, hashLow);
}

//per node holding particles: how many sit in each of its interior voxels, counted as P2G counts them, then the occupied voxels and the histograms of
//particles per voxel, over all of them and over the node's central voxels whose 6 neighbours, all inside the node, hold particles. Integers throughout,
//so the order the nodes and particles come in doesn't matter
__global__ void countVoxelParticles(uint numParticleNodes, uint numParticles, const uint* firstParticles, const uint* gridPosition, const double* px, const double* py, const double* pz,
                                    Grid grid, uint refinementLevel, int apronCells, unsigned long long* integers){
    extern __shared__ int counts[];
    int width = 2<<refinementLevel;
    int voxels = width*width*width;
    for(int voxel = threadIdx.x; voxel < voxels; voxel += blockDim.x){
        counts[voxel] = 0;
    }
    __syncthreads();
    uint first = firstParticles[blockIdx.x];
    uint last = blockIdx.x == numParticleNodes - 1 ? numParticles : firstParticles[blockIdx.x + 1];
    uint cell = gridPosition[first];
    uint cellX = cell % grid.sizeX;
    uint cellY = cell / grid.sizeX % grid.sizeY;
    uint cellZ = cell / (grid.sizeX*grid.sizeY);
    float perVoxel = 1.0f / (float)calcSubCellWidth(refinementLevel, grid);
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        //the voxel P2G counts the particle in: positionInNodeBlock's arithmetic exactly, then back from block to interior coordinates
        int x = (int)((float)(px[index] - grid.negX - cellX*grid.cellSize)*perVoxel + apronCells) - apronCells;
        int y = (int)((float)(py[index] - grid.negY - cellY*grid.cellSize)*perVoxel + apronCells) - apronCells;
        int z = (int)((float)(pz[index] - grid.negZ - cellZ*grid.cellSize)*perVoxel + apronCells) - apronCells;
        x = min(max(x, 0), width - 1);
        y = min(max(y, 0), width - 1);
        z = min(max(z, 0), width - 1);
        atomicAdd(counts + x + y*width + z*width*width, 1);
    }
    __syncthreads();
    for(int voxel = threadIdx.x; voxel < voxels; voxel += blockDim.x){
        int count = counts[voxel];
        int bucket = min(count, DIAGNOSTIC_BUCKETS - 1);
        if(count > 0){
            atomicAdd(integers + OCCUPIED_VOXELS, 1ull);
            atomicAdd(integers + PER_VOXEL + bucket, 1ull);
        }
        int x = voxel % width;
        int y = voxel / width % width;
        int z = voxel / (width*width);
        bool central = x > 0 && y > 0 && z > 0 && x < width - 1 && y < width - 1 && z < width - 1;
        if(central && counts[voxel - 1] && counts[voxel + 1] && counts[voxel - width] && counts[voxel + width] && counts[voxel - width*width] && counts[voxel + width*width]){
            atomicAdd(integers + CORE + bucket, 1ull);
        }
    }
}

FrameDiagnostics computeFrameDiagnostics(Particles& partition){
    cudaStream_t stream = partition.stream;
    const Grid& grid = partition.grid;
    uint numParticles = partition.size;
    double* partials;
    unsigned long long* integers;
    unsigned int* fastestBits;
    double* sums;
    gpuErrchk(cudaMallocAsync((void**)&partials, sizeof(double)*SUM_BLOCKS*PARTIAL_WIDTH, stream));
    gpuErrchk(cudaMallocAsync((void**)&integers, sizeof(unsigned long long)*NUM_INTEGER_TOTALS, stream));
    gpuErrchk(cudaMallocAsync((void**)&fastestBits, sizeof(unsigned int), stream));
    gpuErrchk(cudaMallocAsync((void**)&sums, sizeof(double)*NUM_PARTICLE_SUMS, stream));
    gpuErrchk(cudaMemsetAsync(integers, 0, sizeof(unsigned long long)*NUM_INTEGER_TOTALS, stream));
    gpuErrchk(cudaMemsetAsync(fastestBits, 0, sizeof(unsigned int), stream));
    unsigned long long count = numParticles;
    gpuErrchk(cudaMemcpyAsync(integers + PARTICLE_COUNT, &count, sizeof(count), cudaMemcpyHostToDevice, stream));
    if(numParticles > 0){
        double3 centre = make_double3(grid.negX + 0.5*grid.sizeX*grid.cellSize, grid.negY + 0.5*grid.sizeY*grid.cellSize, grid.negZ + 0.5*grid.sizeZ*grid.cellSize);
        sumParticles<<<SUM_BLOCKS, SUM_THREADS, 0, stream>>>(numParticles, partition.px.devPtr(), partition.py.devPtr(), partition.pz.devPtr(),
            partition.vx.devPtr(), partition.vy.devPtr(), partition.vz.devPtr(), centre, partials, integers, fastestBits);
        gpuErrchk(cudaPeekAtLastError());
    }
    if(partition.numParticleNodes > 0){
        int width = 2<<partition.refinementLevel;
        countVoxelParticles<<<partition.numParticleNodes, 128, sizeof(int)*width*width*width, stream>>>(partition.numParticleNodes, numParticles,
            partition.gridNodeIndicesToFirstParticleIndex.devPtr(), partition.gridCell.devPtr(), partition.px.devPtr(), partition.py.devPtr(), partition.pz.devPtr(),
            grid, partition.refinementLevel, (int)std::floor(partition.radius), integers);
        gpuErrchk(cudaPeekAtLastError());
    }
    //this partition's sums, adding the blocks' shares in block order, and its box
    std::vector<double> blockPartials(SUM_BLOCKS*PARTIAL_WIDTH);
    unsigned int bits = 0;
    if(numParticles > 0){
        gpuErrchk(cudaMemcpyAsync(blockPartials.data(), partials, sizeof(double)*blockPartials.size(), cudaMemcpyDeviceToHost, stream));
    }
    gpuErrchk(cudaMemcpyAsync(&bits, fastestBits, sizeof(bits), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    double partitionSums[NUM_PARTICLE_SUMS] = {};
    double lowest[3] = {INFINITY, INFINITY, INFINITY};
    double highest[3] = {-INFINITY, -INFINITY, -INFINITY};
    if(numParticles > 0){
        for(uint block = 0; block < SUM_BLOCKS; ++block){
            const double* share = blockPartials.data() + block*PARTIAL_WIDTH;
            for(int sum = 0; sum < NUM_PARTICLE_SUMS; ++sum){
                partitionSums[sum] += share[sum];
            }
            for(int axis = 0; axis < 3; ++axis){
                lowest[axis] = std::fmin(lowest[axis], share[NUM_PARTICLE_SUMS + axis]);
                highest[axis] = std::fmax(highest[axis], share[NUM_PARTICLE_SUMS + 3 + axis]);
            }
        }
    }
    float fastest;
    std::memcpy(&fastest, &bits, sizeof(fastest));
    //over every partition, each making the same calls in the same order: the integers add up exactly, the doubles in partition order
    PartitionContext& context = *partition.context;
    context.sumOverPartitions((long long*)integers, NUM_INTEGER_TOTALS, stream);
    gpuErrchk(cudaMemcpyAsync(sums, partitionSums, sizeof(partitionSums), cudaMemcpyHostToDevice, stream));
    for(int sum = 0; sum < NUM_PARTICLE_SUMS; ++sum){
        context.sumOverPartitions(sums + sum, stream);
    }
    unsigned long long totals[NUM_INTEGER_TOTALS];
    gpuErrchk(cudaMemcpyAsync(totals, integers, sizeof(totals), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(partitionSums, sums, sizeof(partitionSums), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    FrameDiagnostics diagnostics;
    for(int axis = 0; axis < 3; ++axis){
        diagnostics.lowest[axis] = -context.maxOverPartitions(-lowest[axis]);
        diagnostics.highest[axis] = context.maxOverPartitions(highest[axis]);
    }
    diagnostics.fastest = context.maxOverPartitions(fastest);
    gpuErrchk(cudaFreeAsync(partials, stream));
    gpuErrchk(cudaFreeAsync(integers, stream));
    gpuErrchk(cudaFreeAsync(fastestBits, stream));
    gpuErrchk(cudaFreeAsync(sums, stream));
    diagnostics.particles = totals[PARTICLE_COUNT];
    diagnostics.hashHigh = totals[HASH_HIGH];
    diagnostics.hashLow = totals[HASH_LOW];
    diagnostics.occupiedVoxels = totals[OCCUPIED_VOXELS];
    for(int bucket = 0; bucket < DIAGNOSTIC_BUCKETS; ++bucket){
        diagnostics.perVoxel[bucket] = totals[PER_VOXEL + bucket];
        diagnostics.core[bucket] = totals[CORE + bucket];
    }
    diagnostics.kineticEnergy = partitionSums[KINETIC_ENERGY];
    for(int axis = 0; axis < 3; ++axis){
        diagnostics.momentum[axis] = partitionSums[MOMENTUM_X + axis];
        diagnostics.angularMomentum[axis] = partitionSums[ANGULAR_MOMENTUM_X + axis];
        diagnostics.positionSum[axis] = partitionSums[POSITION_X + axis];
    }
    return diagnostics;
}
