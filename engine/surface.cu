//Copyright 2023 Aberrant Behavior LLC

//A liquid's surface from its particles, on the GPU (surface.hu), in five passes over compact lists:
//
//1. cells: the particles sorted into a lattice of cells at least as wide as their reach, so a point in the field only hears from the particles in its own
//   cell and the 26 around it. Each cell holds side^3 samples of the field, side along each edge, the first at its lowest corner
//2. band: the cells by the edge of the space the particles fill: each occupied cell with an empty neighbour, and the 26 around each of those. A cell
//   beyond the band is taken to be deep in the liquid if it holds particles, and outside it if not, the field there the constants Lattice gives. Where
//   the field beside such a cell says otherwise (a gap among sparse particles, beside a cell taken to be deep), the band grows to take that cell in,
//   and the field is worked out again, until nothing beyond the band disagrees with it: so the surface lies wholly in the band, and the mesh is closed
//3. field: at each sample of the band, Zhu and Bridson's |x - xbar| - r, where xbar is the nearby particles' positions averaged with the weight
//   (1 - d^2/R^2)^3 over those within R: Houdini's Average Position. The liquid's velocity is averaged alike, for the mesh's points
//4. smoothing: passes of the mean of each sample and its six neighbours
//5. surface nets: a point in each cube of 8 samples the surface cuts, at the mean of where it cuts the cube's edges, and a quad joining the 4 cubes around
//   each edge it cuts, facing outwards. Neighbouring cubes share their points, so the mesh is closed and welded
//
//The fields (fields()) are on a lattice of their own, at their voxel size, over every cell within reach of a particle and a few cells beyond. Their
//velocity is the particles' velocities averaged as the field averages their positions, then carried on past where they reach, a voxel a pass, each voxel
//without one taking its neighbours' mean. Their surface is the field's zero level as a signed distance field: at each voxel beside the surface, the
//distance to where it crosses the edges between them, and from there out to the half width, Godunov's upwind solution of |grad d| = 1, relaxed a pass at
//a time.
//
//Each pass over samples is a block per band cell, whose threads read the cell's 27 neighbours' places in the band (found once, in findNeighbours), so a
//sample's neighbour in any direction is a direct lookup. There are no atomics: every sum runs in a fixed order and every list is compacted by prefix sums,
//so a frame always meshes to the same mesh

#include "surface.hu"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <cub/cub.cuh>

namespace{

using Key = unsigned long long;
constexpr Key NO_CELL = ~0ull;
constexpr int MAX_SIDE = 8;                 //samples along a cell's edge: a block's threads are its cell's samples, so up to 512
constexpr int OUTSIDE = -1, DEEP = -2;      //a neighbour that isn't in the band: outside the liquid, or deep inside it

struct Lattice{
    float3 origin;      //cell (0, 0, 0)'s lowest corner, which is sample (0, 0, 0)
    float cellSize;
    float h;            //between samples
    int side;           //samples along a cell's edge
    int3 cells;         //cells along each axis
    float reach;        //R: how far a particle reaches into the field
    float radius;       //r: each particle's radius
    float outside;      //the field where no particle reaches: R - r, the most it is where one does
    float deep;         //the field deep in the liquid: -r, what it is amid evenly spread particles

    __host__ __device__ Key key(int x, int y, int z) const{
        return (Key)x + (Key)cells.x*((Key)y + (Key)cells.y*(Key)z);
    }
    __device__ int3 cell(Key key) const{
        return make_int3((int)(key % (Key)cells.x), (int)(key / (Key)cells.x % (Key)cells.y), (int)(key / ((Key)cells.x*(Key)cells.y)));
    }
    __device__ bool contains(int x, int y, int z) const{
        return x >= 0 && y >= 0 && z >= 0 && x < cells.x && y < cells.y && z < cells.z;
    }
    __device__ int samples() const{     //in a cell
        return side*side*side;
    }
};

//the place of key in a sorted list, or -1
__device__ inline int findKey(const Key* keys, int count, Key key){
    int low = 0, high = count;
    while(low < high){
        int middle = low + (high - low)/2;
        if(keys[middle] < key){
            low = middle + 1;
        }
        else{
            high = middle;
        }
    }
    return low < count && keys[low] == key ? low : -1;
}

// ---- 1: cells ----

__global__ void cellKeys(int count, const float* x, const float* y, const float* z, Lattice lattice, Key* keys, unsigned int* order){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < count){
        int cx = min(max((int)floorf((x[i] - lattice.origin.x) / lattice.cellSize), 0), lattice.cells.x - 1);
        int cy = min(max((int)floorf((y[i] - lattice.origin.y) / lattice.cellSize), 0), lattice.cells.y - 1);
        int cz = min(max((int)floorf((z[i] - lattice.origin.z) / lattice.cellSize), 0), lattice.cells.z - 1);
        keys[i] = lattice.key(cx, cy, cz);
        order[i] = (unsigned int)i;
    }
}

//the particles in cell order, each position and velocity together
__global__ void gatherParticles(int count, const unsigned int* order, const float* x, const float* y, const float* z, const float* vx, const float* vy, const float* vz,
                                float4* positions, float4* velocities){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < count){
        unsigned int from = order[i];
        positions[i] = make_float4(x[from], y[from], z[from], 0.0f);
        velocities[i] = make_float4(vx[from], vy[from], vz[from], 0.0f);
    }
}

// ---- 2: the band ----

//each occupied cell with an empty neighbour puts itself and its 26 neighbours forward for the band; the rest put forward nothing
__global__ void markBand(int occupiedCount, const Key* occupied, Lattice lattice, Key* candidates){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i >= occupiedCount){
        return;
    }
    int3 c = lattice.cell(occupied[i]);
    bool edge = false;
    for(int n = 0; n < 27 && !edge; ++n){
        int x = c.x + n % 3 - 1, y = c.y + n / 3 % 3 - 1, z = c.z + n / 9 - 1;
        edge = !lattice.contains(x, y, z) || findKey(occupied, occupiedCount, lattice.key(x, y, z)) < 0;
    }
    for(int n = 0; n < 27; ++n){
        int x = c.x + n % 3 - 1, y = c.y + n / 3 % 3 - 1, z = c.z + n / 9 - 1;
        candidates[27*(size_t)i + n] = edge && lattice.contains(x, y, z) ? lattice.key(x, y, z) : NO_CELL;
    }
}

//each band cell's 27 neighbours, x fastest from -1 to 1: its place in the band, or OUTSIDE or DEEP
__global__ void findNeighbours(int bandCount, const Key* band, int occupiedCount, const Key* occupied, Lattice lattice, int* neighbours){
    size_t i = threadIdx.x + (size_t)blockIdx.x*blockDim.x;
    if(i >= 27*(size_t)bandCount){
        return;
    }
    int n = (int)(i % 27);
    int3 c = lattice.cell(band[i / 27]);
    int x = c.x + n % 3 - 1, y = c.y + n / 3 % 3 - 1, z = c.z + n / 9 - 1;
    int place = OUTSIDE;
    if(lattice.contains(x, y, z)){
        Key key = lattice.key(x, y, z);
        place = findKey(band, bandCount, key);
        if(place < 0){
            place = findKey(occupied, occupiedCount, key) >= 0 ? DEEP : OUTSIDE;
        }
    }
    neighbours[i] = place;
}

// ---- the passes over samples: a block per band cell, a thread per sample ----

__device__ inline void loadNeighbours(const int* neighbours, int* around){
    for(int n = threadIdx.x; n < 27; n += blockDim.x){
        around[n] = neighbours[27*(size_t)blockIdx.x + n];
    }
    __syncthreads();
}

__device__ inline int3 localSample(int side){
    return make_int3((int)threadIdx.x % side, (int)threadIdx.x / side % side, (int)threadIdx.x / (side*side));
}

//where the sample offset by (dx, dy, dz) from this thread's lies, each offset -1, 0 or 1: its cell among the 27 around this one, and its index there
__device__ inline void locate(int3 local, int dx, int dy, int dz, int side, int& neighbour, int& index){
    int x = local.x + dx, y = local.y + dy, z = local.z + dz;
    int nx = x < 0 ? -1 : x >= side ? 1 : 0, ny = y < 0 ? -1 : y >= side ? 1 : 0, nz = z < 0 ? -1 : z >= side ? 1 : 0;
    neighbour = (nx + 1) + 3*(ny + 1) + 9*(nz + 1);
    index = (x - nx*side) + side*((y - ny*side) + side*(z - nz*side));
}

//the field at the sample offset from this thread's: from the band, or the constant of the cell it's in
__device__ inline float fieldAt(const int* around, const float* field, const Lattice& lattice, int3 local, int dx, int dy, int dz){
    int neighbour, index;
    locate(local, dx, dy, dz, lattice.side, neighbour, index);
    int place = around[neighbour];
    if(place >= 0){
        return field[(size_t)place*lattice.samples() + index];
    }
    return place == DEEP ? lattice.deep : lattice.outside;
}

//the liquid's velocity at the sample offset from this thread's, its w 1 if any particle reaches there, 0 if none does or it isn't in the band
__device__ inline float4 velocityAt(const int* around, const float4* velocities, const Lattice& lattice, int3 local, int dx, int dy, int dz){
    int neighbour, index;
    locate(local, dx, dy, dz, lattice.side, neighbour, index);
    int place = around[neighbour];
    return place >= 0 ? velocities[(size_t)place*lattice.samples() + index] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// ---- 3: the field ----

__global__ void evaluateField(const Key* band, int occupiedCount, const Key* occupied, const int* starts, const int* counts, const float4* positions,
                              const float4* particleVelocities, Lattice lattice, float* field, float4* velocities){
    __shared__ int2 runs[27];       //each neighbouring cell's particles: the first, and how many
    int3 c = lattice.cell(band[blockIdx.x]);
    for(int n = threadIdx.x; n < 27; n += blockDim.x){
        int x = c.x + n % 3 - 1, y = c.y + n / 3 % 3 - 1, z = c.z + n / 9 - 1;
        int place = lattice.contains(x, y, z) ? findKey(occupied, occupiedCount, lattice.key(x, y, z)) : -1;
        runs[n] = place >= 0 ? make_int2(starts[place], counts[place]) : make_int2(0, 0);
    }
    __syncthreads();
    int3 local = localSample(lattice.side);
    float3 p = make_float3(lattice.origin.x + lattice.h*(c.x*lattice.side + local.x), lattice.origin.y + lattice.h*(c.y*lattice.side + local.y),
                           lattice.origin.z + lattice.h*(c.z*lattice.side + local.z));
    float reach2 = lattice.reach*lattice.reach;
    float weight = 0.0f;
    float3 mean = make_float3(0.0f, 0.0f, 0.0f), velocity = make_float3(0.0f, 0.0f, 0.0f);
    for(int n = 0; n < 27; ++n){
        for(int k = runs[n].x; k < runs[n].x + runs[n].y; ++k){
            float4 q = positions[k];
            float dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
            float d2 = dx*dx + dy*dy + dz*dz;
            if(d2 < reach2){
                float s = 1.0f - d2/reach2;
                float w = s*s*s;
                float4 u = particleVelocities[k];
                weight += w;
                mean.x += w*q.x; mean.y += w*q.y; mean.z += w*q.z;
                velocity.x += w*u.x; velocity.y += w*u.y; velocity.z += w*u.z;
            }
        }
    }
    size_t at = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if(weight > 0.0f){
        float dx = p.x - mean.x/weight, dy = p.y - mean.y/weight, dz = p.z - mean.z/weight;
        field[at] = sqrtf(dx*dx + dy*dy + dz*dz) - lattice.radius;
        velocities[at] = make_float4(velocity.x/weight, velocity.y/weight, velocity.z/weight, 1.0f);
    }
    else{
        field[at] = lattice.outside;
        velocities[at] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

//the cells beyond the band whose constant disagrees with the field beside them: each band cell puts forward each of its 26 neighbours that isn't in the
//band, inside the lattice, and has a sample next to one of the cell's (a corner of a cube they share) on the other side of the surface
__global__ void checkBand(const Key* band, const int* neighbours, Lattice lattice, const float* field, Key* candidates){
    __shared__ int around[27];
    __shared__ int grow[27];
    for(int n = threadIdx.x; n < 27; n += blockDim.x){
        around[n] = neighbours[27*(size_t)blockIdx.x + n];
        grow[n] = 0;
    }
    __syncthreads();
    int3 local = localSample(lattice.side);
    bool inside = field[(size_t)blockIdx.x*blockDim.x + threadIdx.x] < 0.0f;
    for(int d = 0; d < 27; ++d){
        int neighbour, index;
        locate(local, d % 3 - 1, d / 3 % 3 - 1, d / 9 - 1, lattice.side, neighbour, index);
        int place = around[neighbour];
        if(place < 0 && inside != (place == DEEP)){
            grow[neighbour] = 1;    //every thread that writes it writes 1
        }
    }
    __syncthreads();
    int3 c = lattice.cell(band[blockIdx.x]);
    for(int n = threadIdx.x; n < 27; n += blockDim.x){
        int x = c.x + n % 3 - 1, y = c.y + n / 3 % 3 - 1, z = c.z + n / 9 - 1;
        candidates[27*(size_t)blockIdx.x + n] = grow[n] && lattice.contains(x, y, z) ? lattice.key(x, y, z) : NO_CELL;
    }
}

//the cells within rings of each occupied one, which make the velocity field's lattice: (2 rings + 1)^3 per occupied cell, NO_CELL beyond the lattice
__global__ void dilateCells(int occupiedCount, const Key* occupied, Lattice lattice, int rings, Key* candidates){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i >= occupiedCount){
        return;
    }
    int3 c = lattice.cell(occupied[i]);
    int width = 2*rings + 1;
    size_t at = (size_t)i*width*width*width;
    for(int z = -rings; z <= rings; ++z){
        for(int y = -rings; y <= rings; ++y){
            for(int x = -rings; x <= rings; ++x){
                candidates[at++] = lattice.contains(c.x + x, c.y + y, c.z + z) ? lattice.key(c.x + x, c.y + y, c.z + z) : NO_CELL;
            }
        }
    }
}

// ---- 4: smoothing ----

__global__ void smoothField(const int* neighbours, Lattice lattice, const float* from, float* to){
    __shared__ int around[27];
    loadNeighbours(neighbours, around);
    int3 local = localSample(lattice.side);
    float sum = fieldAt(around, from, lattice, local, 0, 0, 0) + fieldAt(around, from, lattice, local, -1, 0, 0) + fieldAt(around, from, lattice, local, 1, 0, 0)
              + fieldAt(around, from, lattice, local, 0, -1, 0) + fieldAt(around, from, lattice, local, 0, 1, 0) + fieldAt(around, from, lattice, local, 0, 0, -1)
              + fieldAt(around, from, lattice, local, 0, 0, 1);
    to[(size_t)blockIdx.x*blockDim.x + threadIdx.x] = sum / 7.0f;
}

// ---- the velocity field ----

//one pass carrying the velocity on past where particles reach: each sample without one takes the mean of its six neighbours' that have one
__global__ void extendVelocity(const int* neighbours, Lattice lattice, const float4* from, float4* to){
    __shared__ int around[27];
    loadNeighbours(neighbours, around);
    int3 local = localSample(lattice.side);
    size_t at = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    float4 own = from[at];
    if(own.w == 0.0f){
        float3 sum = make_float3(0.0f, 0.0f, 0.0f);
        int found = 0;
        for(int d = 0; d < 6; ++d){
            int step = d & 1 ? 1 : -1;
            float4 v = velocityAt(around, from, lattice, local, d >> 1 == 0 ? step : 0, d >> 1 == 1 ? step : 0, d >> 1 == 2 ? step : 0);
            if(v.w > 0.0f){
                sum.x += v.x; sum.y += v.y; sum.z += v.z;
                ++found;
            }
        }
        if(found > 0){
            own = make_float4(sum.x/found, sum.y/found, sum.z/found, 1.0f);
        }
    }
    to[at] = own;
}

//the samples with a velocity, each at its place among them (places: their flags summed before each): its voxel, counted from first, and velocity
__global__ void listVelocities(const Key* cells, Lattice lattice, int3 first, const float4* velocities, const unsigned int* places, int* voxels, float* values){
    size_t sample = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if(places[sample + 1] == places[sample]){
        return;
    }
    int3 c = lattice.cell(cells[blockIdx.x]);
    int3 local = localSample(lattice.side);
    size_t at = 3*(size_t)places[sample];
    voxels[at] = first.x + c.x*lattice.side + local.x;
    voxels[at + 1] = first.y + c.y*lattice.side + local.y;
    voxels[at + 2] = first.z + c.z*lattice.side + local.z;
    float4 v = velocities[sample];
    values[at] = v.x;
    values[at + 1] = v.y;
    values[at + 2] = v.z;
}

__global__ void flagVelocities(const float4* velocities, unsigned int* flags){
    size_t sample = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    flags[sample] = velocities[sample].w > 0.0f ? 1u : 0u;
}

constexpr float FAR = 1e20f;    //a distance not yet known

//the signed distance at the sample offset from this thread's: FAR beyond the lattice's cells, which are all outside
__device__ inline float distanceAt(const int* around, const float* distances, const Lattice& lattice, int3 local, int dx, int dy, int dz){
    int neighbour, index;
    locate(local, dx, dy, dz, lattice.side, neighbour, index);
    int place = around[neighbour];
    if(place >= 0){
        return distances[(size_t)place*lattice.samples() + index];
    }
    return place == DEEP ? -FAR : FAR;
}

//the distances beside the surface: at each sample with a neighbour on the other side of it, the distance to the plane through where it crosses the edges
//to them, from the field interpolated along each; FAR, signed as the field is, everywhere else
__global__ void startDistances(const int* neighbours, Lattice lattice, const float* field, float* distances){
    __shared__ int around[27];
    loadNeighbours(neighbours, around);
    int3 local = localSample(lattice.side);
    size_t at = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    float here = field[at];
    bool inside = here < 0.0f;
    float inverse = 0.0f;       //the sum over the axes the surface crosses of 1/d^2, d the distance to it along each
    for(int axis = 0; axis < 3; ++axis){
        float nearest = FAR;
        for(int step = -1; step <= 1; step += 2){
            float there = fieldAt(around, field, lattice, local, axis == 0 ? step : 0, axis == 1 ? step : 0, axis == 2 ? step : 0);
            if((there < 0.0f) != inside){
                nearest = fminf(nearest, here / (here - there) * lattice.h);
            }
        }
        if(nearest < FAR){
            nearest = fmaxf(nearest, 1e-4f*lattice.h);
            inverse += 1.0f / (nearest*nearest);
        }
    }
    float distance = inverse > 0.0f ? rsqrtf(inverse) : FAR;
    distances[at] = inside ? -distance : distance;
}

//one pass of the distances out from the surface: each sample not beside it (fixed holds FAR there) takes the upwind solution of |grad d| = 1 from its
//nearest neighbours along each axis on its own side, if that's nearer than it had
__global__ void relaxDistances(const int* neighbours, Lattice lattice, const float* fixed, const float* from, float* to){
    __shared__ int around[27];
    loadNeighbours(neighbours, around);
    size_t at = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    float own = from[at];
    if(fabsf(fixed[at]) < FAR){
        to[at] = own;
        return;
    }
    int3 local = localSample(lattice.side);
    bool inside = own < 0.0f;
    float nearest[3];
    for(int axis = 0; axis < 3; ++axis){
        nearest[axis] = FAR;
        for(int step = -1; step <= 1; step += 2){
            float there = distanceAt(around, from, lattice, local, axis == 0 ? step : 0, axis == 1 ? step : 0, axis == 2 ? step : 0);
            if((there < 0.0f) == inside){
                nearest[axis] = fminf(nearest[axis], fabsf(there));
            }
        }
    }
    float a = fminf(fminf(nearest[0], nearest[1]), nearest[2]);     //sorted: a, b, c
    float b = fmaxf(fminf(nearest[0], nearest[1]), fminf(fmaxf(nearest[0], nearest[1]), nearest[2]));
    float c = fmaxf(fmaxf(nearest[0], nearest[1]), nearest[2]);
    float h = lattice.h;
    float t = a + h;
    if(b < FAR && t > b){
        t = 0.5f*(a + b + sqrtf(fmaxf(2.0f*h*h - (a - b)*(a - b), 0.0f)));
        if(c < FAR && t > c){
            float sum = a + b + c;
            t = (sum + sqrtf(fmaxf(sum*sum - 3.0f*(a*a + b*b + c*c - h*h), 0.0f))) / 3.0f;
        }
    }
    float distance = fminf(fabsf(own), t);
    to[at] = inside ? -distance : distance;
}

__global__ void flagDistances(const float* distances, float limit, unsigned int* flags){
    size_t sample = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    flags[sample] = fabsf(distances[sample]) < limit ? 1u : 0u;
}

//the samples in the narrow band, each at its place among them (places: their flags summed before each): its voxel, counted from first, and distance
__global__ void listDistances(const Key* cells, Lattice lattice, int3 first, const float* distances, const unsigned int* places, int* voxels, float* values){
    size_t sample = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if(places[sample + 1] == places[sample]){
        return;
    }
    int3 c = lattice.cell(cells[blockIdx.x]);
    int3 local = localSample(lattice.side);
    size_t at = places[sample];
    voxels[3*at] = first.x + c.x*lattice.side + local.x;
    voxels[3*at + 1] = first.y + c.y*lattice.side + local.y;
    voxels[3*at + 2] = first.z + c.z*lattice.side + local.z;
    values[at] = distances[sample];
}

// ---- 5: surface nets ----

//a cube's 12 edges, as pairs of its corners (a corner's bit 0 is +x, bit 1 +y, bit 2 +z)
__constant__ int CUBE_EDGES[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

//whether the surface cuts the cube whose lowest corner is this thread's sample: whether its corners aren't all on one side
__device__ bool cubeCut(const int* around, const float* field, const Lattice& lattice, int3 local, float value[8]){
    bool any = false, all = true;
    for(int corner = 0; corner < 8; ++corner){
        value[corner] = fieldAt(around, field, lattice, local, corner & 1, corner >> 1 & 1, corner >> 2);
        bool in = value[corner] < 0.0f;
        any = any || in;
        all = all && in;
    }
    return any && !all;
}

//which cubes the surface cuts: 1 or 0 per sample, as the cube's lowest corner
__global__ void flagCubes(const int* neighbours, Lattice lattice, const float* field, unsigned int* flags){
    __shared__ int around[27];
    loadNeighbours(neighbours, around);
    float value[8];
    flags[(size_t)blockIdx.x*blockDim.x + threadIdx.x] = cubeCut(around, field, lattice, localSample(lattice.side), value) ? 1u : 0u;
}

//each cut cube's point, at the mean of where the surface cuts its edges, and the liquid's velocity there, interpolated from the corners any particle
//reaches; at its place among the points (places: the cubes' flags summed before each)
__global__ void writePoints(const Key* band, const int* neighbours, Lattice lattice, const float* field, const float4* velocities, const unsigned int* places,
                            float* points, float* pointVelocities){
    __shared__ int around[27];
    loadNeighbours(neighbours, around);
    size_t sample = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if(places[sample + 1] == places[sample]){
        return;
    }
    int3 local = localSample(lattice.side);
    float value[8];
    cubeCut(around, field, lattice, local, value);
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    int cuts = 0;
    for(int edge = 0; edge < 12; ++edge){
        int a = CUBE_EDGES[edge][0], b = CUBE_EDGES[edge][1];
        if((value[a] < 0.0f) != (value[b] < 0.0f)){
            float t = value[a] / (value[a] - value[b]);
            sum.x += (a & 1) + t*((b & 1) - (a & 1));
            sum.y += (a >> 1 & 1) + t*((b >> 1 & 1) - (a >> 1 & 1));
            sum.z += (a >> 2) + t*((b >> 2) - (a >> 2));
            ++cuts;
        }
    }
    float3 point = make_float3(sum.x/cuts, sum.y/cuts, sum.z/cuts);     //within the cube, in samples
    float3 moving = make_float3(0.0f, 0.0f, 0.0f);
    float total = 0.0f;
    for(int corner = 0; corner < 8; ++corner){
        float4 v = velocityAt(around, velocities, lattice, local, corner & 1, corner >> 1 & 1, corner >> 2);
        float w = v.w*((corner & 1 ? point.x : 1.0f - point.x)*(corner >> 1 & 1 ? point.y : 1.0f - point.y)*(corner >> 2 ? point.z : 1.0f - point.z) + 1e-6f);
        moving.x += w*v.x; moving.y += w*v.y; moving.z += w*v.z;
        total += w;
    }
    int3 c = lattice.cell(band[blockIdx.x]);
    size_t at = 3*(size_t)places[sample];
    points[at] = lattice.origin.x + lattice.h*(c.x*lattice.side + local.x + point.x);
    points[at + 1] = lattice.origin.y + lattice.h*(c.y*lattice.side + local.y + point.y);
    points[at + 2] = lattice.origin.z + lattice.h*(c.z*lattice.side + local.z + point.z);
    pointVelocities[at] = total > 0.0f ? moving.x/total : 0.0f;
    pointVelocities[at + 1] = total > 0.0f ? moving.y/total : 0.0f;
    pointVelocities[at + 2] = total > 0.0f ? moving.z/total : 0.0f;
}

//the point of the cube whose lowest corner is offset from this thread's sample, or -1 if the band doesn't hold it
__device__ inline int cubePoint(const int* around, const Lattice& lattice, const unsigned int* places, int3 local, int dx, int dy, int dz){
    int neighbour, index;
    locate(local, dx, dy, dz, lattice.side, neighbour, index);
    int place = around[neighbour];
    if(place < 0){
        return -1;
    }
    size_t sample = (size_t)place*lattice.samples() + index;
    return places[sample + 1] != places[sample] ? (int)places[sample] : -1;
}

//the quads: each edge from this thread's sample along +x, +y and +z that the surface cuts joins the 4 cubes around it, counter-clockwise seen from
//outside. With quads null, only how many there are (counts); otherwise they're written from the count summed before this sample's (counts, so summed)
__global__ void writeQuads(const int* neighbours, Lattice lattice, const float* field, const unsigned int* cubePlaces, unsigned int* counts, int* quads){
    __shared__ int around[27];
    loadNeighbours(neighbours, around);
    int3 local = localSample(lattice.side);
    size_t sample = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    float here = fieldAt(around, field, lattice, local, 0, 0, 0);
    unsigned int made = 0;
    size_t at = quads == nullptr ? 0 : 4*(size_t)counts[sample];
    for(int axis = 0; axis < 3; ++axis){
        int e[3] = {0, 0, 0}, u[3] = {0, 0, 0}, v[3] = {0, 0, 0};
        e[axis] = 1;
        u[(axis + 1) % 3] = 1;
        v[(axis + 2) % 3] = 1;
        float there = fieldAt(around, field, lattice, local, e[0], e[1], e[2]);
        if((here < 0.0f) == (there < 0.0f)){
            continue;
        }
        int corners[4] = {      //the 4 cubes around the edge, counter-clockwise about +axis: at (0, 0), (-u, 0), (-u, -v) and (0, -v)
            cubePoint(around, lattice, cubePlaces, local, 0, 0, 0),
            cubePoint(around, lattice, cubePlaces, local, -u[0], -u[1], -u[2]),
            cubePoint(around, lattice, cubePlaces, local, -u[0] - v[0], -u[1] - v[1], -u[2] - v[2]),
            cubePoint(around, lattice, cubePlaces, local, -v[0], -v[1], -v[2])};
        if(corners[0] < 0 || corners[1] < 0 || corners[2] < 0 || corners[3] < 0){
            continue;
        }
        ++made;
        if(quads != nullptr){
            bool alongAxis = here < 0.0f;       //inside at this end, so the face looks along +axis, as counter-clockwise about it does
            for(int k = 0; k < 4; ++k){
                quads[at + k] = corners[alongAxis ? k : (4 - k) % 4];
            }
            at += 4;
        }
    }
    if(quads == nullptr){
        counts[sample] = made;
    }
}

// ---- buffers ----

void check(cudaError_t error, const char* what){
    if(error != cudaSuccess){
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
}

//GPU memory that grows as it's asked for more, and is kept for the next frame
template<typename T>
struct Buffer{
    T* data = nullptr;
    size_t capacity = 0;

    T* hold(size_t count){
        count = std::max(count, (size_t)1);
        if(count > capacity){
            if(data != nullptr){
                cudaFree(data);
                data = nullptr;
            }
            size_t grown = std::max(count, capacity + capacity/2);
            if(cudaMalloc((void**)&data, sizeof(T)*grown) != cudaSuccess){
                cudaGetLastError();
                data = nullptr;
                capacity = 0;
                throw std::runtime_error("out of GPU memory meshing (" + std::to_string((sizeof(T)*grown) >> 20) + " MB more)");
            }
            capacity = grown;
        }
        return data;
    }

    ~Buffer(){
        if(data != nullptr){
            cudaFree(data);
        }
    }
};

}

struct SurfaceMesher::Buffers{
    Buffer<float> planes;               //positions then velocities, x, y and z planes
    Buffer<Key> keys, sortedKeys;
    Buffer<unsigned int> order, sortedOrder;
    Buffer<float4> positions, particleVelocities;
    Buffer<Key> occupied;
    Buffer<int> counts, starts;
    Buffer<Key> candidates, sortedCandidates, band, merged;
    Buffer<int> neighbours;
    Buffer<float> field, smoothed;
    Buffer<float4> velocities, spareVelocities;
    Buffer<float> distances, spareDistances, fixedDistances;
    Buffer<unsigned int> cubePlaces, quadPlaces;
    Buffer<float> points, pointVelocities;
    Buffer<int> quads;
    Buffer<int> numbers;                //counts the library passes leave on the GPU
    Buffer<char> scratch;

    //runs a library pass, giving it the scratch space it asks for first
    template<typename Pass>
    void run(Pass pass){
        size_t bytes = 0;
        check(pass(nullptr, bytes), "sizing a library pass");
        check(pass(scratch.hold(bytes), bytes), "a library pass");
    }

    //the count particles in planes (positions, then velocities) sorted into lattice's cells: positions and particleVelocities in cell order, occupied
    //the cells that hold any, each one's first particle in starts and how many in counts. Returns how many cells are occupied
    int sortIntoCells(const Lattice& lattice, int count, int keyBits){
        const int threads = 256;
        int blocks = (count + threads - 1) / threads;
        float* from = planes.data;
        cellKeys<<<blocks, threads>>>(count, from, from + count, from + 2*(size_t)count, lattice, keys.hold(count), order.hold(count));
        sortedKeys.hold(count);
        sortedOrder.hold(count);
        run([&](void* space, size_t& bytes){
            return cub::DeviceRadixSort::SortPairs(space, bytes, keys.data, sortedKeys.data, order.data, sortedOrder.data, count, 0, keyBits);
        });
        gatherParticles<<<blocks, threads>>>(count, sortedOrder.data, from, from + count, from + 2*(size_t)count, from + 3*(size_t)count, from + 4*(size_t)count,
                                             from + 5*(size_t)count, positions.hold(count), particleVelocities.hold(count));
        occupied.hold(count);
        counts.hold(count);
        starts.hold(count);
        run([&](void* space, size_t& bytes){
            return cub::DeviceRunLengthEncode::Encode(space, bytes, sortedKeys.data, occupied.data, counts.data, numbers.hold(1), count);
        });
        int occupiedCount = 0;
        check(cudaMemcpy(&occupiedCount, numbers.data, sizeof(int), cudaMemcpyDeviceToHost), "counting cells");
        run([&](void* space, size_t& bytes){
            return cub::DeviceScan::ExclusiveSum(space, bytes, counts.data, starts.data, occupiedCount);
        });
        return occupiedCount;
    }

    //the number of distinct keys a library pass just sorted and made unique into list, less the NO_CELL at its end if there is one
    int distinct(const Key* list){
        int found = 0;
        check(cudaMemcpy(&found, numbers.data, sizeof(int), cudaMemcpyDeviceToHost), "counting cells");
        if(found > 0){
            Key last = 0;
            check(cudaMemcpy(&last, list + found - 1, sizeof(Key), cudaMemcpyDeviceToHost), "the last cell");
            found -= last == NO_CELL;
        }
        return found;
    }
};

SurfaceMesher::SurfaceMesher() : buffers(new Buffers){}

SurfaceMesher::~SurfaceMesher(){
    delete buffers;
}

std::string FieldSettings::problem() const{
    if(!(voxelScale > 0.0f)){
        return "the fields' voxel scale has to be positive";
    }
    if(!(halfWidth >= 1.0f && halfWidth <= 64.0f)){
        return "the fields' half width has to be from 1 to 64 voxels";
    }
    return "";
}

std::string SurfaceSettings::problem() const{
    if(!(separation > 0.0f)){
        return "the particle separation has to be positive";
    }
    if(!(voxelScale > 0.0f) || !(influenceScale > 0.0f) || !(radiusScale > 0.0f)){
        return "the voxel, influence and radius scales have to be positive";
    }
    if(radiusScale >= influenceScale){
        return "the radius scale has to be less than the influence scale";
    }
    if(influenceScale > MAX_SIDE*voxelScale){
        return "the influence scale can be at most " + std::to_string(MAX_SIDE) + " times the voxel scale";
    }
    if(smoothing < 0){
        return "smoothing can't be negative";
    }
    return "";
}

void SurfaceMesher::mesh(const float* positionPlanes, const float* velocityPlanes, unsigned long long count, const float low[3], const float high[3],
                         const SurfaceSettings& settings, SurfaceMesh& out){
    auto began = std::chrono::steady_clock::now();
    out.points.clear();
    out.velocities.clear();
    out.quads.clear();
    seconds = 0.0;
    rounds = 0;
    bandCells = 0;
    this->count = 0;
    std::string problem = settings.problem();
    if(!problem.empty()){
        throw std::runtime_error(problem);
    }
    if(count == 0){
        return;
    }
    if(count > (unsigned long long)INT_MAX){
        throw std::runtime_error("too many particles to mesh in one piece");
    }
    Buffers& b = *buffers;
    Lattice lattice;
    lattice.h = settings.separation*settings.voxelScale;
    lattice.reach = settings.separation*settings.influenceScale;
    lattice.radius = settings.separation*settings.radiusScale;
    lattice.side = std::min(std::max((int)std::ceil(lattice.reach/lattice.h - 1e-4f), 2), MAX_SIDE);
    lattice.cellSize = lattice.side*lattice.h;
    lattice.reach = std::min(lattice.reach, lattice.cellSize);     //what reaches a cell's samples has to be in it or its neighbours
    lattice.outside = lattice.reach - lattice.radius;
    lattice.deep = -lattice.radius;
    double cellCount = 1.0;
    int extent[3];
    float origin[3];
    for(int axis = 0; axis < 3; ++axis){    //2 empty cells or more on every side, so the band's cells all have their neighbours
        origin[axis] = low[axis] - 2.0f*lattice.cellSize;
        extent[axis] = (int)std::ceil((high[axis] - origin[axis]) / lattice.cellSize) + 3;
        cellCount *= extent[axis];
    }
    if(!(cellCount < 4.0e18)){
        throw std::runtime_error("the particles spread too far to mesh");
    }
    lattice.origin = make_float3(origin[0], origin[1], origin[2]);
    lattice.cells = make_int3(extent[0], extent[1], extent[2]);
    int keyBits = std::max(1, (int)std::ceil(std::log2(cellCount + 1.0)));
    const int threads = 256;
    int n = (int)count;

    // 1: cells
    float* planes = b.planes.hold(6*count);
    check(cudaMemcpy(planes, positionPlanes, sizeof(float)*3*count, cudaMemcpyHostToDevice), "copying positions");
    check(cudaMemcpy(planes + 3*count, velocityPlanes, sizeof(float)*3*count, cudaMemcpyHostToDevice), "copying velocities");
    this->count = count;
    surface = settings;
    for(int axis = 0; axis < 3; ++axis){
        this->low[axis] = low[axis];
        this->high[axis] = high[axis];
    }
    int occupiedCount = b.sortIntoCells(lattice, n, keyBits);
    Key* occupied = b.occupied.data;
    int* counts = b.counts.data;
    int* starts = b.starts.data;
    float4* positions = b.positions.data;
    float4* particleVelocities = b.particleVelocities.data;
    int* numbers = b.numbers.data;

    // 2: the band
    size_t candidateCount = 27*(size_t)occupiedCount;
    if(candidateCount > (size_t)INT_MAX){
        throw std::runtime_error("too many cells to mesh in one piece");
    }
    Key* candidates = b.candidates.hold(candidateCount);
    Key* sortedCandidates = b.sortedCandidates.hold(candidateCount);
    Key* band = b.band.hold(candidateCount);
    markBand<<<(occupiedCount + threads - 1) / threads, threads>>>(occupiedCount, occupied, lattice, candidates);
    int candidateItems = (int)candidateCount;
    b.run([&](void* scratch, size_t& bytes){     //the key's bits only: NO_CELL's are all set, so it still sorts last
        return cub::DeviceRadixSort::SortKeys(scratch, bytes, candidates, sortedCandidates, candidateItems, 0, keyBits);
    });
    b.run([&](void* scratch, size_t& bytes){
        return cub::DeviceSelect::Unique(scratch, bytes, sortedCandidates, band, numbers, candidateItems);
    });
    int bandCount = b.distinct(band);
    if(bandCount == 0){
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
        return;
    }

    // 3 and 4: the field, smoothed; then the band grown and the field worked out again, until nothing beyond the band disagrees with it
    int perCell = lattice.side*lattice.side*lattice.side;
    size_t samples = 0;
    int* neighbours = nullptr;
    float* field = nullptr;
    float4* velocities = nullptr;
    const int MOST_ROUNDS = 16;     //enough for any likely gap; beyond them the mesh is made as it stands
    for(rounds = 1; ; ++rounds){
        samples = (size_t)bandCount*perCell;
        if(samples + 1 > (size_t)INT_MAX || 27*(size_t)bandCount > (size_t)INT_MAX){
            throw std::runtime_error("too large a surface to mesh in one piece");
        }
        neighbours = b.neighbours.hold(27*(size_t)bandCount);
        findNeighbours<<<(int)((27*(size_t)bandCount + threads - 1) / threads), threads>>>(bandCount, band, occupiedCount, occupied, lattice, neighbours);
        field = b.field.hold(samples);
        float* spare = b.smoothed.hold(settings.smoothing > 0 ? samples : 1);
        velocities = b.velocities.hold(samples);
        evaluateField<<<bandCount, perCell>>>(band, occupiedCount, occupied, starts, counts, positions, particleVelocities, lattice, field, velocities);
        for(int pass = 0; pass < settings.smoothing; ++pass){
            smoothField<<<bandCount, perCell>>>(neighbours, lattice, field, spare);
            std::swap(field, spare);
        }
        if(rounds == MOST_ROUNDS){
            break;
        }
        int checked = 27*bandCount;
        candidates = b.candidates.hold(checked);
        sortedCandidates = b.sortedCandidates.hold(checked);
        checkBand<<<bandCount, perCell>>>(band, neighbours, lattice, field, candidates);
        b.run([&](void* scratch, size_t& bytes){
            return cub::DeviceRadixSort::SortKeys(scratch, bytes, candidates, sortedCandidates, checked, 0, keyBits);
        });
        b.run([&](void* scratch, size_t& bytes){
            return cub::DeviceSelect::Unique(scratch, bytes, sortedCandidates, candidates, numbers, checked);
        });
        int added = b.distinct(candidates);
        if(added == 0){
            break;
        }
        Key* merged = b.merged.hold((size_t)bandCount + added);    //the band and the cells it takes in, sorted together into the new band
        check(cudaMemcpy(merged, band, sizeof(Key)*bandCount, cudaMemcpyDeviceToDevice), "growing the band");
        check(cudaMemcpy(merged + bandCount, candidates, sizeof(Key)*added, cudaMemcpyDeviceToDevice), "growing the band");
        bandCount += added;
        band = b.band.hold(bandCount);
        b.run([&](void* scratch, size_t& bytes){
            return cub::DeviceRadixSort::SortKeys(scratch, bytes, merged, band, bandCount, 0, keyBits);
        });
    }
    bandCells = bandCount;

    // 5: surface nets. Each list of flags or counts is summed where it lies, one past its end, so an entry's own is the next sum less its
    int scanned = (int)samples + 1;
    unsigned int* cubePlaces = b.cubePlaces.hold(samples + 1);
    flagCubes<<<bandCount, perCell>>>(neighbours, lattice, field, cubePlaces);
    check(cudaMemset(cubePlaces + samples, 0, sizeof(unsigned int)), "the cubes' flags");
    b.run([&](void* scratch, size_t& bytes){
        return cub::DeviceScan::ExclusiveSum(scratch, bytes, cubePlaces, scanned);
    });
    unsigned int pointCount = 0;
    check(cudaMemcpy(&pointCount, cubePlaces + samples, sizeof(unsigned int), cudaMemcpyDeviceToHost), "counting points");
    float* points = b.points.hold(3*(size_t)pointCount);
    float* pointVelocities = b.pointVelocities.hold(3*(size_t)pointCount);
    writePoints<<<bandCount, perCell>>>(band, neighbours, lattice, field, velocities, cubePlaces, points, pointVelocities);
    unsigned int* quadPlaces = b.quadPlaces.hold(samples + 1);
    writeQuads<<<bandCount, perCell>>>(neighbours, lattice, field, cubePlaces, quadPlaces, nullptr);
    check(cudaMemset(quadPlaces + samples, 0, sizeof(unsigned int)), "the quads' counts");
    b.run([&](void* scratch, size_t& bytes){
        return cub::DeviceScan::ExclusiveSum(scratch, bytes, quadPlaces, scanned);
    });
    unsigned int quadCount = 0;
    check(cudaMemcpy(&quadCount, quadPlaces + samples, sizeof(unsigned int), cudaMemcpyDeviceToHost), "counting quads");
    int* quads = b.quads.hold(4*(size_t)quadCount);
    writeQuads<<<bandCount, perCell>>>(neighbours, lattice, field, cubePlaces, quadPlaces, quads);
    check(cudaGetLastError(), "meshing");
    out.points.resize(3*(size_t)pointCount);
    out.velocities.resize(3*(size_t)pointCount);
    out.quads.resize(4*(size_t)quadCount);
    check(cudaMemcpy(out.points.data(), points, sizeof(float)*out.points.size(), cudaMemcpyDeviceToHost), "copying points");
    check(cudaMemcpy(out.velocities.data(), pointVelocities, sizeof(float)*out.velocities.size(), cudaMemcpyDeviceToHost), "copying velocities");
    check(cudaMemcpy(out.quads.data(), quads, sizeof(int)*out.quads.size(), cudaMemcpyDeviceToHost), "copying quads");
    seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
}

void SurfaceMesher::fields(const FieldSettings& settings, FluidFields& out){
    auto began = std::chrono::steady_clock::now();
    out.surfaceVoxels.clear();
    out.distances.clear();
    out.velocityVoxels.clear();
    out.velocities.clear();
    fieldSeconds = 0.0;
    fieldCells = 0;
    std::string problem = settings.problem();
    if(!problem.empty()){
        throw std::runtime_error(problem);
    }
    out.voxelSize = settings.voxelScale*surface.separation;
    out.halfWidth = settings.halfWidth;
    if(count == 0){
        return;
    }
    Buffers& b = *buffers;
    Lattice lattice;
    lattice.h = out.voxelSize;
    lattice.reach = surface.influenceScale*surface.separation;     //the mesh's field, at these voxels
    lattice.radius = surface.radiusScale*surface.separation;
    lattice.side = std::min(std::max((int)std::ceil(lattice.reach/lattice.h - 1e-4f), 2), MAX_SIDE);
    lattice.cellSize = lattice.side*lattice.h;
    lattice.reach = std::min(lattice.reach, lattice.cellSize);
    lattice.outside = lattice.reach - lattice.radius;
    lattice.deep = -lattice.radius;
    int passes = (int)std::ceil(settings.halfWidth);
    int rings = (int)std::ceil((lattice.reach + passes*lattice.h) / lattice.cellSize);   //cells past the occupied ones: room for the velocity carried on
    int first[3], extent[3];
    float origin[3];
    double cellCount = 1.0;
    for(int axis = 0; axis < 3; ++axis){    //a whole number of voxels from the origin, so each sample is a voxel; and room for the rings and their neighbours
        first[axis] = (int)std::floor(low[axis]/lattice.h) - (rings + 2)*lattice.side;
        origin[axis] = first[axis]*lattice.h;
        extent[axis] = (int)std::ceil((high[axis] - origin[axis]) / lattice.cellSize) + rings + 2;
        cellCount *= extent[axis];
    }
    if(!(cellCount < 4.0e18)){
        throw std::runtime_error("the particles spread too far for the fields");
    }
    lattice.origin = make_float3(origin[0], origin[1], origin[2]);
    lattice.cells = make_int3(extent[0], extent[1], extent[2]);
    int keyBits = std::max(1, (int)std::ceil(std::log2(cellCount + 1.0)));
    const int threads = 256;
    int occupiedCount = b.sortIntoCells(lattice, (int)count, keyBits);

    //the cells within rings of an occupied one, each a cube of voxels
    int width = 2*rings + 1;
    size_t candidateCount = (size_t)occupiedCount*width*width*width;
    if(candidateCount > (size_t)INT_MAX){
        throw std::runtime_error("too large fields to work out in one piece");
    }
    int candidateItems = (int)candidateCount;
    Key* candidates = b.candidates.hold(candidateCount);
    Key* sortedCandidates = b.sortedCandidates.hold(candidateCount);
    Key* cells = b.band.hold(candidateCount);
    dilateCells<<<(occupiedCount + threads - 1) / threads, threads>>>(occupiedCount, b.occupied.data, lattice, rings, candidates);
    b.run([&](void* scratch, size_t& bytes){
        return cub::DeviceRadixSort::SortKeys(scratch, bytes, candidates, sortedCandidates, candidateItems, 0, keyBits);
    });
    b.run([&](void* scratch, size_t& bytes){
        return cub::DeviceSelect::Unique(scratch, bytes, sortedCandidates, cells, b.numbers.data, candidateItems);
    });
    int cellsFound = b.distinct(cells);
    fieldCells = cellsFound;
    int perCell = lattice.side*lattice.side*lattice.side;
    size_t samples = (size_t)cellsFound*perCell;
    if(samples + 1 > (size_t)INT_MAX || 27*(size_t)cellsFound > (size_t)INT_MAX){
        throw std::runtime_error("too large fields to work out in one piece");
    }
    int* neighbours = b.neighbours.hold(27*(size_t)cellsFound);
    findNeighbours<<<(int)((27*(size_t)cellsFound + threads - 1) / threads), threads>>>(cellsFound, cells, occupiedCount, b.occupied.data, lattice, neighbours);

    //the particles' field and velocities, averaged where they reach; the velocities carried on a voxel a pass
    float* field = b.field.hold(samples);
    float4* velocities = b.velocities.hold(samples);
    float4* spare = b.spareVelocities.hold(samples);
    evaluateField<<<cellsFound, perCell>>>(cells, occupiedCount, b.occupied.data, b.starts.data, b.counts.data, b.positions.data, b.particleVelocities.data,
                                           lattice, field, velocities);
    for(int pass = 0; pass < passes; ++pass){
        extendVelocity<<<cellsFound, perCell>>>(neighbours, lattice, velocities, spare);
        std::swap(velocities, spare);
    }

    //the distances to the field's zero level: beside it, then out from it a voxel a pass, twice over for the diagonals
    float* fixed = b.fixedDistances.hold(samples);
    float* distances = b.distances.hold(samples);
    float* spareDistances = b.spareDistances.hold(samples);
    startDistances<<<cellsFound, perCell>>>(neighbours, lattice, field, fixed);
    check(cudaMemcpy(distances, fixed, sizeof(float)*samples, cudaMemcpyDeviceToDevice), "the distances");
    for(int pass = 0; pass < 2*passes + 2; ++pass){
        relaxDistances<<<cellsFound, perCell>>>(neighbours, lattice, fixed, distances, spareDistances);
        std::swap(distances, spareDistances);
    }

    //the voxels in the narrow band, and those with a velocity, listed
    int scanned = (int)samples + 1;
    unsigned int* places = b.cubePlaces.hold(samples + 1);
    auto list = [&](auto flag){     //flags each sample, and sums the flags where they lie: returns how many are flagged
        flag();
        check(cudaMemset(places + samples, 0, sizeof(unsigned int)), "the flags");
        b.run([&](void* scratch, size_t& bytes){
            return cub::DeviceScan::ExclusiveSum(scratch, bytes, places, scanned);
        });
        unsigned int listed = 0;
        check(cudaMemcpy(&listed, places + samples, sizeof(unsigned int), cudaMemcpyDeviceToHost), "counting voxels");
        return (size_t)listed;
    };
    int3 corner = make_int3(first[0], first[1], first[2]);
    size_t banded = list([&]{ flagDistances<<<cellsFound, perCell>>>(distances, settings.halfWidth*lattice.h, places); });
    int* voxels = b.quads.hold(3*banded);
    float* values = b.points.hold(banded);
    listDistances<<<cellsFound, perCell>>>(cells, lattice, corner, distances, places, voxels, values);
    out.surfaceVoxels.resize(3*banded);
    out.distances.resize(banded);
    check(cudaMemcpy(out.surfaceVoxels.data(), voxels, sizeof(int)*out.surfaceVoxels.size(), cudaMemcpyDeviceToHost), "copying voxels");
    check(cudaMemcpy(out.distances.data(), values, sizeof(float)*out.distances.size(), cudaMemcpyDeviceToHost), "copying distances");
    size_t moving = list([&]{ flagVelocities<<<cellsFound, perCell>>>(velocities, places); });
    voxels = b.quads.hold(3*moving);
    values = b.points.hold(3*moving);
    listVelocities<<<cellsFound, perCell>>>(cells, lattice, corner, velocities, places, voxels, values);
    check(cudaGetLastError(), "the fields");
    out.velocityVoxels.resize(3*moving);
    out.velocities.resize(3*moving);
    check(cudaMemcpy(out.velocityVoxels.data(), voxels, sizeof(int)*out.velocityVoxels.size(), cudaMemcpyDeviceToHost), "copying voxels");
    check(cudaMemcpy(out.velocities.data(), values, sizeof(float)*out.velocities.size(), cudaMemcpyDeviceToHost), "copying velocities");
    fieldSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
}
