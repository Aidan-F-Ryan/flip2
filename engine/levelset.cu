//Copyright 2023 Aberrant Behavior LLC

//The liquid's level set on the simulation's own voxels, and surface tension from it.
//
//The pressure solve needs no surface as it comes: its fluid is every voxel a particle's stencil reaches, a voxel or two past the particles. Viscosity and
//surface tension act at the liquid's real surface, and the sharp free surface (freesurface.cu) solves for the liquid inside it, so when any of them is on,
//each substep finds it: the signed distance to the surface at every stored voxel's centre, in voxels, negative inside (Particles::liquidLevel), in three
//steps.
//
//1. The particles' field, after Zhu and Bridson: at a point, how far it is from the mean of the particles within reach of it, weighted by how near
//   each is. Deep in the liquid the mean is the point itself; near the surface it leans inwards, by an amount that says how far off the surface is
//   (surfaceDistance). Each particle adds its weight and weighted position to the voxel centres within reach, a block per node holding particles,
//   summed in fixed point in shared memory and then into the voxels' owners, as P2G sums momentum: integers add up the same in any order, so the field
//   doesn't depend on how the nodes are stored or split. The reach is 1.5 voxels, which every voxel a particle's own stencil reaches covers, and no
//   more. Where a wall or an obstacle cuts into a voxel's reach, the mean would lean away from it and read as a surface along every wetted wall: so
//   the particles beside a wall also count as their mirror images in it (splatParticles), and where there's liquid about, the part of the reach
//   inside an obstacle counts as full of liquid at rest (finishLevelSet).
//2. That's only a distance within a voxel or so of the surface: further in, the particles' own unevenness swamps the lean, and further out there's too
//   little in reach. So the voxels either side of its zero take their distance from where it crosses between them, and three passes carry distances on
//   from those, each solving |grad d| = 1 upwind, as the mesher does for its render level set (surface.cu). A block per node, on a tile of the node's
//   voxels and 4 more on every side (liquidTile.hu): the passes run in shared memory, and each node's own voxels come out as they would from passes
//   over the whole grid.
//3. For surface tension, the surface's curvature at the voxels beside it: the level set smoothed seven times, then div(grad d / |grad d|) by central
//   differences, on the same kind of tile. A particle's place is noise a tenth of a voxel deep in the surface, and curvature is a second derivative:
//   unsmoothed, the noise's curvature is several times a drop's. Smoothing takes it down to a few percent of one 16 voxels across.
//
//Past the domain's walls the level set mirrors what's inside, so a wall isn't a surface, tilted to meet the wall at the contact angle the liquid is
//given (pastWall): 60 degrees by default, which water on most things is near. Obstacles' first layer of voxels reads as liquid where liquid touches them, from step 1, but nothing is
//stored deeper in, which reads as outside: curvature within a couple of voxels of an obstacle is off by about what a drop 20 voxels across has, the
//way a wall that doesn't wet would make it, and the contact angle doesn't reach them.
//
//Surface tension is the pressure jump sigma*kappa across the surface, as a force on the faces it crosses (Brackbill, Kothe and Zemach's continuum
//surface force, in its sharp form): each face between a liquid voxel and one that isn't gets dt*(sigma/rho)*kappa/dx towards the liquid, kappa where
//the surface crosses between them. That's a discrete gradient of sigma*kappa*H over the same faces the pressure update covers, H being 1 in the
//liquid's unknowns and 0 everywhere the pressure is 0, so wherever kappa is constant a pressure of sigma*kappa cancels it exactly: a still drop stays
//still, and only the curvature's changes along the surface move anything. Nothing about the pressure equations changes. It's explicit, so the timestep
//can't pass the time the shortest capillary wave takes to cross a voxel (Particles::capillaryDt), which only applies with surface tension on. The sharp
//free surface takes the same curvature as the pressure at its surface instead, inside the solve (freesurface.cu), and none of this force.

#include "particles.hu"
#include "liquidTile.hu"
#include "algorithms/voxelSolveFunctions.hu"   //NO_VOXEL, WALL_VOXEL
#include <cmath>
#include <iostream>

static constexpr uint SPLAT_THREADS = 256;          //per node block: about 2 particles each in a full node, as P2G
static constexpr float SPLAT_REACH = 1.5f;          //voxels: how far from a particle the voxel centres it weighs on are
static constexpr float SPLAT_SCALE = 262144.0f;     //fixed point: units per unit weight. A voxel deep in liquid sums about 2 per particle in a voxel, so
                                                    //there's room for hundreds of times the rest density before 32 bits overflow
static constexpr int SPLAT_LEAST = 64;              //a weight sum under this many units is too coarse to take a mean from: the voxel reads as outside
static constexpr float SOLID_GATE = 0.25f;          //solids count as liquid in full once the particles' weight is this share of a voxel's deep in liquid
static constexpr int LEVEL_HALO = 4;
static constexpr int LEVEL_WIDTH = 12;              //a tile's side: a node's 4 voxels and LEVEL_HALO either side
static constexpr int LEVEL_SLOTS = LEVEL_WIDTH*LEVEL_WIDTH*LEVEL_WIDTH;
static constexpr uint LEVEL_THREADS = 256;
static constexpr float UNKNOWN_DISTANCE = 1e20f;    //a distance not yet known
static constexpr float KEPT_WITHIN = 1.0f;          //and within which smoothing takes the level set as it is: every voxel that near is stored (the nearest
                                                    //unstored one has no particle in the voxels around it, so it's 1.25 or more outside)

// ---- 1: the particles' field ----

//a point's weight and weighted offset on the voxel centres within reach of it, into a block's sums: p in the block, in voxels. Only the centres inside
//the domain: nothing reads the others
__device__ inline void splatPoint(const float p[3], int* sums, int voxels1D, int3 origin, int3 domainVoxels){
    int voxels3D = voxels1D*voxels1D*voxels1D;
    int base[3] = {(int)floorf(p[0] - 1.0f), (int)floorf(p[1] - 1.0f), (int)floorf(p[2] - 1.0f)};   //the first of the 3 centres along each axis that can be in reach
    #pragma unroll
    for(int k = 0; k < 3; ++k){
        #pragma unroll
        for(int j = 0; j < 3; ++j){
            #pragma unroll
            for(int i = 0; i < 3; ++i){
                int x = base[0] + i, y = base[1] + j, z = base[2] + k;
                float dx = p[0] - (x + 0.5f), dy = p[1] - (y + 0.5f), dz = p[2] - (z + 0.5f);     //the point from the centre
                float s = 1.0f - (dx*dx + dy*dy + dz*dz)/(SPLAT_REACH*SPLAT_REACH);
                if(s > 0.0f && x >= 0 && y >= 0 && z >= 0 && x < voxels1D && y < voxels1D && z < voxels1D
                   && origin.x + x >= 0 && origin.y + y >= 0 && origin.z + z >= 0 && origin.x + x < domainVoxels.x && origin.y + y < domainVoxels.y && origin.z + z < domainVoxels.z){
                    float weight = s*s*s*SPLAT_SCALE;
                    int slot = x + y*voxels1D + z*voxels1D*voxels1D;
                    atomicAdd(sums + slot, __float2int_rn(weight));
                    atomicAdd(sums + voxels3D + slot, __float2int_rn(weight*dx));
                    atomicAdd(sums + 2*voxels3D + slot, __float2int_rn(weight*dy));
                    atomicAdd(sums + 3*voxels3D + slot, __float2int_rn(weight*dz));
                }
            }
        }
    }
}

//each particle's weight and weighted offset, on the voxel centres within reach of it; then the block's sums into the voxels' owners. A particle within
//a voxel of one of the domain's walls is also there as its mirror image in the wall (and in two or three walls at once, by an edge or a corner): beside
//a wall a centre's reach is half inside it, and with the liquid on one side only the mean would lean away from the wall, as it does at a surface. With
//the images it reads as the liquid carrying on through the wall as it is on this side
//With two PHASES (TwoPhase, particles.hu), only the liquid's particles on the grid make its surface: those whose ids aren't marked AIR_PARTICLE or ESCAPED_PARTICLE. A template, so that
//without them the kernel is the one it always was, to the bit
template<bool PHASES>
__global__ void splatParticles(uint numParticleNodes, uint numParticles, const uint* firstParticles, const double* px, const double* py, const double* pz, VoxelPlaces places,
                               int3 domainVoxels, const uint* voxelOwners, int* weights, int* offsetsX, int* offsetsY, int* offsetsZ,
                               const uint* ids){
    extern __shared__ int sums[];   //per slot: the weights, then the weighted offsets along x, y and z, in voxels
    int voxels1D = places.interiorWidth + 2*places.apronCells;
    int voxels3D = voxels1D*voxels1D*voxels1D;
    uint firstParticle = firstParticles[blockIdx.x];
    uint lastParticle = blockIdx.x == numParticleNodes - 1 ? numParticles : firstParticles[blockIdx.x + 1];
    for(int i = threadIdx.x; i < 4*voxels3D; i += blockDim.x){
        sums[i] = 0;
    }
    __syncthreads();
    int3 origin = places.voxelOf(places.nodeCells[blockIdx.x], 0);     //the block's first slot, in the domain's voxels
    int start[3] = {origin.x, origin.y, origin.z};
    int size[3] = {domainVoxels.x, domainVoxels.y, domainVoxels.z};
    double perVoxel = 1.0 / (double)places.voxelSize();
    for(uint index = firstParticle + threadIdx.x; index < lastParticle; index += blockDim.x){    //consecutive threads, consecutive particles
        if constexpr(PHASES){
            if(ids[index] & (AIR_PARTICLE | ESCAPED_PARTICLE)){     //nor its droplets, which aren't the body of the liquid
                continue;
            }
        }
        float p[3] = {(float)((px[index] - places.grid.negX)*perVoxel - origin.x), (float)((py[index] - places.grid.negY)*perVoxel - origin.y),
                      (float)((pz[index] - places.grid.negZ)*perVoxel - origin.z)};     //in the block, in voxels
        float image[3];     //its mirror image along each axis, where a wall is within a voxel of it: past that, no centre inside is in the image's reach
        int walls = 0;
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            float fromLow = p[axis] + start[axis], fromHigh = size[axis] - fromLow;
            image[axis] = fromLow < 1.0f ? p[axis] - 2.0f*fromLow : p[axis] + 2.0f*fromHigh;
            walls |= fromLow < 1.0f || fromHigh < 1.0f ? 1 << axis : 0;
        }
        for(int mirrored = 0; mirrored < 8; ++mirrored){    //itself, then every combination of the walls beside it
            if(mirrored & ~walls){
                continue;
            }
            float at[3] = {mirrored & 1 ? image[0] : p[0], mirrored & 2 ? image[1] : p[1], mirrored & 4 ? image[2] : p[2]};
            splatPoint(at, sums, voxels1D, origin, domainVoxels);
        }
    }
    __syncthreads();
    int* totals[4] = {weights, offsetsX, offsetsY, offsetsZ};
    for(uint i = (blockIdx.x == 0 ? 0 : places.nodeVoxelEnds[blockIdx.x - 1]) + threadIdx.x; i < places.nodeVoxelEnds[blockIdx.x]; i += blockDim.x){
        int slot = places.voxelSlots[i];
        uint owner = voxelOwners[i];
        #pragma unroll
        for(int sum = 0; sum < 4; ++sum){
            int value = sums[sum*voxels3D + slot];
            if(value != 0){
                atomicAdd(totals[sum] + owner, value);
            }
        }
    }
}

//The weight (1 - d^2/R^2)^3 over the part of a ball of radius R beyond a plane t radii from its centre (negative t: the plane is behind the centre, and
//the part is more than half), as a share of the whole ball's; and that part's moment along the plane's normal, as a share of the whole ball's weight
//times R. Both in closed form: the weight over the disc at height z is pi R^2 (1 - z^2/R^2)^4 / 4
__device__ inline float capShare(float t){
    float t2 = t*t;
    float below = t*(1.0f + t2*(-4.0f/3.0f + t2*(6.0f/5.0f + t2*(-4.0f/7.0f + t2/9.0f))));  //the integral of (1 - z^2)^4 from 0 to t
    return 0.5f - below*(315.0f/256.0f);    //and from 0 to 1 it's 128/315
}

__device__ inline float capMoment(float t){
    float s = 1.0f - t*t;
    return s*s*s*s*s*(315.0f/2560.0f);
}

//How far outside the liquid a point is (negative: inside), in voxels, from how far it is from the weighted mean of what's in reach of it, if that's
//liquid at rest filling everything past a flat surface: the mean is then the centre of the part of the reach past the surface, capMoment/capShare
//of a reach away, which grows with the distance, so bisection finds it. This is what makes the field a distance near the surface: Zhu and Bridson's
//own, the mean's distance less a radius, flattens out inside the liquid and steepens outside it, and interpolating that for where it crosses zero
//terraces any surface that lies along the voxels
__device__ inline float surfaceDistance(float toMean){
    float low = -1.0f, high = 1.0f;     //in reaches
    for(int halving = 0; halving < 16; ++halving){
        float t = 0.5f*(low + high);
        float share = capShare(t);
        if(share > 1e-6f && SPLAT_REACH*capMoment(t) < toMean*share){
            low = t;
        }
        else{
            high = t;
        }
    }
    return 0.5f*(low + high)*SPLAT_REACH;
}

//the sums as the field, at every stored voxel, a block per node storing voxels: the distance from the voxel's centre to the weighted mean of what's
//within reach of it, as the distance from the surface that would put the mean there (surfaceDistance). The walls' share of a reach is in the sums
//already, as the particles' images. An obstacle's isn't: where the particles weigh enough for there to be liquid about (gate), the part of the reach
//inside the nearest obstacle counts as liquid at rest, a half space as far off as its surface, facing the way its normal does
__global__ void finishLevelSet(VoxelPlaces places, Obstacles obstacles, const char* walls, const int* weights, const int* offsetsX, const int* offsetsY, const int* offsetsZ,
                               float bulk, float outside, float* raw){
    uint node = blockIdx.x;
    uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
    uint last = places.nodeVoxelEnds[node];
    uint cell = places.nodeCells[node];
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        float value = outside;
        if(!walls[index] && weights[index] >= SPLAT_LEAST){
            float weight = weights[index] / SPLAT_SCALE;
            float offset[3] = {offsetsX[index] / SPLAT_SCALE, offsetsY[index] / SPLAT_SCALE, offsetsZ[index] / SPLAT_SCALE};
            if(obstacles.count > 0){
                float gate = fminf(weight / (SOLID_GATE*bulk), 1.0f)*bulk;
                int3 voxel = places.voxelOf(cell, places.voxelSlots[index]);
                float distance;
                float3 normal;
                if(nearestObstacle(obstacles, places.point(voxel, 0.5f, 0.5f, 0.5f), distance, normal) >= 0){
                    float t = fmaxf(distance / (SPLAT_REACH*places.voxelSize()), -1.0f);
                    if(t < 1.0f){
                        float moment = gate*SPLAT_REACH*capMoment(t);
                        weight += gate*capShare(t);
                        offset[0] -= moment*normal.x;   //the normal points out of the obstacle
                        offset[1] -= moment*normal.y;
                        offset[2] -= moment*normal.z;
                    }
                }
            }
            value = surfaceDistance(sqrtf(offset[0]*offset[0] + offset[1]*offset[1] + offset[2]*offset[2]) / weight);
        }
        raw[index] = value;
    }
}

// ---- 2: distances ----

//What a slot past a wall holds, from its mirror image inside: the image's value, less 2 wetting for every voxel the slot is past the wall, wetting
//being the cosine of the angle the liquid's surface should meet the wall at. The level set's slope across the wall is then that cosine, so its zero
//meets the wall at that angle: square on with none, as a plain mirror image has it, reaching out along the wall with more, and pulling back from it
//with less than none. The surface tension that follows the level set's curvature does the rest: nothing else knows the angle
__device__ inline float pastWall(float image, float depth, float wetting){
    return image - 2.0f*depth*wetting;
}

//a field into a tile: every slot inside the domain takes its stored voxel's value, or missing where nothing's stored, and the slots past the walls
//mirror those (pastWall), unless there's nothing there to mirror. Every thread of the block calls it, after loadTileNodes
__device__ inline void loadLevelTile(const Tile& tile, const uint* nodes, uint numUsedGridNodes, const uint* interiorVoxels, const float* values, float missing, float wetting,
                                     float* field){
    for(int slot = threadIdx.x; slot < LEVEL_SLOTS; slot += blockDim.x){
        int3 t = tile.at(slot);
        if(tile.inDomain(t)){
            uint voxel = tile.voxel(t, nodes, numUsedGridNodes, interiorVoxels);
            field[slot] = voxel != NO_VOXEL ? values[voxel] : missing;
        }
    }
    __syncthreads();
    for(int slot = threadIdx.x; slot < LEVEL_SLOTS; slot += blockDim.x){
        int3 t = tile.at(slot);
        if(!tile.inDomain(t)){
            float image = field[tile.slot(tile.mirrored(t))];
            field[slot] = image < missing ? pastWall(image, tile.pastWalls(t), wetting) : missing;
        }
    }
    __syncthreads();
}

//the level set at a node's own voxels from the particles' field around them: startDistances and three passes of relaxDistances (surface.cu), over a
//tile in shared memory. A pass is only right a slot further in from the tile's edge than the one before, where every neighbour it read was right;
//4 passes leave the node's own voxels right. A tile all on one side of the surface is as far from it as the level set goes
__global__ void redistanceLevelSet(uint numUsedGridNodes, const uint* nodeCells, const uint* cellToNode, const uint* interiorVoxels, Grid grid, float outside, float wetting,
                                   const float* raw, float* level){
    __shared__ float fields[2][LEVEL_SLOTS];
    __shared__ char beside[LEVEL_SLOTS];    //whether the surface passes between the slot and one next to it
    __shared__ uint nodes[27];
    uint cell = nodeCells[blockIdx.x];
    loadTileNodes(nodes, cell, numUsedGridNodes, nodeCells, cellToNode, grid);
    __syncthreads();
    Tile tile(cell, grid, LEVEL_WIDTH - 2*LEVEL_HALO, LEVEL_HALO);
    float* from = fields[0];
    float* to = fields[1];
    loadLevelTile(tile, nodes, numUsedGridNodes, interiorVoxels, raw, outside, wetting, from);
    bool inside = from[0] < 0.0f;
    bool crossed = false;
    for(int slot = threadIdx.x; slot < LEVEL_SLOTS; slot += blockDim.x){
        crossed = crossed || (from[slot] < 0.0f) != inside;
    }
    crossed = __syncthreads_or(crossed);
    const int strides[3] = {1, LEVEL_WIDTH, LEVEL_WIDTH*LEVEL_WIDTH};
    if(crossed){
        //beside the surface: the distance to the plane through where the field crosses 0 on the way to each neighbour on its other side
        for(int slot = threadIdx.x; slot < LEVEL_SLOTS; slot += blockDim.x){
            int3 t = tile.at(slot);
            int coordinates[3] = {t.x, t.y, t.z};
            float here = from[slot];
            bool within = here < 0.0f;
            float inverse = 0.0f;       //the sum over the axes the surface crosses of 1/d^2, d the distance to it along each
            #pragma unroll
            for(int axis = 0; axis < 3; ++axis){
                float nearest = UNKNOWN_DISTANCE;
                for(int step = -1; step <= 1; step += 2){
                    int moved = coordinates[axis] + step;
                    if(moved < 0 || moved >= LEVEL_WIDTH){
                        continue;
                    }
                    float there = from[slot + step*strides[axis]];
                    if((there < 0.0f) != within){
                        nearest = fminf(nearest, here / (here - there));
                    }
                }
                if(nearest < UNKNOWN_DISTANCE){
                    nearest = fmaxf(nearest, 1e-4f);
                    inverse += 1.0f / (nearest*nearest);
                }
            }
            float distance = inverse > 0.0f ? rsqrtf(inverse) : UNKNOWN_DISTANCE;
            to[slot] = within ? -distance : distance;
            beside[slot] = inverse > 0.0f;
        }
        __syncthreads();
        for(int pass = 0; pass < 3; ++pass){    //out from the surface: the upwind solution of |grad d| = 1 from the nearest neighbours on its own side
            float* swap = from;
            from = to;
            to = swap;
            for(int slot = threadIdx.x; slot < LEVEL_SLOTS; slot += blockDim.x){
                float own = from[slot];
                if(beside[slot]){
                    to[slot] = own;
                    continue;
                }
                int3 t = tile.at(slot);
                int coordinates[3] = {t.x, t.y, t.z};
                bool within = own < 0.0f;
                float nearest[3];
                #pragma unroll
                for(int axis = 0; axis < 3; ++axis){
                    nearest[axis] = UNKNOWN_DISTANCE;
                    for(int step = -1; step <= 1; step += 2){
                        int moved = coordinates[axis] + step;
                        if(moved < 0 || moved >= LEVEL_WIDTH){
                            continue;
                        }
                        float there = from[slot + step*strides[axis]];
                        if((there < 0.0f) == within){
                            nearest[axis] = fminf(nearest[axis], fabsf(there));
                        }
                    }
                }
                float a = fminf(fminf(nearest[0], nearest[1]), nearest[2]);     //sorted: a, b, c
                float b = fmaxf(fminf(nearest[0], nearest[1]), fminf(fmaxf(nearest[0], nearest[1]), nearest[2]));
                float c = fmaxf(fmaxf(nearest[0], nearest[1]), nearest[2]);
                float d = a + 1.0f;
                if(b < UNKNOWN_DISTANCE && d > b){
                    d = 0.5f*(a + b + sqrtf(fmaxf(2.0f - (a - b)*(a - b), 0.0f)));
                    if(c < UNKNOWN_DISTANCE && d > c){
                        float sum = a + b + c;
                        d = (sum + sqrtf(fmaxf(sum*sum - 3.0f*(a*a + b*b + c*c - 1.0f), 0.0f))) / 3.0f;
                    }
                }
                float distance = fminf(fabsf(own), d);
                to[slot] = within ? -distance : distance;
            }
            __syncthreads();
        }
    }
    for(int slot = threadIdx.x; slot < LEVEL_SLOTS; slot += blockDim.x){
        int3 t = tile.at(slot);
        if(!tile.interior(t)){
            continue;
        }
        uint voxel = tile.voxel(t, nodes, numUsedGridNodes, interiorVoxels);
        if(voxel != NO_VOXEL){
            level[voxel] = crossed ? fminf(fmaxf(to[slot], -LIQUID_BAND), LIQUID_BAND) : inside ? -LIQUID_BAND : LIQUID_BAND;
        }
    }
}

// ---- 3: curvature ----

//A level set into a tile for smoothing, which reads well past the stored voxels: the first unstored ones are only a voxel or two outside the surface,
//and smoothing across a jump from their neighbours' distances to the level set's limit would bend every surface near one. And more voxels are stored
//above the liquid than below it (the ones holding its upper faces), so taking what's stored and making up the rest would curve a drop's lower side
//more than its upper, which pushes it along: a 2% difference had one drifting. So only the values within KEPT_WITHIN of the surface are taken, which
//are stored on every side of it, and every other slot, stored or not, carries the distances on from those: three passes of the upwind solve, as
//redistanceLevelSet's, and past those the level set's limit. The slots past the walls mirror what's inside (pastWall). Every thread of the block calls
//it, after loadTileNodes; the tile ends up in from, with to as its other buffer, and stored says where a voxel is
__device__ inline void loadSmoothingTile(const Tile& tile, const uint* nodes, uint numUsedGridNodes, const uint* interiorVoxels, const float* values, float wetting, float*& from,
                                         float*& to, char* missing, char* stored){
    for(int slot = threadIdx.x; slot < LEVEL_SLOTS; slot += blockDim.x){
        int3 t = tile.at(slot);
        if(tile.inDomain(t)){
            uint voxel = tile.voxel(t, nodes, numUsedGridNodes, interiorVoxels);
            float value = voxel != NO_VOXEL ? values[voxel] : UNKNOWN_DISTANCE;
            bool kept = fabsf(value) < KEPT_WITHIN;
            from[slot] = kept ? value : value < 0.0f ? -UNKNOWN_DISTANCE : UNKNOWN_DISTANCE;
            missing[slot] = !kept;
            stored[slot] = voxel != NO_VOXEL;
        }
        else{
            stored[slot] = 0;
        }
    }
    __syncthreads();
    for(int slot = threadIdx.x; slot < LEVEL_SLOTS; slot += blockDim.x){
        int3 t = tile.at(slot);
        if(!tile.inDomain(t)){
            int image = tile.slot(tile.mirrored(t));
            from[slot] = from[image];
            missing[slot] = missing[image];
        }
    }
    __syncthreads();
    const int strides[3] = {1, LEVEL_WIDTH, LEVEL_WIDTH*LEVEL_WIDTH};
    for(int pass = 0; pass < 3; ++pass){
        for(int slot = threadIdx.x; slot < LEVEL_SLOTS; slot += blockDim.x){
            float own = from[slot];
            if(missing[slot]){
                int3 t = tile.at(slot);
                int coordinates[3] = {t.x, t.y, t.z};
                bool within = own < 0.0f;
                float nearest[3];
                #pragma unroll
                for(int axis = 0; axis < 3; ++axis){
                    nearest[axis] = UNKNOWN_DISTANCE;
                    for(int step = -1; step <= 1; step += 2){
                        int moved = coordinates[axis] + step;
                        if(moved < 0 || moved >= LEVEL_WIDTH){
                            continue;
                        }
                        float there = from[slot + step*strides[axis]];
                        if((there < 0.0f) == within){
                            nearest[axis] = fminf(nearest[axis], fabsf(there));
                        }
                    }
                }
                float a = fminf(fminf(nearest[0], nearest[1]), nearest[2]);     //sorted: a, b, c
                float b = fmaxf(fminf(nearest[0], nearest[1]), fminf(fmaxf(nearest[0], nearest[1]), nearest[2]));
                float c = fmaxf(fmaxf(nearest[0], nearest[1]), nearest[2]);
                float d = a + 1.0f;
                if(b < UNKNOWN_DISTANCE && d > b){
                    d = 0.5f*(a + b + sqrtf(fmaxf(2.0f - (a - b)*(a - b), 0.0f)));
                    if(c < UNKNOWN_DISTANCE && d > c){
                        float sum = a + b + c;
                        d = (sum + sqrtf(fmaxf(sum*sum - 3.0f*(a*a + b*b + c*c - 1.0f), 0.0f))) / 3.0f;
                    }
                }
                float distance = fminf(fabsf(own), d);
                if(pass == 2){
                    distance = fminf(distance, LIQUID_BAND);
                }
                own = within ? -distance : distance;
            }
            to[slot] = own;
        }
        __syncthreads();
        float* swap = from;
        from = to;
        to = swap;
    }
    if(wetting != 0.0f){    //the slots past the walls, once what they mirror is a distance at every depth: tilted to meet the wall at the contact angle
        for(int slot = threadIdx.x; slot < LEVEL_SLOTS; slot += blockDim.x){
            int3 t = tile.at(slot);
            if(!tile.inDomain(t)){
                from[slot] = pastWall(from[tile.slot(tile.mirrored(t))], tile.pastWalls(t), wetting);
            }
        }
        __syncthreads();
    }
}

//passes of an even blur over each slot and the 26 around it, in a tile's two buffers; the result ends up in from. Even weights take out twice the
//noise a 1 2 1 blur does for the same reach, which is what a tile is short of
__device__ inline void smoothTile(int passes, float*& from, float*& to){
    for(int pass = 0; pass < passes; ++pass){
        for(int slot = threadIdx.x; slot < LEVEL_SLOTS; slot += blockDim.x){
            int x = slot % LEVEL_WIDTH, y = slot / LEVEL_WIDTH % LEVEL_WIDTH, z = slot / (LEVEL_WIDTH*LEVEL_WIDTH);
            float sum = 0.0f;
            #pragma unroll
            for(int dz = -1; dz <= 1; ++dz){
                int rowZ = min(max(z + dz, 0), LEVEL_WIDTH - 1);
                #pragma unroll
                for(int dy = -1; dy <= 1; ++dy){
                    const float* row = from + LEVEL_WIDTH*(min(max(y + dy, 0), LEVEL_WIDTH - 1) + LEVEL_WIDTH*rowZ);
                    sum += row[max(x - 1, 0)] + row[x] + row[min(x + 1, LEVEL_WIDTH - 1)];
                }
            }
            to[slot] = sum / 27.0f;
        }
        __syncthreads();
        float* swap = from;
        from = to;
        to = swap;
    }
}

//The level set smoothed for the curvature, in two steps, as a tile has room for 4 passes that reach a slot each and the curvature's differences take
//one: this one's 4 passes of blur leave the node's own voxels as passes over the whole grid would, and curveLevelSet's 3 more go on from that. Seven
//passes take a particle-sized bump's curvature down to about 4% of a drop's 16 voxels across (3 leave 12%), and cost a drop that size about 1% of its
//frequency. Only tiles holding a voxel near enough the surface for its curvature to be asked do anything: no other is read. The sharp free surface
//takes fewer passes of its own (freesurface.cu)
__global__ void smoothLevelSet(uint numUsedGridNodes, const uint* nodeCells, const uint* cellToNode, const uint* interiorVoxels, Grid grid, float wetting, int passes,
                               const float* level, float* smoothed){
    __shared__ float fields[2][LEVEL_SLOTS];
    __shared__ char flags[2][LEVEL_SLOTS];
    __shared__ uint nodes[27];
    uint cell = nodeCells[blockIdx.x];
    loadTileNodes(nodes, cell, numUsedGridNodes, nodeCells, cellToNode, grid);
    __syncthreads();
    Tile tile(cell, grid, LEVEL_WIDTH - 2*LEVEL_HALO, LEVEL_HALO);
    float* from = fields[0];
    float* to = fields[1];
    loadSmoothingTile(tile, nodes, numUsedGridNodes, interiorVoxels, level, wetting, from, to, flags[0], flags[1]);
    bool near = false;
    for(int slot = threadIdx.x; slot < LEVEL_SLOTS; slot += blockDim.x){
        near = near || fabsf(from[slot]) < CURVED_WITHIN;
    }
    if(__syncthreads_or(near)){
        smoothTile(passes, from, to);
    }
    for(int slot = threadIdx.x; slot < LEVEL_SLOTS; slot += blockDim.x){
        int3 t = tile.at(slot);
        if(tile.interior(t) && flags[1][slot]){
            smoothed[tile.voxel(t, nodes, numUsedGridNodes, interiorVoxels)] = from[slot];
        }
    }
}

//the surface's curvature at a node's own voxels within CURVED_WITHIN of it, in 1/voxels, positive where the liquid bulges; 0 at the rest. The last 3
//passes of blur, then div(grad d / |grad d|) by central differences. Features a voxel across can't have more curvature than 1 a voxel
__global__ void curveLevelSet(uint numUsedGridNodes, const uint* nodeCells, const uint* cellToNode, const uint* interiorVoxels, Grid grid, float wetting, const float* level,
                              const float* smoothed, float* curvature){
    __shared__ float fields[2][LEVEL_SLOTS];
    __shared__ char flags[2][LEVEL_SLOTS];
    __shared__ uint nodes[27];
    uint cell = nodeCells[blockIdx.x];
    loadTileNodes(nodes, cell, numUsedGridNodes, nodeCells, cellToNode, grid);
    __syncthreads();
    Tile tile(cell, grid, LEVEL_WIDTH - 2*LEVEL_HALO, LEVEL_HALO);
    bool near = false;
    for(int slot = threadIdx.x; slot < LEVEL_SLOTS; slot += blockDim.x){
        int3 t = tile.at(slot);
        if(tile.interior(t)){
            uint voxel = tile.voxel(t, nodes, numUsedGridNodes, interiorVoxels);
            near = near || (voxel != NO_VOXEL && fabsf(level[voxel]) < CURVED_WITHIN);
        }
    }
    if(!__syncthreads_or(near)){
        return;     //its voxels' curvatures stay 0
    }
    float* from = fields[0];
    float* to = fields[1];
    loadSmoothingTile(tile, nodes, numUsedGridNodes, interiorVoxels, smoothed, wetting, from, to, flags[0], flags[1]);
    smoothTile(3, from, to);
    for(int slot = threadIdx.x; slot < LEVEL_SLOTS; slot += blockDim.x){
        int3 t = tile.at(slot);
        if(!tile.interior(t) || !flags[1][slot]){
            continue;
        }
        uint voxel = tile.voxel(t, nodes, numUsedGridNodes, interiorVoxels);
        if(!(fabsf(level[voxel]) < CURVED_WITHIN)){
            continue;
        }
        auto f = [&](int dx, int dy, int dz){
            return from[slot + dx + LEVEL_WIDTH*(dy + LEVEL_WIDTH*dz)];
        };
        float centre = f(0, 0, 0);
        float fx = 0.5f*(f(1, 0, 0) - f(-1, 0, 0)), fy = 0.5f*(f(0, 1, 0) - f(0, -1, 0)), fz = 0.5f*(f(0, 0, 1) - f(0, 0, -1));
        float fxx = f(1, 0, 0) - 2.0f*centre + f(-1, 0, 0), fyy = f(0, 1, 0) - 2.0f*centre + f(0, -1, 0), fzz = f(0, 0, 1) - 2.0f*centre + f(0, 0, -1);
        float fxy = 0.25f*(f(1, 1, 0) - f(1, -1, 0) - f(-1, 1, 0) + f(-1, -1, 0));
        float fxz = 0.25f*(f(1, 0, 1) - f(1, 0, -1) - f(-1, 0, 1) + f(-1, 0, -1));
        float fyz = 0.25f*(f(0, 1, 1) - f(0, 1, -1) - f(0, -1, 1) + f(0, -1, -1));
        float squared = fx*fx + fy*fy + fz*fz;
        if(squared > 1e-6f){    //a gradient this small has no direction to speak of: the middle of a sheet a voxel thick
            float value = (fxx*(fy*fy + fz*fz) + fyy*(fx*fx + fz*fz) + fzz*(fx*fx + fy*fy) - 2.0f*(fx*fy*fxy + fx*fz*fxz + fy*fz*fyz)) / (squared*sqrtf(squared));
            curvature[voxel] = fminf(fmaxf(value, -1.0f), 1.0f);
        }
    }
}

// ---- surface tension ----

//Each of this partition's own voxels updates the faces it stores, its lower ones, as the pressure update does (pressureToAcceleration): an unknown's,
//and an air voxel's above an unknown. A face between a liquid unknown and anything else that isn't a wall gets scale*kappa towards the liquid, with
//kappa where the level set crosses 0 between the two voxels' centres. scale is dt*(sigma/rho)/dx^2, kappa being in 1/voxels, or its negative to take
//the same back off
__global__ void surfaceTensionFaces(uint numOwnVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborNy, const uint* neighborNz, const float* level,
                                    const float* curvature, float scale, float* ux, float* uy, float* uz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index >= numOwnVoxels){
        return;
    }
    const uint* lower[3] = {neighborNx, neighborNy, neighborNz};
    float* velocities[3] = {ux, uy, uz};
    bool unknown = solveCodes[index];
    float here = level[index];
    bool liquid = unknown && here < 0.0f;
    #pragma unroll
    for(int dim = 0; dim < 3; ++dim){
        uint below = lower[dim][index];
        if(below == WALL_VOXEL || (!unknown && below == NO_VOXEL)){    //a wall's face, or one with no unknown either side
            continue;
        }
        bool liquidBelow = below != NO_VOXEL && solveCodes[below] && level[below] < 0.0f;
        if(liquid == liquidBelow){
            continue;
        }
        float kappa = curvature[index];
        if(below != NO_VOXEL){
            float there = level[below];
            float t = (there < 0.0f) != (here < 0.0f) ? there / (there - here) : 0.5f;     //how far from the voxel below to this one the surface is
            kappa = curvature[below] + t*(curvature[index] - curvature[below]);
        }
        velocities[dim][index] += liquid ? scale*kappa : -scale*kappa;  //the liquid is this voxel, above the face, or the one below it
    }
}

// ---- the steps pressureSolve takes ----

__global__ void fillLevels(uint count, float value, float* levels){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < count){
        levels[index] = value;
    }
}

void Particles::buildLevelSet(){
    uint numVoxels = voxelIDsUsed.size();
    uint interiorWidth = numVoxels1D - 2*(uint)std::floor(radius);
    float outside = SPLAT_REACH;    //where no particle is within reach, the surface is at least that far
    float wetting = surfaceTension > 0.0 ? (float)wallWetting : 0.0f;   //only surface tension gives the angle the liquid meets the walls at any meaning
    float bulk = (float)(restParticlesPerVoxel*M_PI*SPLAT_REACH*SPLAT_REACH*SPLAT_REACH*64.0/315.0);    //a voxel's weight sum deep in liquid at rest
    liquidLevel.resizeAsync(numVoxels, stream);
    int* sums[4];
    float* raw;
    size_t bytes = sizeof(int)*(size_t)(numVoxels > 0 ? numVoxels : 1);
    for(int*& sum : sums){
        gpuErrchk(cudaMallocAsync((void**)&sum, bytes, stream));
        gpuErrchk(cudaMemsetAsync(sum, 0, bytes, stream));
    }
    gpuErrchk(cudaMallocAsync((void**)&raw, bytes, stream));
    int3 domainVoxels = make_int3(grid.sizeX*interiorWidth, grid.sizeY*interiorWidth, grid.sizeZ*interiorWidth);
    if(numParticleNodes > 0){
        auto splat = twoPhase.on ? splatParticles<true> : splatParticles<false>;
        splat<<<numParticleNodes, SPLAT_THREADS, 4*sizeof(int)*numVoxelsPerNode, stream>>>(numParticleNodes, size, gridNodeIndicesToFirstParticleIndex.devPtr(), px.devPtr(), py.devPtr(),
            pz.devPtr(), voxelPlaces(), domainVoxels, voxelOwners.devPtr(), sums[0], sums[1], sums[2], sums[3], particleIds.devPtr());
    }
    for(int* sum : sums){   //particles near this partition's edge reach into ghost voxels: their owners add those sums in
        context->reduceGhosts(sum, stream);
    }
    if(numStoredNodes > 0 && numVoxels > 0){
        finishLevelSet<<<numStoredNodes, 128, 0, stream>>>(voxelPlaces(), obstacles.state(), solids.devPtr(), sums[0], sums[1], sums[2], sums[3], bulk, outside, raw);
    }
    context->fillGhosts(raw, stream);
    if(numVoxels > 0){  //a voxel in a node's apron that no node's interior holds reads as outside: nothing but the pressure update's air faces is kept there
        fillLevels<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, LIQUID_BAND, liquidLevel.devPtr());
    }
    if(numOwnNodes > 0 && numVoxels > 0){
        redistanceLevelSet<<<numOwnNodes, LEVEL_THREADS, 0, stream>>>(numUsedGridNodes, nodeCells.devPtr(), cellToNode.devPtr(), nodeInteriorVoxels.devPtr(), grid, outside, wetting,
            raw, liquidLevel.devPtr());
    }
    context->fillGhosts(liquidLevel.devPtr(), stream);
    if(needsSurfaceLevel()){    //the pressure solve's surface: smoothed a little, so particles' unevenness doesn't become pressure (freesurface.cu, twophase.cu)
        surfaceLevel.resizeAsync(numVoxels, stream);
        if(numVoxels > 0){
            fillLevels<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, LIQUID_BAND, surfaceLevel.devPtr());
        }
        if(numOwnNodes > 0 && numVoxels > 0){   //mirrored square on at the walls: the contact angle is the curvature's to impose, and tilted into the blur it
                                                //would wet the floor beside a drop, or dry the drop's underside
            smoothLevelSet<<<numOwnNodes, LEVEL_THREADS, 0, stream>>>(numUsedGridNodes, nodeCells.devPtr(), cellToNode.devPtr(), nodeInteriorVoxels.devPtr(), grid, 0.0f,
                SURFACE_PASSES, liquidLevel.devPtr(), surfaceLevel.devPtr());
        }
        context->fillGhosts(surfaceLevel.devPtr(), stream);
    }
    if(surfaceTension > 0.0){
        liquidCurvature.resizeAsync(numVoxels, stream);
        liquidCurvature.zeroDeviceAsync(stream);    //0 wherever it isn't worked out
        if(numOwnNodes > 0 && numVoxels > 0){   //the smoothed level set goes where the particles' field was: every voxel a tile reads gets one
            smoothLevelSet<<<numOwnNodes, LEVEL_THREADS, 0, stream>>>(numUsedGridNodes, nodeCells.devPtr(), cellToNode.devPtr(), nodeInteriorVoxels.devPtr(), grid, wetting, 4,
                liquidLevel.devPtr(), raw);
        }
        context->fillGhosts(raw, stream);
        if(numOwnNodes > 0 && numVoxels > 0){
            curveLevelSet<<<numOwnNodes, LEVEL_THREADS, 0, stream>>>(numUsedGridNodes, nodeCells.devPtr(), cellToNode.devPtr(), nodeInteriorVoxels.devPtr(), grid, wetting,
                liquidLevel.devPtr(), raw, liquidCurvature.devPtr());
        }
        context->fillGhosts(liquidCurvature.devPtr(), stream);
    }
    for(int* sum : sums){
        gpuErrchk(cudaFreeAsync(sum, stream));
    }
    gpuErrchk(cudaFreeAsync(raw, stream));
    gpuErrchk(cudaPeekAtLastError());
}

//adds dt of surface tension to the faces the surface crosses. The ghosts' faces take their owners'. The sharp free surface has none of this: the solve
//itself holds the surface's pressure (freesurface.cu)
void Particles::applySurfaceTension(){
    if(freeSurfaceMode == FreeSurface::sharp){
        return;
    }
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    float scale = (float)(dt*surfaceTension / (voxelSize*voxelSize));
    if(numOwnVoxels > 0){
        surfaceTensionFaces<<<numOwnVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numOwnVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborNy.devPtr(), neighborNz.devPtr(),
            liquidLevel.devPtr(), liquidCurvature.devPtr(), scale, voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr());
        gpuErrchk(cudaPeekAtLastError());
    }
    for(CudaVec<float>* velocity : {&voxelsUx, &voxelsUy, &voxelsUz}){
        context->fillGhosts(velocity->devPtr(), stream);
    }
}

//the longest timestep explicit surface tension holds for: the shortest capillary wave the grid carries mustn't cross a voxel in one (Brackbill, Kothe
//and Zemach 1992), sqrt(rho dx^3 / (2 pi sigma)) with rho the mean of the densities either side of the surface. The footprint's surface has unknowns as
//heavy as liquid on both sides, so that's the liquid's. The sharp surface has air on one, half of it: at the footprint's limit a drop at rest on a wall
//loses a couple of particles in every thousand over 5 s, and at this one none
double Particles::capillaryDt() const{
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    double sides = freeSurfaceMode == FreeSurface::sharp ? 0.5 : 1.0;
    return std::sqrt(sides*voxelSize*voxelSize*voxelSize / (2.0*M_PI*surfaceTension));
}
