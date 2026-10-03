//Copyright 2023 Aberrant Behavior LLC

//Viscosity, implicit, with the free surface it needs (Batty and Bridson, Accurate viscous free surfaces for buckling, coiling and rotating liquids,
//2008), between two pressure solves, which it leaves alone: it has its own equations and its own conjugate gradient, here.
//
//A viscous step takes the face velocities u* to the u that minimizes
//    sum over faces of V (u - u*)^2 / 2  +  dt nu ( sum over voxels of V (exx^2 + eyy^2 + ezz^2)  +  sum over edges of V g^2 / 2 )
//where exx, eyy and ezz are the rates the liquid stretches along each axis at a voxel's centre, g the rates it shears at the middle of each voxel edge
//(g_xy = du/dy + dv/dx on the edges along z, and so on), all by plain differences of the faces around them, and each V is how much of the voxel-sized
//box around that face, centre or edge is liquid, from the level set (levelset.cu). The first sum is the liquid's inertia and the rest is what its
//viscosity dissipates over the step, so the minimum is one backward Euler step of the viscous term, stable at any dt. Weighing each term by its liquid
//is what makes the surface free: nothing outside the liquid resists it, with no boundary condition to impose. A drop spinning in space keeps spinning,
//which a Laplacian of each component alone would stop, and a thread of honey can bend and coil.
//
//Setting the gradient to zero gives, per face, V u + c L(u) = V u*, c = dt nu / dx^2, with L every stretch and shear the face is part of
//(viscousRows): symmetric and positive, as it's the second derivative of a sum of squares, so conjugate gradient solves it, starting from u*, with each
//equation scaled by its own coefficient (Jacobi). The three components are coupled through the shears, so it's one system over every face.
//
//A face's stencil reaches the faces of the voxels around its own in every direction, and into air, which the neighbour arrays don't give: so each node
//gathers its voxels and one more on every side into shared memory (liquidTile.hu) and works out its own faces from that. Nothing new is stored: the
//weights V come from the level set each time.
//
//What's solid holds: a face on one of the domain's walls stays at rest, and the faces in and against obstacles move with them. Viscous liquid sticks
//to both (no slip), whatever an obstacle's friction is: past a wall, the faces along it mirror the liquid's with their sign reversed, so the velocity
//on the wall itself is zero, and the faces inside obstacles take the obstacle's velocity for the solve (stickToObstacles). Obstacles are whole voxels
//to it: the faces their surfaces cut are solved as liquid, and an obstacle holds the liquid half a voxel inside its face. The particles follow the
//faces: how they read the faces past a wall, and how they move beside one, hold to it as far as the viscosity carries (Particles::wallStick).
//
//It isn't solved together with the pressure (Larionov et al. 2017 do that), so each substep's last word is the pressure's: what the second solve adds
//isn't resisted until the next substep. Where gravity drives the flow that's small, as the first solve has already turned gravity into the pressure
//gradient the viscous step then resists: a dam break of syrup keeps to Huppert's spreading law within 2%. Where the pressure itself drives the liquid
//past something solid (squeezed through a gap by a moving wall), or bends a thread, it shows once c is large: the liquid slips along the walls by the
//pressure's push of one substep, and a thread a few voxels across folds from side to side rather than coiling until c is under about 40: so the
//timestep keeps c under 36 by default (Particles::setViscousCfl).
//
//The sums CG needs are added up exactly, per node and then across partitions, as the pressure CG's are (conjugateGradientFunctions.cu), so the result
//is the same bit for bit however the nodes are stored or split. Its iterations are queued in batches sized from the last substep's count, and each
//kernel does nothing once the residual is small enough, which the GPU decides: the host waits once a substep, for whether the batch got there.

#include "particles.hu"
#include "liquidTile.hu"
#include "algorithms/voxelSolveFunctions.hu"   //NO_VOXEL, WALL_VOXEL
#include <algorithm>
#include <cmath>
#include <iostream>

static constexpr int VISCOUS_WIDTH = 6;     //a tile's side: a node's 4 voxels and one either side
static constexpr int VISCOUS_SLOTS = VISCOUS_WIDTH*VISCOUS_WIDTH*VISCOUS_WIDTH;
static constexpr uint VISCOUS_THREADS = 64; //a thread per voxel of a node
static constexpr unsigned char SLOT_STORED = 1;     //a tile slot's state: a voxel is stored there (or it's past a wall, which is as good)
static constexpr unsigned char SLOT_SOLID = 2;      //it's inside an obstacle
static constexpr unsigned char SLOT_LIVE = 4;       //and, << dim, its lower face along dim holds a velocity (liveFaces)
static constexpr double VISCOUS_TOLERANCE = 1e-4;   //how small the residual has to get, relative to where it started
static constexpr uint MOST_VISCOUS_ITERATIONS = 2000;
static constexpr int VISCOUS_DIGITS = 9;
static constexpr uint VISCOUS_NODE_SLOTS = 64;      //a node's interior voxels, 4^3

//what an iteration needs of the ones before, on the GPU: the kernels read it there, so the host never has to
struct ViscousSolve{
    double rz;          //r*z of the current residual
    double dq;          //d*(A*d) of the current direction
    double rzNext;      //r*z after this iteration's step
    double rzStart;     //r*z of the starting guess
    int done;           //the residual is small enough: every kernel after this leaves things as they are
    int iterations;
};

// ---- the tile ----

//a block's tile, in its shared memory
struct ViscousTile{
    float* v[3];            //per slot, its lower faces' values along x, y and z
    float* level;
    float* cell;            //how much of each slot's voxel is liquid, which weighs its stretching
    float* edge[3];         //and of the box around each of the edges at its lower corner: along z (xy shear), y (xz) and x (yz)
    unsigned char* state;
    unsigned char* closed;  //bit per face (-x, +x, -y, +y, -z, +z): obstacles close it
};

//which of each stored voxel's three lower faces hold a velocity, bit per axis: an unknown's, and an air voxel's above an unknown, which are the faces
//the pressure update writes and P2G or its extrapolation filled. The rest of the faces stored in air were never given one: they stay out of the solve,
//or they'd drag on the liquid beside them, and only from above it, where the voxels holding its upper faces are
__global__ void liveFaces(uint numVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborNy, const uint* neighborNz, char* live){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        live[index] = solveCodes[index] ? 7 : (neighborNx[index] < WALL_VOXEL ? 1 : 0) | (neighborNy[index] < WALL_VOXEL ? 2 : 0) | (neighborNz[index] < WALL_VOXEL ? 4 : 0);
    }
}

//Fills a tile: the level set, the faces' values and what obstacles do to each slot, from the stored voxels; past the walls, the mirror image of what's
//inside; then the liquid in each voxel and around each edge. A face on a wall is at rest whatever the array says (P2G leaves particles' velocity
//there until the pressure update zeroes it). Past a wall, a face's value is minus its mirror image's, for every component: the wall's normal one mirrors
//face for face, which leaves the wall's own face its own image, at rest, and the others voxel for voxel, which puts zero on the wall between them.
//Every thread of the block calls it, after loadTileNodes
__device__ inline void fillViscousTile(const Tile& tile, const uint* nodes, uint numUsedGridNodes, const uint* interiorVoxels, const float* levels, const char* live, const char* solid,
                                       const char* near, const float* const faces[3], ViscousTile s){
    const int strides[3] = {1, VISCOUS_WIDTH, VISCOUS_WIDTH*VISCOUS_WIDTH};
    for(int slot = threadIdx.x; slot < VISCOUS_SLOTS; slot += blockDim.x){
        int3 t = tile.at(slot);
        if(!tile.inDomain(t)){
            continue;
        }
        unsigned char state = 0, closed = 0;
        float level = LIQUID_BAND;
        float value[3] = {0.0f, 0.0f, 0.0f};
        uint voxel = tile.voxel(t, nodes, numUsedGridNodes, interiorVoxels);
        if(voxel != NO_VOXEL){
            int3 g = tile.global(t);
            state = SLOT_STORED | (unsigned char)((live[voxel] & 7)*SLOT_LIVE);
            level = levels[voxel];
            value[0] = g.x == 0 ? 0.0f : faces[0][voxel];
            value[1] = g.y == 0 ? 0.0f : faces[1][voxel];
            value[2] = g.z == 0 ? 0.0f : faces[2][voxel];
            state |= (g.x == 0 ? SLOT_LIVE : 0) | (g.y == 0 ? 2*SLOT_LIVE : 0) | (g.z == 0 ? 4*SLOT_LIVE : 0);   //a wall's own faces hold its
            if(solid != nullptr){
                state |= solid[voxel] ? SLOT_SOLID : 0;
                closed = (unsigned char)near[voxel] >> 1;   //past NEAR_SURFACE, CLOSED_FACE << face
            }
        }
        s.state[slot] = state;
        s.closed[slot] = closed;
        s.level[slot] = level;
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            s.v[dim][slot] = value[dim];
        }
    }
    __syncthreads();
    for(int slot = threadIdx.x; slot < VISCOUS_SLOTS; slot += blockDim.x){
        int3 t = tile.at(slot);
        if(tile.inDomain(t)){
            continue;
        }
        int3 g = tile.global(t);
        int out[3] = {g.x < 0 ? -1 : g.x >= tile.domainVoxels.x ? 1 : 0, g.y < 0 ? -1 : g.y >= tile.domainVoxels.y ? 1 : 0, g.z < 0 ? -1 : g.z >= tile.domainVoxels.z ? 1 : 0};
        int image = tile.slot(tile.mirrored(t));
        s.state[slot] = SLOT_STORED | 7*SLOT_LIVE;
        s.closed[slot] = 0;
        s.level[slot] = s.level[image];
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            float sign = 1.0f;
            int from = image;
            #pragma unroll
            for(int axis = 0; axis < 3; ++axis){
                if(out[axis] != 0){
                    sign = -sign;
                    if(axis == dim){    //face for face: below the wall, a voxel further in than its image; above it, the wall's own face
                        from = out[axis] < 0 ? from + strides[axis] : -1;
                    }
                }
            }
            s.v[dim][slot] = from < 0 ? 0.0f : sign*s.v[dim][from];
        }
    }
    __syncthreads();
    //Where nothing's stored, the level set a voxel on from its nearest neighbour that has one, rather than its limit: voxels are stored a layer further
    //above the liquid than below it (the ones holding its upper faces), and the liquid's two sides should weigh alike whichever way they face
    for(int slot = threadIdx.x; slot < VISCOUS_SLOTS; slot += blockDim.x){
        if(s.state[slot] & SLOT_STORED){
            continue;
        }
        int3 t = tile.at(slot);
        int coordinates[3] = {t.x, t.y, t.z};
        float level = LIQUID_BAND;
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            for(int step = -1; step <= 1; step += 2){
                int moved = coordinates[axis] + step;
                if(moved >= 0 && moved < VISCOUS_WIDTH && (s.state[slot + step*strides[axis]] & SLOT_STORED)){
                    level = fminf(level, s.level[slot + step*strides[axis]] + 1.0f);
                }
            }
        }
        s.cell[slot] = level;   //kept apart until every slot has read its neighbours' own
    }
    __syncthreads();
    for(int slot = threadIdx.x; slot < VISCOUS_SLOTS; slot += blockDim.x){
        if(!(s.state[slot] & SLOT_STORED)){
            s.level[slot] = s.cell[slot];
        }
    }
    __syncthreads();
    //A term only counts where every face it reads holds a velocity: the solve's own (live), or what a wall or an obstacle holds it at. A voxel's
    //stretching reads its lower faces and the ones above it, an edge's shear two faces of each component around it. Liquid always has them; this is for
    //the air beside it and the stray voxels that don't
    auto held = [&](int at, int dim){
        if((s.state[at] & (SLOT_SOLID | SLOT_LIVE << dim)) || (s.closed[at] >> 2*dim & 1)){
            return true;
        }
        if(at / strides[dim] % VISCOUS_WIDTH == 0){
            return false;
        }
        int below = at - strides[dim];
        return (s.state[below] & SLOT_SOLID) || (s.closed[below] >> (2*dim + 1) & 1);
    };
    for(int slot = threadIdx.x; slot < VISCOUS_SLOTS; slot += blockDim.x){
        int3 t = tile.at(slot);
        int coordinates[3] = {t.x, t.y, t.z};
        bool whole = s.state[slot] & SLOT_STORED;
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            whole = whole && coordinates[axis] < VISCOUS_WIDTH - 1 && held(slot, axis) && held(slot + strides[axis], axis);
        }
        s.cell[slot] = whole ? liquidFraction(s.level[slot]) : 0.0f;
        #pragma unroll
        for(int pair = 0; pair < 3; ++pair){    //xy, xz, yz
            int a = pair == 2 ? 1 : 0, b = pair == 0 ? 1 : 2;
            float fraction = 0.0f;
            if(coordinates[a] > 0 && coordinates[b] > 0){
                int around[4] = {slot, slot - strides[a], slot - strides[b], slot - strides[a] - strides[b]};
                if((s.state[around[0]] & s.state[around[1]] & s.state[around[2]] & s.state[around[3]] & SLOT_STORED)
                   && held(slot, a) && held(slot - strides[b], a) && held(slot, b) && held(slot - strides[a], b)){
                    fraction = liquidFraction(0.25f*(s.level[around[0]] + s.level[around[1]] + s.level[around[2]] + s.level[around[3]]));
                }
            }
            s.edge[pair][slot] = fraction;
        }
    }
    __syncthreads();
}

//For the three lower faces of one of the node's voxels, at slot: the stretches and shears each is part of, times their liquid (stress, the L(u) of the
//face's equation); the most a face's own value weighs in that (own); and the liquid around the face itself (mass). A shear across one of the domain's
//walls reads the face's own mirror image, minus itself, so there it weighs twice
__device__ inline void viscousRows(const ViscousTile& s, int slot, int3 global, int3 domainVoxels, float stress[3], float own[3], float mass[3]){
    const int sx = 1, sy = VISCOUS_WIDTH, sz = VISCOUS_WIDTH*VISCOUS_WIDTH;
    const float* u = s.v[0];
    const float* v = s.v[1];
    const float* w = s.v[2];
    const float* xy = s.edge[0];
    const float* xz = s.edge[1];
    const float* yz = s.edge[2];
    auto exx = [&](int at){ return u[at + sx] - u[at]; };
    auto eyy = [&](int at){ return v[at + sy] - v[at]; };
    auto ezz = [&](int at){ return w[at + sz] - w[at]; };
    auto gxy = [&](int at){ return (u[at] - u[at - sy]) + (v[at] - v[at - sx]); };
    auto gxz = [&](int at){ return (u[at] - u[at - sz]) + (w[at] - w[at - sx]); };
    auto gyz = [&](int at){ return (v[at] - v[at - sz]) + (w[at] - w[at - sy]); };
    float here = s.cell[slot];
    stress[0] = 2.0f*(s.cell[slot - sx]*exx(slot - sx) - here*exx(slot)) + xy[slot]*gxy(slot) - xy[slot + sy]*gxy(slot + sy) + xz[slot]*gxz(slot) - xz[slot + sz]*gxz(slot + sz);
    stress[1] = 2.0f*(s.cell[slot - sy]*eyy(slot - sy) - here*eyy(slot)) + xy[slot]*gxy(slot) - xy[slot + sx]*gxy(slot + sx) + yz[slot]*gyz(slot) - yz[slot + sz]*gyz(slot + sz);
    stress[2] = 2.0f*(s.cell[slot - sz]*ezz(slot - sz) - here*ezz(slot)) + xz[slot]*gxz(slot) - xz[slot + sx]*gxz(slot + sx) + yz[slot]*gyz(slot) - yz[slot + sy]*gyz(slot + sy);
    float lowX = global.x == 0 ? 2.0f : 1.0f, highX = global.x == domainVoxels.x - 1 ? 2.0f : 1.0f;
    float lowY = global.y == 0 ? 2.0f : 1.0f, highY = global.y == domainVoxels.y - 1 ? 2.0f : 1.0f;
    float lowZ = global.z == 0 ? 2.0f : 1.0f, highZ = global.z == domainVoxels.z - 1 ? 2.0f : 1.0f;
    own[0] = 2.0f*(s.cell[slot - sx] + here) + lowY*xy[slot] + highY*xy[slot + sy] + lowZ*xz[slot] + highZ*xz[slot + sz];
    own[1] = 2.0f*(s.cell[slot - sy] + here) + lowX*xy[slot] + highX*xy[slot + sx] + lowZ*yz[slot] + highZ*yz[slot + sz];
    own[2] = 2.0f*(s.cell[slot - sz] + here) + lowX*xz[slot] + highX*xz[slot + sx] + lowY*yz[slot] + highY*yz[slot + sy];
    mass[0] = liquidFraction(0.5f*(s.level[slot] + s.level[slot - sx]));
    mass[1] = liquidFraction(0.5f*(s.level[slot] + s.level[slot - sy]));
    mass[2] = liquidFraction(0.5f*(s.level[slot] + s.level[slot - sz]));
}

//whether the solve may move the dim face of one of the node's voxels: one that holds a velocity, not on a wall, nor with an obstacle on either side
//or closing it
__device__ inline bool viscousFaceFree(const ViscousTile& s, int slot, int dim, int3 global){
    const int strides[3] = {1, VISCOUS_WIDTH, VISCOUS_WIDTH*VISCOUS_WIDTH};
    int below = slot - strides[dim];
    int along = dim == 0 ? global.x : dim == 1 ? global.y : global.z;
    return along > 0 && (s.state[slot] & SLOT_LIVE << dim) && !((s.state[slot] | s.state[below]) & SLOT_SOLID) && !(s.closed[slot] >> 2*dim & 1)
        && !(s.closed[below] >> (2*dim + 1) & 1);
}

//a block's shared memory for a tile, and the tile filled from faces: the same at the start of both kernels that work on one
#define VISCOUS_TILE(faces) \
    __shared__ float tileValues[8][VISCOUS_SLOTS]; \
    __shared__ unsigned char tileFlags[2][VISCOUS_SLOTS]; \
    __shared__ uint nodes[27]; \
    uint cell = nodeCells[blockIdx.x]; \
    loadTileNodes(nodes, cell, numUsedGridNodes, nodeCells, cellToNode, grid); \
    __syncthreads(); \
    Tile tile(cell, grid, VISCOUS_WIDTH - 2, 1); \
    ViscousTile s = {{tileValues[0], tileValues[1], tileValues[2]}, tileValues[3], tileValues[4], {tileValues[5], tileValues[6], tileValues[7]}, tileFlags[0], tileFlags[1]}; \
    fillViscousTile(tile, nodes, numUsedGridNodes, interiorVoxels, levels, live, solid, near, faces, s); \
    int3 t = make_int3(1 + threadIdx.x % 4, 1 + threadIdx.x / 4 % 4, 1 + threadIdx.x / 16); \
    uint voxel = tile.voxel(t, nodes, numUsedGridNodes, interiorVoxels); \
    int slot = tile.slot(t); \
    int3 global = tile.global(t);

// ---- the solve ----

//the start, a block per node of this partition's own: which of its faces the solve moves, each one's own coefficient, and the residual of the starting
//guess u*, r = V u* - (V u* + c L(u*)) = -c L(u*). The first direction is that, preconditioned. The faces it doesn't move keep 0 in all three
__global__ void startViscosity(uint numUsedGridNodes, const uint* nodeCells, const uint* cellToNode, const uint* interiorVoxels, Grid grid, const float* levels, const char* live,
                               const char* solid, const char* near, uint numVoxels, const float* ux, const float* uy, const float* uz, float c, float* r, float* diagonal, float* d){
    const float* const faces[3] = {ux, uy, uz};
    VISCOUS_TILE(faces)
    if(voxel == NO_VOXEL){
        return;
    }
    float stress[3], own[3], mass[3];
    viscousRows(s, slot, global, tile.domainVoxels, stress, own, mass);
    #pragma unroll
    for(int dim = 0; dim < 3; ++dim){
        float coefficient = mass[dim] + c*own[dim];
        if(coefficient > 0.0f && viscousFaceFree(s, slot, dim, global)){
            size_t at = (size_t)dim*numVoxels + voxel;
            float residual = -c*stress[dim];
            r[at] = residual;
            diagonal[at] = coefficient;
            d[at] = residual / coefficient;
        }
    }
}

//q = A*d over the faces the solve moves, a block per node of this partition's own. d is 0 on every other face, so the walls and obstacles hold
__global__ void multiplyViscosity(uint numUsedGridNodes, const uint* nodeCells, const uint* cellToNode, const uint* interiorVoxels, Grid grid, const float* levels, const char* live,
                                  const char* solid, const char* near, uint numVoxels, const float* diagonal, const float* d, float c, const ViscousSolve* solve, float* q){
    if(solve->done){
        return;
    }
    const float* const faces[3] = {d, d + numVoxels, d + 2*(size_t)numVoxels};
    VISCOUS_TILE(faces)
    if(voxel == NO_VOXEL){
        return;
    }
    float stress[3], own[3], mass[3];
    viscousRows(s, slot, global, tile.domainVoxels, stress, own, mass);
    #pragma unroll
    for(int dim = 0; dim < 3; ++dim){
        size_t at = (size_t)dim*numVoxels + voxel;
        if(diagonal[at] > 0.0f){
            q[at] = mass[dim]*s.v[dim][slot] + c*stress[dim];
        }
    }
}

// ---- exact sums, as the pressure CG's (conjugateGradientFunctions.cu) ----

//an exact sum of floats in fixed point: digit k counts units of 2^(32k - 149), each a 64-bit integer with room for 2^31 additions' carries, so the
//digits come out the same in any order and add up across partitions word by word
struct ViscousSum{
    long long digits[VISCOUS_DIGITS];
    long long nonFinite;    //how many infs or NaNs went in
};
static constexpr uint VISCOUS_SUM_WORDS = sizeof(ViscousSum) / sizeof(long long);

__device__ inline void addViscousExactly(float value, ViscousSum& sum){
    uint bits = __float_as_uint(value);
    int exponent = bits >> 23 & 0xFF;
    if(value == 0.0f){
        return;
    }
    if(exponent == 0xFF){
        atomicAdd((unsigned long long*)&sum.nonFinite, 1ull);
        return;
    }
    unsigned long long significand = bits & 0x7FFFFF;
    if(exponent != 0){
        significand |= 0x800000;    //the hidden bit
    }
    else{
        exponent = 1;               //a subnormal
    }
    int position = exponent - 1;    //value = significand*2^(exponent - 150), and 2^-149 is bit 0
    significand <<= position % 32;
    long long sign = bits >> 31 ? -1 : 1;
    atomicAdd((unsigned long long*)sum.digits + position/32, (unsigned long long)(sign*(long long)(significand & 0xFFFFFFFFull)));
    atomicAdd((unsigned long long*)sum.digits + position/32 + 1, (unsigned long long)(sign*(long long)(significand >> 32)));
}

//a*b over this partition's own faces, or with a diagonal, a*(a/diagonal) over the ones with one (r*z, z never being stored), into an exact sum: a
//warp per node, its lanes taking interior slots lane and lane + 32 and their three faces in a fixed order, then pairing off in a fixed tree to the
//node's share, which goes in exactly. A share depends only on the node's own values, not on where its voxels are stored or which GPU takes it
__global__ void addViscousShares(uint numNodes, const uint* interiorVoxels, uint numVoxels, const float* a, const float* b, const float* diagonal, const ViscousSolve* solve,
                                 ViscousSum* sum){
    __shared__ ViscousSum blockSum;
    if(solve->done){
        return;
    }
    if(threadIdx.x < VISCOUS_DIGITS){
        blockSum.digits[threadIdx.x] = 0;
    }
    if(threadIdx.x == 0){
        blockSum.nonFinite = 0;
    }
    __syncthreads();
    uint lane = threadIdx.x % 32;
    uint warps = gridDim.x*blockDim.x/32;
    for(uint node = (threadIdx.x + blockIdx.x*blockDim.x)/32; node < numNodes; node += warps){    //the same for the whole warp
        float share = 0.0f;
        #pragma unroll
        for(int half = 0; half < 2; ++half){
            uint voxel = interiorVoxels[node*VISCOUS_NODE_SLOTS + lane + 32*half];
            if(voxel == NO_VOXEL){
                continue;
            }
            #pragma unroll
            for(int dim = 0; dim < 3; ++dim){
                size_t at = (size_t)dim*numVoxels + voxel;
                float other = diagonal == nullptr ? b[at] : diagonal[at] > 0.0f ? a[at] / diagonal[at] : 0.0f;
                share = __fadd_rn(share, __fmul_rn(a[at], other));  //no fused multiply-adds: every build rounds alike
            }
        }
        for(int lanes = 16; lanes > 0; lanes >>= 1){
            share = __fadd_rn(share, __shfl_xor_sync(0xffffffff, share, lanes));   //a + b == b + a exactly, so every lane ends with the same share
        }
        if(lane == 0){
            addViscousExactly(share, blockSum);
        }
    }
    __syncthreads();
    if(threadIdx.x < VISCOUS_DIGITS && blockSum.digits[threadIdx.x] != 0){
        atomicAdd((unsigned long long*)sum->digits + threadIdx.x, (unsigned long long)blockSum.digits[threadIdx.x]);
    }
    if(threadIdx.x == 0 && blockSum.nonFinite){
        atomicAdd((unsigned long long*)&sum->nonFinite, (unsigned long long)blockSum.nonFinite);
    }
}

//an exact sum back to a double (finishExactSum's conversion): carries passed up, then the digits added in from the top, a negative sum negated first
__global__ void finishViscousSum(const ViscousSum* sum, const ViscousSolve* solve, double* total){
    if(solve->done){
        return;
    }
    if(sum->nonFinite){
        *total = nan("");
        return;
    }
    unsigned long long digits[VISCOUS_DIGITS];
    long long carry = 0;
    for(int k = 0; k < VISCOUS_DIGITS; ++k){
        long long value = sum->digits[k] + carry;
        digits[k] = (unsigned long long)value & 0xFFFFFFFFull;
        carry = value >> 32;
    }
    bool negative = carry < 0;
    if(negative){
        unsigned long long add = 1;
        for(int k = 0; k < VISCOUS_DIGITS; ++k){
            unsigned long long value = (~digits[k] & 0xFFFFFFFFull) + add;
            digits[k] = value & 0xFFFFFFFFull;
            add = value >> 32;
        }
        carry = ~carry + (long long)add;
    }
    double magnitude = (double)carry*scalbn(1.0, 32*VISCOUS_DIGITS - 149);
    for(int k = VISCOUS_DIGITS - 1; k >= 0; --k){
        magnitude += (double)digits[k]*scalbn(1.0, 32*k - 149);
    }
    *total = negative ? -magnitude : magnitude;
}

// ---- CG's steps over every face ----

//the starting residual is what every later one is measured against; with none (nothing shears), there's nothing to do
__global__ void beginViscosity(ViscousSolve* solve){
    solve->rz = solve->rzNext;
    solve->rzStart = solve->rzNext;
    solve->done = !(solve->rzNext > 0.0);
}

//u += alpha*d and r -= alpha*q, alpha = r*z / d*q: as far along d as lowers the energy most. Over every stored voxel's faces, ghosts' too: d's ghosts
//hold their owners' values, so u's stay equal to their owners' as well
__global__ void stepViscosity(uint numVoxels, const float* d, const float* q, float* ux, float* uy, float* uz, float* r, const ViscousSolve* solve){
    __shared__ float alpha;
    if(solve->done){
        return;
    }
    if(threadIdx.x == 0){
        alpha = solve->dq != 0.0 ? solve->rz / solve->dq : 0.0;
    }
    __syncthreads();
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        float* velocities[3] = {ux, uy, uz};
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            size_t at = (size_t)dim*numVoxels + index;
            velocities[dim][index] += alpha*d[at];
            r[at] -= alpha*q[at];
        }
    }
}

//the next direction: the preconditioned residual, plus beta = r*z(new) / r*z(old) of the last one
__global__ void turnViscosity(uint count, const float* r, const float* diagonal, float* d, const ViscousSolve* solve){
    __shared__ float beta;
    if(solve->done){
        return;
    }
    if(threadIdx.x == 0){
        beta = solve->rz != 0.0 ? solve->rzNext / solve->rz : 0.0;
    }
    __syncthreads();
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < count){
        d[index] = (diagonal[index] > 0.0f ? r[index] / diagonal[index] : 0.0f) + beta*d[index];
    }
}

//the step's r*z becomes the current one, and the solve is done once it's small enough (or isn't a number: nothing more would help)
__global__ void nextViscosity(ViscousSolve* solve, double tolerance){
    if(solve->done){
        return;
    }
    solve->rz = solve->rzNext;
    ++solve->iterations;
    solve->done = !(solve->rzNext > tolerance*tolerance*solve->rzStart);
}

// ---- obstacles ----

//Viscous liquid sticks to obstacles: for the solve, each face of this partition's own voxels with an obstacle on either side, or closed by one, moves
//with the obstacle there. The faces between fluid and obstacle already do; the ones inside hold the fluid's velocity carried on along the surface,
//unless the obstacle has friction (obstacleGhostVelocities), and the pressure update gives them that back. A block per node of its own
__global__ void stickToObstacles(Obstacles obstacles, VoxelPlaces places, const char* solid, const char* near, float* ux, float* uy, float* uz){
    uint node = blockIdx.x;
    uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
    uint last = places.nodeVoxelEnds[node];
    uint cell = places.nodeCells[node];
    int voxels1D = places.interiorWidth + 2*places.apronCells;
    float* velocities[3] = {ux, uy, uz};
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        if(!solid[index] && !((unsigned char)near[index] >> 1)){
            continue;
        }
        int slot = places.voxelSlots[index];
        int x = slot % voxels1D, y = slot / voxels1D % voxels1D, z = slot / (voxels1D*voxels1D);
        if(x < places.apronCells || y < places.apronCells || z < places.apronCells || x >= voxels1D - places.apronCells || y >= voxels1D - places.apronCells || z >= voxels1D - places.apronCells){
            continue;   //only the node's own voxels: their neighbours own the apron's
        }
        int3 voxel = places.voxelOf(cell, slot);
        for(int dim = 0; dim < 3; ++dim){
            if(!solid[index] && !(near[index] & CLOSED_FACE << 2*dim)){
                continue;
            }
            float3 face = places.point(voxel, dim == 0 ? 0.0f : 0.5f, dim == 1 ? 0.0f : 0.5f, dim == 2 ? 0.0f : 0.5f);
            float distance;
            float3 normal;
            int nearest = nearestObstacle(obstacles, face, distance, normal);
            if(nearest >= 0){
                velocities[dim][index] = component(obstacleVelocity(obstacles.items[nearest], face), dim);
            }
        }
    }
}

// ---- the steps pressureSolve takes ----

//the face velocities as the forces left them, kept for undoViscosity: before the first of pressureSolve's two solves around the viscous step
void Particles::keepVelocitiesForViscosity(){
    uint numVoxels = voxelIDsUsed.size();
    CudaVec<float>* velocities[3] = {&voxelsUx, &voxelsUy, &voxelsUz};
    for(int dim = 0; dim < 3; ++dim){
        if(viscousBefore[dim].size() != numVoxels){
            viscousBefore[dim].resizeAsync(numVoxels, stream);
        }
        gpuErrchk(cudaMemcpyAsync(viscousBefore[dim].devPtr(), velocities[dim]->devPtr(), sizeof(float)*numVoxels, cudaMemcpyDeviceToDevice, stream));
    }
}

//one viscous step of dt on the face velocities, in place, once the first solve's pressure is on them
void Particles::applyViscosity(){
    uint numVoxels = voxelIDsUsed.size();
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    float c = (float)(dt*viscosity / (voxelSize*voxelSize));
    CudaVec<float>* velocities[3] = {&voxelsUx, &voxelsUy, &voxelsUz};
    bool hasObstacles = obstacles.count() > 0;
    if(hasObstacles && numOwnNodes > 0 && numVoxels > 0){
        stickToObstacles<<<numOwnNodes, 128, 0, stream>>>(obstacles.state(), voxelPlaces(), obstacleSolids.devPtr(), obstacleNear.devPtr(), voxelsUx.devPtr(), voxelsUy.devPtr(),
            voxelsUz.devPtr());
    }
    for(CudaVec<float>* velocity : velocities){     //the ghosts' faces as their owners have them, after the first solve's pressure and the obstacles
        context->fillGhosts(velocity->devPtr(), stream);
    }
    if(viscousSeen == nullptr){
        gpuErrchk(cudaMallocHost((void**)&viscousSeen, sizeof(ViscousSolve)));
    }
    size_t count = 3*(size_t)numVoxels;
    size_t bytes = sizeof(float)*(count > 0 ? count : 1);
    float* r;
    float* d;
    float* q;
    float* diagonal;
    char* live;
    ViscousSolve* solve;
    ViscousSum* sum;
    for(float** vector : {&r, &d, &q, &diagonal}){
        gpuErrchk(cudaMallocAsync((void**)vector, bytes, stream));
        gpuErrchk(cudaMemsetAsync(*vector, 0, bytes, stream));
    }
    gpuErrchk(cudaMallocAsync((void**)&live, numVoxels > 0 ? numVoxels : 1, stream));
    if(numVoxels > 0){
        liveFaces<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborNy.devPtr(), neighborNz.devPtr(), live);
    }
    context->fillGhosts(live, stream);  //a ghost's lower neighbours past the ghost plane are missing here: its owner knows them
    gpuErrchk(cudaMallocAsync((void**)&solve, sizeof(ViscousSolve), stream));
    gpuErrchk(cudaMallocAsync((void**)&sum, sizeof(ViscousSum), stream));
    gpuErrchk(cudaMemsetAsync(solve, 0, sizeof(ViscousSolve), stream));
    const char* solid = hasObstacles ? obstacleSolids.devPtr() : nullptr;
    const char* near = hasObstacles ? obstacleNear.devPtr() : nullptr;
    bool mine = numOwnNodes > 0 && numVoxels > 0;   //whether this partition has anything to solve: it makes every exchange either way
    uint entryBlocks = (uint)(count / BLOCKSIZE + 1);
    //total = a*b, or with a diagonal a*(a/diagonal), over every partition's faces
    auto dotProduct = [&](const float* a, const float* b, const float* scaling, double* total){
        gpuErrchk(cudaMemsetAsync(sum, 0, sizeof(ViscousSum), stream));
        if(mine){
            uint blocks = numOwnNodes*32 / BLOCKSIZE + 1;
            addViscousShares<<<blocks < 1024 ? blocks : 1024, BLOCKSIZE, 0, stream>>>(numOwnNodes, nodeInteriorVoxels.devPtr(), numVoxels, a, b, scaling, solve, sum);
        }
        context->sumOverPartitions((long long*)sum, VISCOUS_SUM_WORDS, stream);
        finishViscousSum<<<1, 1, 0, stream>>>(sum, solve, total);
    };
    auto fillDirection = [&](){    //A*d reads the ghosts' d
        for(int dim = 0; dim < 3; ++dim){
            context->fillGhosts(d + (size_t)dim*numVoxels, stream);
        }
    };
    if(mine){
        startViscosity<<<numOwnNodes, VISCOUS_THREADS, 0, stream>>>(numUsedGridNodes, nodeCells.devPtr(), cellToNode.devPtr(), nodeInteriorVoxels.devPtr(), grid, liquidLevel.devPtr(), live,
            solid, near, numVoxels, voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr(), c, r, diagonal, d);
    }
    dotProduct(r, nullptr, diagonal, &solve->rzNext);
    beginViscosity<<<1, 1, 0, stream>>>(solve);
    fillDirection();
    gpuErrchk(cudaPeekAtLastError());
    uint queued = 0;
    uint batch = viscousIterations + viscousIterations/4 + 4;   //what the last substep took, and some: a substep is much like the one before
    while(true){
        for(uint iteration = 0; iteration < batch; ++iteration){
            if(mine){
                multiplyViscosity<<<numOwnNodes, VISCOUS_THREADS, 0, stream>>>(numUsedGridNodes, nodeCells.devPtr(), cellToNode.devPtr(), nodeInteriorVoxels.devPtr(), grid,
                    liquidLevel.devPtr(), live, solid, near, numVoxels, diagonal, d, c, solve, q);
            }
            dotProduct(d, q, nullptr, &solve->dq);
            if(numVoxels > 0){
                stepViscosity<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, d, q, voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr(), r, solve);
            }
            dotProduct(r, nullptr, diagonal, &solve->rzNext);
            if(numVoxels > 0){
                turnViscosity<<<entryBlocks, BLOCKSIZE, 0, stream>>>((uint)count, r, diagonal, d, solve);
            }
            nextViscosity<<<1, 1, 0, stream>>>(solve, VISCOUS_TOLERANCE);
            fillDirection();
        }
        queued += batch;
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaMemcpyAsync(viscousSeen, solve, sizeof(ViscousSolve), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));   //the one wait: whether the batch got there. Every partition sees the same answer
        if(((ViscousSolve*)viscousSeen)->done || queued >= MOST_VISCOUS_ITERATIONS){
            break;
        }
        batch = std::min(2*batch, 64u);     //it's taking more than the last substep did: fewer waits than batches that size would need
    }
    const ViscousSolve& seen = *(ViscousSolve*)viscousSeen;
    viscousIterations = seen.iterations;
    if(verbose()){
        std::cout<<"Viscosity: "<<seen.iterations<<" iterations, residual "<<(seen.rzStart > 0.0 ? std::sqrt(seen.rz / seen.rzStart) : 0.0)<<" of its start\n";
        if(!seen.done){
            std::cerr<<"Viscosity: not solved to "<<VISCOUS_TOLERANCE<<" in "<<queued<<" iterations: carrying on with what it reached\n";
        }
    }
    for(CudaVec<float>* velocity : velocities){     //the ghosts' faces as their owners left them
        context->fillGhosts(velocity->devPtr(), stream);
    }
    for(float* vector : {r, d, q, diagonal}){
        gpuErrchk(cudaFreeAsync(vector, stream));
    }
    gpuErrchk(cudaFreeAsync(live, stream));
    gpuErrchk(cudaFreeAsync(solve, stream));
    gpuErrchk(cudaFreeAsync(sum, stream));
    gpuErrchk(cudaPeekAtLastError());
}

//the face velocities as keepVelocitiesForViscosity kept them: pressureSolve's retry takes the forces back off those, and does it all again at half the dt
void Particles::undoViscosity(){
    CudaVec<float>* velocities[3] = {&voxelsUx, &voxelsUy, &voxelsUz};
    for(int dim = 0; dim < 3; ++dim){
        gpuErrchk(cudaMemcpyAsync(velocities[dim]->devPtr(), viscousBefore[dim].devPtr(), sizeof(float)*velocities[dim]->size(), cudaMemcpyDeviceToDevice, stream));
    }
}
