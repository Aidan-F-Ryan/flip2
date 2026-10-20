//Copyright 2023 Aberrant Behavior LLC

//Two-way coupled rigid bodies (see bodies.hu)

#include "bodies.hu"
#include "algorithms/voxelSolveFunctions.hu"    //NO_VOXEL, WALL_VOXEL
#include "cacheWriter.hu"     //BODY_PLACEMENT
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iterator>
#include <stdexcept>

static constexpr int COUPLING_THREADS = 256;    //a warp per node in the coupling's passes
static constexpr int MASS_THREADS = 256;
static constexpr uint MASS_BLOCKS = 1024;       //fixed, so the mass's sums are taken in the same order every run
static constexpr int MOMENTS = 10;              //a mass's: its sum, its first moments, then xx, yy, zz, xy, xz, yz
static constexpr double MAX_MASS_SAMPLES = 64.0*1024*1024;

// ---- its mass: its density over the inside of its shape ----

//the moments of the mass inside an obstacle's shape, in its own space, over a lattice of samples spacing h apart from low: each sample is a cube of the
//shape's density, inside it by as much as a plane at its distance from the cube's centre would cut it (the half-sample ramp coveredFraction uses), so
//the sums move smoothly with the surface. Relative to the lattice's centre, which keeps the second moments' cancelling small. A thread's samples go in
//in a fixed order, then the block's threads in a fixed tree, so every run and every partition gets the same bits
__global__ void integrateMass(ObstacleState shape, double3 low, double h, int3 samples, double3 middle, double* partials){
    __shared__ double sums[MOMENTS][MASS_THREADS];
    double moments[MOMENTS] = {};
    unsigned long long count = (unsigned long long)samples.x*samples.y*samples.z;
    for(unsigned long long sample = threadIdx.x + (unsigned long long)blockIdx.x*blockDim.x; sample < count; sample += (unsigned long long)blockDim.x*gridDim.x){
        int i = (int)(sample % samples.x), j = (int)(sample / samples.x % samples.y), k = (int)(sample / ((unsigned long long)samples.x*samples.y));
        double x = low.x + (i + 0.5)*h, y = low.y + (j + 0.5)*h, z = low.z + (k + 0.5)*h;
        float3 normal;
        float distance = obstacleDistance(shape, make_float3((float)x, (float)y, (float)z), normal);
        double inside = fmin(fmax(0.5 - distance / h, 0.0), 1.0);
        if(inside == 0.0){
            continue;
        }
        double dx = x - middle.x, dy = y - middle.y, dz = z - middle.z;
        moments[0] += inside;
        moments[1] += inside*dx;
        moments[2] += inside*dy;
        moments[3] += inside*dz;
        moments[4] += inside*dx*dx;
        moments[5] += inside*dy*dy;
        moments[6] += inside*dz*dz;
        moments[7] += inside*dx*dy;
        moments[8] += inside*dx*dz;
        moments[9] += inside*dy*dz;
    }
    for(int moment = 0; moment < MOMENTS; ++moment){
        sums[moment][threadIdx.x] = moments[moment];
    }
    __syncthreads();
    for(int half = MASS_THREADS/2; half > 0; half /= 2){
        if(threadIdx.x < half){
            for(int moment = 0; moment < MOMENTS; ++moment){
                sums[moment][threadIdx.x] += sums[moment][threadIdx.x + half];
            }
        }
        __syncthreads();
    }
    if(threadIdx.x < MOMENTS){
        partials[MOMENTS*blockIdx.x + threadIdx.x] = sums[threadIdx.x][0];
    }
}

//the blocks' moments added up, in block order
__global__ void sumMassPartials(uint blocks, const double* partials, double* moments){
    if(threadIdx.x < MOMENTS){
        double sum = 0.0;
        for(uint block = 0; block < blocks; ++block){
            sum += partials[MOMENTS*block + threadIdx.x];
        }
        moments[threadIdx.x] = sum;
    }
}

//3x3 inverse of a symmetric positive definite tensor, by its cofactors
static bool invert(const double m[9], double out[9]){
    double c[9] = {m[4]*m[8] - m[5]*m[7], m[2]*m[7] - m[1]*m[8], m[1]*m[5] - m[2]*m[4],
                   m[5]*m[6] - m[3]*m[8], m[0]*m[8] - m[2]*m[6], m[2]*m[3] - m[0]*m[5],
                   m[3]*m[7] - m[4]*m[6], m[1]*m[6] - m[0]*m[7], m[0]*m[4] - m[1]*m[3]};
    double determinant = m[0]*c[0] + m[1]*c[3] + m[2]*c[6];
    if(!(determinant > 0.0)){
        return false;
    }
    for(int i = 0; i < 9; ++i){
        out[i] = c[i] / determinant;
    }
    return true;
}

// ---- where they are and how they move, each a thread ----

//a body's frame where it is: the world to its own space, (R^T, its centre in its own space - R^T x); its reach, the box around its bounds' corners
//3 voxels wider every way, as ObstacleSet::update has it for the others; how it moves; and what the coupling weighs it at
__global__ void writeBodyFrames(int numBodies, const BodyState* states, const BodyMass* masses, BodyFrame* frames, BodyCoupling* couplings, double voxelSize){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body >= numBodies){
        return;
    }
    const BodyState& state = states[body];
    const BodyMass& mass = masses[body];
    BodyFrame& frame = frames[body];
    double r[9];
    rotationOf(state.orientation, r);
    for(int row = 0; row < 3; ++row){
        for(int column = 0; column < 3; ++column){
            frame.worldToObject[4*row + column] = (float)r[3*column + row];
        }
        frame.worldToObject[4*row + 3] = (float)(mass.centre[row] - (r[row]*state.position[0] + r[3 + row]*state.position[1] + r[6 + row]*state.position[2]));
    }
    double low[3] = {INFINITY, INFINITY, INFINITY}, high[3] = {-INFINITY, -INFINITY, -INFINITY};
    for(int corner = 0; corner < 8; ++corner){
        double local[3] = {(corner & 1 ? mass.objectHigh.x : mass.objectLow.x) - mass.centre[0], (corner & 2 ? mass.objectHigh.y : mass.objectLow.y) - mass.centre[1],
                           (corner & 4 ? mass.objectHigh.z : mass.objectLow.z) - mass.centre[2]};
        for(int row = 0; row < 3; ++row){
            double world = r[3*row]*local[0] + r[3*row + 1]*local[1] + r[3*row + 2]*local[2] + state.position[row];
            low[row] = fmin(low[row], world);
            high[row] = fmax(high[row], world);
        }
    }
    double reach = 3.0*voxelSize;
    frame.reachLow = make_float3((float)(low[0] - reach), (float)(low[1] - reach), (float)(low[2] - reach));
    frame.reachHigh = make_float3((float)(high[0] + reach), (float)(high[1] + reach), (float)(high[2] + reach));
    writeVelocity(state, frame);
    BodyCoupling& coupling = couplings[body];
    coupling.centre = make_float3((float)state.position[0], (float)state.position[1], (float)state.position[2]);
    coupling.inverseMass = mass.liquidVoxelMass / mass.mass;
    double inverse[9];
    turned(r, mass.inverseInertia, inverse);
    for(int i = 0; i < 9; ++i){
        coupling.inverseInertia[i] = mass.liquidVoxelMass*inverse[i];
    }
}

//gravity's dt on each body, from where the last substep and the warm start (contacts.cu) left it
__global__ void predictBodies(int numBodies, BodyState* states, BodyFrame* frames, double dt, double3 gravity){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body >= numBodies){
        return;
    }
    BodyState& state = states[body];
    state.velocity[0] += dt*gravity.x;
    state.velocity[1] += dt*gravity.y;
    state.velocity[2] += dt*gravity.z;
    writeVelocity(state, frames[body]);
}

//the pressure's impulse on each body, s M~^-1 B^T p (in responses): onto its velocity, and its turning's onto its angular momentum
__global__ void takeBodyImpulses(int numBodies, BodyState* states, const BodyMass* masses, BodyFrame* frames, const double* responses){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body >= numBodies){
        return;
    }
    BodyState& state = states[body];
    const double* response = responses + BODY_LANES*body;
    for(int axis = 0; axis < 3; ++axis){
        state.velocity[axis] += response[axis];
        state.angularVelocity[axis] += response[3 + axis];
    }
    double r[9], inertia[9];
    rotationOf(state.orientation, r);
    turned(r, masses[body].inertia, inertia);
    times(inertia, state.angularVelocity, state.angularMomentum);
    writeVelocity(state, frames[body]);
}

//the end of a substep: each body goes dt along its velocity, and turns by w dt (exactly, as a quaternion: by |w| dt about w), each with its contacts'
//push out of what it's sunk into on top, which is gone after (contacts.cu); its angular momentum stays,
//and its angular velocity is what that momentum makes of its inertia where it's turned to now. How fast its surface's velocity changed over the substep
//is for the next one's timestep
__global__ void moveBodies(int numBodies, BodyState* states, const BodyMass* masses, double dt, double measureOver){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body >= numBodies){
        return;
    }
    BodyState& state = states[body];
    const BodyMass& mass = masses[body];
    double change = 0.0, turning = 0.0;
    double w[3];    //what turns it this substep: its angular velocity, and its contacts' push out of what it's sunk into
    for(int axis = 0; axis < 3; ++axis){
        change += (state.velocity[axis] - state.before[axis])*(state.velocity[axis] - state.before[axis]);
        turning += (state.angularVelocity[axis] - state.before[3 + axis])*(state.angularVelocity[axis] - state.before[3 + axis]);
        state.position[axis] += dt*(state.velocity[axis] + state.pseudo[axis]);
        w[axis] = state.angularVelocity[axis] + state.pseudo[3 + axis];
    }
    for(double& pushed : state.pseudo){
        pushed = 0.0;
    }
    state.acceleration = measureOver > 0.0 ? (sqrt(change) + sqrt(turning)*mass.reach) / measureOver : 0.0;
    double speed = sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]);
    if(speed > 0.0){
        double half = 0.5*speed*dt;
        double s = sin(half) / speed;
        double d[4] = {cos(half), s*w[0], s*w[1], s*w[2]};
        const double* q = state.orientation;
        double turned4[4] = {d[0]*q[0] - d[1]*q[1] - d[2]*q[2] - d[3]*q[3],
                             d[0]*q[1] + d[1]*q[0] + d[2]*q[3] - d[3]*q[2],
                             d[0]*q[2] - d[1]*q[3] + d[2]*q[0] + d[3]*q[1],
                             d[0]*q[3] + d[1]*q[2] - d[2]*q[1] + d[3]*q[0]};
        double length = sqrt(turned4[0]*turned4[0] + turned4[1]*turned4[1] + turned4[2]*turned4[2] + turned4[3]*turned4[3]);
        for(int i = 0; i < 4; ++i){
            state.orientation[i] = turned4[i] / length;
        }
    }
    double r[9], inverse[9];
    rotationOf(state.orientation, r);
    turned(r, mass.inverseInertia, inverse);
    times(inverse, state.angularMomentum, state.angularVelocity);
}

// ---- which body is across each face of an unknown ----

//Per near unknown, a block per node near obstacles: each face obstacleFaceFlux counts an obstacle's flow through, the body nearest its centre if that's a
//dynamic one, plus 1, in 5 bits a face (0: none). The same faces and the same lookups as obstacleFaceFlux, so the coupling's B is the flow the divergence
//took. Only this partition's own nodes: the sums and the products are over its own unknowns
struct BodiesOf{
    signed char of[MAX_OBSTACLES];  //per obstacle, its body, or -1
};

__global__ void findFaceBodiesKernel(Obstacles obstacles, BodiesOf bodies, VoxelPlaces places, uint3 domainVoxels, const uint* nodes, uint numOwnNodes, const char* solveCodes,
                                     const char* near, const uint* opens, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy,
                                     const uint* neighborNz, const uint* neighborPz, uint* faceBodies){
    for(uint listed = blockIdx.x; listed < nodes[0]; listed += gridDim.x){
        uint node = nodes[1 + listed];
        if(node >= numOwnNodes){
            continue;
        }
        uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
        uint last = places.nodeVoxelEnds[node];
        uint cell = places.nodeCells[node];
        const uint* lower[3] = {neighborNx, neighborNy, neighborNz};
        const uint* upper[3] = {neighborPx, neighborPy, neighborPz};
        uint size[3] = {domainVoxels.x, domainVoxels.y, domainVoxels.z};
        for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
            if(!solveCodes[index] || !(near[index] & NEAR_SURFACE)){
                continue;
            }
            int3 voxel = places.voxelOf(cell, places.voxelSlots[index]);
            int coordinates[3] = {voxel.x, voxel.y, voxel.z};
            CutFaces cut = cachedCut(opens, index);
            uint code = 0;
            for(int face = 0; face < 6; ++face){
                if(near[index] & CLOSED_FACE << face){
                    cut.open[face] = 0.0f;
                }
                int axis = face/2, side = face % 2;
                uint neighbor = (side ? upper : lower)[axis][index];
                int across = coordinates[axis] + (side ? 1 : -1);
                if(cut.open[face] == 1.0f || (neighbor == WALL_VOXEL && (across < 0 || across >= (int)size[axis]))){
                    continue;
                }
                float offset[3] = {0.5f, 0.5f, 0.5f};
                offset[axis] = side ? 1.0f : 0.0f;
                float distance;
                float3 normal;
                int obstacle = nearestObstacle(obstacles, places.point(voxel, offset[0], offset[1], offset[2]), distance, normal);
                int body = obstacle >= 0 ? bodies.of[obstacle] : -1;
                if(body >= 0){
                    code |= (uint)(body + 1) << 5*face;
                }
            }
            faceBodies[index] = code;
        }
    }
}

// ---- the coupling: B^T x, s M~^-1 of it, and B times that ----

//an own unknown's row of B for one body: over its faces that body's surface cuts, the solid part of each, the way the flow through it leaves the voxel,
//times what the body's motion does across it: sign (1 - open)(e, (x - c) x e), x the solid part's centroid, where the motion's flow through it and the
//pressure's push on it act (obstacleCentroids). In a fixed order of faces, rounded the same way in every pass that asks (no fused multiply-adds), so B^T
//and B are each other's transpose to the bit
__device__ inline bool couplingRow(const CouplingGrid& where, uint voxel, int3 at, int body, float3 centre, float row[BODY_LANES]){
    for(int lane = 0; lane < BODY_LANES; ++lane){
        row[lane] = 0.0f;
    }
    uint faces = where.faceBodies[voxel];
    bool any = false;
    float h = where.grid.cellSize / where.interiorWidth;
    float corner[3] = {(float)at.x, (float)at.y, (float)at.z};
    float origin[3] = {where.grid.negX, where.grid.negY, where.grid.negZ};
    float c[3] = {centre.x, centre.y, centre.z};
    for(int face = 0; face < 6; ++face){
        if((int)(faces >> 5*face & 31u) != body + 1){
            continue;
        }
        any = true;
        int axis = face/2, side = face % 2;
        uint word = where.opens[4*(size_t)voxel + axis];
        bool closed = where.near[voxel] & CLOSED_FACE << face;
        float open = closed ? 0.0f : (side ? word >> 16 : word & 0xFFFFu) / OPEN_FULL;
        float solid = __fsub_rn(1.0f, open);
        float flow = side ? solid : -solid;
        float2 centre = cachedCentroid(where.centroids, voxel, face, closed);
        float offsets[3];
        offsets[axis] = side ? 1.0f : 0.0f;
        offsets[(axis + 1) % 3] = centre.x;
        offsets[(axis + 2) % 3] = centre.y;
        float lever[3];
        for(int dim = 0; dim < 3; ++dim){   //from its centre of mass to the solid part's centroid
            lever[dim] = __fsub_rn(__fadd_rn(origin[dim], __fmul_rn(__fadd_rn(corner[dim], offsets[dim]), h)), c[dim]);
        }
        float turn[3] = {axis == 0 ? 0.0f : axis == 1 ? -lever[2] : lever[1],      //lever x e
                         axis == 0 ? lever[2] : axis == 1 ? 0.0f : -lever[0],
                         axis == 0 ? -lever[1] : axis == 1 ? lever[0] : 0.0f};
        row[axis] = __fadd_rn(row[axis], flow);
        for(int dim = 0; dim < 3; ++dim){
            row[3 + dim] = __fadd_rn(row[3 + dim], __fmul_rn(flow, turn[dim]));
        }
    }
    return any;
}

//an own node's interior slot's voxel and where it is in the domain, for a node near obstacles
__device__ inline uint slotVoxel(const CouplingGrid& where, uint node, int slot, int3& at){
    uint cell = where.nodeCells[node];
    int w = where.interiorWidth;
    at = make_int3((int)(cell % where.grid.sizeX)*w + slot % w, (int)(cell / where.grid.sizeX % where.grid.sizeY)*w + slot / w % w,
                   (int)(cell / (where.grid.sizeX*where.grid.sizeY))*w + slot / (w*w));
    uint voxel = where.interiorVoxels[(size_t)node*w*w*w + slot];
    return voxel != NO_VOXEL && where.solveCodes[voxel] && (where.near[voxel] & NEAR_SURFACE) && where.faceBodies[voxel] != 0 ? voxel : NO_VOXEL;
}

//B^T x into each body's exact sums: a warp per own node near obstacles, its lanes taking interior slots lane and lane + 32 (a node's 64), each body's six
//shares paired off in a fixed tree, then in exactly, as the CG's dot products are (addNodeDotShares): a node's share depends only on its own values,
//so the sums are the same however the nodes are stored or split between partitions
__global__ void addBodyShares(int numBodies, CouplingGrid where, const BodyCoupling* couplings, const float* x, const uint* onlyIf, ExactSum* sums){
    __shared__ ExactSum blockSums[MAX_OBSTACLES*BODY_LANES];
    if(onlyIf != nullptr && !*onlyIf){
        return;
    }
    for(int i = threadIdx.x; i < numBodies*BODY_LANES; i += blockDim.x){
        for(int digit = 0; digit < EXACT_DIGITS; ++digit){
            blockSums[i].digits[digit] = 0;
        }
        blockSums[i].nonFinite = 0;
    }
    __syncthreads();
    uint lane = threadIdx.x % 32;
    uint warps = gridDim.x*blockDim.x/32;
    for(uint listed = (threadIdx.x + blockIdx.x*blockDim.x)/32; listed < where.nodes[0]; listed += warps){     //the same for the whole warp
        uint node = where.nodes[1 + listed];
        if(node >= where.numOwnNodes){
            continue;
        }
        int3 at[2];
        uint voxels[2];
        uint present = 0;   //the bodies across this lane's voxels' faces, a bit each
        for(int half = 0; half < 2; ++half){
            voxels[half] = slotVoxel(where, node, lane + 32*half, at[half]);
            if(voxels[half] != NO_VOXEL){
                uint faces = where.faceBodies[voxels[half]];
                for(int face = 0; face < 6; ++face){
                    uint code = faces >> 5*face & 31u;
                    present |= code ? 1u << (code - 1) : 0u;
                }
            }
        }
        for(int lanes = 16; lanes > 0; lanes >>= 1){
            present |= __shfl_xor_sync(0xffffffffu, present, lanes);
        }
        for(uint mask = present; mask != 0; mask &= mask - 1){
            int body = __ffs(mask) - 1;
            float terms[2][BODY_LANES];
            for(int half = 0; half < 2; ++half){
                float row[BODY_LANES];
                bool any = voxels[half] != NO_VOXEL && couplingRow(where, voxels[half], at[half], body, couplings[body].centre, row);
                float value = any ? x[voxels[half]] : 0.0f;
                for(int k = 0; k < BODY_LANES; ++k){
                    terms[half][k] = any ? __fmul_rn(value, row[k]) : 0.0f;
                }
            }
            for(int k = 0; k < BODY_LANES; ++k){
                float share = __fadd_rn(terms[0][k], terms[1][k]);
                for(int lanes = 16; lanes > 0; lanes >>= 1){
                    share = __fadd_rn(share, __shfl_xor_sync(0xffffffffu, share, lanes));
                }
                if(lane == 0){
                    addExactly(share, blockSums[BODY_LANES*body + k]);
                }
            }
        }
    }
    __syncthreads();
    for(int i = threadIdx.x; i < numBodies*BODY_LANES; i += blockDim.x){
        for(int digit = 0; digit < EXACT_DIGITS; ++digit){
            if(blockSums[i].digits[digit] != 0){
                atomicAdd((unsigned long long*)sums[i].digits + digit, (unsigned long long)blockSums[i].digits[digit]);
            }
        }
        if(blockSums[i].nonFinite){
            atomicAdd((unsigned long long*)&sums[i].nonFinite, (unsigned long long)blockSums[i].nonFinite);
        }
    }
}

//each body's B^T x, every partition's, as doubles, and the response to it, s M~^-1 B^T x; after them, the energy (B^T x).(s M~^-1 B^T x)
__global__ void finishBodyShares(int numBodies, const ExactSum* sums, const BodyCoupling* couplings, double scale, const uint* onlyIf, double* responses){
    if(onlyIf != nullptr && !*onlyIf){
        return;
    }
    double energy = 0.0;
    for(int body = 0; body < numBodies; ++body){
        double shares[BODY_LANES];
        for(int k = 0; k < BODY_LANES; ++k){
            shares[k] = exactValue(sums[BODY_LANES*body + k]);
        }
        const BodyCoupling& coupling = couplings[body];
        double* response = responses + BODY_LANES*body;
        for(int axis = 0; axis < 3; ++axis){
            response[axis] = scale*coupling.inverseMass*shares[axis];
        }
        times(coupling.inverseInertia, shares + 3, response + 3);
        for(int axis = 0; axis < 3; ++axis){
            response[3 + axis] *= scale;
        }
        for(int k = 0; k < BODY_LANES; ++k){
            energy += shares[k]*response[k];
        }
    }
    responses[BODY_LANES*numBodies] = energy;
}

//y += sign B (s M~^-1 B^T x) on the own unknowns: each body's response, through its row, a voxel a lane
__global__ void applyBodyResponses(int numBodies, CouplingGrid where, const BodyCoupling* couplings, const double* responses, float sign, const uint* onlyIf, float* y){
    if(onlyIf != nullptr && !*onlyIf){
        return;
    }
    uint lane = threadIdx.x % 32;
    uint warps = gridDim.x*blockDim.x/32;
    for(uint listed = (threadIdx.x + blockIdx.x*blockDim.x)/32; listed < where.nodes[0]; listed += warps){
        uint node = where.nodes[1 + listed];
        if(node >= where.numOwnNodes){
            continue;
        }
        for(int half = 0; half < 2; ++half){
            int3 at;
            uint voxel = slotVoxel(where, node, lane + 32*half, at);
            if(voxel == NO_VOXEL){
                continue;
            }
            float sum = 0.0f;
            for(int body = 0; body < numBodies; ++body){
                float row[BODY_LANES];
                if(!couplingRow(where, voxel, at, body, couplings[body].centre, row)){
                    continue;
                }
                for(int k = 0; k < BODY_LANES; ++k){
                    sum = __fadd_rn(sum, __fmul_rn(row[k], (float)responses[BODY_LANES*body + k]));
                }
            }
            y[voxel] = __fadd_rn(y[voxel], __fmul_rn(sign, sum));
        }
    }
}

//the friction the substep's faces gave the particles, every partition's, taken off each body: its momentum off the velocity, its moment off the turning
__global__ void takeBodyFriction(int numBodies, BodyState* states, const BodyMass* masses, BodyFrame* frames, const BodyCoupling* couplings, const ExactSum* sums){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body >= numBodies){
        return;
    }
    BodyState& state = states[body];
    const BodyCoupling& coupling = couplings[body];
    double given[BODY_LANES];
    for(int lane = 0; lane < BODY_LANES; ++lane){
        given[lane] = exactValue(sums[BODY_LANES*body + lane]);
    }
    double turn[3];
    times(coupling.inverseInertia, given + 3, turn);
    for(int axis = 0; axis < 3; ++axis){
        state.velocity[axis] -= coupling.inverseMass*given[axis];
        state.angularVelocity[axis] -= turn[axis];
    }
    double r[9], inertia[9];
    rotationOf(state.orientation, r);
    turned(r, masses[body].inertia, inertia);
    times(inertia, state.angularVelocity, state.angularMomentum);
    writeVelocity(state, frames[body]);
}

__global__ void addBodyEnergy(int numBodies, const double* responses, double* total){
    *total += responses[BODY_LANES*numBodies];
}

//the term's norm, bounded by its trace: s times the sum over each body's unknowns of B_i M~^-1 B_i^T, in exact sums as addBodyShares takes them
__global__ void addBodyBounds(int numBodies, CouplingGrid where, const BodyCoupling* couplings, ExactSum* sums){
    __shared__ ExactSum blockSums[MAX_OBSTACLES];
    for(int i = threadIdx.x; i < numBodies; i += blockDim.x){
        for(int digit = 0; digit < EXACT_DIGITS; ++digit){
            blockSums[i].digits[digit] = 0;
        }
        blockSums[i].nonFinite = 0;
    }
    __syncthreads();
    uint lane = threadIdx.x % 32;
    uint warps = gridDim.x*blockDim.x/32;
    for(uint listed = (threadIdx.x + blockIdx.x*blockDim.x)/32; listed < where.nodes[0]; listed += warps){
        uint node = where.nodes[1 + listed];
        if(node >= where.numOwnNodes){
            continue;
        }
        int3 at[2];
        uint voxels[2];
        for(int half = 0; half < 2; ++half){
            voxels[half] = slotVoxel(where, node, lane + 32*half, at[half]);
        }
        for(int body = 0; body < numBodies; ++body){
            float terms[2];
            for(int half = 0; half < 2; ++half){
                float row[BODY_LANES];
                terms[half] = 0.0f;
                if(voxels[half] != NO_VOXEL && couplingRow(where, voxels[half], at[half], body, couplings[body].centre, row)){
                    const BodyCoupling& coupling = couplings[body];
                    double turn[3];
                    double lever[3] = {row[3], row[4], row[5]};
                    times(coupling.inverseInertia, lever, turn);
                    terms[half] = (float)(coupling.inverseMass*((double)row[0]*row[0] + (double)row[1]*row[1] + (double)row[2]*row[2]) + lever[0]*turn[0] + lever[1]*turn[1] +
                                          lever[2]*turn[2]);
                }
            }
            float share = __fadd_rn(terms[0], terms[1]);
            for(int lanes = 16; lanes > 0; lanes >>= 1){
                share = __fadd_rn(share, __shfl_xor_sync(0xffffffffu, share, lanes));
            }
            if(lane == 0){
                addExactly(share, blockSums[body]);
            }
        }
    }
    __syncthreads();
    for(int i = threadIdx.x; i < numBodies; i += blockDim.x){
        for(int digit = 0; digit < EXACT_DIGITS; ++digit){
            if(blockSums[i].digits[digit] != 0){
                atomicAdd((unsigned long long*)sums[i].digits + digit, (unsigned long long)blockSums[i].digits[digit]);
            }
        }
    }
}

__global__ void finishBodyBound(int numBodies, const ExactSum* sums, double scale, double* bound){
    double total = 0.0;
    for(int body = 0; body < numBodies; ++body){
        total += exactValue(sums[body]);
    }
    *bound = 1.01*scale*total;     //a little over, for the floats it was summed from
}

// ---- the host's side ----

static uint couplingBlocks(){
    int device, processors;
    gpuErrchk(cudaGetDevice(&device));
    gpuErrchk(cudaDeviceGetAttribute(&processors, cudaDevAttrMultiProcessorCount, device));
    return 2*processors;
}

void Bodies::release(){
    for(void* device : {(void*)states, (void*)deviceMasses, (void*)frames, (void*)couplings, (void*)sums, (void*)responses, (void*)frictionSums, (void*)boundSums,
                        (void*)bound, (void*)surfacePoints, (void*)surfaceBodies, (void*)picks, (void*)contacts, (void*)lastContacts, (void*)contactOrder,
                        (void*)batchStarts}){
        if(device != nullptr){
            cudaFree(device);
        }
    }
    for(void* host : {(void*)seen, (void*)seenBound}){
        if(host != nullptr){
            cudaFreeHost(host);
        }
    }
    states = nullptr;
    deviceMasses = nullptr;
    frames = nullptr;
    couplings = nullptr;
    sums = nullptr;
    responses = nullptr;
    frictionSums = nullptr;
    boundSums = nullptr;
    bound = nullptr;
    seen = nullptr;
    seenBound = nullptr;
    surfacePoints = nullptr;
    surfaceBodies = nullptr;
    numSurfacePoints = 0;
    picks = nullptr;
    contacts = nullptr;
    lastContacts = nullptr;
    contactOrder = nullptr;
    batchStarts = nullptr;
    numBodies = 0;
    obstacles.clear();
    names.clear();
    masses.clear();
}

Bodies::~Bodies(){
    release();
}

void Bodies::setUp(const std::vector<SceneObstacle>& descriptions, ObstacleSet& obstacleSet, double liquidDensity, double voxelSize, PartitionContext* partitionContext,
                   cudaStream_t onStream){
    release();
    context = partitionContext;
    stream = onStream;
    voxel = voxelSize;
    std::vector<BodyState> initial;
    for(size_t description = 0; description < descriptions.size(); ++description){
        const SceneObstacle& obstacle = descriptions[description];
        int built = obstacleSet.builtFrom(description);
        if(!obstacle.dynamic || built < 0){
            continue;
        }
        //its shape where its own space is the world's, sampled a quarter voxel apart over its bounds and a voxel past them (an open mesh's shell
        //reaches past its triangles), or coarser for a body so large that would be more than MAX_MASS_SAMPLES
        ObstacleState shape = obstacleSet.state().items[built];
        for(int i = 0; i < 12; ++i){
            shape.worldToObject[i] = i % 5 == 0 ? 1.0f : 0.0f;
        }
        shape.frame = nullptr;
        shape.reachLow = make_float3(-INFINITY, -INFINITY, -INFINITY);
        shape.reachHigh = make_float3(INFINITY, INFINITY, INFINITY);
        float3 objectLow, objectHigh;
        obstacleSet.objectBounds(built, objectLow, objectHigh);
        double low[3] = {objectLow.x - voxelSize - 0.5*obstacle.thickness, objectLow.y - voxelSize - 0.5*obstacle.thickness, objectLow.z - voxelSize - 0.5*obstacle.thickness};
        double high[3] = {objectHigh.x + voxelSize + 0.5*obstacle.thickness, objectHigh.y + voxelSize + 0.5*obstacle.thickness, objectHigh.z + voxelSize + 0.5*obstacle.thickness};
        double h = 0.25*voxelSize;
        double volume = (high[0] - low[0])*(high[1] - low[1])*(high[2] - low[2]);
        if(volume / (h*h*h) > MAX_MASS_SAMPLES){
            h = std::cbrt(volume / MAX_MASS_SAMPLES);
        }
        int3 samples = make_int3((int)std::ceil((high[0] - low[0]) / h), (int)std::ceil((high[1] - low[1]) / h), (int)std::ceil((high[2] - low[2]) / h));
        double3 middle = make_double3(0.5*(low[0] + high[0]), 0.5*(low[1] + high[1]), 0.5*(low[2] + high[2]));
        double* partials;
        double* moments;
        gpuErrchk(cudaMallocAsync((void**)&partials, sizeof(double)*MOMENTS*MASS_BLOCKS, stream));
        gpuErrchk(cudaMallocAsync((void**)&moments, sizeof(double)*MOMENTS, stream));
        integrateMass<<<MASS_BLOCKS, MASS_THREADS, 0, stream>>>(shape, make_double3(low[0], low[1], low[2]), h, samples, middle, partials);
        sumMassPartials<<<1, 32, 0, stream>>>(MASS_BLOCKS, partials, moments);
        gpuErrchk(cudaPeekAtLastError());
        double m[MOMENTS];
        gpuErrchk(cudaMemcpyAsync(m, moments, sizeof(m), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));   //once, at the start
        cudaFreeAsync(partials, stream);
        cudaFreeAsync(moments, stream);
        double cube = h*h*h;
        BodyMass mass = {};
        mass.volume = m[0]*cube;
        if(!(mass.volume > 0.0)){
            std::cerr<<"Bodies: obstacle "<<description<<" has nothing inside it to weigh; it's left still\n";
            continue;
        }
        //its density over its volume, or the mass given; its centre of mass, and its inertia about that: the sum of each sample's (|r|^2 I - r r^T), plus
        //each sample cube's own h^2/6 along every axis
        double density = obstacle.body.mass > 0.0 ? obstacle.body.mass / mass.volume : obstacle.body.density;
        mass.mass = density*mass.volume;
        double mean[3] = {m[1] / m[0], m[2] / m[0], m[3] / m[0]};
        for(int axis = 0; axis < 3; ++axis){
            mass.centre[axis] = (axis == 0 ? middle.x : axis == 1 ? middle.y : middle.z) + mean[axis];
        }
        double s[9] = {m[4] - m[0]*mean[0]*mean[0], m[7] - m[0]*mean[0]*mean[1], m[8] - m[0]*mean[0]*mean[2],
                       m[7] - m[0]*mean[0]*mean[1], m[5] - m[0]*mean[1]*mean[1], m[9] - m[0]*mean[1]*mean[2],
                       m[8] - m[0]*mean[0]*mean[2], m[9] - m[0]*mean[1]*mean[2], m[6] - m[0]*mean[2]*mean[2]};
        double trace = s[0] + s[4] + s[8];
        for(int row = 0; row < 3; ++row){
            for(int column = 0; column < 3; ++column){
                mass.inertia[3*row + column] = density*cube*((row == column ? trace + m[0]*h*h/6.0 : 0.0) - s[3*row + column]);
            }
        }
        if(!invert(mass.inertia, mass.inverseInertia)){
            std::cerr<<"Bodies: obstacle "<<description<<" has no inertia to speak of; it's left still\n";
            continue;
        }
        mass.objectLow = objectLow;
        mass.objectHigh = objectHigh;
        mass.reach = 0.0;
        for(int corner = 0; corner < 8; ++corner){
            double d[3] = {(corner & 1 ? objectHigh.x : objectLow.x) - mass.centre[0], (corner & 2 ? objectHigh.y : objectLow.y) - mass.centre[1],
                           (corner & 4 ? objectHigh.z : objectLow.z) - mass.centre[2]};
            mass.reach = std::max(mass.reach, std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]));
        }
        mass.liquidVoxelMass = liquidDensity*voxelSize*voxelSize*voxelSize;
        mass.bounce = (float)obstacle.body.bounce;
        mass.contactFriction = (float)obstacle.body.contactFriction;
        //where it starts: its transform's rotation and translation, which take its own space to the world, its centre of mass with them
        double translation[3], rotation[4];
        obstacleSet.restPlacement(built, translation, rotation);
        BodyState state = {};
        double r[9];
        rotationOf(rotation, r);
        for(int i = 0; i < 4; ++i){
            state.orientation[i] = rotation[i];
        }
        times(r, mass.centre, state.position);
        for(int axis = 0; axis < 3; ++axis){
            state.position[axis] += translation[axis];
            state.velocity[axis] = obstacle.body.velocity[axis];
            state.angularVelocity[axis] = obstacle.body.angularVelocity[axis];
        }
        double inertia[9];
        turned(r, mass.inertia, inertia);
        times(inertia, state.angularVelocity, state.angularMomentum);
        std::cerr<<"Body "<<masses.size()<<" (obstacle "<<description<<"): "<<mass.mass<<" kg, "<<mass.volume<<" m^3 (density "<<density<<" kg/m^3, "
                 <<density / liquidDensity<<" of the liquid's), centre of mass ("<<state.position[0]<<", "<<state.position[1]<<", "<<state.position[2]<<")\n";
        obstacles.push_back(built);
        names.push_back(obstacle.name);
        masses.push_back(mass);
        initial.push_back(state);
    }
    numBodies = (int)masses.size();
    if(numBodies == 0){
        return;
    }
    gpuErrchk(cudaMalloc((void**)&states, sizeof(BodyState)*numBodies));
    gpuErrchk(cudaMalloc((void**)&deviceMasses, sizeof(BodyMass)*numBodies));
    gpuErrchk(cudaMalloc((void**)&frames, sizeof(BodyFrame)*numBodies));
    gpuErrchk(cudaMalloc((void**)&couplings, sizeof(BodyCoupling)*numBodies));
    gpuErrchk(cudaMalloc((void**)&sums, sizeof(ExactSum)*BODY_LANES*numBodies));
    gpuErrchk(cudaMalloc((void**)&responses, sizeof(double)*(BODY_LANES*numBodies + 1)));
    gpuErrchk(cudaMalloc((void**)&boundSums, sizeof(ExactSum)*numBodies));
    gpuErrchk(cudaMalloc((void**)&frictionSums, sizeof(ExactSum)*BODY_LANES*numBodies));
    gpuErrchk(cudaMemsetAsync(frictionSums, 0, sizeof(ExactSum)*BODY_LANES*numBodies, stream));
    gpuErrchk(cudaMalloc((void**)&bound, sizeof(double)));
    gpuErrchk(cudaMallocHost((void**)&seen, sizeof(BodyState)*numBodies));
    gpuErrchk(cudaMallocHost((void**)&seenBound, sizeof(double)));
    *seenBound = 0.0;
    std::memcpy(seen, initial.data(), sizeof(BodyState)*numBodies);
    gpuErrchk(cudaMemcpyAsync(states, seen, sizeof(BodyState)*numBodies, cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(deviceMasses, masses.data(), sizeof(BodyMass)*numBodies, cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemsetAsync(responses, 0, sizeof(double)*(BODY_LANES*numBodies + 1), stream));
    gpuErrchk(cudaStreamSynchronize(stream));   //before seen is written again, and masses' copy is gone
    for(int body = 0; body < numBodies; ++body){
        obstacleSet.setFrame(obstacles[body], frames + body);
    }
    setUpContacts(obstacleSet, voxelSize);
    writeFrames();
}

void Bodies::writeFrames(){
    if(numBodies == 0){
        return;
    }
    writeBodyFrames<<<1, 32, 0, stream>>>(numBodies, states, deviceMasses, frames, couplings, voxel);
    gpuErrchk(cudaPeekAtLastError());
}

void Bodies::predict(double dt, float3 gravity){
    if(numBodies == 0){
        return;
    }
    predictBodies<<<1, 32, 0, stream>>>(numBodies, states, frames, dt, make_double3(gravity.x, gravity.y, gravity.z));
    gpuErrchk(cudaPeekAtLastError());
}

void Bodies::findFaceBodies(const Obstacles& obstacleState, const VoxelPlaces& places, uint3 domainVoxels, const uint* nodes, uint numOwnNodes, const char* solveCodes,
                            const char* near, const uint* opens, const uint* const neighbors[6], uint numVoxels){
    if(numBodies == 0){
        return;
    }
    faceBodyCodes.resizeAsync(numVoxels, stream);
    if(numVoxels == 0){
        return;
    }
    BodiesOf of;
    for(int obstacle = 0; obstacle < MAX_OBSTACLES; ++obstacle){
        of.of[obstacle] = -1;
    }
    for(int body = 0; body < numBodies; ++body){
        of.of[obstacles[body]] = (signed char)body;
    }
    uint* faceBodies = faceBodyCodes.devPtr();
    gpuErrchk(cudaMemsetAsync(faceBodies, 0, sizeof(uint)*numVoxels, stream));
    findFaceBodiesKernel<<<couplingBlocks(), 512, 0, stream>>>(obstacleState, of, places, domainVoxels, nodes, numOwnNodes, solveCodes, near, opens, neighbors[0], neighbors[1],
        neighbors[2], neighbors[3], neighbors[4], neighbors[5], faceBodies);
    gpuErrchk(cudaPeekAtLastError());
}

void Bodies::prepare(double solveScale, const uint* nodes, uint numOwnNodes, const uint* nodeCells, const uint* interiorVoxels, const char* solveCodes, const char* near,
                     const uint* opens, const uint* centroids, const Grid& grid, int interiorWidth, bool bounded){
    if(numBodies == 0){
        return;
    }
    scale = solveScale;
    where.nodes = nodes;
    where.numOwnNodes = numOwnNodes;
    where.nodeCells = nodeCells;
    where.interiorVoxels = interiorVoxels;
    where.solveCodes = solveCodes;
    where.near = near;
    where.opens = opens;
    where.centroids = centroids;
    where.faceBodies = faceBodyCodes.devPtr();
    where.grid = grid;
    where.interiorWidth = interiorWidth;
    if(bounded){
        gpuErrchk(cudaMemsetAsync(boundSums, 0, sizeof(ExactSum)*numBodies, stream));
        addBodyBounds<<<couplingBlocks(), COUPLING_THREADS, 0, stream>>>(numBodies, where, couplings, boundSums);
        context->sumOverPartitions((long long*)boundSums, numBodies*EXACT_WORDS, stream);
        finishBodyBound<<<1, 1, 0, stream>>>(numBodies, boundSums, scale, bound);
        gpuErrchk(cudaMemcpyAsync(seenBound, bound, sizeof(double), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaPeekAtLastError());
    }
}

void Bodies::sumShares(const float* x, const uint* onlyIf){
    gpuErrchk(cudaMemsetAsync(sums, 0, sizeof(ExactSum)*BODY_LANES*numBodies, stream));
    addBodyShares<<<couplingBlocks(), COUPLING_THREADS, 0, stream>>>(numBodies, where, couplings, x, onlyIf, sums);
    context->sumOverPartitions((long long*)sums, BODY_LANES*numBodies*EXACT_WORDS, stream);
    finishBodyShares<<<1, 1, 0, stream>>>(numBodies, sums, couplings, scale, onlyIf, responses);
    gpuErrchk(cudaPeekAtLastError());
}

void Bodies::addTimes(const float* x, float* y, float sign, const uint* onlyIf, cudaStream_t onStream){
    (void)onStream;     //the bodies' own, which is the solve's
    sumShares(x, onlyIf);
    applyBodyResponses<<<couplingBlocks(), COUPLING_THREADS, 0, stream>>>(numBodies, where, couplings, responses, sign, onlyIf, y);
    gpuErrchk(cudaPeekAtLastError());
}

void Bodies::addLastEnergy(double* total, cudaStream_t onStream){
    (void)onStream;
    addBodyEnergy<<<1, 1, 0, stream>>>(numBodies, responses, total);
    gpuErrchk(cudaPeekAtLastError());
}

float Bodies::normBound() const{
    return (float)*seenBound;
}

void Bodies::takePressure(const float* p){
    if(numBodies == 0){
        return;
    }
    sumShares(p, nullptr);
    takeBodyImpulses<<<1, 32, 0, stream>>>(numBodies, states, deviceMasses, frames, responses);
    gpuErrchk(cudaPeekAtLastError());
}

void Bodies::beginFriction(){
    if(numBodies > 0){
        gpuErrchk(cudaMemsetAsync(frictionSums, 0, sizeof(ExactSum)*BODY_LANES*numBodies, stream));
    }
}

BodyFriction Bodies::frictionFor(float share, double particleMass, const float* weightsX, const float* weightsY, const float* weightsZ){
    BodyFriction friction = {};
    for(int obstacle = 0; obstacle < MAX_OBSTACLES; ++obstacle){
        friction.of[obstacle] = -1;
    }
    if(numBodies == 0 || share == 0.0f){
        return friction;
    }
    friction.sums = frictionSums;
    friction.couplings = couplings;
    for(int body = 0; body < numBodies; ++body){
        friction.of[obstacles[body]] = (signed char)body;
    }
    friction.weights[0] = weightsX;
    friction.weights[1] = weightsY;
    friction.weights[2] = weightsZ;
    friction.scale = (float)(share*particleMass);
    return friction;
}

void Bodies::takeFriction(){
    if(numBodies == 0){
        return;
    }
    context->sumOverPartitions((long long*)frictionSums, BODY_LANES*numBodies*EXACT_WORDS, stream);
    takeBodyFriction<<<1, 32, 0, stream>>>(numBodies, states, deviceMasses, frames, couplings, frictionSums);
    gpuErrchk(cudaPeekAtLastError());
}

void Bodies::moveOnce(double dt, double measureOver){
    moveBodies<<<1, 32, 0, stream>>>(numBodies, states, deviceMasses, dt, measureOver);
    gpuErrchk(cudaPeekAtLastError());
}

float Bodies::fastest() const{
    double fastest = 0.0;
    for(int body = 0; body < numBodies; ++body){
        const BodyState& state = seen[body];
        double v = std::sqrt(state.velocity[0]*state.velocity[0] + state.velocity[1]*state.velocity[1] + state.velocity[2]*state.velocity[2]);
        double w = std::sqrt(state.angularVelocity[0]*state.angularVelocity[0] + state.angularVelocity[1]*state.angularVelocity[1] +
                             state.angularVelocity[2]*state.angularVelocity[2]);
        fastest = std::max(fastest, v + w*masses[body].reach);
    }
    return (float)fastest;
}

float Bodies::acceleration() const{
    double most = 0.0;
    for(int body = 0; body < numBodies; ++body){
        most = std::max(most, seen[body].acceleration);
    }
    return (float)most;
}

// ---- the cache ----

static_assert(offsetof(BodyState, position) == 0 && offsetof(BodyState, orientation) == 3*sizeof(double) && offsetof(BodyState, velocity) == 7*sizeof(double) &&
              offsetof(BodyState, angularVelocity) == 10*sizeof(double) && offsetof(BodyState, angularMomentum) == BODY_PLACEMENT*sizeof(double),
              "a cache frame's body is the doubles BodyState starts with: position, orientation, velocity, angularVelocity (cacheWriter.hu)");

void Bodies::copyPlacements(double* host){
    if(numBodies == 0){
        return;
    }
    gpuErrchk(cudaMemcpy2DAsync(host, BODY_PLACEMENT*sizeof(double), states, sizeof(BodyState), BODY_PLACEMENT*sizeof(double), numBodies, cudaMemcpyDeviceToHost, stream));
}

// ---- checkpoints ----

//a field of a body's state or of a contact, as a checkpoint's record has it: its name, where it is, how many values, and of what (doubles, floats, ints)
struct SavedField{
    const char* name;
    size_t offset;
    int count;
    char type;  //'d', 'f' or 'i'
};

//Every field of both, so a resume carries on from exactly what the substep after the checkpoint reads: the warm start's and the carry's of the contacts
//(warmStartBodies, carryContacts), the state's, and the host's copy of it, which the timestep and the contact steps go by. A contact's slot is its
//partner (the domain's 6 walls, then the obstacles) and its place among that partner's CONTACT_SLOTS, not its index, which the partners' count would change
static constexpr SavedField STATE_FIELDS[] = {
    {"position", offsetof(BodyState, position), 3, 'd'}, {"orientation", offsetof(BodyState, orientation), 4, 'd'}, {"velocity", offsetof(BodyState, velocity), 3, 'd'},
    {"angularVelocity", offsetof(BodyState, angularVelocity), 3, 'd'}, {"angularMomentum", offsetof(BodyState, angularMomentum), 3, 'd'},
    {"before", offsetof(BodyState, before), BODY_LANES, 'd'}, {"acceleration", offsetof(BodyState, acceleration), 1, 'd'}, {"pseudo", offsetof(BodyState, pseudo), BODY_LANES, 'd'},
    {"held", offsetof(BodyState, held), BODY_LANES, 'd'}, {"perSecond", offsetof(BodyState, perSecond), BODY_LANES, 'd'}};
static constexpr SavedField CONTACT_FIELDS[] = {
    {"point", offsetof(BodyContact, point), 1, 'i'}, {"other", offsetof(BodyContact, other), 1, 'i'}, {"at", offsetof(BodyContact, at), 3, 'f'},
    {"normal", offsetof(BodyContact, normal), 3, 'f'}, {"depth", offsetof(BodyContact, depth), 1, 'f'}, {"tangents", offsetof(BodyContact, tangents), 6, 'f'},
    {"partnerVelocity", offsetof(BodyContact, partnerVelocity), 3, 'f'}, {"lever", offsetof(BodyContact, lever), 6, 'f'}, {"mass", offsetof(BodyContact, mass), 3, 'f'},
    {"friction", offsetof(BodyContact, friction), 1, 'f'}, {"target", offsetof(BodyContact, target), 1, 'f'}, {"bias", offsetof(BodyContact, bias), 1, 'f'},
    {"impulse", offsetof(BodyContact, impulse), 3, 'f'}, {"applied", offsetof(BodyContact, applied), 1, 'f'}, {"pseudo", offsetof(BodyContact, pseudo), 1, 'f'},
    {"local", offsetof(BodyContact, local), 3, 'f'}, {"dt", offsetof(BodyContact, dt), 1, 'f'}, {"color", offsetof(BodyContact, color), 1, 'i'}};

template<size_t N>
static constexpr size_t savedBytes(const SavedField (&fields)[N]){
    size_t bytes = 0;
    for(size_t i = 0; i < N; ++i){
        bytes += fields[i].count*(fields[i].type == 'd' ? sizeof(double) : sizeof(float));
    }
    return bytes;
}
static_assert(savedBytes(STATE_FIELDS) == sizeof(BodyState), "a field of BodyState that checkpoints don't keep: add it to STATE_FIELDS");
static_assert(savedBytes(CONTACT_FIELDS) == sizeof(BodyContact), "a field of BodyContact that checkpoints don't keep: add it to CONTACT_FIELDS");

//the fields of the struct at base as JSON, "name":value or "name":[values]: doubles and floats to the bit (%.17g reads back as the same double, and a
//float's double is the float itself), ints as they are, and what isn't finite as null (only a body that's already blown up has any)
static void writeFields(std::string& out, const char* base, const SavedField* fields, size_t numFields){
    char text[32];
    for(size_t f = 0; f < numFields; ++f){
        const SavedField& field = fields[f];
        out += std::string(f ? ",\"" : "\"") + field.name + "\":" + (field.count > 1 ? "[" : "");
        const char* at = base + field.offset;
        for(int i = 0; i < field.count; ++i){
            double value = field.type == 'd' ? ((const double*)at)[i] : field.type == 'f' ? (double)((const float*)at)[i] : (double)((const int*)at)[i];
            if(field.type == 'i'){
                std::snprintf(text, sizeof(text), "%d", ((const int*)at)[i]);
            }
            else if(std::isfinite(value)){
                std::snprintf(text, sizeof(text), "%.17g", value);
            }
            else{
                std::snprintf(text, sizeof(text), "null");
            }
            out += (i ? "," : "") + std::string(text);
        }
        out += field.count > 1 ? "]" : "";
    }
}

//and back into the struct at base: null as NaN, and a field the record doesn't have keeps what base had
static void readFields(const Json& record, char* base, const SavedField* fields, size_t numFields, const std::string& where){
    for(size_t f = 0; f < numFields; ++f){
        const SavedField& field = fields[f];
        const Json* value = record.find(field.name);
        if(value == nullptr){
            continue;
        }
        bool fits = field.count > 1 ? value->kind == Json::ARRAY && (int)value->items.size() == field.count : true;
        for(int i = 0; fits && i < field.count; ++i){
            const Json& item = field.count > 1 ? value->items[i] : *value;
            fits = item.kind == Json::NUMBER || (item.kind == Json::NUL && field.type != 'i');
        }
        if(!fits){
            throw std::runtime_error(where + ": \"" + field.name + "\" isn't " + (field.count > 1 ? std::to_string(field.count) + " numbers" : "a number"));
        }
        char* at = base + field.offset;
        for(int i = 0; i < field.count; ++i){
            const Json& item = field.count > 1 ? value->items[i] : *value;
            double number = item.kind == Json::NUMBER ? item.number : NAN;
            if(field.type == 'd'){
                ((double*)at)[i] = number;
            }
            else if(field.type == 'f'){
                ((float*)at)[i] = (float)number;
            }
            else{
                ((int*)at)[i] = (int)number;
            }
        }
    }
}

std::string Bodies::checkpoint(){
    if(numBodies == 0){
        return "";
    }
    int slots = numBodies*CONTACT_PARTNERS*CONTACT_SLOTS;
    std::vector<BodyState> now(numBodies);
    std::vector<BodyContact> carried(slots);    //the last contact step's, which the next substep starts from (touch swaps them in)
    gpuErrchk(cudaMemcpyAsync(now.data(), states, sizeof(BodyState)*numBodies, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(carried.data(), lastContacts, sizeof(BodyContact)*slots, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    std::string out = "[";
    for(int body = 0; body < numBodies; ++body){
        out += std::string(body ? ",\n  " : "\n  ") + "{\"obstacle\":" + std::to_string(obstacles[body]) + ",";
        writeFields(out, (const char*)&now[body], STATE_FIELDS, std::size(STATE_FIELDS));
        out += ",\"contacts\":[";
        int written = 0;
        for(int slot = body*CONTACT_PARTNERS*CONTACT_SLOTS; slot < (body + 1)*CONTACT_PARTNERS*CONTACT_SLOTS; ++slot){
            if(carried[slot].point < 0){
                continue;   //an empty slot, which nothing reads
            }
            out += std::string(written++ ? "," : "") + "{\"partner\":" + std::to_string(slot / CONTACT_SLOTS % CONTACT_PARTNERS) + ",\"slot\":" +
                   std::to_string(slot % CONTACT_SLOTS) + ",";
            writeFields(out, (const char*)&carried[slot], CONTACT_FIELDS, std::size(CONTACT_FIELDS));
            out += "}";
        }
        out += "]}";
    }
    return out + "]";
}

std::string Bodies::restore(const Json& saved){
    if(saved.kind != Json::ARRAY || (int)saved.items.size() != numBodies){
        return "its checkpoint has " + std::to_string(saved.items.size()) + " dynamic bodies, and the scene has " + std::to_string(numBodies);
    }
    if(numBodies == 0){
        return "";
    }
    int slots = numBodies*CONTACT_PARTNERS*CONTACT_SLOTS;
    std::vector<BodyState> restored(seen, seen + numBodies);    //where setUp started them, for any field the record doesn't have
    std::vector<BodyContact> carried(slots);
    for(BodyContact& contact : carried){
        contact = BodyContact{};
        contact.point = -1;
    }
    try{
        for(int body = 0; body < numBodies; ++body){
            const Json& record = saved.items[body];
            std::string where = "its checkpoint's body " + std::to_string(body);
            const Json* obstacle = record.find("obstacle");
            if(obstacle == nullptr || obstacle->kind != Json::NUMBER || (int)obstacle->number != obstacles[body]){
                return where + " is another of the scene's obstacles than the scene's body " + std::to_string(body);
            }
            readFields(record, (char*)&restored[body], STATE_FIELDS, std::size(STATE_FIELDS), where);
            const Json* contacts = record.find("contacts");
            for(size_t index = 0; contacts != nullptr && index < contacts->items.size(); ++index){
                const Json& entry = contacts->items[index];
                std::string which = where + "'s contact " + std::to_string(index);
                const Json* partner = entry.find("partner");
                const Json* k = entry.find("slot");
                if(partner == nullptr || k == nullptr || partner->kind != Json::NUMBER || k->kind != Json::NUMBER || partner->number < 0 ||
                   partner->number >= CONTACT_PARTNERS || k->number < 0 || k->number >= CONTACT_SLOTS){
                    return which + " has no partner and slot this build has";
                }
                BodyContact& contact = carried[(body*CONTACT_PARTNERS + (int)partner->number)*CONTACT_SLOTS + (int)k->number];
                readFields(entry, (char*)&contact, CONTACT_FIELDS, std::size(CONTACT_FIELDS), which);
                if(contact.point < 0){
                    return which + " has no surface point";
                }
            }
        }
    }
    catch(const std::exception& error){
        return error.what();
    }
    std::memcpy(seen, restored.data(), sizeof(BodyState)*numBodies);   //what the first substep's timestep and contact steps go by, as the checkpoint's last left it
    gpuErrchk(cudaMemcpyAsync(states, seen, sizeof(BodyState)*numBodies, cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(lastContacts, carried.data(), sizeof(BodyContact)*slots, cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaStreamSynchronize(stream));   //once, on resuming: before seen is written again, and carried is gone
    return "";
}
