//Copyright 2023 Aberrant Behavior LLC

#include "forces.hu"
#include <algorithm>
#include <cmath>

//Gravity, as applyGravity and removeGravity did it on y, on whichever axes it has. The same expressions, so the same bits
__global__ void addGravity(uint numVoxels, float3 gravity, float dt, const char* solids, float* ux, float* uy, float* uz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels && !solids[index]){
        if(gravity.x != 0.0f){
            ux[index] += gravity.x*dt;
        }
        if(gravity.y != 0.0f){
            uy[index] += gravity.y*dt;
        }
        if(gravity.z != 0.0f){
            uz[index] += gravity.z*dt;
        }
    }
}

// ---- turbulence: curl noise ----

__device__ inline uint hashCorner(int x, int y, int z, uint seed){
    uint h = seed ^ (uint)x*0x8da6b343u ^ (uint)y*0xd8163841u ^ (uint)z*0xcb1ab31fu;
    h = (h ^ (h >> 16))*0x7feb352du;
    h = (h ^ (h >> 15))*0x846ca68bu;
    return h ^ (h >> 16);
}

//one of Perlin's 12 edge gradients, picked by a hash of the lattice corner
__device__ inline float3 cornerGradient(int x, int y, int z, uint seed){
    switch(hashCorner(x, y, z, seed) % 12){
        case 0: return make_float3(1.0f, 1.0f, 0.0f);
        case 1: return make_float3(-1.0f, 1.0f, 0.0f);
        case 2: return make_float3(1.0f, -1.0f, 0.0f);
        case 3: return make_float3(-1.0f, -1.0f, 0.0f);
        case 4: return make_float3(1.0f, 0.0f, 1.0f);
        case 5: return make_float3(-1.0f, 0.0f, 1.0f);
        case 6: return make_float3(1.0f, 0.0f, -1.0f);
        case 7: return make_float3(-1.0f, 0.0f, -1.0f);
        case 8: return make_float3(0.0f, 1.0f, 1.0f);
        case 9: return make_float3(0.0f, -1.0f, 1.0f);
        case 10: return make_float3(0.0f, 1.0f, -1.0f);
        default: return make_float3(0.0f, -1.0f, -1.0f);
    }
}

__device__ inline float dot3(float3 a, float3 b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

//the gradient of Perlin noise at p, analytically: the corners' gradients blended by the quintic fade, plus the fade's own slope times the corners' values
__device__ float3 noiseGradient(float3 p, uint seed){
    int3 cell = make_int3((int)floorf(p.x), (int)floorf(p.y), (int)floorf(p.z));
    float3 f = make_float3(p.x - cell.x, p.y - cell.y, p.z - cell.z);
    float3 u = make_float3(f.x*f.x*f.x*(f.x*(f.x*6.0f - 15.0f) + 10.0f), f.y*f.y*f.y*(f.y*(f.y*6.0f - 15.0f) + 10.0f), f.z*f.z*f.z*(f.z*(f.z*6.0f - 15.0f) + 10.0f));
    float3 du = make_float3(30.0f*f.x*f.x*(f.x - 1.0f)*(f.x - 1.0f), 30.0f*f.y*f.y*(f.y - 1.0f)*(f.y - 1.0f), 30.0f*f.z*f.z*(f.z - 1.0f)*(f.z - 1.0f));
    float3 g[8];
    float v[8];
    #pragma unroll
    for(int corner = 0; corner < 8; ++corner){  //x fastest: a, b, c, d on z = 0, then e, f, g, h
        int dx = corner & 1, dy = corner >> 1 & 1, dz = corner >> 2;
        g[corner] = cornerGradient(cell.x + dx, cell.y + dy, cell.z + dz, seed);
        v[corner] = dot3(g[corner], make_float3(f.x - dx, f.y - dy, f.z - dz));
    }
    float k1 = v[1] - v[0], k2 = v[2] - v[0], k3 = v[4] - v[0];
    float k4 = v[0] - v[1] - v[2] + v[3];
    float k5 = v[0] - v[2] - v[4] + v[6];
    float k6 = v[0] - v[1] - v[4] + v[5];
    float k7 = -v[0] + v[1] + v[2] - v[3] + v[4] - v[5] - v[6] + v[7];
    float3 gradient;
    float* out = &gradient.x;
    #pragma unroll
    for(int axis = 0; axis < 3; ++axis){
        float a = (&g[0].x)[axis], b = (&g[1].x)[axis], c = (&g[2].x)[axis], d = (&g[3].x)[axis];
        float e = (&g[4].x)[axis], ff = (&g[5].x)[axis], gg = (&g[6].x)[axis], h = (&g[7].x)[axis];
        out[axis] = a + u.x*(b - a) + u.y*(c - a) + u.z*(e - a) + u.x*u.y*(a - b - c + d) + u.y*u.z*(a - c - e + gg) + u.z*u.x*(a - b - e + ff)
                  + u.x*u.y*u.z*(-a + b + c - d + e - ff - gg + h);
    }
    gradient.x += du.x*(k1 + u.y*k4 + u.z*k6 + u.y*u.z*k7);
    gradient.y += du.y*(k2 + u.x*k4 + u.z*k5 + u.x*u.z*k7);
    gradient.z += du.z*(k3 + u.y*k5 + u.x*k6 + u.x*u.y*k7);
    return gradient;
}

//the curl of a vector potential whose 3 components are independent noises: a divergence-free swirl, about 1 in size
__device__ float3 curlNoise(float3 p, uint seed){
    float3 gx = noiseGradient(p, seed*3 + 0x1234567u);
    float3 gy = noiseGradient(make_float3(p.x + 31.4f, p.y + 15.9f, p.z + 26.5f), seed*3 + 0x2345678u);
    float3 gz = noiseGradient(make_float3(p.x - 27.1f, p.y + 82.8f, p.z - 18.3f), seed*3 + 0x3456789u);
    return make_float3(gz.y - gy.z, gx.z - gz.x, gy.x - gx.y);
}

//The most a component of curlNoise can be, whatever gradients the hash deals the corners. One noise's derivative along an axis is linear in its eight
//corner gradients, each any of the 12 edge vectors, so at a point of the cell the most it can be is, summed over the corners, the two largest
//components of what multiplies that corner's gradient: 3.75 at most, at the cell's middle. A component of the curl is the difference of two noises'
//derivatives, sampled the potentials' fixed offsets apart, which keeps both from their worst at once: 6.473 at most. (Its components' rms is 1, and
//the largest over 2e8 points and 48 seeds was 5.13)
static constexpr float CURL_NOISE_MOST = 6.48f;

// ---- the fields ----

//(1 - r/radius)^falloff inside radius, 0 beyond it; 1 everywhere with no radius
__device__ inline float fade(float r, float radius, float falloff){
    return radius <= 0.0f ? 1.0f : r >= radius ? 0.0f : powf(1.0f - r/radius, falloff);
}

//a volume's vector at p: trilinear between the 8 samples around it, as sdfVelocity reads a level set's velocities. Where the volume has nothing, past
//its bounds, in its bricks without samples or at samples past its active voxels, it's 0, and coverage is how much of the weight fell on what is the
//volume's own: a velocity of 0 can then be told from no velocity at all
__device__ inline float3 volumeVector(const SolidSDF& volume, float3 p, float& coverage){
    const int samples = SDF_BRICK*SDF_BRICK*SDF_BRICK;
    float q[3] = {(p.x - volume.origin.x) / volume.spacing, (p.y - volume.origin.y) / volume.spacing, (p.z - volume.origin.z) / volume.spacing};
    int i[3];
    float f[3];
    #pragma unroll
    for(int axis = 0; axis < 3; ++axis){
        i[axis] = (int)floorf(q[axis]);
        f[axis] = q[axis] - i[axis];
    }
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    coverage = 0.0f;
    if(i[0] < -1 || i[1] < -1 || i[2] < -1 || i[0] >= volume.bricks.x*SDF_BRICK || i[1] >= volume.bricks.y*SDF_BRICK || i[2] >= volume.bricks.z*SDF_BRICK){
        return sum;     //well outside it: most faces, for a small volume
    }
    #pragma unroll
    for(int corner = 0; corner < 8; ++corner){
        int x = i[0] + (corner & 1), y = i[1] + (corner >> 1 & 1), z = i[2] + (corner >> 2);
        if(x < 0 || y < 0 || z < 0 || x >= volume.bricks.x*SDF_BRICK || y >= volume.bricks.y*SDF_BRICK || z >= volume.bricks.z*SDF_BRICK){
            continue;
        }
        int entry = volume.table[x/SDF_BRICK + volume.bricks.x*(y/SDF_BRICK + volume.bricks.y*(z/SDF_BRICK))];
        if(entry < 0){
            continue;
        }
        float weight = (corner & 1 ? f[0] : 1.0f - f[0])*(corner >> 1 & 1 ? f[1] : 1.0f - f[1])*(corner >> 2 ? f[2] : 1.0f - f[2]);
        int within = x%SDF_BRICK + SDF_BRICK*(y%SDF_BRICK + SDF_BRICK*(z%SDF_BRICK));
        size_t at = (size_t)entry*3*samples + within;
        sum.x += weight*volume.velocities[at];
        sum.y += weight*volume.velocities[at + samples];
        sum.z += weight*volume.velocities[at + 2*samples];
        coverage += weight*volume.pool[(size_t)entry*samples + within];
    }
    return sum;
}

//a field's acceleration of the dim face at point over a substep of dt, which has velocity before any force and lies depth voxels inside the fluid's
//footprint
__device__ float fieldAcceleration(const ForceField& field, int dim, float3 point, float time, float dt, float before, int depth){
    switch(field.kind){
        case FORCE_POINT:{
            float3 towards = make_float3(field.position.x - point.x, field.position.y - point.y, field.position.z - point.z);
            float r = sqrtf(dot3(towards, towards));
            return r > 1e-6f ? field.strength*fade(r, field.radius, field.falloff)*(&towards.x)[dim]/r : 0.0f;
        }
        case FORCE_VORTEX:{
            float3 offset = make_float3(point.x - field.position.x, point.y - field.position.y, point.z - field.position.z);
            float along = dot3(offset, field.axis);
            float3 radial = make_float3(offset.x - along*field.axis.x, offset.y - along*field.axis.y, offset.z - along*field.axis.z);
            float r = sqrtf(dot3(radial, radial));
            float3 around = make_float3(field.axis.y*radial.z - field.axis.z*radial.y, field.axis.z*radial.x - field.axis.x*radial.z, field.axis.x*radial.y - field.axis.y*radial.x);
            return r > 1e-6f ? field.strength*fade(r, field.radius, field.falloff)*(&around.x)[dim]/r : 0.0f;
        }
        case FORCE_TURBULENCE:{
            float drift = time*field.speed;
            float3 p = make_float3(point.x/field.scale + 0.31f*drift, point.y/field.scale + 0.83f*drift, point.z/field.scale + 0.47f*drift);
            float3 swirl = curlNoise(p, field.seed);
            return field.strength*(&swirl.x)[dim];
        }
        case FORCE_VOLUME:{
            float coverage;
            float3 vector = volumeVector(field.volume, point, coverage);
            if(field.mode == VOLUME_FORCE){
                return field.strength*(&vector.x)[dim];
            }
            //towards the volume's velocity where it has one, by as much of the gap as drag closes in dt, so a strong drag over a long substep can't
            //overshoot it; from the velocity before any force, as wind is, so taking it back off is exact
            return -expm1f(-field.drag*dt)/dt*(field.strength*(&vector.x)[dim] - coverage*before);
        }
        default:    //FORCE_WIND
            return depth <= field.depth ? field.drag*((&field.velocity.x)[dim] - before) : 0.0f;
    }
}

//a block per node storing voxels: every field's acceleration of each face of the node's voxels outside the walls
__global__ void addForceFields(Forces forces, float dt, float time, ForceVoxels voxels){
    uint node = blockIdx.x;
    uint first = node == 0 ? 0 : voxels.nodeVoxelEnds[node - 1];
    uint last = voxels.nodeVoxelEnds[node];
    const Grid& grid = voxels.grid;
    int interiorWidth = 2<<voxels.refinementLevel;
    int voxels1D = interiorWidth + 2*voxels.apronCells;
    float voxelSize = grid.cellSize / interiorWidth;
    uint cell = voxels.nodeCells[node];
    int3 origin = make_int3((int)(cell % grid.sizeX)*interiorWidth - voxels.apronCells, (int)(cell / grid.sizeX % grid.sizeY)*interiorWidth - voxels.apronCells,
                            (int)(cell / (grid.sizeX*grid.sizeY))*interiorWidth - voxels.apronCells);  //the block's first slot, in the domain's voxels
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        if(voxels.solids[index]){
            continue;
        }
        int slot = voxels.voxelSlots[index];
        int3 voxel = make_int3(origin.x + slot % voxels1D, origin.y + slot / voxels1D % voxels1D, origin.z + slot / (voxels1D*voxels1D));
        int depth = voxels.depth[index];
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){   //a dim face sits on the voxel's lower boundary along dim, and mid-voxel along the others
            float3 point = make_float3(grid.negX + (voxel.x + (dim == 0 ? 0.0f : 0.5f))*voxelSize, grid.negY + (voxel.y + (dim == 1 ? 0.0f : 0.5f))*voxelSize,
                                       grid.negZ + (voxel.z + (dim == 2 ? 0.0f : 0.5f))*voxelSize);
            float before = voxels.before[dim][index];
            float acceleration = 0.0f;
            for(int field = 0; field < forces.numFields; ++field){
                acceleration += fieldAcceleration(forces.fields[field], dim, point, time, dt, before, depth);
            }
            voxels.velocities[dim][index] += acceleration*dt;
        }
    }
}

void applyForces(const Forces& forces, float dt, double time, const ForceVoxels& voxels, cudaStream_t stream){
    if(voxels.numVoxels == 0){
        return;
    }
    addGravity<<<voxels.numVoxels / WORKSIZE + 1, WORKSIZE, 0, stream>>>(voxels.numVoxels, forces.gravity, dt, voxels.solids, voxels.velocities[0], voxels.velocities[1], voxels.velocities[2]);
    gpuErrchk(cudaPeekAtLastError());
    if(forces.numFields > 0 && voxels.numNodes > 0){
        addForceFields<<<voxels.numNodes, 128, 0, stream>>>(forces, dt, (float)time, voxels);
        gpuErrchk(cudaPeekAtLastError());
    }
}

void addFieldLimits(const ForceField& field, float longest, double& steady, double& drag){
    switch(field.kind){
        case FORCE_POINT:
        case FORCE_VORTEX:
            steady += std::abs(field.strength);
            break;
        case FORCE_TURBULENCE:
            steady += std::abs(field.strength)*CURL_NOISE_MOST;
            break;
        case FORCE_VOLUME:
            if(field.mode == VOLUME_FORCE){
                steady += std::abs(field.strength)*longest;
            }
            else{
                steady += field.drag*std::abs(field.strength)*longest;
                drag += field.drag;
            }
            break;
        default:    //FORCE_WIND
            steady += field.drag*std::max({std::abs(field.velocity.x), std::abs(field.velocity.y), std::abs(field.velocity.z)});
            drag += field.drag;
    }
}

SolidSDF uploadForceVolume(const SceneField& field, cudaStream_t stream){
    static_assert(SDF_BRICK == 8 && SDF_OUTSIDE == -1, "SceneField's layout (scene.hpp) has to be SolidSDF's");
    SolidSDF volume = {};
    volume.origin = make_float3((float)field.origin[0], (float)field.origin[1], (float)field.origin[2]);
    volume.spacing = (float)field.spacing;
    volume.bricks = make_int3(field.bricks[0], field.bricks[1], field.bricks[2]);
    int* table = nullptr;
    float* vectors = nullptr;
    float* own = nullptr;
    gpuErrchk(cudaMalloc((void**)&table, sizeof(int)*std::max(field.table.size(), (size_t)1)));
    gpuErrchk(cudaMalloc((void**)&vectors, sizeof(float)*std::max(field.velocities.size(), (size_t)1)));
    gpuErrchk(cudaMalloc((void**)&own, sizeof(float)*std::max(field.pool.size(), (size_t)1)));
    if(!field.table.empty()){    //from pageable memory, so they're copied before this returns
        gpuErrchk(cudaMemcpyAsync(table, field.table.data(), sizeof(int)*field.table.size(), cudaMemcpyHostToDevice, stream));
    }
    if(!field.velocities.empty()){
        gpuErrchk(cudaMemcpyAsync(vectors, field.velocities.data(), sizeof(float)*field.velocities.size(), cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(own, field.pool.data(), sizeof(float)*field.pool.size(), cudaMemcpyHostToDevice, stream));
    }
    volume.table = table;
    volume.velocities = vectors;
    volume.pool = own;
    return volume;
}

void freeForceVolume(SolidSDF& volume){
    if(volume.table != nullptr){
        cudaFree(const_cast<int*>(volume.table));
        cudaFree(const_cast<float*>(volume.velocities));
        cudaFree(const_cast<float*>(volume.pool));
    }
    volume = {};
}
