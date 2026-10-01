//Copyright 2023 Aberrant Behavior LLC

//Obstacles: their distance fields, built on the GPU, and what the simulation does with them each substep. Nothing here changes the pressure solve; it
//only changes what the solve is given:
//- the voxels whose centres are inside an obstacle stop being unknowns, and the unknowns beside them see them as walls (WALL_VOXEL), as they do the
//  domain's own walls (markObstacleSolids, run as the voxels are built)
//- where an obstacle moves, the divergence gets the flow its surface pushes through those wall faces (addObstacleFlux, after each cudaCalcDivU)
//- after the pressure solve, and before it for FLIP's sake, the faces between fluid and obstacle take the obstacle's velocity across them, and the faces
//  inside it continue the fluid's velocity along the surface, slipping freely with friction 0 and moving with the obstacle with friction 1, so particles
//  next to it read sensible velocities (obstacleGhostVelocities)
//- particles that end up inside one anyway are pushed back out, losing their velocity into it (pushOutParticles); and fluid that starts inside one is
//  removed
//The obstacles' fields are in their own space and replicated on every partition, and every pass computes the same thing from the same inputs, so
//results don't depend on how the domain is split

#include "particles.hu"
#include "algorithms/voxelSolveFunctions.hu"   //NO_VOXEL, WALL_VOXEL
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cub/cub.cuh>

// ---- small vector helpers ----

__host__ __device__ inline float3 operator+(float3 a, float3 b){
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(float3 a, float3 b){
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(float s, float3 a){
    return make_float3(s*a.x, s*a.y, s*a.z);
}

__host__ __device__ inline float dot(float3 a, float3 b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ inline float component(float3 v, int axis){
    return axis == 0 ? v.x : axis == 1 ? v.y : v.z;
}

// ---- reading the fields ----

__device__ inline float sdfSample(const SolidSDF& sdf, int x, int y, int z){
    if(x < 0 || y < 0 || z < 0 || x >= sdf.bricks.x*SDF_BRICK || y >= sdf.bricks.y*SDF_BRICK || z >= sdf.bricks.z*SDF_BRICK){
        return sdf.band;    //past the bounds: well outside
    }
    int entry = sdf.table[x/SDF_BRICK + sdf.bricks.x*(y/SDF_BRICK + sdf.bricks.y*(z/SDF_BRICK))];
    if(entry < 0){
        return entry == SDF_INSIDE ? -sdf.band : sdf.band;
    }
    return sdf.pool[(size_t)entry*SDF_BRICK*SDF_BRICK*SDF_BRICK + x%SDF_BRICK + SDF_BRICK*(y%SDF_BRICK + SDF_BRICK*(z%SDF_BRICK))];
}

//trilinear between the 8 samples around p, in the field's own space, and its gradient
__device__ inline float sdfTrilinear(const SolidSDF& sdf, float3 p, float3& gradient){
    float q[3] = {(p.x - sdf.origin.x) / sdf.spacing, (p.y - sdf.origin.y) / sdf.spacing, (p.z - sdf.origin.z) / sdf.spacing};
    int i[3];
    float f[3];
    for(int axis = 0; axis < 3; ++axis){
        i[axis] = (int)floorf(q[axis]);
        f[axis] = q[axis] - i[axis];
    }
    float c[8];
    for(int corner = 0; corner < 8; ++corner){
        c[corner] = sdfSample(sdf, i[0] + (corner & 1), i[1] + (corner >> 1 & 1), i[2] + (corner >> 2));
    }
    float gx = (c[1] - c[0])*(1 - f[1])*(1 - f[2]) + (c[3] - c[2])*f[1]*(1 - f[2]) + (c[5] - c[4])*(1 - f[1])*f[2] + (c[7] - c[6])*f[1]*f[2];
    float gy = (c[2] - c[0])*(1 - f[0])*(1 - f[2]) + (c[3] - c[1])*f[0]*(1 - f[2]) + (c[6] - c[4])*(1 - f[0])*f[2] + (c[7] - c[5])*f[0]*f[2];
    float gz = (c[4] - c[0])*(1 - f[0])*(1 - f[1]) + (c[5] - c[1])*f[0]*(1 - f[1]) + (c[6] - c[2])*(1 - f[0])*f[1] + (c[7] - c[3])*f[0]*f[1];
    gradient = make_float3(gx / sdf.spacing, gy / sdf.spacing, gz / sdf.spacing);
    float x0 = c[0] + f[0]*(c[1] - c[0]), x1 = c[2] + f[0]*(c[3] - c[2]), x2 = c[4] + f[0]*(c[5] - c[4]), x3 = c[6] + f[0]*(c[7] - c[6]);
    float y0 = x0 + f[1]*(x1 - x0), y1 = x2 + f[1]*(x3 - x2);
    return y0 + f[2]*(y1 - y0);
}

__device__ inline float boxDistance(float3 low, float3 high, float3 p, float3& gradient){
    float3 centre = 0.5f*(low + high);
    float3 half = 0.5f*(high - low);
    float3 d = p - centre;
    float q[3] = {fabsf(d.x) - half.x, fabsf(d.y) - half.y, fabsf(d.z) - half.z};
    float out[3] = {fmaxf(q[0], 0.0f), fmaxf(q[1], 0.0f), fmaxf(q[2], 0.0f)};
    float outside = sqrtf(out[0]*out[0] + out[1]*out[1] + out[2]*out[2]);
    float signs[3] = {d.x < 0.0f ? -1.0f : 1.0f, d.y < 0.0f ? -1.0f : 1.0f, d.z < 0.0f ? -1.0f : 1.0f};
    if(outside > 0.0f){
        gradient = make_float3(signs[0]*out[0] / outside, signs[1]*out[1] / outside, signs[2]*out[2] / outside);
        return outside;
    }
    int axis = q[0] >= q[1] && q[0] >= q[2] ? 0 : q[1] >= q[2] ? 1 : 2;     //inside: out through the nearest face
    gradient = make_float3(axis == 0 ? signs[0] : 0.0f, axis == 1 ? signs[1] : 0.0f, axis == 2 ? signs[2] : 0.0f);
    return q[axis];
}

//how far a world point is outside the obstacle (negative inside), and the way out of it there: its unit normal in the world
__device__ inline float obstacleDistance(const ObstacleState& obstacle, float3 x, float3& normal){
    const float* m = obstacle.worldToObject;
    float3 p = make_float3(m[0]*x.x + m[1]*x.y + m[2]*x.z + m[3], m[4]*x.x + m[5]*x.y + m[6]*x.z + m[7], m[8]*x.x + m[9]*x.y + m[10]*x.z + m[11]);
    float3 g;
    float distance;
    if(obstacle.kind == OBSTACLE_MESH){
        distance = sdfTrilinear(obstacle.sdf, p, g);
    }
    else if(obstacle.kind == OBSTACLE_BOX){
        distance = boxDistance(obstacle.low, obstacle.high, p, g);
    }
    else{
        float length = sqrtf(dot(p, p));
        g = length > 0.0f ? (1.0f / length)*p : make_float3(0.0f, 1.0f, 0.0f);
        distance = length - obstacle.radius;
    }
    float3 world = make_float3(m[0]*g.x + m[4]*g.y + m[8]*g.z, m[1]*g.x + m[5]*g.y + m[9]*g.z, m[2]*g.x + m[6]*g.y + m[10]*g.z);  //rigid: the transpose rotates back
    float length = sqrtf(dot(world, world));
    normal = length > 0.0f ? (1.0f / length)*world : make_float3(0.0f, 1.0f, 0.0f);
    return distance;
}

//a deforming mesh's surface velocity at p: trilinear between the 8 samples around it, each the velocity of the nearest point on the mesh to it; in bricks
//without samples, deep inside or well outside, the mesh's mean velocity
__device__ inline float3 sdfVelocity(const SolidSDF& sdf, float3 p){
    const int samples = SDF_BRICK*SDF_BRICK*SDF_BRICK;
    float q[3] = {(p.x - sdf.origin.x) / sdf.spacing, (p.y - sdf.origin.y) / sdf.spacing, (p.z - sdf.origin.z) / sdf.spacing};
    int i[3];
    float f[3];
    for(int axis = 0; axis < 3; ++axis){
        i[axis] = (int)floorf(q[axis]);
        f[axis] = q[axis] - i[axis];
    }
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for(int corner = 0; corner < 8; ++corner){
        int x = i[0] + (corner & 1), y = i[1] + (corner >> 1 & 1), z = i[2] + (corner >> 2);
        float weight = (corner & 1 ? f[0] : 1.0f - f[0])*(corner >> 1 & 1 ? f[1] : 1.0f - f[1])*(corner >> 2 ? f[2] : 1.0f - f[2]);
        float3 velocity = sdf.farVelocity;
        if(x >= 0 && y >= 0 && z >= 0 && x < sdf.bricks.x*SDF_BRICK && y < sdf.bricks.y*SDF_BRICK && z < sdf.bricks.z*SDF_BRICK){
            int entry = sdf.table[x/SDF_BRICK + sdf.bricks.x*(y/SDF_BRICK + sdf.bricks.y*(z/SDF_BRICK))];
            if(entry >= 0){
                size_t at = (size_t)entry*3*samples + x%SDF_BRICK + SDF_BRICK*(y%SDF_BRICK + SDF_BRICK*(z%SDF_BRICK));
                velocity = make_float3(sdf.velocities[at], sdf.velocities[at + samples], sdf.velocities[at + 2*samples]);
            }
        }
        sum = sum + weight*velocity;
    }
    return sum;
}

//how fast the obstacle's surface moves at world point x: a rigid one's from its motion, a deforming mesh's from its field, which is in the world
__device__ inline float3 obstacleVelocity(const ObstacleState& obstacle, float3 x){
    if(obstacle.kind == OBSTACLE_MESH && obstacle.sdf.velocities != nullptr){
        return sdfVelocity(obstacle.sdf, x);
    }
    const float* v = obstacle.velocity;
    return make_float3(v[0]*x.x + v[1]*x.y + v[2]*x.z + v[3], v[4]*x.x + v[5]*x.y + v[6]*x.z + v[7], v[8]*x.x + v[9]*x.y + v[10]*x.z + v[11]);
}

//the obstacle a point is deepest inside, or nearest to: its index and the distance and normal there
__device__ inline int nearestObstacle(const Obstacles& obstacles, float3 x, float& distance, float3& normal){
    int nearest = -1;
    distance = INFINITY;
    for(int index = 0; index < obstacles.count; ++index){
        float3 n;
        float d = obstacleDistance(obstacles.items[index], x, n);
        if(d < distance){
            distance = d;
            normal = n;
            nearest = index;
        }
    }
    return nearest;
}

// ---- building a mesh's field ----

//the closest point to p on triangle abc (Ericson, Real-Time Collision Detection, 5.1.5), and its barycentric weights for b and c. A deforming mesh's
//triangles can lose their area for a moment, so it only divides by what's positive: d1 - d3 is |ab|^2, d2 - d6 is |ac|^2, the bc edge's sum is |bc|^2,
//and va + vb + vc is |ab x ac|^2
__device__ inline float3 closestOnTriangle(float3 p, float3 a, float3 b, float3 c, float& wb, float& wc){
    float3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    wb = 0.0f;
    wc = 0.0f;
    if(d1 <= 0.0f && d2 <= 0.0f){
        return a;
    }
    float3 bp = p - b;
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if(d3 >= 0.0f && d4 <= d3){
        wb = 1.0f;
        return b;
    }
    float vc = d1*d4 - d3*d2;
    if(vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f){
        wb = d1 - d3 > 0.0f ? d1 / (d1 - d3) : 0.0f;
        return a + wb*ab;
    }
    float3 cp = p - c;
    float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if(d6 >= 0.0f && d5 <= d6){
        wc = 1.0f;
        return c;
    }
    float vb = d5*d2 - d1*d6;
    if(vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f){
        wc = d2 - d6 > 0.0f ? d2 / (d2 - d6) : 0.0f;
        return a + wc*ac;
    }
    float va = d3*d6 - d5*d4;
    if(va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f){
        float length = (d4 - d3) + (d5 - d6);
        wc = length > 0.0f ? (d4 - d3) / length : 0.0f;
        wb = 1.0f - wc;
        return b + wc*(c - b);
    }
    float area = va + vb + vc;
    if(!(area > 0.0f)){
        return a;
    }
    float scale = 1.0f / area;
    wb = vb*scale;
    wc = vc*scale;
    return a + wb*ab + wc*ac;
}

//the bins a triangle's bounds, grown by reach, overlap: bricks of the field, or with flatAxis 0, 1 or 2, columns of bricks along that axis
__device__ inline void binRange(const float3* vertices, int3 triangle, float3 origin, float binSize, float reach, int3 bins, int flatAxis, int first[3], int last[3]){
    float3 a = vertices[triangle.x], b = vertices[triangle.y], c = vertices[triangle.z];
    float low[3] = {fminf(a.x, fminf(b.x, c.x)), fminf(a.y, fminf(b.y, c.y)), fminf(a.z, fminf(b.z, c.z))};
    float high[3] = {fmaxf(a.x, fmaxf(b.x, c.x)), fmaxf(a.y, fmaxf(b.y, c.y)), fmaxf(a.z, fmaxf(b.z, c.z))};
    float start[3] = {origin.x, origin.y, origin.z};
    int size[3] = {bins.x, bins.y, bins.z};
    for(int axis = 0; axis < 3; ++axis){
        first[axis] = min(max((int)floorf((low[axis] - reach - start[axis]) / binSize), 0), size[axis] - 1);
        last[axis] = min(max((int)floorf((high[axis] + reach - start[axis]) / binSize), 0), size[axis] - 1);
    }
    if(flatAxis >= 0){
        first[flatAxis] = last[flatAxis] = 0;
    }
}

__global__ void countTriangleBins(uint numTriangles, const float3* vertices, const int3* triangles, float3 origin, float binSize, float reach, int3 bins, int flatAxis, uint* counts){
    uint triangle = threadIdx.x + blockIdx.x*blockDim.x;
    if(triangle < numTriangles){
        int first[3], last[3];
        binRange(vertices, triangles[triangle], origin, binSize, reach, bins, flatAxis, first, last);
        counts[triangle] = (last[0] - first[0] + 1)*(last[1] - first[1] + 1)*(last[2] - first[2] + 1);
    }
}

//each triangle's (bin, triangle) pairs, from its offset: the bin as an index into the bins, x fastest (with a flat axis, its coordinate is 0). Pairs past
//capacity are dropped: only rounding at a bin's edge could make that many (see pairCapacity), and a bin a triangle only reaches by rounding is band away
__global__ void listTriangleBins(uint numTriangles, const float3* vertices, const int3* triangles, float3 origin, float binSize, float reach, int3 bins, int flatAxis,
                                 const uint* offsets, uint capacity, uint* keys, uint* values){
    uint triangle = threadIdx.x + blockIdx.x*blockDim.x;
    if(triangle < numTriangles){
        int first[3], last[3];
        binRange(vertices, triangles[triangle], origin, binSize, reach, bins, flatAxis, first, last);
        uint out = offsets[triangle];
        for(int z = first[2]; z <= last[2]; ++z){
            for(int y = first[1]; y <= last[1]; ++y){
                for(int x = first[0]; x <= last[0]; ++x){
                    if(out < capacity){
                        keys[out] = x + bins.x*(y + bins.y*z);
                        values[out] = triangle;
                    }
                    ++out;
                }
            }
        }
    }
}

//past the last pair, a key no bin has, which sorts after them all
__global__ void padPairs(uint capacity, uint numTriangles, const uint* counts, const uint* offsets, uint padding, uint* keys){
    uint total = offsets[numTriangles - 1] + counts[numTriangles - 1];
    for(uint index = total + threadIdx.x + blockIdx.x*blockDim.x; index < capacity; index += blockDim.x*gridDim.x){
        keys[index] = padding;
    }
}

//each bin's run of sorted pairs: its first, and one past its last. Bins with none keep (0, 0)
__global__ void findRuns(uint numPairs, uint numBins, const uint* sortedKeys, int2* runs){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numPairs){
        uint bin = sortedKeys[index];
        if(bin < numBins){
            if(index == 0 || sortedKeys[index - 1] != bin){
                runs[bin].x = (int)index;
            }
            if(index + 1 == numPairs || sortedKeys[index + 1] != bin){
                runs[bin].y = (int)index + 1;
            }
        }
    }
}

//per brick near the surface, a block, and a thread per sample: the distance to the nearest of the triangles near the brick. Those are every triangle within
//band of it, so the distance is exact up to band, and past it just more than band. A deforming mesh's sample also takes the velocity of that nearest
//point, blended from its triangle's vertices'. Blocks past the bricks in use, or past the pool, have nothing to do
__global__ void brickDistances(const uint* activeBricks, const uint* numActive, const int2* runs, const uint* triangleOfPair, const float3* vertices, const int3* triangles,
                               const float3* vertexVelocities, float3 origin, float spacing, int3 bricks, float* pool, float* velocities){
    __shared__ float3 corners[64*3];
    __shared__ uint chunkTriangles[64];
    if(blockIdx.x >= *numActive){
        return;
    }
    uint brick = activeBricks[blockIdx.x];
    int3 brickCell = make_int3(brick % bricks.x, brick / bricks.x % bricks.y, brick / (bricks.x*bricks.y));
    int3 sample = make_int3(threadIdx.x % SDF_BRICK, threadIdx.x / SDF_BRICK % SDF_BRICK, threadIdx.x / (SDF_BRICK*SDF_BRICK));
    float3 p = origin + spacing*make_float3(brickCell.x*SDF_BRICK + sample.x, brickCell.y*SDF_BRICK + sample.y, brickCell.z*SDF_BRICK + sample.z);
    int2 run = runs[brick];
    float best = INFINITY;
    uint nearest = 0;
    float nearestB = 0.0f, nearestC = 0.0f;
    for(int chunk = run.x; chunk < run.y; chunk += 64){
        int n = min(64, run.y - chunk);
        __syncthreads();
        if((int)threadIdx.x < 3*n){
            uint which = triangleOfPair[chunk + threadIdx.x/3];
            int3 triangle = triangles[which];
            int corner = threadIdx.x % 3;
            corners[threadIdx.x] = vertices[corner == 0 ? triangle.x : corner == 1 ? triangle.y : triangle.z];
            if(corner == 0){
                chunkTriangles[threadIdx.x/3] = which;
            }
        }
        __syncthreads();
        for(int k = 0; k < n; ++k){
            float wb, wc;
            float3 offset = p - closestOnTriangle(p, corners[3*k], corners[3*k + 1], corners[3*k + 2], wb, wc);
            float squared = dot(offset, offset);
            if(squared < best){
                best = squared;
                nearest = chunkTriangles[k];
                nearestB = wb;
                nearestC = wc;
            }
        }
    }
    const int samples = SDF_BRICK*SDF_BRICK*SDF_BRICK;
    pool[(size_t)blockIdx.x*samples + threadIdx.x] = sqrtf(best);
    if(velocities != nullptr){
        float3 velocity = make_float3(0.0f, 0.0f, 0.0f);
        if(best < INFINITY){
            int3 triangle = triangles[nearest];
            velocity = (1.0f - nearestB - nearestC)*vertexVelocities[triangle.x] + nearestB*vertexVelocities[triangle.y] + nearestC*vertexVelocities[triangle.z];
        }
        size_t at = (size_t)blockIdx.x*3*samples + threadIdx.x;
        velocities[at] = velocity.x;
        velocities[at + samples] = velocity.y;
        velocities[at + 2*samples] = velocity.z;
    }
}

//the triangles of a column the ray from p along +axis crosses: how many
struct ColumnBins{
    const int2* runs[3];            //per axis, per column of bricks along it (indexed by its two other brick coordinates), its run of triangles: first, one past last
    const uint* triangleOfPair[3];
};

__device__ inline int crossings(float3 p, int axis, const ColumnBins& columns, const float3* vertices, const int3* triangles, float3 origin, float brickSize, int3 bricks){
    int u = (axis + 1) % 3, v = (axis + 2) % 3;
    int size[3] = {bricks.x, bricks.y, bricks.z};
    int column[3] = {0, 0, 0};
    column[u] = min(max((int)floorf((component(p, u) - component(origin, u)) / brickSize), 0), size[u] - 1);
    column[v] = min(max((int)floorf((component(p, v) - component(origin, v)) / brickSize), 0), size[v] - 1);
    int2 run = columns.runs[axis][column[0] + bricks.x*(column[1] + bricks.y*column[2])];
    float pu = component(p, u), pv = component(p, v), pa = component(p, axis);
    int hits = 0;
    for(int k = run.x; k < run.y; ++k){
        int3 triangle = triangles[columns.triangleOfPair[axis][k]];
        float3 a = vertices[triangle.x], b = vertices[triangle.y], c = vertices[triangle.z];
        float au = component(a, u) - pu, av = component(a, v) - pv;
        float bu = component(b, u) - pu, bv = component(b, v) - pv;
        float cu = component(c, u) - pu, cv = component(c, v) - pv;
        float e0 = bu*cv - bv*cu;   //twice the areas of the triangles p makes with each edge: p's barycentric weights for a, b and c
        float e1 = cu*av - cv*au;
        float e2 = au*bv - av*bu;
        if((e0 > 0.0f && e1 > 0.0f && e2 > 0.0f) || (e0 < 0.0f && e1 < 0.0f && e2 < 0.0f)){
            float hit = (e0*component(a, axis) + e1*component(b, axis) + e2*component(c, axis)) / (e0 + e1 + e2);
            hits += hit > pa;
        }
    }
    return hits;
}

//whether p is inside the mesh: the parity of the crossings of a ray along each axis, the majority of the three, which survives a ray through a crack or
//along an edge. p is nudged off the lattice first, so rays don't run exactly along the mesh's own axis-aligned edges
__device__ inline bool insideMesh(float3 p, const ColumnBins& columns, const float3* vertices, const int3* triangles, float3 origin, float brickSize, int3 bricks, float spacing){
    p = p + spacing*make_float3(1.234567e-4f, 2.345678e-4f, 3.456789e-4f);
    int votes = 0;
    for(int axis = 0; axis < 3; ++axis){
        votes += crossings(p, axis, columns, vertices, triangles, origin, brickSize, bricks) & 1;
    }
    return votes >= 2;
}

//the samples' signs, and the band they're clamped to; with a shell, the distance less half its thickness instead
__global__ void signBrickSamples(const uint* activeBricks, const uint* numActive, ColumnBins columns, const float3* vertices, const int3* triangles, float3 origin, float spacing,
                                 int3 bricks, float band, float shell, float* pool){
    if(blockIdx.x >= *numActive){
        return;
    }
    uint brick = activeBricks[blockIdx.x];
    int3 brickCell = make_int3(brick % bricks.x, brick / bricks.x % bricks.y, brick / (bricks.x*bricks.y));
    int3 sample = make_int3(threadIdx.x % SDF_BRICK, threadIdx.x / SDF_BRICK % SDF_BRICK, threadIdx.x / (SDF_BRICK*SDF_BRICK));
    float3 p = origin + spacing*make_float3(brickCell.x*SDF_BRICK + sample.x, brickCell.y*SDF_BRICK + sample.y, brickCell.z*SDF_BRICK + sample.z);
    float& value = pool[(size_t)blockIdx.x*SDF_BRICK*SDF_BRICK*SDF_BRICK + threadIdx.x];
    float signedDistance = shell > 0.0f ? value - shell : insideMesh(p, columns, vertices, triangles, origin, spacing*SDF_BRICK, bricks, spacing) ? -value : value;
    value = fminf(fmaxf(signedDistance, -band), band);
}

//the bricks far from the surface: inside or out, by their centres
__global__ void signFarBricks(uint numBricks, ColumnBins columns, const float3* vertices, const int3* triangles, float3 origin, float spacing, int3 bricks, int* table){
    uint brick = threadIdx.x + blockIdx.x*blockDim.x;
    if(brick < numBricks && table[brick] == SDF_OUTSIDE){
        float half = 0.5f*SDF_BRICK;
        float3 p = origin + spacing*make_float3((brick % bricks.x)*SDF_BRICK + half, (brick / bricks.x % bricks.y)*SDF_BRICK + half, (brick / (bricks.x*bricks.y))*SDF_BRICK + half);
        if(insideMesh(p, columns, vertices, triangles, origin, spacing*SDF_BRICK, bricks, spacing)){
            table[brick] = SDF_INSIDE;
        }
    }
}

//which bricks have triangles near them, and so samples
__global__ void flagBricksInUse(uint numBricks, const int2* runs, uint* flags){
    uint brick = threadIdx.x + blockIdx.x*blockDim.x;
    if(brick < numBricks){
        flags[brick] = runs[brick].y > runs[brick].x;
    }
}

//the bricks in use take their places in the pool in order, as many as it has room for; the rest, and the bricks not in use, are outside until
//signFarBricks says otherwise. numActive counts them all, room or not
__global__ void placeBricks(uint numBricks, const uint* flags, const uint* slots, uint poolBricks, int* table, uint* activeBricks, uint* numActive){
    uint brick = threadIdx.x + blockIdx.x*blockDim.x;
    if(brick < numBricks){
        uint slot = slots[brick];
        bool placed = flags[brick] && slot < poolBricks;
        table[brick] = placed ? (int)slot : SDF_OUTSIDE;
        if(placed){
            activeBricks[slot] = brick;
        }
        if(brick == numBricks - 1){
            *numActive = slot + flags[brick];
        }
    }
}

//a deforming mesh's vertices a blend of the way between two samples, and their velocities: the change between the samples, per second
__global__ void interpolateVertices(uint numVertices, const float3* from, const float3* to, float blend, float perSecond, float3* vertices, float3* velocities){
    uint vertex = threadIdx.x + blockIdx.x*blockDim.x;
    if(vertex < numVertices){
        float3 a = from[vertex], b = to[vertex];
        vertices[vertex] = a + blend*(b - a);
        velocities[vertex] = perSecond*(b - a);
    }
}

//One way of sorting a mesh's triangles into bins, and the buffers it takes, kept between builds: per triangle, the bins its bounds grown by reach overlap;
//those (bin, triangle) pairs sorted by bin, the triangles in bin order; and each bin's run of them. The pairs are sorted capacity at a time, the real ones
//first and padding after, so nothing has to come back to the host
struct TriangleBins{
    float reach = 0.0f;
    int flatAxis = -1;          //-1: bricks; 0, 1 or 2: columns of bricks along that axis
    uint numBins = 0;
    uint capacity = 0;          //pairs
    int keyBits = 32;           //enough for numBins, the padding's key
    uint* counts = nullptr;     //per triangle
    uint* offsets = nullptr;
    uint* keys = nullptr;       //per pair
    uint* values = nullptr;
    uint* sortedKeys = nullptr;
    uint* triangleOfPair = nullptr;
    int2* runs = nullptr;       //per bin: its first pair, and one past its last
    void* scratch = nullptr;    //the scan's and the sort's
    size_t scratchBytes = 0;
};

//the most (bin, triangle) pairs binning can make wherever the mesh goes: along each axis a triangle's bounds are never wider than at the widest of its
//samples, since its corners move linearly between them, and bounds of width w grown by reach either side overlap at most (w + 2 reach)/size + 2 bins;
//with a little more for rounding at bins' edges
static uint pairCapacity(const std::vector<float3>& widths, float reach, float binSize, int flatAxis, int3 bins){
    int size[3] = {bins.x, bins.y, bins.z};
    unsigned long long total = 0;
    for(const float3& width : widths){
        unsigned long long pairs = 1;
        for(int axis = 0; axis < 3; ++axis){
            if(axis != flatAxis){
                pairs *= (unsigned long long)std::min((double)size[axis], std::floor((component(width, axis) + 2.0*reach) / binSize) + 2.0);
            }
        }
        total += pairs;
    }
    total += total/64 + 1024;
    if(total > 0x7FFFFFFFull){
        throw std::runtime_error("a mesh obstacle's triangles reach " + std::to_string(total) + " bins of its field, too many: are some of them huge?");
    }
    return (uint)total;
}

static void allocateBins(TriangleBins& bins, uint numTriangles, cudaStream_t stream){
    gpuErrchk(cudaMallocAsync((void**)&bins.counts, sizeof(uint)*numTriangles, stream));
    gpuErrchk(cudaMallocAsync((void**)&bins.offsets, sizeof(uint)*numTriangles, stream));
    for(uint** pairs : {&bins.keys, &bins.values, &bins.sortedKeys, &bins.triangleOfPair}){
        gpuErrchk(cudaMallocAsync((void**)pairs, sizeof(uint)*bins.capacity, stream));
    }
    gpuErrchk(cudaMallocAsync((void**)&bins.runs, sizeof(int2)*bins.numBins, stream));
    bins.keyBits = 1;
    while(bins.keyBits < 32 && (1ull << bins.keyBits) <= bins.numBins){
        ++bins.keyBits;
    }
    size_t scanBytes = 0, sortBytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, bins.counts, bins.offsets, (int)numTriangles, stream);
    cub::DeviceRadixSort::SortPairs(nullptr, sortBytes, bins.keys, bins.sortedKeys, bins.values, bins.triangleOfPair, (int)bins.capacity, 0, bins.keyBits, stream);
    bins.scratchBytes = std::max(scanBytes, sortBytes);
    gpuErrchk(cudaMallocAsync(&bins.scratch, bins.scratchBytes, stream));
}

static void freeBins(TriangleBins& bins){
    for(void* buffer : {(void*)bins.counts, (void*)bins.offsets, (void*)bins.keys, (void*)bins.values, (void*)bins.sortedKeys, (void*)bins.triangleOfPair, (void*)bins.runs,
                        bins.scratch}){
        if(buffer != nullptr){
            cudaFree(buffer);
        }
    }
    bins = TriangleBins();
}

//sorts the mesh's triangles into the bins where they are now, queued on stream
static void sortIntoBins(TriangleBins& bins, uint numTriangles, const float3* vertices, const int3* triangles, float3 origin, float binSize, int3 lattice, cudaStream_t stream){
    countTriangleBins<<<numTriangles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numTriangles, vertices, triangles, origin, binSize, bins.reach, lattice, bins.flatAxis, bins.counts);
    size_t bytes = bins.scratchBytes;
    cub::DeviceScan::ExclusiveSum(bins.scratch, bytes, bins.counts, bins.offsets, (int)numTriangles, stream);
    listTriangleBins<<<numTriangles / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numTriangles, vertices, triangles, origin, binSize, bins.reach, lattice, bins.flatAxis, bins.offsets,
        bins.capacity, bins.keys, bins.values);
    padPairs<<<256, BLOCKSIZE, 0, stream>>>(bins.capacity, numTriangles, bins.counts, bins.offsets, bins.numBins, bins.keys);
    bytes = bins.scratchBytes;
    cub::DeviceRadixSort::SortPairs(bins.scratch, bytes, bins.keys, bins.sortedKeys, bins.values, bins.triangleOfPair, (int)bins.capacity, 0, bins.keyBits, stream);
    gpuErrchk(cudaMemsetAsync(bins.runs, 0, sizeof(int2)*bins.numBins, stream));
    findRuns<<<bins.capacity / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(bins.capacity, bins.numBins, bins.sortedKeys, bins.runs);
}

//A triangle mesh's distance field on its partition's GPU, and what building it takes. The field covers a fixed lattice of bricks over everywhere the
//mesh goes, and only the bricks within band of a triangle hold samples. A rigid mesh is built once and keeps just the field. A deforming one is rebuilt at
//every update, from its vertices between their samples, without waiting on the GPU: the binning's buffers are sized up front for the most pairs the mesh
//can make, and the pool keeps room for half again as many bricks as the GPU last said it used, growing when that runs short
struct MeshField{
    uint numVertices = 0;
    uint numTriangles = 0;
    int3* triangles = nullptr;
    float3* vertices = nullptr;             //where the mesh is now
    float3* vertexVelocities = nullptr;     //deforming: how fast each vertex moves now
    float3 origin;
    float spacing = 0.0f;
    float band = 0.0f;
    float shell = 0.0f;                     //an open mesh's half thickness; 0: it's closed, and signed by ray parity
    int3 bricks;
    uint numBricks = 0;
    int* table = nullptr;
    float* pool = nullptr;
    float* velocities = nullptr;            //deforming
    uint poolBricks = 0;                    //what pool has room for
    uint firstUsed = 0;                     //the bricks the first build used
    uint* flags = nullptr;                  //per brick: whether it's in use
    uint* slots = nullptr;                  //and its place in pool if it is
    uint* activeBricks = nullptr;           //per place in pool, its brick
    uint* numActive = nullptr;              //how many bricks are in use
    void* scanScratch = nullptr;
    size_t scanBytes = 0;
    TriangleBins bins[4];                   //into bricks; for a closed mesh's signs, also into columns of bricks along x, y and z
    //deforming
    std::shared_ptr<const std::vector<float>> samples;
    std::vector<double> sampleTimes;
    std::vector<float3> meanVelocities;     //per interval between samples: its vertices' mean velocity
    float3* fromSample = nullptr;           //the samples either end of the interval being built, on the GPU
    float3* toSample = nullptr;
    int fromIndex = -1;
    int toIndex = -1;
    float3* staging = nullptr;              //pinned: samples on their way there
    cudaEvent_t staged = nullptr;
    uint* usedSeen = nullptr;               //pinned: numActive, copied back after each build for a later one to read
    double builtAt = NAN;
    float3 farVelocity = make_float3(0.0f, 0.0f, 0.0f);

    MeshField() = default;
    MeshField(const MeshField&) = delete;
    MeshField& operator=(const MeshField&) = delete;

    ~MeshField(){
        dropBuilders();
        for(void* buffer : {(void*)table, (void*)pool, (void*)velocities}){
            if(buffer != nullptr){
                cudaFree(buffer);
            }
        }
    }

    bool deforms() const{
        return samples != nullptr;
    }

    bool closed() const{
        return shell <= 0.0f;
    }

    bool movesAt(double time) const{
        return deforms() && sampleTimes.size() > 1 && time > sampleTimes.front() - 1e-4 && time < sampleTimes.back() + 1e-4;
    }

    SolidSDF sdf() const{
        return {origin, spacing, bricks, table, pool, band, velocities, farVelocity};
    }

    //everything but the field itself, which a rigid mesh needs no more once it's built
    void dropBuilders(){
        for(TriangleBins& binning : bins){
            freeBins(binning);
        }
        for(void* buffer : {(void*)triangles, (void*)vertices, (void*)vertexVelocities, (void*)flags, (void*)slots, (void*)activeBricks, (void*)numActive, scanScratch,
                            (void*)fromSample, (void*)toSample}){
            if(buffer != nullptr){
                cudaFree(buffer);
            }
        }
        triangles = nullptr;
        vertices = vertexVelocities = fromSample = toSample = nullptr;
        flags = slots = activeBricks = numActive = nullptr;
        scanScratch = nullptr;
        if(staging != nullptr){
            cudaFreeHost(staging);
            staging = nullptr;
        }
        if(usedSeen != nullptr){
            cudaFreeHost(usedSeen);
            usedSeen = nullptr;
        }
        if(staged != nullptr){
            cudaEventDestroy(staged);
            staged = nullptr;
        }
    }

    //the lattice over everywhere the mesh goes (its vertices, or every sample of a deforming one's), and the buffers for building on it
    void setUp(const std::vector<float3>& meshVertices, const std::vector<int3>& meshTriangles, float voxelSize, float halfThickness,
               std::shared_ptr<const std::vector<float>> deformingSamples, const std::vector<double>& times, cudaStream_t stream){
        numVertices = (uint)meshVertices.size();
        numTriangles = (uint)meshTriangles.size();
        samples = deformingSamples;
        sampleTimes = times;
        size_t numSamples = deforms() ? sampleTimes.size() : 1;
        float3 low = make_float3(INFINITY, INFINITY, INFINITY), high = make_float3(-INFINITY, -INFINITY, -INFINITY);
        std::vector<float3> widths(numTriangles, make_float3(0.0f, 0.0f, 0.0f));   //each triangle's widest bounds over the samples
        std::vector<float3> at(numVertices);
        for(size_t sample = 0; sample < numSamples; ++sample){
            for(uint vertex = 0; vertex < numVertices; ++vertex){
                const float* xyz = deforms() ? samples->data() + 3*((size_t)numVertices*sample + vertex) : nullptr;
                at[vertex] = deforms() ? make_float3(xyz[0], xyz[1], xyz[2]) : meshVertices[vertex];
                low = make_float3(std::min(low.x, at[vertex].x), std::min(low.y, at[vertex].y), std::min(low.z, at[vertex].z));
                high = make_float3(std::max(high.x, at[vertex].x), std::max(high.y, at[vertex].y), std::max(high.z, at[vertex].z));
            }
            for(uint index = 0; index < numTriangles; ++index){
                float3 a = at[meshTriangles[index].x], b = at[meshTriangles[index].y], c = at[meshTriangles[index].z];
                float3& width = widths[index];
                width.x = std::max(width.x, std::max(a.x, std::max(b.x, c.x)) - std::min(a.x, std::min(b.x, c.x)));
                width.y = std::max(width.y, std::max(a.y, std::max(b.y, c.y)) - std::min(a.y, std::min(b.y, c.y)));
                width.z = std::max(width.z, std::max(a.z, std::max(b.z, c.z)) - std::min(a.z, std::min(b.z, c.z)));
            }
        }
        spacing = voxelSize;
        shell = halfThickness;
        band = 3.0f*spacing + shell;
        float pad = band + 2.0f*spacing;
        origin = low - make_float3(pad, pad, pad);
        float brickSize = spacing*SDF_BRICK;
        bricks = make_int3((int)std::ceil((high.x - low.x + 2*pad) / brickSize) + 1, (int)std::ceil((high.y - low.y + 2*pad) / brickSize) + 1,
                           (int)std::ceil((high.z - low.z + 2*pad) / brickSize) + 1);
        numBricks = (uint)bricks.x*bricks.y*bricks.z;
        for(int which = 0; which < (closed() ? 4 : 1); ++which){
            TriangleBins& binning = bins[which];
            binning.reach = which == 0 ? band : 0.0f;
            binning.flatAxis = which - 1;
            binning.numBins = numBricks;
            binning.capacity = pairCapacity(widths, binning.reach, brickSize, binning.flatAxis, bricks);
            allocateBins(binning, numTriangles, stream);
        }
        gpuErrchk(cudaMallocAsync((void**)&triangles, sizeof(int3)*numTriangles, stream));
        gpuErrchk(cudaMemcpyAsync(triangles, meshTriangles.data(), sizeof(int3)*numTriangles, cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaMallocAsync((void**)&vertices, sizeof(float3)*numVertices, stream));
        if(deforms()){
            for(float3** buffer : {&vertexVelocities, &fromSample, &toSample}){
                gpuErrchk(cudaMallocAsync((void**)buffer, sizeof(float3)*numVertices, stream));
            }
            gpuErrchk(cudaMallocHost((void**)&staging, 2*sizeof(float3)*numVertices));
            gpuErrchk(cudaEventCreateWithFlags(&staged, cudaEventDisableTiming));
            for(size_t interval = 0; interval + 1 < sampleTimes.size(); ++interval){
                double sum[3] = {0.0, 0.0, 0.0};
                const float* from = samples->data() + 3*(size_t)numVertices*interval;
                const float* to = from + 3*(size_t)numVertices;
                for(size_t i = 0; i < 3*(size_t)numVertices; ++i){
                    sum[i % 3] += (double)to[i] - from[i];
                }
                double scale = 1.0 / (std::max(numVertices, 1u)*(sampleTimes[interval + 1] - sampleTimes[interval]));
                meanVelocities.push_back(make_float3((float)(sum[0]*scale), (float)(sum[1]*scale), (float)(sum[2]*scale)));
            }
        }
        else{
            gpuErrchk(cudaMemcpyAsync(vertices, meshVertices.data(), sizeof(float3)*numVertices, cudaMemcpyHostToDevice, stream));
        }
        gpuErrchk(cudaMallocAsync((void**)&table, sizeof(int)*numBricks, stream));
        for(uint** buffer : {&flags, &slots, &activeBricks}){
            gpuErrchk(cudaMallocAsync((void**)buffer, sizeof(uint)*numBricks, stream));
        }
        gpuErrchk(cudaMallocAsync((void**)&numActive, sizeof(uint), stream));
        cub::DeviceScan::ExclusiveSum(nullptr, scanBytes, flags, slots, (int)numBricks, stream);
        gpuErrchk(cudaMallocAsync(&scanScratch, scanBytes, stream));
        gpuErrchk(cudaMallocHost((void**)&usedSeen, sizeof(uint)));
    }

    void resizePool(uint capacity, cudaStream_t stream){
        for(float** buffer : {&pool, &velocities}){
            if(*buffer != nullptr){
                gpuErrchk(cudaFreeAsync(*buffer, stream));     //after whatever's queued that reads it
                *buffer = nullptr;
            }
        }
        poolBricks = capacity;
        const size_t samplesPerBrick = SDF_BRICK*SDF_BRICK*SDF_BRICK;
        gpuErrchk(cudaMallocAsync((void**)&pool, sizeof(float)*samplesPerBrick*capacity, stream));
        if(deforms()){
            gpuErrchk(cudaMallocAsync((void**)&velocities, 3*sizeof(float)*samplesPerBrick*capacity, stream));
        }
    }

    //the field of the mesh where its vertices are now, queued on stream. Only the first build waits, to size the pool: a rigid mesh's exactly
    void build(cudaStream_t stream){
        float brickSize = spacing*SDF_BRICK;
        for(int which = 0; which < (closed() ? 4 : 1); ++which){
            sortIntoBins(bins[which], numTriangles, vertices, triangles, origin, brickSize, bricks, stream);
        }
        flagBricksInUse<<<numBricks / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numBricks, bins[0].runs, flags);
        size_t bytes = scanBytes;
        cub::DeviceScan::ExclusiveSum(scanScratch, bytes, flags, slots, (int)numBricks, stream);
        if(poolBricks == 0){
            uint last[2];
            gpuErrchk(cudaMemcpyAsync(&last[0], slots + numBricks - 1, sizeof(uint), cudaMemcpyDeviceToHost, stream));
            gpuErrchk(cudaMemcpyAsync(&last[1], flags + numBricks - 1, sizeof(uint), cudaMemcpyDeviceToHost, stream));
            gpuErrchk(cudaStreamSynchronize(stream));
            firstUsed = last[0] + last[1];
            *usedSeen = firstUsed;
            resizePool(deforms() ? firstUsed + firstUsed/2 + 64 : std::max(firstUsed, 1u), stream);
        }
        else if(deforms()){
            uint used = *(volatile uint*)usedSeen;     //a build or so behind, which is all the pool's size needs
            if(used > poolBricks){
                std::cerr<<"A deforming obstacle's field needed "<<used<<" bricks, more than the "<<poolBricks<<" it had room for, so for a substep the rest were coarse\n";
            }
            if(used + used/4 > poolBricks){
                resizePool(used + used/2 + 64, stream);
            }
        }
        placeBricks<<<numBricks / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numBricks, flags, slots, poolBricks, table, activeBricks, numActive);
        brickDistances<<<poolBricks, SDF_BRICK*SDF_BRICK*SDF_BRICK, 0, stream>>>(activeBricks, numActive, bins[0].runs, bins[0].triangleOfPair, vertices, triangles, vertexVelocities,
            origin, spacing, bricks, pool, velocities);
        ColumnBins columns = {};
        if(closed()){
            for(int axis = 0; axis < 3; ++axis){
                columns.runs[axis] = bins[1 + axis].runs;
                columns.triangleOfPair[axis] = bins[1 + axis].triangleOfPair;
            }
        }
        signBrickSamples<<<poolBricks, SDF_BRICK*SDF_BRICK*SDF_BRICK, 0, stream>>>(activeBricks, numActive, columns, vertices, triangles, origin, spacing, bricks, band, shell, pool);
        if(closed()){
            signFarBricks<<<numBricks / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numBricks, columns, vertices, triangles, origin, spacing, bricks, table);
        }
        if(deforms()){
            gpuErrchk(cudaMemcpyAsync(usedSeen, numActive, sizeof(uint), cudaMemcpyDeviceToHost, stream));
        }
        gpuErrchk(cudaPeekAtLastError());
    }

    //puts samples from and to on the GPU, through the pinned staging buffer: the copies through it last time, an interval ago, are long done
    void load(int from, int to, cudaStream_t stream){
        if(from == toIndex && from != fromIndex){   //the next interval starts where the last one ended
            std::swap(fromSample, toSample);
            std::swap(fromIndex, toIndex);
        }
        if(from == fromIndex && to == toIndex){
            return;
        }
        gpuErrchk(cudaEventSynchronize(staged));
        size_t bytes = sizeof(float3)*numVertices;
        if(from != fromIndex){
            std::memcpy(staging, samples->data() + 3*(size_t)numVertices*from, bytes);
            gpuErrchk(cudaMemcpyAsync(fromSample, staging, bytes, cudaMemcpyHostToDevice, stream));
            fromIndex = from;
        }
        if(to != toIndex){
            std::memcpy(staging + numVertices, samples->data() + 3*(size_t)numVertices*to, bytes);
            gpuErrchk(cudaMemcpyAsync(toSample, staging + numVertices, bytes, cudaMemcpyHostToDevice, stream));
            toIndex = to;
        }
        gpuErrchk(cudaEventRecord(staged, stream));
    }

    //a deforming mesh where it is at time, linearly between the samples either side, and still before the first and from the last on; and its field
    //there. At a sample's time it moves as it will until the next, as a substep starting there will see
    void deformTo(double time, cudaStream_t stream){
        if(time == builtAt){
            return;
        }
        builtAt = time;
        int last = (int)sampleTimes.size() - 1;
        int from = 0, to = 0;
        double blend = 0.0, perSecond = 0.0;
        if(time >= sampleTimes[last]){
            from = to = last;
        }
        else if(time >= sampleTimes[0]){
            while(from + 1 < last && time >= sampleTimes[from + 1]){
                ++from;
            }
            to = from + 1;
            double span = sampleTimes[to] - sampleTimes[from];
            blend = (time - sampleTimes[from]) / span;
            perSecond = 1.0 / span;
        }
        load(from, to, stream);
        interpolateVertices<<<numVertices / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVertices, fromSample, toSample, (float)blend, (float)perSecond, vertices, vertexVelocities);
        farVelocity = to != from ? meanVelocities[from] : make_float3(0.0f, 0.0f, 0.0f);
        build(stream);
    }
};

// ---- placing them in time ----

//a row-major 4x4 transform's translation, per-axis scale and rotation (a unit quaternion, w x y z), taking a mirroring as a negative z scale
static void decompose(const std::array<double, 16>& m, double translation[3], double scale[3], double quaternion[4]){
    double r[3][3];
    for(int column = 0; column < 3; ++column){
        translation[column] = m[4*column + 3];
        scale[column] = std::sqrt(m[column]*m[column] + m[4 + column]*m[4 + column] + m[8 + column]*m[8 + column]);
        for(int row = 0; row < 3; ++row){
            r[row][column] = scale[column] > 0.0 ? m[4*row + column] / scale[column] : (row == column ? 1.0 : 0.0);
        }
    }
    double determinant = r[0][0]*(r[1][1]*r[2][2] - r[1][2]*r[2][1]) - r[0][1]*(r[1][0]*r[2][2] - r[1][2]*r[2][0]) + r[0][2]*(r[1][0]*r[2][1] - r[1][1]*r[2][0]);
    if(determinant < 0.0){
        scale[2] = -scale[2];
        for(int row = 0; row < 3; ++row){
            r[row][2] = -r[row][2];
        }
    }
    double trace = r[0][0] + r[1][1] + r[2][2];     //Shepperd's method
    if(trace > 0.0){
        double s = 2.0*std::sqrt(trace + 1.0);
        quaternion[0] = 0.25*s;
        quaternion[1] = (r[2][1] - r[1][2]) / s;
        quaternion[2] = (r[0][2] - r[2][0]) / s;
        quaternion[3] = (r[1][0] - r[0][1]) / s;
    }
    else if(r[0][0] > r[1][1] && r[0][0] > r[2][2]){
        double s = 2.0*std::sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]);
        quaternion[0] = (r[2][1] - r[1][2]) / s;
        quaternion[1] = 0.25*s;
        quaternion[2] = (r[0][1] + r[1][0]) / s;
        quaternion[3] = (r[0][2] + r[2][0]) / s;
    }
    else if(r[1][1] > r[2][2]){
        double s = 2.0*std::sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]);
        quaternion[0] = (r[0][2] - r[2][0]) / s;
        quaternion[1] = (r[0][1] + r[1][0]) / s;
        quaternion[2] = 0.25*s;
        quaternion[3] = (r[1][2] + r[2][1]) / s;
    }
    else{
        double s = 2.0*std::sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]);
        quaternion[0] = (r[1][0] - r[0][1]) / s;
        quaternion[1] = (r[0][2] + r[2][0]) / s;
        quaternion[2] = (r[1][2] + r[2][1]) / s;
        quaternion[3] = 0.25*s;
    }
}

//a rigid transform at time: the keys' translations interpolated linearly and their rotations by slerp, held before the first and after the last. Row-major
//3x4, own space to world
void ObstacleSet::rigidAt(const Built& obstacle, double time, double matrix[12]) const{
    size_t keys = obstacle.keyTimes.size();
    size_t key = 0;
    double blend = 0.0;
    if(keys > 1 && time > obstacle.keyTimes[0]){
        while(key + 2 < keys && time >= obstacle.keyTimes[key + 1]){
            ++key;
        }
        blend = std::min(1.0, (time - obstacle.keyTimes[key]) / (obstacle.keyTimes[key + 1] - obstacle.keyTimes[key]));
    }
    size_t next = keys > 1 ? key + 1 : key;
    const double* q0 = &obstacle.rotations[4*key];
    double q1[4] = {obstacle.rotations[4*next], obstacle.rotations[4*next + 1], obstacle.rotations[4*next + 2], obstacle.rotations[4*next + 3]};
    double cosine = q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3];
    if(cosine < 0.0){   //the short way round
        cosine = -cosine;
        for(double& component : q1){
            component = -component;
        }
    }
    double w0 = 1.0 - blend, w1 = blend;
    if(cosine < 0.9995){
        double angle = std::acos(cosine);
        w0 = std::sin((1.0 - blend)*angle) / std::sin(angle);
        w1 = std::sin(blend*angle) / std::sin(angle);
    }
    double q[4];
    double length = 0.0;
    for(int i = 0; i < 4; ++i){
        q[i] = w0*q0[i] + w1*q1[i];
        length += q[i]*q[i];
    }
    length = std::sqrt(length);
    double w = q[0] / length, x = q[1] / length, y = q[2] / length, z = q[3] / length;
    double rotation[9] = {1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
                          2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
                          2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)};
    for(int row = 0; row < 3; ++row){
        for(int column = 0; column < 3; ++column){
            matrix[4*row + column] = rotation[3*row + column];
        }
        matrix[4*row + 3] = (1.0 - blend)*obstacle.translations[3*key + row] + blend*obstacle.translations[3*next + row];
    }
}

void ObstacleSet::update(double time){
    current.count = (int)built.size();
    current.moving = false;
    for(size_t index = 0; index < built.size(); ++index){
        Built& obstacle = built[index];
        ObstacleState& state = current.items[index];
        if(obstacle.field != nullptr && obstacle.field->deforms()){
            obstacle.field->deformTo(time, stream);
            obstacle.sdf = obstacle.field->sdf();
            current.moving = current.moving || obstacle.field->movesAt(time);
        }
        state.kind = obstacle.kind;
        state.sdf = obstacle.sdf;
        state.low = obstacle.low;
        state.high = obstacle.high;
        state.radius = obstacle.radius;
        state.friction = obstacle.friction;
        double now[12];
        rigidAt(obstacle, time, now);
        //the inverse of a rigid transform: the transposed rotation, and the translation rotated back and negated
        for(int row = 0; row < 3; ++row){
            for(int column = 0; column < 3; ++column){
                state.worldToObject[4*row + column] = (float)now[4*column + row];
            }
            state.worldToObject[4*row + 3] = (float)-(now[row]*now[3] + now[4 + row]*now[7] + now[8 + row]*now[11]);
        }
        //the velocity of the point at x: d/dt of where its own-space point goes, M'(t) M(t)^-1 (x, 1), M' by central differences
        double velocity[12] = {};
        bool moves = obstacle.keyTimes.size() > 1 && time > obstacle.keyTimes.front() - 1e-4 && time < obstacle.keyTimes.back() + 1e-4;
        if(moves){
            double h = 1e-4;
            double before[12], after[12];
            rigidAt(obstacle, time - h, before);
            rigidAt(obstacle, time + h, after);
            double inverse[12];
            for(int i = 0; i < 12; ++i){
                inverse[i] = state.worldToObject[i];
            }
            for(int row = 0; row < 3; ++row){
                for(int column = 0; column < 4; ++column){
                    double sum = column == 3 ? (after[4*row + 3] - before[4*row + 3]) / (2.0*h) : 0.0;
                    for(int k = 0; k < 3; ++k){
                        sum += (after[4*row + k] - before[4*row + k]) / (2.0*h)*inverse[4*k + column];
                    }
                    velocity[4*row + column] = sum;
                }
            }
            current.moving = true;
        }
        for(int i = 0; i < 12; ++i){
            state.velocity[i] = (float)velocity[i];
        }
    }
}

void ObstacleSet::release(){     //with its GPU the current one; its stream may be gone already, as Particles destroys it before its members
    bool fields = false;
    for(const Built& obstacle : built){
        fields = fields || obstacle.field != nullptr;
    }
    if(fields){
        cudaDeviceSynchronize();    //nothing queued still reads them
    }
    for(Built& obstacle : built){
        delete obstacle.field;
    }
    built.clear();
    current.count = 0;
}

ObstacleSet::~ObstacleSet(){
    release();
}

void ObstacleSet::build(const std::vector<SceneObstacle>& obstacles, double voxelSize, cudaStream_t stream){
    release();
    this->stream = stream;
    if(obstacles.size() > MAX_OBSTACLES){
        std::cerr<<"ObstacleSet: only the first "<<MAX_OBSTACLES<<" obstacles count\n";
    }
    for(size_t index = 0; index < obstacles.size() && index < MAX_OBSTACLES; ++index){
        const SceneObstacle& obstacle = obstacles[index];
        Built made;
        made.kind = obstacle.kind == SceneObstacle::MESH ? OBSTACLE_MESH : obstacle.kind == SceneObstacle::BOX ? OBSTACLE_BOX : OBSTACLE_SPHERE;
        made.friction = (float)obstacle.friction;
        made.keyTimes = obstacle.keyTimes;
        //each key's rigid motion; the first key's scale is baked into the shape
        double scale[3] = {1.0, 1.0, 1.0};
        for(size_t key = 0; key < obstacle.transforms.size(); ++key){
            double translation[3], keyScale[3], quaternion[4];
            decompose(obstacle.transforms[key], translation, keyScale, quaternion);
            if(key == 0){
                std::copy(keyScale, keyScale + 3, scale);
            }
            made.translations.insert(made.translations.end(), translation, translation + 3);
            made.rotations.insert(made.rotations.end(), quaternion, quaternion + 4);
        }
        if(made.kind == OBSTACLE_BOX){
            float low[3], high[3];
            for(int axis = 0; axis < 3; ++axis){
                double a = obstacle.min[axis]*scale[axis], b = obstacle.max[axis]*scale[axis];
                low[axis] = (float)std::min(a, b);
                high[axis] = (float)std::max(a, b);
            }
            made.low = make_float3(low[0], low[1], low[2]);
            made.high = make_float3(high[0], high[1], high[2]);
        }
        else if(made.kind == OBSTACLE_SPHERE){  //a sphere stays one, scaled by its largest scale; it's centred on its own origin
            double largest = std::max(std::fabs(scale[0]), std::max(std::fabs(scale[1]), std::fabs(scale[2])));
            made.radius = (float)(obstacle.radius*largest);
            for(int axis = 0; axis < 3; ++axis){    //its centre moves into the translation, so it turns about itself
                double offset = obstacle.centre[axis]*scale[axis];
                for(size_t key = 0; key < obstacle.transforms.size(); ++key){
                    double translation[3], keyScale[3], quaternion[4];
                    decompose(obstacle.transforms[key], translation, keyScale, quaternion);
                    (void)translation;
                    (void)keyScale;
                    double w = quaternion[0], x = quaternion[1], y = quaternion[2], z = quaternion[3];
                    double rotation[9] = {1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y), 2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x), 2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)};
                    for(int row = 0; row < 3; ++row){
                        made.translations[3*key + row] += rotation[3*row + axis]*offset;
                    }
                }
            }
        }
        else{
            //a rigid mesh scaled, leaving out triangles with no area, which can't be anyone's nearest; a deforming one's samples are in the world, and
            //it keeps every triangle, as one can lose its area for a while and get it back
            bool deforms = !obstacle.sampleTimes.empty();
            std::vector<float3> vertices(obstacle.vertices.size() / 3);
            for(size_t vertex = 0; vertex < vertices.size(); ++vertex){
                double factor[3] = {deforms ? 1.0 : scale[0], deforms ? 1.0 : scale[1], deforms ? 1.0 : scale[2]};
                vertices[vertex] = make_float3((float)(obstacle.vertices[3*vertex]*factor[0]), (float)(obstacle.vertices[3*vertex + 1]*factor[1]), (float)(obstacle.vertices[3*vertex + 2]*factor[2]));
            }
            std::vector<int3> triangles;
            for(size_t corner = 0; corner + 2 < obstacle.triangles.size(); corner += 3){
                int3 triangle = make_int3(obstacle.triangles[corner], obstacle.triangles[corner + 1], obstacle.triangles[corner + 2]);
                float3 a = vertices[triangle.x], b = vertices[triangle.y], c = vertices[triangle.z];
                float3 ab = b - a, ac = c - a;
                float3 normal = make_float3(ab.y*ac.z - ab.z*ac.y, ab.z*ac.x - ab.x*ac.z, ab.x*ac.y - ab.y*ac.x);
                if(deforms || dot(normal, normal) > 0.0f){
                    triangles.push_back(triangle);
                }
            }
            if(triangles.empty()){
                std::cerr<<"ObstacleSet: obstacle "<<index<<" has no triangles with any area; it's left out\n";
                continue;
            }
            made.field = new MeshField();
            made.field->setUp(vertices, triangles, (float)voxelSize, (float)(0.5*obstacle.thickness), deforms ? obstacle.samples : nullptr, obstacle.sampleTimes, stream);
            if(deforms){
                made.field->deformTo(0.0, stream);
            }
            else{
                made.field->build(stream);
                gpuErrchk(cudaStreamSynchronize(stream));
                made.field->dropBuilders();
            }
            made.sdf = made.field->sdf();
            const MeshField& field = *made.field;
            size_t bytes = sizeof(float)*(size_t)field.poolBricks*SDF_BRICK*SDF_BRICK*SDF_BRICK*(deforms ? 4 : 1) + sizeof(int)*field.numBricks;
            std::cerr<<"Obstacle "<<index<<": "<<triangles.size()<<" triangles, "<<field.firstUsed<<" of "<<field.numBricks<<" bricks near its surface";
            if(deforms){
                std::cerr<<" at the start, deforming through "<<obstacle.sampleTimes.size()<<" samples from "<<obstacle.sampleTimes.front()<<" s to "<<obstacle.sampleTimes.back()<<" s";
            }
            std::cerr<<" ("<<bytes / (1<<20)<<" MB)\n";
        }
        built.push_back(std::move(made));
    }
    update(0.0);
}

// ---- what the simulation does with them ----

void Particles::setObstacles(const std::vector<SceneObstacle>& descriptions){
    obstacles.build(descriptions, grid.cellSize / (2<<refinementLevel), stream);
    obstacles.update(elapsedTime);
}

void Particles::updateObstacles(){
    if(obstacles.count() > 0){
        obstacles.update(elapsedTime);
    }
}

//a particle inside an obstacle goes back out along the way out, a twentieth of a voxel past the surface, and loses the part of its velocity relative to
//the obstacle that points into it
__global__ void pushOutOfObstacles(uint numParticles, double* px, double* py, double* pz, float* vx, float* vy, float* vz, Obstacles obstacles, float clearance){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        for(int which = 0; which < obstacles.count; ++which){
            const ObstacleState& obstacle = obstacles.items[which];
            float3 x = make_float3((float)px[index], (float)py[index], (float)pz[index]);
            float3 normal;
            float distance = obstacleDistance(obstacle, x, normal);
            if(distance < 0.0f){
                float push = clearance - distance;
                px[index] += push*normal.x;
                py[index] += push*normal.y;
                pz[index] += push*normal.z;
                float3 solid = obstacleVelocity(obstacle, x + push*normal);
                float3 relative = make_float3(vx[index], vy[index], vz[index]) - solid;
                float inward = dot(relative, normal);
                if(inward < 0.0f){
                    vx[index] -= inward*normal.x;
                    vy[index] -= inward*normal.y;
                    vz[index] -= inward*normal.z;
                }
            }
        }
    }
}

void Particles::pushOutParticles(){
    if(obstacles.count() == 0 || size == 0 || substepIndex == 0){   //at the start, fluid inside an obstacle is removed instead
        return;
    }
    float voxelSize = grid.cellSize / (2<<refinementLevel);
    pushOutOfObstacles<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr(), obstacles.state(), 0.05f*voxelSize);
    gpuErrchk(cudaPeekAtLastError());
}

__global__ void markInsideObstacles(uint numParticles, const double* px, const double* py, const double* pz, Obstacles obstacles, char* removed){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        float distance;
        float3 normal;
        if(nearestObstacle(obstacles, make_float3((float)px[index], (float)py[index], (float)pz[index]), distance, normal) >= 0 && distance < 0.0f){
            removed[index] = 1;
        }
    }
}

void Particles::markParticlesInsideObstacles(){
    if(obstacles.count() == 0 || size == 0 || substepIndex != 0){
        return;
    }
    markInsideObstacles<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, px.devPtr(), py.devPtr(), pz.devPtr(), obstacles.state(), removedFlags.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}

VoxelPlaces Particles::voxelPlaces(){
    return {nodeIndexUsedVoxels.devPtr(), nodeCells.devPtr(), voxelIDsUsed.devPtr(), grid, (int)(2<<refinementLevel), (int)std::floor(radius)};
}

//per stored voxel, a block per node: whether its centre is inside an obstacle. The domain's walls stay the domain's
__global__ void findSolidVoxels(Obstacles obstacles, VoxelPlaces places, const char* walls, char* solid){
    uint node = blockIdx.x;
    uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
    uint last = places.nodeVoxelEnds[node];
    uint cell = places.nodeCells[node];
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        float distance;
        float3 normal;
        int3 voxel = places.voxelOf(cell, places.voxelSlots[index]);
        solid[index] = !walls[index] && nearestObstacle(obstacles, places.point(voxel, 0.5f, 0.5f, 0.5f), distance, normal) >= 0 && distance < 0.0f;
    }
}

//an unknown outside the obstacles sees a neighbour inside one as a wall
__global__ void wallOffSolidNeighbors(uint numVoxels, const char* solid, const char* solveCodes, uint* neighborNx, uint* neighborPx, uint* neighborNy, uint* neighborPy, uint* neighborNz, uint* neighborPz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels && solveCodes[index] && !solid[index]){
        uint* neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
        for(int face = 0; face < 6; ++face){
            uint neighbor = neighbors[face][index];
            if(neighbor < WALL_VOXEL && solid[neighbor]){
                neighbors[face][index] = WALL_VOXEL;
            }
        }
    }
}

//a voxel inside an obstacle stops being an unknown, and the pressure update leaves its faces alone: it names no unknown below it, and an air voxel above it
//that named it no longer does. Only this voxel writes either place
__global__ void retireSolidVoxels(uint numVoxels, const char* solid, char* solveCodes, uint* neighborNx, const uint* neighborPx, uint* neighborNy, const uint* neighborPy, uint* neighborNz,
                                  const uint* neighborPz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels && solid[index]){
        uint* lower[3] = {neighborNx, neighborNy, neighborNz};
        const uint* upper[3] = {neighborPx, neighborPy, neighborPz};
        if(solveCodes[index]){
            for(int axis = 0; axis < 3; ++axis){
                uint above = upper[axis][index];
                if(above < WALL_VOXEL && !solid[above] && !solveCodes[above] && lower[axis][above] == index){
                    lower[axis][above] = NO_VOXEL;
                }
            }
        }
        for(int axis = 0; axis < 3; ++axis){
            lower[axis][index] = NO_VOXEL;
        }
        solveCodes[index] = 0;
    }
}

void Particles::markObstacleSolids(){
    if(obstacles.count() == 0){
        return;
    }
    uint numVoxels = voxelIDsUsed.size();
    obstacleSolids.resizeAsync(numVoxels, stream);
    if(numVoxels == 0){
        return;
    }
    findSolidVoxels<<<numStoredNodes, 128, 0, stream>>>(obstacles.state(), voxelPlaces(), solids.devPtr(), obstacleSolids.devPtr());
    wallOffSolidNeighbors<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, obstacleSolids.devPtr(), solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(),
        neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr());
    retireSolidVoxels<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, obstacleSolids.devPtr(), solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(),
        neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}

//a moving obstacle's surface pushes fluid through the wall faces between it and the unknowns: the divergence takes that flow, the obstacle's velocity
//across each face, where cudaCalcDivU took none
__global__ void obstacleFaceFlux(Obstacles obstacles, VoxelPlaces places, uint3 domainVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy,
                                 const uint* neighborPy, const uint* neighborNz, const uint* neighborPz, float* divU){
    uint node = blockIdx.x;
    uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
    uint last = places.nodeVoxelEnds[node];
    uint cell = places.nodeCells[node];
    const uint* lower[3] = {neighborNx, neighborNy, neighborNz};
    const uint* upper[3] = {neighborPx, neighborPy, neighborPz};
    uint size[3] = {domainVoxels.x, domainVoxels.y, domainVoxels.z};
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        if(!solveCodes[index]){
            continue;
        }
        int3 voxel = places.voxelOf(cell, places.voxelSlots[index]);
        int coordinates[3] = {voxel.x, voxel.y, voxel.z};
        float flux = 0.0f;
        for(int axis = 0; axis < 3; ++axis){
            for(int side = 0; side < 2; ++side){    //0 the lower face, 1 the upper
                uint neighbor = (side ? upper : lower)[axis][index];
                int across = coordinates[axis] + (side ? 1 : -1);
                if(neighbor != WALL_VOXEL || across < 0 || across >= (int)size[axis]){
                    continue;   //not a wall, or the domain's own
                }
                float offset[3] = {0.5f, 0.5f, 0.5f};
                offset[axis] = side ? 1.5f : -0.5f;
                float distance;
                float3 normal;
                int solid = nearestObstacle(obstacles, places.point(voxel, offset[0], offset[1], offset[2]), distance, normal);
                if(solid >= 0 && distance < 0.0f){
                    offset[axis] = side ? 1.0f : 0.0f;
                    float velocity = component(obstacleVelocity(obstacles.items[solid], places.point(voxel, offset[0], offset[1], offset[2])), axis);
                    flux += side ? velocity : -velocity;
                }
            }
        }
        divU[index] += flux;
    }
}

void Particles::addObstacleFlux(){
    if(obstacles.count() == 0 || !obstacles.state().moving || numStoredNodes == 0){
        return;
    }
    uint interiorWidth = 2<<refinementLevel;
    uint3 domainVoxels = make_uint3(grid.sizeX*interiorWidth, grid.sizeY*interiorWidth, grid.sizeZ*interiorWidth);
    obstacleFaceFlux<<<numStoredNodes, 128, 0, stream>>>(obstacles.state(), voxelPlaces(), domainVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(),
        neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(), divU.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}

//One layer of the faces around and inside obstacles, a block per node of this partition's own: the node's block of voxels (with its apron, from their
//owners) goes into shared memory, and each of its interior voxels' faces with an obstacle on either side gets a velocity:
//- a face between fluid and obstacle: the obstacle's velocity across it, the flow the pressure solve gave it (addObstacleFlux)
//- a face inside the obstacle: the fluid's velocity along the surface from the faces of the same component next to it, blended towards the obstacle's
//  by its friction; across the surface, the obstacle's. Layer 1 fills the faces next to fluid faces, layer 2 the faces next to those
__global__ void obstacleGhostFaces(int layer, Obstacles obstacles, VoxelPlaces places, const uint* voxelOwners, const char* solid, float* ux, float* uy, float* uz){
    extern __shared__ float blockVelocities[];  //per component, a value per slot; then per slot, whether it's stored and whether it's solid
    int voxels1D = places.interiorWidth + 2*places.apronCells;
    int voxels3D = voxels1D*voxels1D*voxels1D;
    char* stored = (char*)(blockVelocities + 3*voxels3D);
    char* inside = stored + voxels3D;
    uint node = blockIdx.x;
    uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
    uint last = places.nodeVoxelEnds[node];
    uint cell = places.nodeCells[node];
    float* velocities[3] = {ux, uy, uz};
    for(int slot = threadIdx.x; slot < voxels3D; slot += blockDim.x){
        stored[slot] = 0;
        inside[slot] = 0;
    }
    __syncthreads();
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        int slot = places.voxelSlots[index];
        uint owner = voxelOwners[index];
        for(int dim = 0; dim < 3; ++dim){
            blockVelocities[dim*voxels3D + slot] = velocities[dim][owner];
        }
        stored[slot] = 1;
        inside[slot] = solid[owner];
    }
    __syncthreads();
    int strides[3] = {1, voxels1D, voxels1D*voxels1D};
    auto along = [&](int slot, int axis){   //a slot's coordinate along an axis of the block
        return slot / strides[axis] % voxels1D;
    };
    //the slot a step along an axis from slot, or -1 past the block's edge
    auto step = [&](int slot, int axis, int by){
        int moved = along(slot, axis) + by;
        return moved >= 0 && moved < voxels1D ? slot + by*strides[axis] : -1;
    };
    //a dim face is a fluid face if it's stored and neither voxel either side of it is in an obstacle; an inside face if both are
    auto fluidFace = [&](int slot, int dim){
        int below = step(slot, dim, -1);
        return below >= 0 && stored[slot] && !inside[slot] && !inside[below];
    };
    auto insideFace = [&](int slot, int dim){
        int below = step(slot, dim, -1);
        return below >= 0 && stored[slot] && inside[slot] && inside[below];
    };
    //whether an inside face has a fluid face of the same component beside it: layer 1 fills those
    auto nextToFluid = [&](int slot, int dim){
        for(int axis = 0; axis < 3; ++axis){
            for(int by = -1; by <= 1; by += 2){
                int neighbor = step(slot, axis, by);
                if(neighbor >= 0 && fluidFace(neighbor, dim)){
                    return true;
                }
            }
        }
        return false;
    };
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        int slot = places.voxelSlots[index];
        int x = slot % voxels1D, y = slot / voxels1D % voxels1D, z = slot / (voxels1D*voxels1D);
        if(x < places.apronCells || y < places.apronCells || z < places.apronCells || x >= voxels1D - places.apronCells || y >= voxels1D - places.apronCells || z >= voxels1D - places.apronCells){
            continue;   //only the node's own voxels: their neighbours own the apron's
        }
        int3 voxel = places.voxelOf(cell, slot);
        for(int dim = 0; dim < 3; ++dim){
            int below = slot - strides[dim];     //interior, so never past the block's edge
            bool here = inside[slot], there = inside[below];
            if(!here && !there){
                continue;
            }
            float offset[3] = {0.5f, 0.5f, 0.5f};
            offset[dim] = 0.0f;
            float3 face = places.point(voxel, offset[0], offset[1], offset[2]);
            float distance;
            float3 normal;
            int obstacle = nearestObstacle(obstacles, face, distance, normal);
            if(obstacle < 0){
                continue;
            }
            float solidVelocity = component(obstacleVelocity(obstacles.items[obstacle], face), dim);
            float value;
            if(here != there){  //between fluid and obstacle
                if(layer != 1){
                    continue;
                }
                value = solidVelocity;
            }
            else{
                bool firstLayer = nextToFluid(slot, dim);
                if(firstLayer != (layer == 1)){
                    continue;
                }
                float sum = 0.0f;
                int count = 0;
                for(int axis = 0; axis < 3; ++axis){
                    for(int by = -1; by <= 1; by += 2){
                        int neighbor = step(slot, axis, by);
                        if(neighbor < 0){
                            continue;
                        }
                        if(layer == 1 ? fluidFace(neighbor, dim) : insideFace(neighbor, dim) && nextToFluid(neighbor, dim)){
                            sum += blockVelocities[dim*voxels3D + neighbor];
                            ++count;
                        }
                    }
                }
                float tangential = 1.0f - component(normal, dim)*component(normal, dim);   //how much of this component runs along the surface
                float fluid = count > 0 ? sum / count : solidVelocity;
                value = solidVelocity + tangential*(1.0f - obstacles.items[obstacle].friction)*(fluid - solidVelocity);
            }
            velocities[dim][index] = value;
        }
    }
}

void Particles::obstacleGhostVelocities(){
    if(obstacles.count() == 0 || numOwnNodes == 0){
        return;
    }
    int voxels1D = numVoxels1D;
    size_t shared = (3*sizeof(float) + 2)*voxels1D*voxels1D*voxels1D;
    for(int layer = 1; layer <= 2; ++layer){
        obstacleGhostFaces<<<numOwnNodes, 128, shared, stream>>>(layer, obstacles.state(), voxelPlaces(), voxelOwners.devPtr(), obstacleSolids.devPtr(),
            voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr());
        gpuErrchk(cudaPeekAtLastError());
        for(CudaVec<float>* velocity : {&voxelsUx, &voxelsUy, &voxelsUz}){   //the next layer, and the ghosts, read what this one wrote
            context->fillGhosts(velocity->devPtr(), stream);
        }
    }
}

//the density correction counts each voxel's particles against the rest count, but a voxel an obstacle partly covers can only hold its uncovered part's
//worth, so it would read as thin and the correction would keep pulling fluid towards the obstacle. Each voxel outside the obstacles gets credit for its
//covered part: the rest count times the fraction covered, measured on a 4^3 lattice of points in it
__global__ void creditCoveredVolume(Obstacles obstacles, VoxelPlaces places, const char* walls, const char* solid, float restParticlesPerVoxel, float* particleCounts){
    uint node = blockIdx.x;
    uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
    uint last = places.nodeVoxelEnds[node];
    uint cell = places.nodeCells[node];
    float voxelSize = places.voxelSize();
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        if(walls[index] || solid[index]){
            continue;
        }
        float distance;
        float3 normal;
        int3 voxel = places.voxelOf(cell, places.voxelSlots[index]);
        if(nearestObstacle(obstacles, places.point(voxel, 0.5f, 0.5f, 0.5f), distance, normal) >= 0 && distance < 0.87f*voxelSize){   //within half a diagonal
            int covered = 0;    //the fraction of a 4^3 lattice of points in the voxel inside an obstacle
            for(int point = 0; point < 64; ++point){
                float inside;
                covered += nearestObstacle(obstacles, places.point(voxel, (point % 4 + 0.5f)*0.25f, (point / 4 % 4 + 0.5f)*0.25f, (point / 16 + 0.5f)*0.25f), inside, normal) >= 0 && inside < 0.0f;
            }
            particleCounts[index] += restParticlesPerVoxel*covered/64.0f;
        }
    }
}

void Particles::creditObstacleVolume(){
    if(obstacles.count() == 0 || numStoredNodes == 0){
        return;
    }
    creditCoveredVolume<<<numStoredNodes, 128, 0, stream>>>(obstacles.state(), voxelPlaces(), solids.devPtr(), obstacleSolids.devPtr(), (float)restParticlesPerVoxel, particleCounts.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}
