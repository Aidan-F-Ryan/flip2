//Copyright 2023 Aberrant Behavior LLC

//Obstacles: their distance fields, built on the GPU, and what the simulation does with them each substep. Nothing here changes the pressure solve; it
//only changes what the solve is given:
//- cut cells: a voxel obstacles close on every face stops being an unknown, and every face they close is a wall (WALL_VOXEL) to the unknown beside it, as
//  the domain's own walls are, whether or not a voxel is stored there (markObstacleSolids, run as the voxels are built); a face they only cut weighs in
//  the pressure equation as much as it's open (weighCutCells, after each cudaGetA)
//- the divergence gets the flow on the faces they cut or close: the fluid's on the open part, the surface's on the rest (addObstacleFlux, after each
//  cudaCalcDivU)
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
#include <unordered_map>
#include <cub/cub.cuh>

// ---- building a mesh's field ----

//which part of a triangle a closest point is on
enum TriangleFeature{AT_A, AT_B, AT_C, ON_AB, ON_BC, ON_CA, ON_FACE};

//the closest point to p on triangle abc (Ericson, Real-Time Collision Detection, 5.1.5), its barycentric weights for b and c, and with feature, which part
//of the triangle it's on. A deforming mesh's triangles can lose their area for a moment, so it only divides by what's positive: d1 - d3 is |ab|^2, d2 - d6
//is |ac|^2, the bc edge's sum is |bc|^2, and va + vb + vc is |ab x ac|^2
__device__ inline float3 closestOnTriangle(float3 p, float3 a, float3 b, float3 c, float& wb, float& wc, int* feature = nullptr){
    float3 ab = b - a, ac = c - a, ap = p - a;
    float d1 = dot(ab, ap), d2 = dot(ac, ap);
    wb = 0.0f;
    wc = 0.0f;
    if(d1 <= 0.0f && d2 <= 0.0f){
        if(feature){
            *feature = AT_A;
        }
        return a;
    }
    float3 bp = p - b;
    float d3 = dot(ab, bp), d4 = dot(ac, bp);
    if(d3 >= 0.0f && d4 <= d3){
        if(feature){
            *feature = AT_B;
        }
        wb = 1.0f;
        return b;
    }
    float vc = d1*d4 - d3*d2;
    if(vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f){
        if(feature){
            *feature = ON_AB;
        }
        wb = d1 - d3 > 0.0f ? d1 / (d1 - d3) : 0.0f;
        return a + wb*ab;
    }
    float3 cp = p - c;
    float d5 = dot(ab, cp), d6 = dot(ac, cp);
    if(d6 >= 0.0f && d5 <= d6){
        if(feature){
            *feature = AT_C;
        }
        wc = 1.0f;
        return c;
    }
    float vb = d5*d2 - d1*d6;
    if(vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f){
        if(feature){
            *feature = ON_CA;
        }
        wc = d2 - d6 > 0.0f ? d2 / (d2 - d6) : 0.0f;
        return a + wc*ac;
    }
    float va = d3*d6 - d5*d4;
    if(va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f){
        if(feature){
            *feature = ON_BC;
        }
        float length = (d4 - d3) + (d5 - d6);
        wc = length > 0.0f ? (d4 - d3) / length : 0.0f;
        wb = 1.0f - wc;
        return b + wc*(c - b);
    }
    float area = va + vb + vc;
    if(!(area > 0.0f)){
        if(feature){
            *feature = AT_A;
        }
        return a;
    }
    if(feature){
        *feature = ON_FACE;
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

//per axis, the triangles in each column of bricks along it
struct ColumnBins{
    const int2* runs[3];            //per axis, per column of bricks along it (indexed by its two other brick coordinates), its run of triangles: first, one past last
    const uint* triangleOfPair[3];
};

//how many of the triangles of p's column a ray from p along +axis crosses; with first and stride, of every stride-th of them from first on, a lane's share
__device__ inline int crossings(float3 p, int axis, const ColumnBins& columns, const float3* vertices, const int3* triangles, float3 origin, float brickSize, int3 bricks,
                                int first = 0, int stride = 1){
    int u = (axis + 1) % 3, v = (axis + 2) % 3;
    int size[3] = {bricks.x, bricks.y, bricks.z};
    int column[3] = {0, 0, 0};
    column[u] = min(max((int)floorf((component(p, u) - component(origin, u)) / brickSize), 0), size[u] - 1);
    column[v] = min(max((int)floorf((component(p, v) - component(origin, v)) / brickSize), 0), size[v] - 1);
    int2 run = columns.runs[axis][column[0] + bricks.x*(column[1] + bricks.y*column[2])];
    float pu = component(p, u), pv = component(p, v), pa = component(p, axis);
    int hits = 0;
    for(int k = run.x + first; k < run.y; k += stride){
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

//The winding number along a ray. Each triangle the ray from p along +axis crosses past p counts +1 or -1 by which way it faces along the ray (the sign of
//its area seen down the axis), and for a closed mesh whose triangles all wind the same way the sum is how many times the mesh wraps p: 0 outside, 1
//inside, 2 where it overlaps or folds through itself (a skinned joint bent hard), -1 where it's turned inside out. Parity counts a fold as outside, and so
//can pseudonormals, which see only the nearest sheet. Each edge's side test takes the same bits in both triangles that share it (no fused multiply-adds,
//which would round them two ways), and a ray that meets an edge or vertex exactly goes through just one of the triangles there, by which way the edge
//runs (as if p were moved a hair along -u, and a hair less along -v), so it neither slips between two triangles nor counts one twice
__device__ inline float edgeSide(float fromU, float fromV, float toU, float toV){
    return __fsub_rn(__fmul_rn(fromU, toV), __fmul_rn(fromV, toU));
}

//whether the ray along axis through (pu, pv) crosses triangle abc: 1 or -1 by which way the triangle faces along it, and where along it, or 0
__device__ inline int windingCrossing(float3 a, float3 b, float3 c, float pu, float pv, int axis, int u, int v, float& hit){
    float au = __fsub_rn(component(a, u), pu), av = __fsub_rn(component(a, v), pv);
    float bu = __fsub_rn(component(b, u), pu), bv = __fsub_rn(component(b, v), pv);
    float cu = __fsub_rn(component(c, u), pu), cv = __fsub_rn(component(c, v), pv);
    float ab = edgeSide(au, av, bu, bv), bc = edgeSide(bu, bv, cu, cv), ca = edgeSide(cu, cv, au, av);
    float area = ab + bc + ca;
    if(area == 0.0f){   //edge on, seen down the axis
        return 0;
    }
    float s = area > 0.0f ? 1.0f : -1.0f;
    auto takes = [s](float side, float du, float dv){   //inside this edge, or on it and the edge runs the one way, which its twin in the next triangle doesn't
        return s*side > 0.0f || (side == 0.0f && (s*dv > 0.0f || (dv == 0.0f && s*du < 0.0f)));
    };
    if(!takes(ab, __fsub_rn(bu, au), __fsub_rn(bv, av)) || !takes(bc, __fsub_rn(cu, bu), __fsub_rn(cv, bv)) || !takes(ca, __fsub_rn(au, cu), __fsub_rn(av, cv))){
        return 0;
    }
    hit = (bc*component(a, axis) + ca*component(b, axis) + ab*component(c, axis)) / area;
    return (int)s;
}

//the run of triangles in the column of bricks along axis that holds p
__device__ inline int2 columnRun(float3 p, int axis, const ColumnBins& columns, float3 origin, float brickSize, int3 bricks){
    int u = (axis + 1) % 3, v = (axis + 2) % 3;
    int size[3] = {bricks.x, bricks.y, bricks.z};
    int column[3] = {0, 0, 0};
    column[u] = min(max((int)floorf((component(p, u) - component(origin, u)) / brickSize), 0), size[u] - 1);
    column[v] = min(max((int)floorf((component(p, v) - component(origin, v)) / brickSize), 0), size[v] - 1);
    return columns.runs[axis][column[0] + bricks.x*(column[1] + bricks.y*column[2])];
}

//the winding number along +axis from p, over the triangles of p's column; with first and stride, of every stride-th of them from first on, a lane's share
__device__ inline int windingAlong(float3 p, int axis, const ColumnBins& columns, const float3* vertices, const int3* triangles, float3 origin, float brickSize, int3 bricks,
                                   int first = 0, int stride = 1){
    int u = (axis + 1) % 3, v = (axis + 2) % 3;
    int2 run = columnRun(p, axis, columns, origin, brickSize, bricks);
    float pu = component(p, u), pv = component(p, v), pa = component(p, axis);
    int sum = 0;
    for(int k = run.x + first; k < run.y; k += stride){
        int3 triangle = triangles[columns.triangleOfPair[axis][k]];
        float hit;
        int facing = windingCrossing(vertices[triangle.x], vertices[triangle.y], vertices[triangle.z], pu, pv, axis, u, v, hit);
        if(facing != 0 && hit > pa){
            sum += facing;
        }
    }
    return sum;
}

//points are nudged off the lattice before rays are cast from them, so the rays don't run exactly along the mesh's own axis-aligned edges
__device__ inline float3 offLattice(float3 p, float spacing){
    return p + spacing*make_float3(1.234567e-4f, 2.345678e-4f, 3.456789e-4f);
}

//whether p is inside the mesh: the parity of the crossings of a ray along each axis, the majority of the three, which survives a ray through a crack or
//along an edge
__device__ inline bool insideMesh(float3 p, const ColumnBins& columns, const float3* vertices, const int3* triangles, float3 origin, float brickSize, int3 bricks, float spacing){
    p = offLattice(p, spacing);
    int votes = 0;
    for(int axis = 0; axis < 3; ++axis){
        votes += crossings(p, axis, columns, vertices, triangles, origin, brickSize, bricks) & 1;
    }
    return votes >= 2;
}

//insideMesh by a whole warp, each lane taking every 32nd of the triangles; every lane gets the answer. A mesh that winds consistently counts p inside
//where it wraps p any number of times (its winding number isn't 0), by the majority of the three rays again
__device__ inline bool insideMeshByWarp(float3 p, const ColumnBins& columns, const float3* vertices, const int3* triangles, float3 origin, float brickSize, int3 bricks, float spacing,
                                        bool wound){
    p = offLattice(p, spacing);
    int lane = threadIdx.x % 32;
    int votes = 0;
    for(int axis = 0; axis < 3; ++axis){
        int count = wound ? windingAlong(p, axis, columns, vertices, triangles, origin, brickSize, bricks, lane, 32)
                          : crossings(p, axis, columns, vertices, triangles, origin, brickSize, bricks, lane, 32);
        for(int offset = 16; offset > 0; offset /= 2){
            count += __shfl_xor_sync(0xFFFFFFFFu, count, offset);
        }
        votes += wound ? count != 0 : count & 1;
    }
    return votes >= 2;
}

//the winding numbers of a row of a brick's samples, those along axis from at (whose own coordinate along it is 0), for a mesh that winds consistently:
//one ray from the first, whose crossings count for each sample before them. With first and stride, of every stride-th of the column's triangles from
//first on, a lane's share. One axis is enough: the edge tests leave no cracks, so the count is exact up to rounding where a ray all but grazes the surface
__device__ inline void windingRow(int axis, int3 at, int3 brickCell, const ColumnBins& columns, const float3* vertices, const int3* triangles, float3 origin, float spacing,
                                  int3 bricks, int first, int stride, int windings[SDF_BRICK]){
    int u = (axis + 1) % 3, v = (axis + 2) % 3;
    float3 p = offLattice(origin + spacing*make_float3(brickCell.x*SDF_BRICK + at.x, brickCell.y*SDF_BRICK + at.y, brickCell.z*SDF_BRICK + at.z), spacing);
    float firstAlong = component(p, axis);
    float pu = component(p, u), pv = component(p, v);
    int2 run = columnRun(p, axis, columns, origin, spacing*SDF_BRICK, bricks);
    #pragma unroll
    for(int j = 0; j < SDF_BRICK; ++j){
        windings[j] = 0;
    }
    for(int k = run.x + first; k < run.y; k += stride){
        int3 triangle = triangles[columns.triangleOfPair[axis][k]];
        float hit;
        int facing = windingCrossing(vertices[triangle.x], vertices[triangle.y], vertices[triangle.z], pu, pv, axis, u, v, hit);
        if(facing != 0){
            #pragma unroll
            for(int j = 0; j < SDF_BRICK; ++j){
                windings[j] += hit > firstAlong + j*spacing ? facing : 0;
            }
        }
    }
}

//per brick near the surface, a block, and a thread per sample: the distance to the nearest of the triangles near the brick. Those are every triangle within
//band of it, so the distance is exact up to band, and past it just more than band. A deforming mesh's sample also takes the velocity of that nearest
//point, blended from its triangle's vertices'. Blocks past the bricks in use, or past the pool, have nothing to do.
//A closed mesh that winds consistently is signed here too, by its winding numbers, and clamped to the band: a ray along each row of samples (windingRow),
//along whichever axis the brick's column of triangles is shortest, each row's triangles shared between SDF_BRICK lanes
__global__ void brickDistances(const uint* activeBricks, const uint* numActive, const int2* runs, const uint* triangleOfPair, const float3* vertices, const int3* triangles,
                               const float3* vertexVelocities, float3 origin, float spacing, int3 bricks, float* pool, float* velocities, bool signs, float band,
                               ColumnBins columns){
    __shared__ float3 corners[64*3];
    __shared__ uint chunkTriangles[64];
    __shared__ bool wraps[SDF_BRICK*SDF_BRICK*SDF_BRICK];           //whether the mesh wraps each sample
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
    float distance = sqrtf(best);
    if(signs){
        static_assert(SDF_BRICK*SDF_BRICK*SDF_BRICK == 512 && 32 % SDF_BRICK == 0, "a row's lanes share a warp, and the block has a lane for each of a row's samples");
        int axis = 0, fewest = 0;
        for(int along = 0; along < 3; ++along){
            int3 column = along == 0 ? make_int3(0, brickCell.y, brickCell.z) : along == 1 ? make_int3(brickCell.x, 0, brickCell.z) : make_int3(brickCell.x, brickCell.y, 0);
            int2 run = columns.runs[along][column.x + bricks.x*(column.y + bricks.y*column.z)];
            if(along == 0 || run.y - run.x < fewest){
                fewest = run.y - run.x;
                axis = along;
            }
        }
        int row = threadIdx.x / SDF_BRICK, lane = threadIdx.x % SDF_BRICK;
        int across = row % SDF_BRICK, beyond = row / SDF_BRICK;     //the row's place along the next axis round and the one after
        int3 at = axis == 0 ? make_int3(0, across, beyond) : axis == 1 ? make_int3(beyond, 0, across) : make_int3(across, beyond, 0);
        int windings[SDF_BRICK];
        windingRow(axis, at, brickCell, columns, vertices, triangles, origin, spacing, bricks, lane, SDF_BRICK, windings);
        bool mine = false;
        #pragma unroll
        for(int j = 0; j < SDF_BRICK; ++j){
            for(int offset = SDF_BRICK/2; offset > 0; offset /= 2){
                windings[j] += __shfl_xor_sync(0xFFFFFFFFu, windings[j], offset);
            }
            mine = lane == j ? windings[j] != 0 : mine;
        }
        int stride = axis == 0 ? 1 : axis == 1 ? SDF_BRICK : SDF_BRICK*SDF_BRICK;
        wraps[at.x + SDF_BRICK*(at.y + SDF_BRICK*at.z) + lane*stride] = mine;    //each lane the sample its own number along the row
        __syncthreads();
        bool inside = wraps[threadIdx.x];
        distance = fminf(fmaxf(inside ? -distance : distance, -band), band);
    }
    pool[(size_t)blockIdx.x*samples + threadIdx.x] = distance;
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

//the bricks far from the surface: inside or out, by their centres, a warp per brick
__global__ void signFarBricks(uint numBricks, ColumnBins columns, const float3* vertices, const int3* triangles, float3 origin, float spacing, int3 bricks, bool wound, int* table){
    uint brick = (threadIdx.x + blockIdx.x*blockDim.x) / 32;
    if(brick >= numBricks || table[brick] != SDF_OUTSIDE){     //the same for the whole warp
        return;
    }
    float half = 0.5f*SDF_BRICK;
    float3 p = origin + spacing*make_float3((brick % bricks.x)*SDF_BRICK + half, (brick / bricks.x % bricks.y)*SDF_BRICK + half, (brick / (bricks.x*bricks.y))*SDF_BRICK + half);
    if(insideMeshByWarp(p, columns, vertices, triangles, origin, spacing*SDF_BRICK, bricks, spacing, wound) && threadIdx.x % 32 == 0){
        table[brick] = SDF_INSIDE;
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

//which interval between keys or samples a substep starting at time moves through: the k with times[k] <= time < times[k + 1], counting a time a hair
//before a key as at it, since the simulation's clock is a sum of substeps; -1 before the first and from the last on, where it holds still. At a key it
//moves as it will until the next, since that's what the substep will see
static int movingInterval(const std::vector<double>& times, double time){
    const double early = 1e-9;
    if(times.size() < 2 || time < times.front() - early || time >= times.back() - early){
        return -1;
    }
    int interval = 0;
    while(interval + 2 < (int)times.size() && time >= times[interval + 1] - early){
        ++interval;
    }
    return interval;
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
    float shell = 0.0f;                     //an open mesh's half thickness; 0: it's closed, and signed by rays
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
    bool wound = false;                     //closed, and a manifold whose triangles all wind the same way: signed by winding numbers, otherwise by parity
    //deforming
    std::shared_ptr<const std::vector<float>> samples;
    std::vector<double> sampleTimes;
    std::vector<float3> meanVelocities;     //per interval between samples: its vertices' mean velocity
    std::vector<float> fastestVertices;     //and the fastest of them's speed
    float3 meshLow, meshHigh;               //the bounds of everywhere its vertices go
    float speed = 0.0f;                     //how fast its fastest vertex moves now
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

    //whether a closed mesh's triangles make a manifold that winds consistently, which its winding numbers need: every edge in exactly two triangles,
    //running along it in opposite directions in each. Which way they wind doesn't matter: wrapped either way round is inside
    static bool windsConsistently(const std::vector<int3>& meshTriangles){
        std::unordered_map<unsigned long long, int> open;      //per edge seen once, which way it ran: 1 from its lower vertex, -1 from its higher; 0 seen twice
        open.reserve(2*meshTriangles.size());
        for(const int3& t : meshTriangles){
            int corners[3] = {t.x, t.y, t.z};
            for(int edge = 0; edge < 3; ++edge){
                int from = corners[edge], to = corners[(edge + 1) % 3];
                if(from == to){
                    return false;
                }
                unsigned long long key = (unsigned long long)std::min(from, to) << 32 | (unsigned)std::max(from, to);
                int way = from < to ? 1 : -1;
                auto found = open.find(key);
                if(found == open.end()){
                    open.emplace(key, way);
                }
                else if(found->second != -way){
                    return false;   //a third triangle on the edge, or two running along it the same way
                }
                else{
                    found->second = 0;
                }
            }
        }
        for(const auto& edge : open){
            if(edge.second != 0){
                return false;       //an edge with one triangle: the mesh has a hole
            }
        }
        return true;
    }

    bool movesAt(double time) const{
        return deforms() && movingInterval(sampleTimes, time) >= 0;
    }

    SolidSDF sdf() const{
        return {origin, spacing, bricks, table, pool, band, velocities, farVelocity};
    }

    //a level set sampled already, on the host (a VDB's, volumes.hpp): its bricks go onto the GPU as they are
    void upload(const SceneField& field, cudaStream_t stream){
        static_assert(SDF_BRICK == 8 && SDF_OUTSIDE == -1 && SDF_INSIDE == -2, "SceneField's layout (scene.hpp) has to be SolidSDF's");
        const size_t samplesPerBrick = SDF_BRICK*SDF_BRICK*SDF_BRICK;
        origin = make_float3((float)field.origin[0], (float)field.origin[1], (float)field.origin[2]);
        spacing = (float)field.spacing;
        band = field.band;
        bricks = make_int3(field.bricks[0], field.bricks[1], field.bricks[2]);
        numBricks = (uint)field.table.size();
        firstUsed = (uint)(field.pool.size() / samplesPerBrick);
        poolBricks = std::max(firstUsed, 1u);
        meshLow = make_float3((float)field.low[0], (float)field.low[1], (float)field.low[2]);
        meshHigh = make_float3((float)field.high[0], (float)field.high[1], (float)field.high[2]);
        gpuErrchk(cudaMallocAsync((void**)&table, sizeof(int)*numBricks, stream));
        gpuErrchk(cudaMemcpyAsync(table, field.table.data(), sizeof(int)*numBricks, cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaMallocAsync((void**)&pool, sizeof(float)*samplesPerBrick*poolBricks, stream));
        gpuErrchk(cudaMemcpyAsync(pool, field.pool.data(), sizeof(float)*field.pool.size(), cudaMemcpyHostToDevice, stream));
        if(!field.velocities.empty()){
            gpuErrchk(cudaMallocAsync((void**)&velocities, 3*sizeof(float)*samplesPerBrick*poolBricks, stream));
            gpuErrchk(cudaMemcpyAsync(velocities, field.velocities.data(), sizeof(float)*field.velocities.size(), cudaMemcpyHostToDevice, stream));
        }
        gpuErrchk(cudaStreamSynchronize(stream));
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
        meshLow = low;
        meshHigh = high;
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
                double fastest = 0.0;
                for(uint vertex = 0; vertex < numVertices; ++vertex){
                    double d[3] = {(double)to[3*vertex] - from[3*vertex], (double)to[3*vertex + 1] - from[3*vertex + 1], (double)to[3*vertex + 2] - from[3*vertex + 2]};
                    fastest = std::max(fastest, std::sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2]));
                }
                fastestVertices.push_back((float)(fastest / (sampleTimes[interval + 1] - sampleTimes[interval])));
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
        wound = closed() && windsConsistently(meshTriangles);
        if(closed() && !wound){
            std::cerr<<"A mesh isn't a closed manifold whose triangles all wind the same way, so it's signed by the parity of rays through it: where it overlaps "
                       "itself reads as outside"<<(deforms() ? ", and rebuilding it is much slower" : "")<<"\n";
        }
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
        ColumnBins columns = {};
        if(closed()){
            for(int axis = 0; axis < 3; ++axis){
                columns.runs[axis] = bins[1 + axis].runs;
                columns.triangleOfPair[axis] = bins[1 + axis].triangleOfPair;
            }
        }
        brickDistances<<<poolBricks, SDF_BRICK*SDF_BRICK*SDF_BRICK, 0, stream>>>(activeBricks, numActive, bins[0].runs, bins[0].triangleOfPair, vertices, triangles, vertexVelocities,
            origin, spacing, bricks, pool, velocities, wound, band, columns);
        if(!wound){
            signBrickSamples<<<poolBricks, SDF_BRICK*SDF_BRICK*SDF_BRICK, 0, stream>>>(activeBricks, numActive, columns, vertices, triangles, origin, spacing, bricks, band, shell, pool);
        }
        if(closed()){
            signFarBricks<<<numBricks / 4 + 1, 128, 0, stream>>>(numBricks, columns, vertices, triangles, origin, spacing, bricks, wound, table);
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

    //a deforming mesh where it is at time, linearly between the samples either side, and still before the first and from the last on (movingInterval);
    //and its field there
    void deformTo(double time, cudaStream_t stream){
        if(time == builtAt){
            return;
        }
        builtAt = time;
        int interval = movingInterval(sampleTimes, time);
        int from = time < sampleTimes[0] ? 0 : (int)sampleTimes.size() - 1, to = from;
        double blend = 0.0, perSecond = 0.0;
        if(interval >= 0){
            from = interval;
            to = interval + 1;
            double span = sampleTimes[to] - sampleTimes[from];
            blend = std::min(std::max((time - sampleTimes[from]) / span, 0.0), 1.0);
            perSecond = 1.0 / span;
        }
        load(from, to, stream);
        interpolateVertices<<<numVertices / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVertices, fromSample, toSample, (float)blend, (float)perSecond, vertices, vertexVelocities);
        farVelocity = interval >= 0 ? meanVelocities[interval] : make_float3(0.0f, 0.0f, 0.0f);
        speed = interval >= 0 ? fastestVertices[interval] : 0.0f;
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
//3x4, own space to world. With an interval, that interval's interpolation, carried on past its ends
void ObstacleSet::rigidAt(const Built& obstacle, double time, double matrix[12], int interval) const{
    size_t keys = obstacle.keyTimes.size();
    size_t key = 0;
    double blend = 0.0;
    if(interval >= 0){
        key = interval;
        blend = (time - obstacle.keyTimes[key]) / (obstacle.keyTimes[key + 1] - obstacle.keyTimes[key]);
    }
    else if(keys > 1 && time > obstacle.keyTimes[0]){
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
    fastestSurface = 0.0f;
    for(size_t index = 0; index < built.size(); ++index){
        Built& obstacle = built[index];
        ObstacleState& state = current.items[index];
        if(obstacle.field != nullptr && obstacle.field->deforms()){
            obstacle.field->deformTo(time, stream);
            obstacle.sdf = obstacle.field->sdf();
            current.moving = current.moving || obstacle.field->movesAt(time);
            fastestSurface = std::max(fastestSurface, obstacle.field->speed);
        }
        state.kind = obstacle.kind;
        state.sdf = obstacle.sdf;
        bool deforming = obstacle.field != nullptr && obstacle.field->deforms();
        if(obstacle.sdf.velocities != nullptr && !deforming){   //a level set with velocities: its surface moves as they say, where it is
            current.moving = true;
            fastestSurface = std::max(fastestSurface, obstacle.fieldSpeed);
        }
        state.low = obstacle.low;
        state.high = obstacle.high;
        state.radius = obstacle.radius;
        state.friction = obstacle.friction;
        float reach = (float)(3.0*voxel);
        double now[12];
        rigidAt(obstacle, time, now);
        for(int axis = 0; axis < 3; ++axis){
            obstacle.worldLow[axis] = deforming ? component(obstacle.field->meshLow, axis) : INFINITY;
            obstacle.worldHigh[axis] = deforming ? component(obstacle.field->meshHigh, axis) : -INFINITY;
        }
        for(int corner = 0; corner < 8 && !deforming; ++corner){    //a rigid one's own bounds, turned and moved: the box around their corners
            double local[3] = {corner & 1 ? obstacle.objectHigh.x : obstacle.objectLow.x, corner & 2 ? obstacle.objectHigh.y : obstacle.objectLow.y,
                               corner & 4 ? obstacle.objectHigh.z : obstacle.objectLow.z};
            for(int row = 0; row < 3; ++row){
                double world = now[4*row]*local[0] + now[4*row + 1]*local[1] + now[4*row + 2]*local[2] + now[4*row + 3];
                obstacle.worldLow[row] = std::min(obstacle.worldLow[row], world);
                obstacle.worldHigh[row] = std::max(obstacle.worldHigh[row], world);
            }
        }
        state.reachLow = make_float3((float)obstacle.worldLow[0] - reach, (float)obstacle.worldLow[1] - reach, (float)obstacle.worldLow[2] - reach);
        state.reachHigh = make_float3((float)obstacle.worldHigh[0] + reach, (float)obstacle.worldHigh[1] + reach, (float)obstacle.worldHigh[2] + reach);
        //the inverse of a rigid transform: the transposed rotation, and the translation rotated back and negated
        for(int row = 0; row < 3; ++row){
            for(int column = 0; column < 3; ++column){
                state.worldToObject[4*row + column] = (float)now[4*column + row];
            }
            state.worldToObject[4*row + 3] = (float)-(now[row]*now[3] + now[4 + row]*now[7] + now[8 + row]*now[11]);
        }
        //the velocity of the point at x: d/dt of where its own-space point goes, M'(t) M(t)^-1 (x, 1), M' by central differences on the interval it's
        //moving through (movingInterval)
        double velocity[12] = {};
        int interval = movingInterval(obstacle.keyTimes, time);
        if(interval >= 0){
            double h = 1e-4;
            double before[12], after[12];
            rigidAt(obstacle, time - h, before, interval);
            rigidAt(obstacle, time + h, after, interval);
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
            for(int corner = 0; corner < 8; ++corner){  //its velocity is affine in x, so over its bounds it's fastest at a corner
                double local[3] = {corner & 1 ? obstacle.objectHigh.x : obstacle.objectLow.x, corner & 2 ? obstacle.objectHigh.y : obstacle.objectLow.y,
                                   corner & 4 ? obstacle.objectHigh.z : obstacle.objectLow.z};
                double world[3], moving[3];
                for(int row = 0; row < 3; ++row){
                    world[row] = now[4*row]*local[0] + now[4*row + 1]*local[1] + now[4*row + 2]*local[2] + now[4*row + 3];
                }
                for(int row = 0; row < 3; ++row){
                    moving[row] = velocity[4*row]*world[0] + velocity[4*row + 1]*world[1] + velocity[4*row + 2]*world[2] + velocity[4*row + 3];
                }
                fastestSurface = std::max(fastestSurface, (float)std::sqrt(moving[0]*moving[0] + moving[1]*moving[1] + moving[2]*moving[2]));
            }
        }
        for(int i = 0; i < 12; ++i){
            state.velocity[i] = (float)velocity[i];
        }
    }
}

void ObstacleSet::bounds(int index, double low[3], double high[3]) const{
    for(int axis = 0; axis < 3; ++axis){
        low[axis] = built[index].worldLow[axis];
        high[axis] = built[index].worldHigh[axis];
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
    voxel = voxelSize;
    builtIndex.assign(obstacles.size(), -1);
    if(obstacles.size() > MAX_OBSTACLES){
        std::cerr<<"ObstacleSet: only the first "<<MAX_OBSTACLES<<" obstacles count\n";
    }
    for(size_t index = 0; index < obstacles.size() && index < MAX_OBSTACLES; ++index){
        const SceneObstacle& obstacle = obstacles[index];
        Built made;
        made.kind = obstacle.kind == SceneObstacle::MESH || obstacle.kind == SceneObstacle::FIELD ? OBSTACLE_MESH : obstacle.kind == SceneObstacle::BOX ? OBSTACLE_BOX : OBSTACLE_SPHERE;
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
            made.objectLow = made.low;
            made.objectHigh = made.high;
        }
        else if(made.kind == OBSTACLE_SPHERE){  //a sphere stays one, scaled by its largest scale; it's centred on its own origin
            double largest = std::max(std::fabs(scale[0]), std::max(std::fabs(scale[1]), std::fabs(scale[2])));
            made.radius = (float)(obstacle.radius*largest);
            made.objectLow = make_float3(-made.radius, -made.radius, -made.radius);
            made.objectHigh = make_float3(made.radius, made.radius, made.radius);
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
        else if(obstacle.kind == SceneObstacle::FIELD){     //a level set, sampled on the host already
            made.field = new MeshField();
            made.field->upload(*obstacle.field, stream);
            made.sdf = made.field->sdf();
            made.objectLow = made.field->meshLow;
            made.objectHigh = made.field->meshHigh;
            made.fieldSpeed = obstacle.field->fastest;
            size_t bytes = sizeof(float)*(size_t)made.field->poolBricks*SDF_BRICK*SDF_BRICK*SDF_BRICK*(obstacle.field->velocities.empty() ? 1 : 4) + sizeof(int)*made.field->numBricks;
            std::cerr<<"Obstacle "<<index<<": a level set, "<<made.field->firstUsed<<" of "<<made.field->numBricks<<" bricks near its surface"
                     <<(obstacle.field->velocities.empty() ? "" : ", with velocities")<<" ("<<bytes / (1<<20)<<" MB)\n";
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
            made.objectLow = made.field->meshLow;
            made.objectHigh = made.field->meshHigh;
            const MeshField& field = *made.field;
            size_t bytes = sizeof(float)*(size_t)field.poolBricks*SDF_BRICK*SDF_BRICK*SDF_BRICK*(deforms ? 4 : 1) + sizeof(int)*field.numBricks;
            std::cerr<<"Obstacle "<<index<<": "<<triangles.size()<<" triangles, "<<field.firstUsed<<" of "<<field.numBricks<<" bricks near its surface";
            if(deforms){
                std::cerr<<" at the start, deforming through "<<obstacle.sampleTimes.size()<<" samples from "<<obstacle.sampleTimes.front()<<" s to "<<obstacle.sampleTimes.back()<<" s";
            }
            std::cerr<<" ("<<bytes / (1<<20)<<" MB)\n";
        }
        builtIndex[index] = (int)built.size();
        built.push_back(std::move(made));
    }
    update(0.0);
}

// ---- what the simulation does with them ----

//threads per node block in the obstacle passes: a block's 8^3 slots a thread each. The few nodes near obstacles are all the work, a chain of distance
//lookups per slot, and with only a few warps of them on each SM there's little to hide its latency behind; a thread per slot keeps each chain short
static constexpr int OBSTACLE_THREADS = 512;

//blocks for the passes over the nodes near obstacles: two per SM, each taking the listed nodes in turn. Launching a block per node would spend most of
//the pass starting and ending the thousands of blocks with nothing to do
static uint obstacleBlocks(){
    int device, processors;
    gpuErrchk(cudaGetDevice(&device));
    gpuErrchk(cudaDeviceGetAttribute(&processors, cudaDevAttrMultiProcessorCount, device));
    return 2*processors;
}

void Particles::setObstacles(const std::vector<SceneObstacle>& descriptions){
    obstacles.build(descriptions, grid.cellSize / (2<<refinementLevel), stream);
    obstacles.update(elapsedTime);
}

void Particles::updateObstacles(){
    if(obstacles.count() > 0){
        obstacles.update(elapsedTime);
    }
    placeSourceMeshes();    //meshes of emitters, sinks and fluids move as obstacles do
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

//At the start, the fluid seeded inside an obstacle goes. With air (ids not nullptr), so does the air from a cell of the seeding lattice (spacing a
//side, from the domain's corner) whose centre is inside one, though the particle itself landed outside. The lattice gives a cell to the fluid that
//holds its centre: where liquid is modelled up to a wall, the cells the wall cuts through are the air's, and what they leave outside the wall is air
//in a sliver between the liquid and the wall it was meant to touch, which then rises through the liquid from every wall
__global__ void markInsideObstacles(uint numParticles, const double* px, const double* py, const double* pz, const uint* ids, Grid grid, double spacing, Obstacles obstacles,
                                    char* removed){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        float distance;
        float3 normal;
        bool inside = nearestObstacle(obstacles, make_float3((float)px[index], (float)py[index], (float)pz[index]), distance, normal) >= 0 && distance < 0.0f;
        if(!inside && ids != nullptr && (ids[index] & AIR_PARTICLE)){
            double position[3] = {px[index], py[index], pz[index]};
            double low[3] = {grid.negX, grid.negY, grid.negZ};
            float centre[3];
            for(int axis = 0; axis < 3; ++axis){
                centre[axis] = (float)(low[axis] + (floor((position[axis] - low[axis]) / spacing) + 0.5)*spacing);
            }
            inside = nearestObstacle(obstacles, make_float3(centre[0], centre[1], centre[2]), distance, normal) >= 0 && distance < 0.0f;
        }
        if(inside){
            removed[index] = 1;
        }
    }
}

void Particles::markParticlesInsideObstacles(){
    if(obstacles.count() == 0 || size == 0 || substepIndex != 0){
        return;
    }
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    markInsideObstacles<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, px.devPtr(), py.devPtr(), pz.devPtr(), twoPhase.particles() ? particleIds.devPtr() : nullptr, grid,
        voxelSize / sources.latticePerSide, obstacles.state(), removedFlags.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}

VoxelPlaces Particles::voxelPlaces(){
    return {nodeIndexUsedVoxels.devPtr(), nodeCells.devPtr(), voxelIDsUsed.devPtr(), grid, (int)(2<<refinementLevel), (int)std::floor(radius)};
}

// ---- cut cells: how much of each voxel face is open to the fluid ----
//
//Cut cells (Batty, Bertails and Bridson, A fast variational framework for accurate solid-fluid coupling, 2007): a voxel partly inside an obstacle stays
//an unknown, and each face of its pressure equation weighs as much as the face is open, so the solve sees the surface as it is rather than a staircase of
//whole voxels. Only the values the solvers are handed change (weighCutFaces, after cudaGetA, and the divergence in obstacleFaceFlux), never which voxels
//they couple. A face's open fraction is a pure function of where it is, so the two voxels it divides, on whichever partitions, always agree on it

//of the segment between two points, the fraction inside a solid, with the distance linear between them
__device__ inline float segmentInside(float a, float b){
    if(a < 0.0f && b < 0.0f){
        return 1.0f;
    }
    if(a < 0.0f){
        return a / (a - b);
    }
    if(b < 0.0f){
        return b / (b - a);
    }
    return 0.0f;
}

//of a square, the fraction inside a solid, from the distances at its corners in order around it: where all or none are inside, all or none of it; where
//some are, the polygon their edges' crossings cut off. Two inside on a diagonal are joined, or not, as the square's centre is inside or not
__device__ inline float squareInside(float c0, float c1, float c2, float c3){
    float c[4] = {c0, c1, c2, c3};
    int inside = (c0 < 0.0f) + (c1 < 0.0f) + (c2 < 0.0f) + (c3 < 0.0f);
    auto turn = [&](){
        float first = c[0];
        c[0] = c[1];
        c[1] = c[2];
        c[2] = c[3];
        c[3] = first;
    };
    if(inside == 4 || inside == 0){
        return inside == 4 ? 1.0f : 0.0f;
    }
    if(inside == 3){    //all but the triangle at the one outside
        while(c[0] < 0.0f){
            turn();
        }
        return 1.0f - 0.5f*(1.0f - segmentInside(c[0], c[3]))*(1.0f - segmentInside(c[0], c[1]));
    }
    if(inside == 1){    //the triangle at the one inside
        while(c[0] >= 0.0f){
            turn();
        }
        return 0.5f*segmentInside(c[0], c[3])*segmentInside(c[0], c[1]);
    }
    while(c[0] >= 0.0f || !(c[1] < 0.0f || c[2] < 0.0f)){     //two inside: one first, and the other next or opposite
        turn();
    }
    if(c[1] < 0.0f){    //next to each other: a trapezoid
        return 0.5f*(segmentInside(c[0], c[3]) + segmentInside(c[1], c[2]));
    }
    if(0.25f*(c[0] + c[1] + c[2] + c[3]) < 0.0f){  //opposite, joined: all but the triangles at the two outside
        return 1.0f - 0.5f*(1.0f - segmentInside(c[0], c[3]))*(1.0f - segmentInside(c[2], c[3])) - 0.5f*(1.0f - segmentInside(c[0], c[1]))*(1.0f - segmentInside(c[2], c[1]));
    }
    return 0.5f*segmentInside(c[0], c[1])*segmentInside(c[0], c[3]) + 0.5f*segmentInside(c[2], c[1])*segmentInside(c[2], c[3]);    //opposite, apart
}

//how open a cut voxel is, the mean of its faces: what scales its density correction (obstacleFaceFlux), as its row of the pressure equation is scaled.
//Not how much fluid it holds at rest, which goes by volume (coveredFraction): for a corner a surface cuts off at x + y + z = c, the faces' mean is
//c^2/4, the volume c^3/6
__device__ inline float openVolume(const CutFaces& cut){
    float sum = 0.0f;
    for(int face = 0; face < 6; ++face){
        sum += cut.open[face];
    }
    return sum / 6.0f;
}

//the fraction of a voxel inside obstacles, from the distances at the centres of its eighths: each inside by as much as a plane that far from its centre
//would cut it (a ramp half a voxel wide), so it moves smoothly with the surface. What P2G's count of the particles in a voxel falls short by at rest. Off
//the exact volume by 1.4% at most around a sphere 13 voxels across and 4% at a box's edges, where a count of 64 points inside is off by up to 9%, and
//trilinear blends of the corners by 24%, as they round the edge off
__device__ inline float coveredFraction(const float eighths[8], float h){
    float sum = 0.0f;
    #pragma unroll
    for(int eighth = 0; eighth < 8; ++eighth){
        sum += fminf(fmaxf(0.5f - 2.0f*eighths[eighth]/h, 0.0f), 1.0f);
    }
    return sum / 8.0f;
}

__device__ inline CutFaces cutFaces(const Obstacles& obstacles, const VoxelPlaces& places, int3 voxel){
    CutFaces cut;
    float distance;
    float3 normal;
    bool any = nearestObstacle(obstacles, places.point(voxel, 0.5f, 0.5f, 0.5f), distance, normal) >= 0;
    cut.near = any && fabsf(distance) <= 0.9f*places.voxelSize();
    cut.centre = any ? distance : INFINITY;
    if(!cut.near){
        for(int face = 0; face < 6; ++face){
            cut.open[face] = !any || distance > 0.0f ? 1.0f : 0.0f;
        }
        cut.covered = !any || distance > 0.0f ? 0.0f : 1.0f;
        return cut;
    }
    float corners[8];   //x fastest; offsets are whole voxels, so a corner two voxels share is the same point to the bit
    float eighths[8];   //at the centres of its eighths, for how much of it is covered
    #pragma unroll      //sixteen independent lookups, in flight together rather than one after another
    for(int corner = 0; corner < 8; ++corner){
        float x = (float)(corner & 1), y = (float)(corner >> 1 & 1), z = (float)(corner >> 2);
        nearestObstacle(obstacles, places.point(voxel, x, y, z), corners[corner], normal);
        nearestObstacle(obstacles, places.point(voxel, 0.25f + 0.5f*x, 0.25f + 0.5f*y, 0.25f + 0.5f*z), eighths[corner], normal);
    }
    for(int face = 0; face < 6; ++face){
        int axis = face/2, side = face % 2, u = (axis + 1) % 3, w = (axis + 2) % 3;
        auto at = [&](int du, int dw){
            return corners[side << axis | du << u | dw << w];
        };
        cut.open[face] = 1.0f - squareInside(at(0, 0), at(1, 0), at(1, 1), at(0, 1));
    }
    cut.covered = coveredFraction(eighths, places.voxelSize());
    return cut;
}

//how fast an obstacle's velocity field spreads out at x, its divergence: 0 for a rigid one (its velocity matrix's trace, to rounding); for a field's,
//central differences of it half a voxel either side
__device__ inline float obstacleDivergence(const ObstacleState& obstacle, float3 x, float h){
    if(obstacle.kind == OBSTACLE_MESH && obstacle.sdf.velocities != nullptr){
        float sum = 0.0f;
        for(int axis = 0; axis < 3; ++axis){
            float3 step = make_float3(axis == 0 ? 0.5f*h : 0.0f, axis == 1 ? 0.5f*h : 0.0f, axis == 2 ? 0.5f*h : 0.0f);
            sum += component(sdfVelocity(obstacle.sdf, x + step), axis) - component(sdfVelocity(obstacle.sdf, x - step), axis);
        }
        return sum / h;
    }
    return obstacle.velocity[0] + obstacle.velocity[5] + obstacle.velocity[10];
}

//per stored voxel, a block per node: whether it's solid. One the fluid has reached is once obstacles close every face of it: partly inside one, it stays
//fluid, through the open part of its faces. One the fluid hasn't, once its centre is inside one, as every voxel was before cut cells: an empty sliver
//left as air, behind a moving obstacle, would hold p = 0 deep in the fluid, and the solve, weighing its faces only as much as they're open, would let
//their velocities run away (the hydrostatic head across a face 1% open drives it at metres a second), and advection would carry particles through them
//many voxels a substep. The domain's walls stay the domain's
__global__ void findSolidVoxels(Obstacles obstacles, VoxelPlaces places, const char* walls, const char* solveCodes, const uint* voxelOwners, char* solid, char* near, uint* opens,
                                uint* nodes){
    uint node = blockIdx.x;
    uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
    uint last = places.nodeVoxelEnds[node];
    uint cell = places.nodeCells[node];
    bool touches = false;   //whether this node's block has anything for the obstacle passes: a voxel within 1.5 voxels of a surface, or in one
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        CutFaces cut = cutFaces(obstacles, places, places.voxelOf(cell, places.voxelSlots[index]));
        bool closed = true;
        int bits = cut.near ? NEAR_SURFACE : 0;
        uint words[3] = {0, 0, 0};
        for(int face = 0; face < 6; ++face){
            uint open = __float2uint_rn(cut.open[face]*OPEN_FULL);   //what every pass reads, so they all agree
            words[face/2] |= face % 2 ? open << 16 : open;
            closed = closed && open == 0;
            bits |= open == 0 ? CLOSED_FACE << face : 0;
        }
        if(cut.near){
            for(int axis = 0; axis < 3; ++axis){
                opens[4*(size_t)index + axis] = words[axis];
            }
            opens[4*(size_t)index + 3] = __float2uint_rn(cut.covered*OPEN_FULL);
        }
        bool empty = !solveCodes[voxelOwners[index]];    //no fluid reached it: it's no unknown, and the solve would take it for air
        solid[index] = !walls[index] && (closed || (empty && cut.centre < 0.0f));
        near[index] = (char)bits;
        touches = touches || cut.centre < 1.5f*places.voxelSize();
    }
    if(__syncthreads_or(touches) && threadIdx.x == 0){     //onto the list of nodes near obstacles: its count, then them, in whatever order they come
        nodes[1 + atomicAdd(nodes, 1u)] = node;
    }
}

//an unknown sees each of its faces obstacles close completely as a wall, a block per node: towards a solid voxel, an unstored one inside an obstacle
//(behind a moving obstacle particles can fall back and reach none of it, and the solve would take it for air and pull the fluid in), or even another
//unknown across a wall thinner than a voxel. So does a face that's open itself but leads into a solid voxel (findSolidVoxels' empty slivers) or an
//unstored one whose centre is inside an obstacle: its closed bit says the obstacle moves across all of it (obstacleFaceFlux), as the ghost velocities
//will, while its cached fraction stays what the surface makes it, which the voxel's volume is measured by. An unknown with no face left that isn't a
//wall has no pressure equation, and is retired with the solid voxels.
//An air voxel above that named it across a closed face no longer does, so the pressure update leaves that face to the obstacle; only the voxel below
//writes that place
__global__ void wallOffSolidNeighbors(Obstacles obstacles, VoxelPlaces places, const uint* nodes, char* solid, char* near, uint* opens, const char* solveCodes, uint* neighborNx,
                                      uint* neighborPx, uint* neighborNy, uint* neighborPy, uint* neighborNz, uint* neighborPz){
    for(uint listed = blockIdx.x; listed < nodes[0]; listed += gridDim.x){    //the nodes near obstacles (findSolidVoxels), a block each in turn
        uint node = nodes[1 + listed];
        uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
        uint last = places.nodeVoxelEnds[node];
        uint cell = places.nodeCells[node];
        uint* neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
        for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
            if(!solveCodes[index] || solid[index]){
                continue;
            }
            int bits = near[index];
            bool reclosed = false;  //whether a face open itself got closed here
            int walls = 0;
            for(int face = 0; face < 6; ++face){
                uint neighbor = neighbors[face][index];
                if(neighbor == WALL_VOXEL){
                    ++walls;
                    continue;
                }
                bool closed = bits & CLOSED_FACE << face;
                if(!closed){
                    bool beyond;    //solid past it
                    if(neighbor < WALL_VOXEL){
                        beyond = solid[neighbor];
                    }
                    else{
                        int3 voxel = places.voxelOf(cell, places.voxelSlots[index]);
                        int axis = face/2, by = face % 2 ? 1 : -1;
                        voxel = make_int3(voxel.x + (axis == 0)*by, voxel.y + (axis == 1)*by, voxel.z + (axis == 2)*by);
                        float distance;
                        float3 normal;
                        beyond = nearestObstacle(obstacles, places.point(voxel, 0.5f, 0.5f, 0.5f), distance, normal) >= 0 && distance < 0.0f;
                    }
                    if(!beyond){
                        continue;
                    }
                    reclosed = true;
                    bits |= CLOSED_FACE << face;
                }
                if(face % 2 && neighbor < WALL_VOXEL && !solveCodes[neighbor] && !solid[neighbor] && neighbors[face - 1][neighbor] == index){
                    neighbors[face - 1][neighbor] = NO_VOXEL;
                }
                neighbors[face][index] = WALL_VOXEL;
                ++walls;
            }
            if(reclosed){
                if(!(bits & NEAR_SURFACE)){     //no surface cut it, so it has no cached fractions yet: all its faces are open, and none of it covered
                    for(int axis = 0; axis < 3; ++axis){
                        opens[4*(size_t)index + axis] = (uint)OPEN_FULL | (uint)OPEN_FULL << 16;
                    }
                    opens[4*(size_t)index + 3] = 0;
                }
                near[index] = (char)(bits | NEAR_SURFACE);
            }
            if(walls == 6){     //nothing on any side but walls: no unknown reads it (a face it closed is closed from the other side too), so this is safe now
                solid[index] = 1;
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

//each face of a cut unknown's pressure equation weighed by how open it is, a block per node: cudaGetA gave every face that isn't a wall scale, towards
//the neighbouring unknown (negative) and on the diagonal. Voxels no surface cuts keep cudaGetA's values exactly
__global__ void weighCutFaces(Obstacles obstacles, VoxelPlaces places, const uint* nodes, const char* near, const uint* opens, const char* solveCodes, const uint* neighborNx, const uint* neighborPx,
                              const uint* neighborNy, const uint* neighborPy, const uint* neighborNz, const uint* neighborPz, float* Anx, float* Apx, float* Any, float* Apy, float* Anz, float* Apz,
                              float* Adiag, float scale){
    for(uint listed = blockIdx.x; listed < nodes[0]; listed += gridDim.x){    //the nodes near obstacles (findSolidVoxels), a block each in turn
        uint node = nodes[1 + listed];
        uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
        uint last = places.nodeVoxelEnds[node];
        const uint* neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
        float* coefficients[6] = {Anx, Apx, Any, Apy, Anz, Apz};
        for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
            if(!solveCodes[index] || !(near[index] & NEAR_SURFACE)){
                continue;
            }
            CutFaces cut = cachedCut(opens, index);
            float diagonal = 0.0f;  //summed in cudaGetA's order, so a voxel with every face open gets its value back to the bit
            for(int face = 0; face < 6; ++face){
                if(neighbors[face][index] != WALL_VOXEL){
                    //the face's weight rounded on its own, never folded into the sum as a multiply-add: the diagonal then takes the very number the
                    //coupling stores, and an unknown with no face to air has a diagonal that's its couplings' sum to the bit (Stencil::air)
                    float weight = __fmul_rn(scale, cut.open[face]);
                    coefficients[face][index] = coefficients[face][index] != 0.0f ? -weight : 0.0f;
                    diagonal += weight;
                }
            }
            Adiag[index] = diagonal;
        }
    }
}

void Particles::weighCutCells(float scale){
    if(obstacles.count() == 0 || numStoredNodes == 0){
        return;
    }
    weighCutFaces<<<obstacleBlocks(), OBSTACLE_THREADS, 0, stream>>>(obstacles.state(), voxelPlaces(), obstacleNodes.devPtr(), obstacleNear.devPtr(), obstacleOpen.devPtr(), solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(),
        neighborNz.devPtr(), neighborPz.devPtr(), Anx.devPtr(), Apx.devPtr(), Any.devPtr(), Apy.devPtr(), Anz.devPtr(), Apz.devPtr(), Adiag.devPtr(), scale);
    gpuErrchk(cudaPeekAtLastError());
}

void Particles::markObstacleSolids(){
    if(obstacles.count() == 0){
        return;
    }
    uint numVoxels = voxelIDsUsed.size();
    obstacleSolids.resizeAsync(numVoxels, stream);
    obstacleNear.resizeAsync(numVoxels, stream);
    obstacleOpen.resizeAsync(4*numVoxels, stream);
    obstacleNodes.resizeAsync(numStoredNodes + 1, stream);
    gpuErrchk(cudaMemsetAsync(obstacleNodes.devPtr(), 0, sizeof(uint), stream));    //the list starts empty
    if(numVoxels == 0){
        return;
    }
    findSolidVoxels<<<numStoredNodes, OBSTACLE_THREADS, 0, stream>>>(obstacles.state(), voxelPlaces(), solids.devPtr(), solveCodes.devPtr(), voxelOwners.devPtr(), obstacleSolids.devPtr(),
        obstacleNear.devPtr(), obstacleOpen.devPtr(), obstacleNodes.devPtr());
    wallOffSolidNeighbors<<<obstacleBlocks(), OBSTACLE_THREADS, 0, stream>>>(obstacles.state(), voxelPlaces(), obstacleNodes.devPtr(), obstacleSolids.devPtr(), obstacleNear.devPtr(), obstacleOpen.devPtr(),
        solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr());
    retireSolidVoxels<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, obstacleSolids.devPtr(), solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(),
        neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}

//the flow obstacles' surfaces make through the unknowns' faces, which cudaCalcDivU took as all fluid: through a face open a fraction w, the fluid flows
//on the open part and the obstacle's surface moves across the rest, so (1 - w)(u_obstacle - u_fluid) more than cudaCalcDivU took. A closed face is a
//wall, where it took none, which comes to the obstacle's velocity across it; a still obstacle only takes the fluid's flow away. The obstacle's flow through
//the closed parts is how fast it moves into the voxel only if its velocity field doesn't spread out inside it, as a rigid one's doesn't: a deforming
//one's, extended from its surface, can, and what spreads inside the solid part of the voxel moves no fluid, so it's taken back out, (1 - open) dx
//div(u_obstacle). Without that, a breathing sphere asks a voxel it nearly fills to shed most of a voxel's worth of fluid through its last open sliver. And the density
//correction cudaCalcDivU asked of a cut voxel is a source in its fluid, which only fills the open part of it: it's scaled by how open the voxel is, the
//mean of its faces, as its row of the pressure equation is. A sliver of fluid would otherwise ask a whole voxel's worth of its few faces, and its
//pressure, and their velocities, would run away
__global__ void obstacleFaceFlux(Obstacles obstacles, VoxelPlaces places, uint3 domainVoxels, const uint* nodes, const char* near, const uint* opens, const char* solveCodes, const uint* neighborNx,
                                 const uint* neighborPx, const uint* neighborNy, const uint* neighborPy, const uint* neighborNz, const uint* neighborPz, const float* ux, const float* uy, const float* uz,
                                 float* divU){
    for(uint listed = blockIdx.x; listed < nodes[0]; listed += gridDim.x){    //the nodes near obstacles (findSolidVoxels), a block each in turn
        uint node = nodes[1 + listed];
        uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
        uint last = places.nodeVoxelEnds[node];
        uint cell = places.nodeCells[node];
        const uint* lower[3] = {neighborNx, neighborNy, neighborNz};
        const uint* upper[3] = {neighborPx, neighborPy, neighborPz};
        const float* velocities[3] = {ux, uy, uz};
        uint size[3] = {domainVoxels.x, domainVoxels.y, domainVoxels.z};
        for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
            if(!solveCodes[index] || !(near[index] & NEAR_SURFACE)){
                continue;
            }
            int3 voxel = places.voxelOf(cell, places.voxelSlots[index]);
            CutFaces cut = cachedCut(opens, index);
            int coordinates[3] = {voxel.x, voxel.y, voxel.z};
            //cudaCalcDivU's flow out of the voxel, worked out as it does, so what it added on top is the density correction
            float flow = 0.0f;
            for(int axis = 0; axis < 3; ++axis){
                uint above = upper[axis][index];
                flow += (above < WALL_VOXEL ? velocities[axis][above] : 0.0f) - (lower[axis][index] == WALL_VOXEL ? 0.0f : velocities[axis][index]);
            }
            float voxelOpen = openVolume(cut);
            float correction = divU[index] - flow;
            float flux = (voxelOpen - 1.0f)*correction;
            for(int face = 0; face < 6; ++face){    //a face open itself but closed as it leads into an empty sliver (wallOffSolidNeighbors) is closed all over
                if(near[index] & CLOSED_FACE << face){
                    cut.open[face] = 0.0f;
                }
            }
            //the faces obstacles cut or close (not the domain's own walls, nor faces all fluid), and the obstacles at them and at the centre, looked up
            //together so their latencies overlap
            bool cutFace[6];
            float3 points[6];
            int solids[6];
            for(int face = 0; face < 6; ++face){
                int axis = face/2, side = face % 2;    //side 0 the lower face, 1 the upper
                uint neighbor = (side ? upper : lower)[axis][index];
                int across = coordinates[axis] + (side ? 1 : -1);
                cutFace[face] = cut.open[face] != 1.0f && !(neighbor == WALL_VOXEL && (across < 0 || across >= (int)size[axis]));
                float offset[3] = {0.5f, 0.5f, 0.5f};
                offset[axis] = side ? 1.0f : 0.0f;
                points[face] = places.point(voxel, offset[0], offset[1], offset[2]);
            }
            float3 centre = places.point(voxel, 0.5f, 0.5f, 0.5f);
            int middle;
            {
                float distance;
                float3 normal;
                middle = voxelOpen < 1.0f ? nearestObstacle(obstacles, centre, distance, normal) : -1;
                for(int face = 0; face < 6; ++face){
                    solids[face] = cutFace[face] ? nearestObstacle(obstacles, points[face], distance, normal) : -1;
                }
            }
            if(middle >= 0){
                flux -= (1.0f - voxelOpen)*places.voxelSize()*obstacleDivergence(obstacles.items[middle], centre, places.voxelSize());
            }
            for(int face = 0; face < 6; ++face){
                if(!cutFace[face]){
                    continue;
                }
                int axis = face/2, side = face % 2;
                uint neighbor = (side ? upper : lower)[axis][index];
                float moving = solids[face] >= 0 ? component(obstacleVelocity(obstacles.items[solids[face]], points[face]), axis) : 0.0f;
                float fluid = neighbor == WALL_VOXEL ? 0.0f : side ? (neighbor < WALL_VOXEL ? velocities[axis][neighbor] : 0.0f) : velocities[axis][index];  //as cudaCalcDivU took it
                float more = (1.0f - cut.open[face])*(moving - fluid);
                flux += side ? more : -more;
            }
            divU[index] += flux;
        }
    }
}

void Particles::addObstacleFlux(){
    if(obstacles.count() == 0 || numStoredNodes == 0){
        return;
    }
    uint interiorWidth = 2<<refinementLevel;
    uint3 domainVoxels = make_uint3(grid.sizeX*interiorWidth, grid.sizeY*interiorWidth, grid.sizeZ*interiorWidth);
    obstacleFaceFlux<<<obstacleBlocks(), OBSTACLE_THREADS, 0, stream>>>(obstacles.state(), voxelPlaces(), domainVoxels, obstacleNodes.devPtr(), obstacleNear.devPtr(), obstacleOpen.devPtr(),
        solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(),
        neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(), voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr(), divU.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}

//One layer of the faces around and inside obstacles, a block per node of this partition's own: the node's block of voxels (with its apron, from their
//owners) goes into shared memory, with the places no voxel is stored taken as inside an obstacle if their centres are, and each of its interior voxels'
//faces with an obstacle on either side gets a velocity:
//- a face between fluid and obstacle: the obstacle's velocity across it, the flow the pressure solve gave it (addObstacleFlux)
//- a face inside the obstacle: the fluid's velocity along the surface from the faces of the same component next to it, blended towards the obstacle's
//  by its friction; across the surface, the obstacle's. Layer 1 fills the faces next to fluid faces, layer 2 the faces next to those
__global__ void obstacleGhostFaces(int layer, Obstacles obstacles, VoxelPlaces places, const uint* nodes, uint numOwnNodes, const uint* voxelOwners, const char* solid, const char* near,
                                   const float* beforeX, const float* beforeY, const float* beforeZ, float* ux, float* uy, float* uz){
    extern __shared__ float blockVelocities[];  //per component, a value per slot; then per slot, whether it's stored, whether it's solid, its near bits
    int voxels1D = places.interiorWidth + 2*places.apronCells;
    int voxels3D = voxels1D*voxels1D*voxels1D;
    char* stored = (char*)(blockVelocities + 3*voxels3D);
    char* inside = stored + voxels3D;
    char* bits = inside + voxels3D;
    for(uint listed = blockIdx.x; listed < nodes[0]; listed += gridDim.x){    //the nodes near obstacles (findSolidVoxels), a block each in turn
        uint node = nodes[1 + listed];
        if(node >= numOwnNodes){    //another partition's, which it fills itself
            continue;
        }
        uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
        uint last = places.nodeVoxelEnds[node];
        uint cell = places.nodeCells[node];
        float* velocities[3] = {ux, uy, uz};
        for(int slot = threadIdx.x; slot < voxels3D; slot += blockDim.x){
            stored[slot] = 0;
            inside[slot] = 0;
            bits[slot] = 0;
        }
        __syncthreads();
        const float* before[3] = {beforeX, beforeY, beforeZ};
        for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
            int slot = places.voxelSlots[index];
            uint owner = voxelOwners[index];
            for(int dim = 0; dim < 3; ++dim){
                blockVelocities[dim*voxels3D + slot] = before[dim][owner];
            }
            stored[slot] = 1;
            inside[slot] = solid[owner];
            bits[slot] = near[owner];
        }
        __syncthreads();
        for(int slot = threadIdx.x; slot < voxels3D; slot += blockDim.x){
            int3 voxel = places.voxelOf(cell, slot);
            if(!stored[slot] && places.inDomain(voxel)){    //past the domain, the domain's walls hold
                float distance;
                float3 normal;
                inside[slot] = nearestObstacle(obstacles, places.point(voxel, 0.5f, 0.5f, 0.5f), distance, normal) >= 0 && distance < 0.0f;
            }
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
        //a dim face is a fluid face if it's stored, neither voxel either side of it is in an obstacle, and obstacles don't close it (a wall thinner than a
        //voxel, whose velocity this pass writes, so another block may be writing it as this one reads); an inside face if both voxels are in obstacles
        auto fluidFace = [&](int slot, int dim){
            int below = step(slot, dim, -1);
            return below >= 0 && stored[slot] && !inside[slot] && !inside[below] && !(bits[slot] & CLOSED_FACE << 2*dim);
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
            //which of its lower faces have anything to do, and the obstacles' distance at those, looked up together so their latencies overlap: the blocks
            //here are few, and each waits on its chain of lookups
            bool here = inside[slot];
            bool there[3], closed[3];
            float3 faces[3], normals[3];
            int nearest[3];
            for(int dim = 0; dim < 3; ++dim){
                there[dim] = inside[slot - strides[dim]];   //interior, so never past the block's edge
                closed[dim] = !here && !there[dim] && (near[index] & CLOSED_FACE << 2*dim);    //a face obstacles close between two voxels that aren't solid
                float offset[3] = {0.5f, 0.5f, 0.5f};
                offset[dim] = 0.0f;
                faces[dim] = places.point(voxel, offset[0], offset[1], offset[2]);
            }
            for(int dim = 0; dim < 3; ++dim){
                float distance;
                nearest[dim] = here || there[dim] || closed[dim] ? nearestObstacle(obstacles, faces[dim], distance, normals[dim]) : -1;
            }
            for(int dim = 0; dim < 3; ++dim){
                int obstacle = nearest[dim];
                if(obstacle < 0){
                    continue;
                }
                float3 face = faces[dim];
                float3 normal = normals[dim];
                float solidVelocity = component(obstacleVelocity(obstacles.items[obstacle], face), dim);
                float value;
                if(here != there[dim] || closed[dim]){     //between fluid and obstacle
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
        __syncthreads();    //before the next node reuses the shared memory
    }
}

//What crosses a face an obstacle cuts is the fluid's velocity over its open part w and the obstacle's over the rest: w u + (1 - w) u_obstacle, as the
//divergence counts it (obstacleFaceFlux). That's what G2P and advection take there, FLIP's old velocity before the solve and the new one after it, rather
//than the open part's alone: the solve holds only w u, and where a surface closes in on fluid, u at a face open a sliver runs to many times the obstacle's
//speed (water a foot squeezes against the floor leaves through a corner of a face at tens of metres a second), which particles would carry off. Mixed, it
//goes smoothly to the obstacle's velocity as w goes to 0, where the face closes and the ghost velocities give it just that. A face is the lower face of
//the voxel above it, a block per node of this partition's own; the ghost pass after it rewrites the faces it owns (closed ones, ones into solids), and the
//domain's walls stay walls
__global__ void mixCutFaces(Obstacles obstacles, VoxelPlaces places, const uint* nodes, uint numOwnNodes, const char* near, const uint* opens, float* ux, float* uy,
                            float* uz){
    int voxels1D = places.interiorWidth + 2*places.apronCells;
    float* velocities[3] = {ux, uy, uz};
    for(uint listed = blockIdx.x; listed < nodes[0]; listed += gridDim.x){    //the nodes near obstacles (findSolidVoxels), a block each in turn
        uint node = nodes[1 + listed];
        if(node >= numOwnNodes){    //another partition's, which it mixes itself
            continue;
        }
        uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
        uint last = places.nodeVoxelEnds[node];
        uint cell = places.nodeCells[node];
        for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
            if(!(near[index] & NEAR_SURFACE)){
                continue;
            }
            int slot = places.voxelSlots[index];
            int x = slot % voxels1D, y = slot / voxels1D % voxels1D, z = slot / (voxels1D*voxels1D);
            if(x < places.apronCells || y < places.apronCells || z < places.apronCells || x >= voxels1D - places.apronCells || y >= voxels1D - places.apronCells ||
               z >= voxels1D - places.apronCells){
                continue;   //only the node's own voxels: their neighbours own the apron's
            }
            int3 voxel = places.voxelOf(cell, slot);
            CutFaces cut = cachedCut(opens, index);
            float3 points[3];
            int nearest[3];
            bool mixes[3];
            for(int dim = 0; dim < 3; ++dim){
                int3 below = make_int3(voxel.x - (dim == 0), voxel.y - (dim == 1), voxel.z - (dim == 2));
                float open = cut.open[2*dim];
                mixes[dim] = open > 0.0f && open < 1.0f && places.inDomain(below) && !(near[index] & CLOSED_FACE << 2*dim);
                float offset[3] = {0.5f, 0.5f, 0.5f};
                offset[dim] = 0.0f;
                points[dim] = places.point(voxel, offset[0], offset[1], offset[2]);
            }
            for(int dim = 0; dim < 3; ++dim){   //looked up together, so their latencies overlap
                float distance;
                float3 normal;
                nearest[dim] = mixes[dim] ? nearestObstacle(obstacles, points[dim], distance, normal) : -1;
            }
            for(int dim = 0; dim < 3; ++dim){
                if(nearest[dim] < 0){
                    continue;
                }
                float open = cut.open[2*dim];
                float solid = component(obstacleVelocity(obstacles.items[nearest[dim]], points[dim]), dim);
                velocities[dim][index] = open*velocities[dim][index] + (1.0f - open)*solid;
            }
        }
    }
}

void Particles::mixObstacleFaces(CudaVec<float>& ux, CudaVec<float>& uy, CudaVec<float>& uz){
    if(obstacles.count() == 0 || numOwnNodes == 0){
        return;
    }
    mixCutFaces<<<obstacleBlocks(), OBSTACLE_THREADS, 0, stream>>>(obstacles.state(), voxelPlaces(), obstacleNodes.devPtr(), numOwnNodes, obstacleNear.devPtr(),
        obstacleOpen.devPtr(), ux.devPtr(), uy.devPtr(), uz.devPtr());
    gpuErrchk(cudaPeekAtLastError());
    for(CudaVec<float>* velocity : {&ux, &uy, &uz}){   //the ghost pass next reads the ghosts' faces, as their owners mixed them
        context->fillGhosts(velocity->devPtr(), stream);
    }
}

void Particles::obstacleGhostVelocities(CudaVec<float>& ux, CudaVec<float>& uy, CudaVec<float>& uz){
    if(obstacles.count() == 0 || numOwnNodes == 0){
        return;
    }
    int voxels1D = numVoxels1D;
    size_t shared = (3*sizeof(float) + 3)*voxels1D*voxels1D*voxels1D;
    Obstacles state = obstacles.state();
    if(viscosity > 0.0){    //a viscous liquid holds to them at least as much as to the walls
        for(int obstacle = 0; obstacle < state.count; ++obstacle){
            state.items[obstacle].friction = std::max(state.items[obstacle].friction, (float)wallStick());
        }
    }
    //Each layer reads the faces as they were before it and writes the faces it sets. Nodes store their own sparse voxels, so where one node's block lacks a
    //voxel another stores, the two can see a face differently (inside by its centre, or fluid by its solid flag): one block would read as fluid a face
    //another is setting, and get either value, by timing
    for(CudaVec<float>& before : ghostBefore){
        if(before.size() < ux.size()){
            before.resizeAsync(ux.size(), stream);
        }
    }
    for(int layer = 1; layer <= 2; ++layer){
        CudaVec<float>* velocity[3] = {&ux, &uy, &uz};
        for(int dim = 0; dim < 3; ++dim){
            gpuErrchk(cudaMemcpyAsync(ghostBefore[dim].devPtr(), velocity[dim]->devPtr(), sizeof(float)*ux.size(), cudaMemcpyDeviceToDevice, stream));
        }
        obstacleGhostFaces<<<obstacleBlocks(), OBSTACLE_THREADS, shared, stream>>>(layer, state, voxelPlaces(), obstacleNodes.devPtr(), numOwnNodes, voxelOwners.devPtr(), obstacleSolids.devPtr(), obstacleNear.devPtr(),
            ghostBefore[0].devPtr(), ghostBefore[1].devPtr(), ghostBefore[2].devPtr(), ux.devPtr(), uy.devPtr(), uz.devPtr());
        gpuErrchk(cudaPeekAtLastError());
        for(CudaVec<float>* velocity : {&ux, &uy, &uz}){   //the next layer, and the ghosts, read what this one wrote
            context->fillGhosts(velocity->devPtr(), stream);
        }
    }
}

//the density correction counts each voxel's particles against the rest count, but a voxel an obstacle partly covers can only hold its uncovered part's
//worth, so it would read as thin and the correction would keep pulling fluid towards the obstacle. Each cut voxel gets credit for its covered part: the
//rest count times the fraction of its volume inside obstacles (coveredFraction), which moves smoothly with the surface
__global__ void creditCoveredVolume(Obstacles obstacles, VoxelPlaces places, const uint* nodes, const char* walls, const char* solid, const char* near, const uint* opens,
                                    float restParticlesPerVoxel,
                                    float* particleCounts){
    for(uint listed = blockIdx.x; listed < nodes[0]; listed += gridDim.x){    //the nodes near obstacles (findSolidVoxels), a block each in turn
        uint node = nodes[1 + listed];
        uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
        uint last = places.nodeVoxelEnds[node];
        for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
            if(walls[index] || solid[index] || !(near[index] & NEAR_SURFACE)){
                continue;
            }
            particleCounts[index] += restParticlesPerVoxel*cachedCut(opens, index).covered;
        }
    }
}

void Particles::creditObstacleVolume(){
    if(obstacles.count() == 0 || numStoredNodes == 0){
        return;
    }
    creditCoveredVolume<<<obstacleBlocks(), OBSTACLE_THREADS, 0, stream>>>(obstacles.state(), voxelPlaces(), obstacleNodes.devPtr(), solids.devPtr(), obstacleSolids.devPtr(), obstacleNear.devPtr(),
        obstacleOpen.devPtr(), (float)restParticlesPerVoxel, particleCounts.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}
