//Copyright 2023 Aberrant Behavior LLC

#include "particles.hu"

#include "algorithms/radixSortKernels.hu"
#include "algorithms/particleToGridFunctions.hu"
#include "algorithms/perVoxelParticleListFunctions.hu"
#include "algorithms/voxelSolveFunctions.hu"
#include "algorithms/conjugateGradientFunctions.hu"
#include "algorithms/parallelPrefixSumKernels.hu"

#include "typedefs.h"
#include <cmath>
#include <iostream>

Particles::Particles(uint size)
: size(size)
, radius(2)
, refinementLevel(1)
, frameWriter(3*sizeof(float)*size)
{
    setupCudaDevices();

    cudaStreamCreate(&stream);
    px.resize(size);
    py.resize(size);
    pz.resize(size);
    
    vx.resize(size);
    vy.resize(size);
    vz.resize(size);

    gridCell.resize(size);

    reorderedGridIndices.resize(size);

    uniqueGridNodeIndices.resize(size);
    nodeCount.resize(1);
    freeSurface.resize(1);

    numVoxels1D = 2*(uint)(std::floor(radius)) + (2<<refinementLevel);
    numVoxelsPerNode = numVoxels1D*numVoxels1D*numVoxels1D;
    frameDt = 1.0/24.0;
    prevDt = 0.0;
}

void Particles::setDomain(double nx, double ny, double nz, uint x, uint y, uint z, double cellSize){
    grid.setNegativeCorner(nx, ny, nz);
    grid.setSize(x, y, z, cellSize);
    cellToNode.resizeAsync(x*y*z, stream);
    cellToNode.zeroDeviceAsync(stream);   //holds node indices, stale or not, but never CLAIMED_NODE
    }

void Particles::alignParticlesToGrid(){
    cudaFindGridCell(px.devPtr(), py.devPtr(), pz.devPtr(), size, grid, gridCell.devPtr(), stream);
}

void Particles::sortParticles(){
    uint* tempGridCell = gridCell.devPtr();
    uint* tempSortedIndices = reorderedGridIndices.devPtr();
    cudaStreamSynchronize(stream);
    cudaSortParticlesByGridNode(size, tempGridCell, tempSortedIndices, stream);
    // cudaStreamSynchronize(stream);

    //@TODO: need to create per-CudaVec stream for allocation/dealloc to get around this overlap issue when freeing on constant stream, leave compute streams in place for all else, CudaVec_stream sync on devPtr call
    gridCell.swapDevicePtrAsync(tempGridCell, stream);
    reorderedGridIndices.swapDevicePtrAsync(tempSortedIndices, stream);

    auto reorder = [&](auto* particleData){   //every per-particle array has to follow the sort
        auto* sortedData = particleData->devPtr();
        gpuErrchk(cudaMallocAsync((void**)&sortedData, sizeof(*sortedData)*size, stream));
        reorderGridIndices<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, reorderedGridIndices.devPtr(), particleData->devPtr(), sortedData);
        particleData->swapDevicePtrAsync(sortedData, stream);
    };
    for(CudaVec<double>* position : {&px, &py, &pz}){
        reorder(position);
    }
    for(CudaVec<float>* velocity : {&vx, &vy, &vz}){
        reorder(velocity);
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaStreamSynchronize(stream));
}

//a particle's position in its node's voxel block, in voxels; the block includes the apron, so the node's interior starts at apronCells
__device__ inline float3 positionInNodeBlock(uint index, const uint* gridPosition, const double* px, const double* py, const double* pz, Grid grid, uint refinementLevel, int apronCells){
    uint cell = gridPosition[index];
    uint cellX = cell % grid.sizeX;
    uint cellY = cell / grid.sizeX % grid.sizeY;
    uint cellZ = cell / (grid.sizeX*grid.sizeY);
    float perVoxel = 1.0f / (float)calcSubCellWidth(refinementLevel, grid);   //the offset into the cell is small, so float from here on
    return make_float3((float)(px[index] - grid.negX - cellX*grid.cellSize)*perVoxel + apronCells,
                       (float)(py[index] - grid.negY - cellY*grid.cellSize)*perVoxel + apronCells,
                       (float)(pz[index] - grid.negZ - cellZ*grid.cellSize)*perVoxel + apronCells);
}

//threads per node block in the particle kernels, which give each thread a particle: about 2 each in a full node
static constexpr uint NODE_THREADS = 256;

//the quadratic B-spline's three nonzero weights for a point at q, with nodes at the integers; base is the first node
struct Spline3{
    int base;
    float w[3];
};

__device__ inline Spline3 quadraticBSpline(float q){
    Spline3 s;
    s.base = (int)floorf(q - 0.5f);
    float g = q - s.base - 1.0f;    //offset from the middle node, in [-1/2, 1/2)
    s.w[0] = 0.5f*(0.5f - g)*(0.5f - g);
    s.w[1] = 0.75f - g*g;
    s.w[2] = 0.5f*(0.5f + g)*(0.5f + g);
    return s;
}

//a particle's per-axis weights: a component's faces sit on voxel boundaries along its own axis (onFaces) and mid-voxel along the other two (onCentres).
//Component dim's face at (x, y, z) weighs axis(dim, 0).w[x]*axis(dim, 1).w[y]*axis(dim, 2).w[z], and its slot in the block is the voxel it belongs to
struct FaceStencil{
    Spline3 onFaces[3];
    Spline3 onCentres[3];
    __device__ explicit FaceStencil(float3 p){
        float q[3] = {p.x, p.y, p.z};
        #pragma unroll
        for(int a = 0; a < 3; ++a){
            onFaces[a] = quadraticBSpline(q[a]);
            onCentres[a] = quadraticBSpline(q[a] - 0.5f);
        }
    }
    __device__ const Spline3& axis(int dim, int a) const{
        return a == dim ? onFaces[a] : onCentres[a];
    }
};

//whether a slot of a node's voxel block lies outside the domain
__device__ inline bool isWallVoxel(uint cell, int slot, int voxels1D, int apronCells, Grid grid, uint refinementLevel){
    int interiorWidth = 2<<refinementLevel;
    int x = (int)(cell % grid.sizeX)*interiorWidth + slot % voxels1D - apronCells;
    int y = (int)(cell / grid.sizeX % grid.sizeY)*interiorWidth + slot / voxels1D % voxels1D - apronCells;
    int z = (int)(cell / (grid.sizeX*grid.sizeY))*interiorWidth + slot / (voxels1D*voxels1D) - apronCells;
    return x < 0 || y < 0 || z < 0 || x >= (int)grid.sizeX*interiorWidth || y >= (int)grid.sizeY*interiorWidth || z >= (int)grid.sizeZ*interiorWidth;
}

//the slot whose dim face a particle reads in place of voxel's: a face inside a wall reads its mirror image in the domain, reversed (sign) when it's the
//wall's normal component, so that's 0 on the wall itself, and as is when tangential, so walls don't drag. origin is the block's first voxel in domain coordinates
__device__ inline int mirroredSlot(int3 voxel, int dim, int3 origin, int3 domainVoxels, int voxels1D, float& sign){
    int local[3] = {voxel.x, voxel.y, voxel.z};
    int start[3] = {origin.x, origin.y, origin.z};
    int size[3] = {domainVoxels.x, domainVoxels.y, domainVoxels.z};
    #pragma unroll
    for(int axis = 0; axis < 3; ++axis){
        int global = start[axis] + local[axis];
        int tangential = axis != dim;   //a dim face sits on a voxel boundary along dim, and mid-voxel along the other axes
        if(global < 0 || global >= size[axis]){
            local[axis] += (global < 0 ? -2*global : 2*(size[axis] - global)) - tangential;
            sign = tangential ? sign : -sign;
        }
    }
    return local[0] + local[1]*voxels1D + local[2]*voxels1D*voxels1D;
}

//the cell of a node's offset-th neighbour (13 is itself), or the number of cells if that's outside the domain
__device__ inline uint neighborCell(uint cell, int offset, Grid grid){
    int x = (int)(cell % grid.sizeX) + offset % 3 - 1;
    int y = (int)(cell / grid.sizeX % grid.sizeY) + offset / 3 % 3 - 1;
    int z = (int)(cell / (grid.sizeX*grid.sizeY)) + offset / 9 - 1;
    bool inside = x >= 0 && y >= 0 && z >= 0 && x < (int)grid.sizeX && y < (int)grid.sizeY && z < (int)grid.sizeZ;
    return inside ? x + y*grid.sizeX + z*grid.sizeX*grid.sizeY : grid.sizeX*grid.sizeY*grid.sizeZ;
}

//a node's 27 neighbours' node indices (13 is itself), numUsedGridNodes where there's none. cellToNode is never cleared, so an entry only counts if nodeCells agrees
__device__ void loadNeighborNodes(uint* neighborNodes, uint cell, uint numUsedGridNodes, const uint* nodeCells, const uint* cellToNode, Grid grid){
    if(threadIdx.x < 27){
        uint neighbor = neighborCell(cell, threadIdx.x, grid);
        uint node = neighbor < grid.sizeX*grid.sizeY*grid.sizeZ ? cellToNode[neighbor] : numUsedGridNodes;
        neighborNodes[threadIdx.x] = node < numUsedGridNodes && nodeCells[node] == neighbor ? node : numUsedGridNodes;
    }
}

//register nodes firstNode..lastNode - 1 in cellToNode. The nodes holding particles come first and read their cell off their first particle;
//the empty nodes after them already know their cell, and get no particles
__global__ void mapCellsToNodes(uint firstNode, uint lastNode, uint numParticleNodes, uint numParticles, const uint* gridPosition, uint* gridNodeIndicesToFirstParticleIndex, uint* nodeCells, uint* cellToNode){
    uint node = firstNode + threadIdx.x + blockIdx.x*blockDim.x;
    if(node < lastNode){
        if(node < numParticleNodes){
            nodeCells[node] = gridPosition[gridNodeIndicesToFirstParticleIndex[node]];
        }
        else{
            gridNodeIndicesToFirstParticleIndex[node] = numParticles;
        }
        cellToNode[nodeCells[node]] = node;
    }
}

#define CLAIMED_NODE 0xFFFFFFFFu

//particles reach into their node's 26 neighbours, and whichever node's interior holds a reached voxel has to exist to solve for it, so append the
//neighbours without particles: the thread that swaps a neighbour's stale cellToNode entry for CLAIMED_NODE appends it, and mapCellsToNodes then registers it
__global__ void claimEmptyNeighborNodes(uint numParticleNodes, uint* nodeCells, uint* cellToNode, uint* numNodes, Grid grid){
    uint cell = neighborCell(nodeCells[blockIdx.x], threadIdx.x, grid);
    if(threadIdx.x < 27 && cell < grid.sizeX*grid.sizeY*grid.sizeZ){
        uint entry = cellToNode[cell];
        bool particleNode = entry < numParticleNodes && nodeCells[entry] == cell;
        if(!particleNode && entry != CLAIMED_NODE && atomicCAS(cellToNode + cell, entry, CLAIMED_NODE) == entry){
            nodeCells[atomicAdd(numNodes, 1)] = cell;
        }
    }
}

//marking pass 1, per node holding particles: every voxel of its block with a face in one of its particles' stencils. voxels1D is 8, so a block row
//starts on a multiple of 8 and the stencil's 3 voxels of it are 3 bits of one word. Each node's masks are laid out stored, fluid, reached
__global__ void markReachedVoxels(uint numParticleNodes, uint numParticles, uint numVoxels1D, const uint* gridNodeIndicesToFirstParticleIndex, const uint* gridPosition, const double* px, const double* py, const double* pz,
                                    uint* usedVoxelMasks, double radius, Grid grid, uint refinementLevel){
    extern __shared__ uint reached[];
    int voxels1D = numVoxels1D;
    int maskWords = (voxels1D*voxels1D*voxels1D + 31) / 32;
    int apronCells = floor(radius);
    uint firstParticle = gridNodeIndicesToFirstParticleIndex[blockIdx.x];
    uint lastParticle = blockIdx.x == numParticleNodes - 1 ? numParticles : gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1];
    for(int word = threadIdx.x; word < maskWords; word += blockDim.x){
        reached[word] = 0;
    }
    __syncthreads();
    for(uint index = firstParticle + threadIdx.x; index < lastParticle; index += blockDim.x){    //consecutive threads, consecutive particles
        float3 pos = positionInNodeBlock(index, gridPosition, px, py, pz, grid, refinementLevel, apronCells);
        int onFaces[3] = {(int)floorf(pos.x - 0.5f), (int)floorf(pos.y - 0.5f), (int)floorf(pos.z - 0.5f)};
        int onCentres[3] = {(int)floorf(pos.x - 1.0f), (int)floorf(pos.y - 1.0f), (int)floorf(pos.z - 1.0f)};
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            int x = dim == 0 ? onFaces[0] : onCentres[0];
            int y = dim == 1 ? onFaces[1] : onCentres[1];
            int z = dim == 2 ? onFaces[2] : onCentres[2];
            #pragma unroll
            for(int k = 0; k < 3; ++k){
                #pragma unroll
                for(int j = 0; j < 3; ++j){
                    int rowStart = x + (y + j)*voxels1D + (z + k)*voxels1D*voxels1D;
                    atomicOr(reached + rowStart/32, 7u << rowStart%32);
                }
            }
        }
    }
    __syncthreads();
    for(int word = threadIdx.x; word < maskWords; word += blockDim.x){
        usedVoxelMasks[(3*blockIdx.x + 2)*maskWords + word] = reached[word];
    }
}

//marking pass 2, per node: its fluid is every interior voxel any particle reaches, its own or a neighbour's, so the node owning a voxel solves for it
//wherever the particles reaching it live. It stores its fluid, the voxels above each fluid voxel in x, y and z (which hold its upper faces), the voxels
//its own particles reach (to find their owners) and, if it has particles, its block's wall voxels. Write the stored mask, then the fluid mask, and count the stored voxels
__global__ void markUsedVoxels(uint numUsedGridNodes, uint numParticleNodes, uint numVoxels1D, const uint* nodeCells, const uint* cellToNode, uint* usedVoxelMasks, uint* numVoxelsEachNode, double radius, Grid grid, uint refinementLevel){
    extern __shared__ uint mask[];
    __shared__ uint neighborNodes[27];
    int voxels1D = numVoxels1D;
    int voxels3D = voxels1D*voxels1D*voxels1D;
    int maskWords = (voxels3D + 31) / 32;
    uint* fluid = mask + maskWords;
    int apronCells = floor(radius);
    int interiorWidth = voxels1D - 2*apronCells;
    uint cell = nodeCells[blockIdx.x];
    bool hasParticles = blockIdx.x < numParticleNodes;
    loadNeighborNodes(neighborNodes, cell, numUsedGridNodes, nodeCells, cellToNode, grid);
    for(int word = threadIdx.x; word < maskWords; word += blockDim.x){
        mask[word] = hasParticles ? usedVoxelMasks[(3*blockIdx.x + 2)*maskWords + word] : 0;
        fluid[word] = 0;
    }
    __syncthreads();
    for(int slot = threadIdx.x; slot < voxels3D; slot += blockDim.x){
        int x = slot % voxels1D;
        int y = slot / voxels1D % voxels1D;
        int z = slot / (voxels1D*voxels1D);
        if(hasParticles && isWallVoxel(cell, slot, voxels1D, apronCells, grid, refinementLevel)){
            atomicOr(mask + slot/32, 1u << slot%32);
        }
        bool interior = x >= apronCells && y >= apronCells && z >= apronCells && x < voxels1D - apronCells && y < voxels1D - apronCells && z < voxels1D - apronCells;
        for(int neighbor = 0; interior && neighbor < 27; ++neighbor){   //the same voxel in each neighbour's block
            uint node = neighborNodes[neighbor];
            int neighborX = x - (neighbor % 3 - 1)*interiorWidth;
            int neighborY = y - (neighbor / 3 % 3 - 1)*interiorWidth;
            int neighborZ = z - (neighbor / 9 - 1)*interiorWidth;
            int neighborSlot = neighborX + neighborY*voxels1D + neighborZ*voxels1D*voxels1D;
            if(node < numParticleNodes && neighborX >= 0 && neighborY >= 0 && neighborZ >= 0 && neighborX < voxels1D && neighborY < voxels1D && neighborZ < voxels1D
               && usedVoxelMasks[(3*node + 2)*maskWords + neighborSlot/32] >> neighborSlot%32 & 1){
                atomicOr(fluid + slot/32, 1u << slot%32);
                break;
            }
        }
    }
    __syncthreads();
    for(int slot = threadIdx.x; slot < voxels3D; slot += blockDim.x){
        if(fluid[slot/32] & 1u << slot%32){
            int storedSlots[4] = {slot, slot + 1, slot + voxels1D, slot + voxels1D*voxels1D};    //it, and the voxels holding its upper faces
            for(int i = 0; i < 4; ++i){
                atomicOr(mask + storedSlots[i]/32, 1u << storedSlots[i]%32);
            }
        }
    }
    __syncthreads();
    for(int word = threadIdx.x; word < 2*maskWords; word += blockDim.x){
        usedVoxelMasks[3*blockIdx.x*maskWords + word] = mask[word];    //stored mask, then fluid mask
    }
    if(threadIdx.x == 0){
        uint count = 0;
        for(int word = 0; word < maskWords; ++word){
            count += __popc(mask[word]);
        }
        numVoxelsEachNode[blockIdx.x] = count;
    }
}

//expand each node's stored mask into its used voxel IDs, in ascending order; wall voxels are flagged solid, and solveCodes gets whether each is fluid
__global__ void writeUsedVoxelIDs(uint numVoxels1D, const uint* usedVoxelMasks, const uint* numVoxelsEachNode, const uint* nodeCells, uint* voxelIDs, char* solids, char* solveCodes, double radius, Grid grid, uint refinementLevel){
    int voxels1D = numVoxels1D;
    int maskWords = (voxels1D*voxels1D*voxels1D + 31) / 32;
    int apronCells = floor(radius);
    const uint* mask = usedVoxelMasks + 3*blockIdx.x*maskWords;
    const uint* fluid = mask + maskWords;
    uint cell = nodeCells[blockIdx.x];
    for(int word = threadIdx.x; word < maskWords; word += blockDim.x){
        uint out = blockIdx.x == 0 ? 0 : numVoxelsEachNode[blockIdx.x - 1];
        for(int previous = 0; previous < word; ++previous){
            out += __popc(mask[previous]);
        }
        for(uint bits = mask[word]; bits != 0; bits &= bits - 1, ++out){
            int slot = word*32 + __ffs(bits) - 1;
            voxelIDs[out] = slot;
            solids[out] = isWallVoxel(cell, slot, voxels1D, apronCells, grid, refinementLevel);
            solveCodes[out] = fluid[word] >> slot%32 & 1;
        }
    }
}

//apron voxels mirror voxels in the 26 neighbouring nodes' interiors. Point every used voxel at the used voxel that owns its value (itself if interior, or if the owning node doesn't use it).
//Interior fluid voxels are the pressure solve's unknowns: also record their red/black colour and the owners of their six face neighbours (NO_VOXEL for air, WALL_VOXEL outside the domain)
__global__ void buildVoxelTopology(uint numUsedGridNodes, uint numVoxels1D, double radius, const uint* nodeCells, const uint* cellToNode, const uint* numVoxelsEachNode, const uint* voxelIDs, uint* voxelOwners,
                                    uint* neighborNx, uint* neighborPx, uint* neighborNy, uint* neighborPy, uint* neighborNz, uint* neighborPz, char* solveCodes, Grid grid, uint refinementLevel){
    extern __shared__ uint sharedOwners[];
    __shared__ uint neighborNodes[27];
    int voxels1D = numVoxels1D;
    int apronCells = floor(radius);
    int interiorWidth = voxels1D - 2*apronCells;
    uint cell = nodeCells[blockIdx.x];
    loadNeighborNodes(neighborNodes, cell, numUsedGridNodes, nodeCells, cellToNode, grid);
    for(int slot = threadIdx.x; slot < voxels1D*voxels1D*voxels1D; slot += blockDim.x){
        sharedOwners[slot] = NO_VOXEL;
    }
    __syncthreads();
    uint startVoxelIndex = blockIdx.x == 0 ? 0 : numVoxelsEachNode[blockIdx.x - 1];
    for(uint i = startVoxelIndex + threadIdx.x; i < numVoxelsEachNode[blockIdx.x]; i += blockDim.x){
        sharedOwners[voxelIDs[i]] = i;
    }
    __syncthreads();
    for(int neighbor = 0; neighbor < 27; ++neighbor){
        uint node = neighborNodes[neighbor];
        if(neighbor == 13 || node == numUsedGridNodes){
            continue;
        }
        for(uint i = (node == 0 ? 0 : numVoxelsEachNode[node - 1]) + threadIdx.x; i < numVoxelsEachNode[node]; i += blockDim.x){
            int x = voxelIDs[i] % voxels1D;
            int y = voxelIDs[i] / voxels1D % voxels1D;
            int z = voxelIDs[i] / (voxels1D*voxels1D);
            if(x < apronCells || y < apronCells || z < apronCells || x >= voxels1D - apronCells || y >= voxels1D - apronCells || z >= voxels1D - apronCells){
                continue;   //only interior voxels are authoritative
            }
            x += (neighbor % 3 - 1)*interiorWidth;  //neighbour's local coordinates -> this node's
            y += (neighbor / 3 % 3 - 1)*interiorWidth;
            z += (neighbor / 9 - 1)*interiorWidth;
            if(x >= 0 && y >= 0 && z >= 0 && x < voxels1D && y < voxels1D && z < voxels1D){
                sharedOwners[x + y*voxels1D + z*voxels1D*voxels1D] = i;
            }
        }
    }
    __syncthreads();
    for(uint i = startVoxelIndex + threadIdx.x; i < numVoxelsEachNode[blockIdx.x]; i += blockDim.x){
        int slot = voxelIDs[i];
        int x = slot % voxels1D;
        int y = slot / voxels1D % voxels1D;
        int z = slot / (voxels1D*voxels1D);
        bool unknown = solveCodes[i] && x >= apronCells && y >= apronCells && z >= apronCells && x < voxels1D - apronCells && y < voxels1D - apronCells && z < voxels1D - apronCells;  //fluid and interior
        voxelOwners[i] = sharedOwners[slot];
        solveCodes[i] = unknown ? 1 + (x + y + z) % 2 : 0;    //1 red, 2 black
        if(unknown){
            uint* neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
            #pragma unroll
            for(int face = 0; face < 6; ++face){
                int neighborSlot = slot + (face % 2 ? 1 : -1)*(face < 2 ? 1 : face < 4 ? voxels1D : voxels1D*voxels1D);
                neighbors[face][i] = isWallVoxel(cell, neighborSlot, voxels1D, apronCells, grid, refinementLevel) ? WALL_VOXEL : sharedOwners[neighborSlot];
            }
        }
    }
}

void Particles::generateVoxels(){
    numParticleNodes = cudaMarkUniqueGridCellsAndCount(size, gridCell.devPtr(), uniqueGridNodeIndices.devPtr(), stream);

    //the used nodes: the ones holding particles, then their neighbours without any, which the particles reach into
    uint numCells = grid.sizeX*grid.sizeY*grid.sizeZ;
    uint maxNodes = 27*numParticleNodes < numCells ? 27*numParticleNodes : numCells;
    gridNodeIndicesToFirstParticleIndex.resizeAsync(maxNodes, stream);
    nodeCells.resizeAsync(maxNodes, stream);
    cudaMapNodeIndicesToParticles(size, uniqueGridNodeIndices.devPtr(), gridNodeIndicesToFirstParticleIndex.devPtr(), stream);
    mapCellsToNodes<<<numParticleNodes / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(0, numParticleNodes, numParticleNodes, size, gridCell.devPtr(), gridNodeIndicesToFirstParticleIndex.devPtr(), nodeCells.devPtr(), cellToNode.devPtr());
    cudaMemcpyAsync(nodeCount.devPtr(), &numParticleNodes, sizeof(uint), cudaMemcpyHostToDevice, stream);
    claimEmptyNeighborNodes<<<numParticleNodes, 32, 0, stream>>>(numParticleNodes, nodeCells.devPtr(), cellToNode.devPtr(), nodeCount.devPtr(), grid);
    cudaMemcpyAsync(&numUsedGridNodes, nodeCount.devPtr(), sizeof(uint), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    mapCellsToNodes<<<(numUsedGridNodes - numParticleNodes) / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticleNodes, numUsedGridNodes, numParticleNodes, size, gridCell.devPtr(), gridNodeIndicesToFirstParticleIndex.devPtr(), nodeCells.devPtr(), cellToNode.devPtr());

    uint maskWords = (numVoxelsPerNode + 31) / 32;
    nodeIndexUsedVoxels.resizeAsync(numUsedGridNodes, stream);
    usedVoxelMasks.resizeAsync(3*numUsedGridNodes*maskWords, stream);
    markReachedVoxels<<<numParticleNodes, NODE_THREADS, sizeof(uint)*maskWords, stream>>>(numParticleNodes, size, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), usedVoxelMasks.devPtr(), radius, grid, refinementLevel);
    markUsedVoxels<<<numUsedGridNodes, 64, 2*sizeof(uint)*maskWords, stream>>>(numUsedGridNodes, numParticleNodes, numVoxels1D, nodeCells.devPtr(), cellToNode.devPtr(), usedVoxelMasks.devPtr(), nodeIndexUsedVoxels.devPtr(), radius, grid, refinementLevel);
    cudaStreamSynchronize(stream);

    cudaParallelPrefixSum(numUsedGridNodes, nodeIndexUsedVoxels.devPtr(), stream);
    cudaStreamSynchronize(stream);

    uint numUsedVoxels;
    cudaMemcpyAsync(&numUsedVoxels, nodeIndexUsedVoxels.devPtr() + numUsedGridNodes - 1, sizeof(uint), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    voxelIDsUsed.resizeAsync(numUsedVoxels, stream);
    voxelOwners.resizeAsync(numUsedVoxels, stream);
    solids.resizeAsync(numUsedVoxels, stream);
    solveCodes.resizeAsync(numUsedVoxels, stream);
    footprintDepth.resizeAsync(numUsedVoxels, stream);
    for(CudaVec<uint>* neighbor : {&neighborNx, &neighborPx, &neighborNy, &neighborPy, &neighborNz, &neighborPz}){
        neighbor->resizeAsync(numUsedVoxels, stream);
    }
    for(CudaVec<float>* voxelData : {&voxelsUx, &voxelsUy, &voxelsUz, &voxelsUxOld, &voxelsUyOld, &voxelsUzOld, &voxelWeightsX, &voxelWeightsY, &voxelWeightsZ, &particleCounts, &divU, &p, &residuals, &Anx, &Apx, &Any, &Apy, &Anz, &Apz, &Adiag}){
        voxelData->resizeAsync(numUsedVoxels, stream);
    }
    p.zeroDeviceAsync(stream);

    writeUsedVoxelIDs<<<numUsedGridNodes, 32, 0, stream>>>(numVoxels1D, usedVoxelMasks.devPtr(), nodeIndexUsedVoxels.devPtr(), nodeCells.devPtr(), voxelIDsUsed.devPtr(), solids.devPtr(), solveCodes.devPtr(), radius, grid, refinementLevel);
    buildVoxelTopology<<<numUsedGridNodes, 64, sizeof(uint)*numVoxelsPerNode, stream>>>(numUsedGridNodes, numVoxels1D, radius, nodeCells.devPtr(), cellToNode.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelOwners.devPtr(),
        neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(), solveCodes.devPtr(), grid, refinementLevel);
    cudaStreamSynchronize(stream);

    std::cout<<"Nodes: "<<numParticleNodes<<" with particles, "<<numUsedGridNodes<<" in all, holding "<<numUsedVoxels<<" voxels\n";
    std::cout<<"Using "<<(CudaVec<uint>::GPU_MEMORY_ALLOCATED + CudaVec<float>::GPU_MEMORY_ALLOCATED + CudaVec<double>::GPU_MEMORY_ALLOCATED + CudaVec<char>::GPU_MEMORY_ALLOCATED) / (1<<20)<<" MB on GPU\n";
}

//P2G: a block per node and a thread per particle. Each particle adds its momentum and weight on the 27 faces of each component around it into the
//node's voxel block in shared memory, and 1 to the count of the voxel holding it; then each voxel the node stores adds the block's sums into its owner.
//It sums in 32-bit fixed point, as sm_86 has native shared integer atomics but not float ones. 7 ints per slot: 14 KB for an 8^3 block
__global__ void scatterParticleVelsToVoxels(uint numParticleNodes, uint numParticles, uint numVoxels1D, const uint* gridNodeIndicesToFirstParticleIndex, const uint* gridPosition,
                                            const double* px, const double* py, const double* pz, const float* vx, const float* vy, const float* vz,
                                            const uint* numVoxelsEachNode, const uint* voxelIDs, const uint* voxelOwners,
                                            float* ux, float* uy, float* uz, float* weightsX, float* weightsY, float* weightsZ, float* particleCounts, float momentumScale, float weightScale, double radius, Grid grid, uint refinementLevel){
    extern __shared__ int fixedSums[];    //per slot: x, y, z momentum, then x, y, z weight, then particle count
    int voxels1D = numVoxels1D;
    int voxels3D = voxels1D*voxels1D*voxels1D;
    int apronCells = floor(radius);
    uint firstParticle = gridNodeIndicesToFirstParticleIndex[blockIdx.x];
    uint lastParticle = blockIdx.x == numParticleNodes - 1 ? numParticles : gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1];
    for(int i = threadIdx.x; i < 7*voxels3D; i += blockDim.x){
        fixedSums[i] = 0;
    }
    __syncthreads();
    const float* velocities[3] = {vx, vy, vz};
    for(uint index = firstParticle + threadIdx.x; index < lastParticle; index += blockDim.x){    //consecutive threads, consecutive particles
        float3 pos = positionInNodeBlock(index, gridPosition, px, py, pz, grid, refinementLevel, apronCells);
        FaceStencil stencil(pos);
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            const Spline3& x = stencil.axis(dim, 0);
            const Spline3& y = stencil.axis(dim, 1);
            const Spline3& z = stencil.axis(dim, 2);
            float velocity = velocities[dim][index];
            #pragma unroll
            for(int k = 0; k < 3; ++k){
                #pragma unroll
                for(int j = 0; j < 3; ++j){
                    int rowStart = x.base + (y.base + j)*voxels1D + (z.base + k)*voxels1D*voxels1D;
                    float yz = y.w[j]*z.w[k];
                    #pragma unroll
                    for(int i = 0; i < 3; ++i){
                        float weight = x.w[i]*yz;
                        atomicAdd(fixedSums + dim*voxels3D + rowStart + i, __float2int_rn(weight*velocity*momentumScale));
                        atomicAdd(fixedSums + (3 + dim)*voxels3D + rowStart + i, __float2int_rn(weight*weightScale));
                    }
                }
            }
        }
        atomicAdd(fixedSums + 6*voxels3D + (int)pos.x + (int)pos.y*voxels1D + (int)pos.z*voxels1D*voxels1D, 1);
    }
    __syncthreads();
    float* accumulators[7] = {ux, uy, uz, weightsX, weightsY, weightsZ, particleCounts};
    float unscale[7] = {1.0f/momentumScale, 1.0f/momentumScale, 1.0f/momentumScale, 1.0f/weightScale, 1.0f/weightScale, 1.0f/weightScale, 1.0f};
    for(uint i = (blockIdx.x == 0 ? 0 : numVoxelsEachNode[blockIdx.x - 1]) + threadIdx.x; i < numVoxelsEachNode[blockIdx.x]; i += blockDim.x){    //coalesced over the node's voxels
        int slot = voxelIDs[i];
        uint owner = voxelOwners[i];    //itself for interior voxels, so these atomics are mostly consecutive
        #pragma unroll
        for(int sum = 0; sum < 7; ++sum){
            int value = fixedSums[sum*voxels3D + slot];
            if(value != 0){
                atomicAdd(accumulators[sum] + owner, value*unscale[sum]);
            }
        }
    }
}

#include "algorithms/reductionKernels.hu"
#include <cmath>

double Particles::getCourantDt(){    //every substep: the fastest particle moves at most 0.7 voxels. P2G finds the fastest
    double voxelSize = (grid.cellSize / (numVoxels1D - 2*std::floor(radius)));
    std::cout<<"maxVel: "<<maxVelocity<<" voxelSize: "<<voxelSize<<"\n";
    return 0.7 * (voxelSize / maxVelocity + 0.0001);
}

void Particles::particleVelToVoxels(){
    for(CudaVec<float>* accumulator : {&voxelsUx, &voxelsUy, &voxelsUz, &voxelWeightsX, &voxelWeightsY, &voxelWeightsZ, &particleCounts}){
        accumulator->zeroDeviceAsync(stream);
    }
    maxVelocity = fmax(fabs(vx.getMax(stream, true)), fmax(fabs(vy.getMax(stream, true)), fabs(vz.getMax(stream, true))));  //also sets the CFL timestep
    //P2G sums in fixed point: a face's weight sum is about the particles per voxel, so budget 128 (15x rest) and keep every sum under 2^30
    float weightScale = (1 << 30) / 128.0f;
    float momentumScale = weightScale / fmax(maxVelocity, 1e-6);
    scatterParticleVelsToVoxels<<<numParticleNodes, NODE_THREADS, 7*sizeof(int)*numVoxelsPerNode, stream>>>(numParticleNodes, size, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr(),
        nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelOwners.devPtr(), voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr(), voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), particleCounts.devPtr(), momentumScale, weightScale, radius, grid, refinementLevel);
    cudaNormalizeVoxelVelocities(solids, voxelWeightsX, voxelWeightsY, voxelWeightsZ, voxelsUx, voxelsUy, voxelsUz, stream);
    cudaExtrapolateUnreachedFaces(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, voxelWeightsX, voxelWeightsY, voxelWeightsZ, voxelsUx, voxelsUy, voxelsUz, stream);
    cudaFindFootprintDepth(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, footprintDepth, freeSurface, stream);
    for(auto [velocity, before] : {std::pair{&voxelsUx, &voxelsUxOld}, {&voxelsUy, &voxelsUyOld}, {&voxelsUz, &voxelsUzOld}}){    //for FLIP's velocity change over the solve
        cudaMemcpyAsync(before->devPtr(), velocity->devPtr(), sizeof(float)*velocity->size(), cudaMemcpyDeviceToDevice, stream);
    }
    cudaStreamSynchronize(stream);
}

void Particles::pressureSolve(){
    double density = 0.014;
    double tolerance = 0.001;   //largest residual, relative to the largest divergence
    uint maxIterations = 16384;
    double previousTerminatingResidual = 100;
    double terminatingResidual;
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    uint hasFreeSurface;
    cudaMemcpyAsync(&hasFreeSurface, freeSurface.devPtr(), sizeof(uint), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    //divergence per unit of relative density error. Without air the fluid can't change volume, and the solve has nowhere to take a net divergence, so it's off
    double correctionRate = hasFreeSurface && densityCorrectionTime > 0.0 ? voxelSize / densityCorrectionTime : 0.0;
    //@TODO: need to use courant number for dt from max voxel u and voxel dimensions
    dt = getCourantDt();
    std::cout<<"initial dt: "<<dt<<std::endl;
    cudaStreamSynchronize(stream);
    if(frameDt - elapsedTimeThisFrame < dt){
        dt = frameDt - elapsedTimeThisFrame;
    }
    applyGravity(solids, voxelsUy, dt, stream);
    cudaCalcDivU(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, voxelsUx, voxelsUy, voxelsUz, particleCounts, footprintDepth, restParticlesPerVoxel, correctionRate, divU, stream);
    cudaGetA(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag, dt/(density*voxelSize*voxelSize), stream);
    gpuErrchk(cudaPeekAtLastError());
    auto solve = useConjugateGradient ? cudaConjugateGradient : cudaGSiteration;    //the two pressure solvers take the same arguments
    while(previousTerminatingResidual - (terminatingResidual = solve(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag, divU, p, residuals, tolerance, maxIterations, stream)) > 0.0){    //while residual getting smaller
        gpuErrchk(cudaPeekAtLastError());
        if(terminatingResidual < tolerance){
            break;
        }
        removeGravity(solids, voxelsUy, dt, stream);
        dt /= 2.0f;
        p.zeroDeviceAsync(stream);
        applyGravity(solids, voxelsUy, dt, stream);
        previousTerminatingResidual = terminatingResidual;
        cudaCalcDivU(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, voxelsUx, voxelsUy, voxelsUz, particleCounts, footprintDepth, restParticlesPerVoxel, correctionRate, divU, stream);
        cudaGetA(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag, dt/(density*voxelSize*voxelSize), stream);
        gpuErrchk(cudaPeekAtLastError());
    }
    elapsedTime += dt;
    elapsedTimeThisFrame += dt;
    prevDt = dt;
    std::cout<<"dt: "<<dt<<" ElapsedTime: "<<elapsedTime<<" elapsedTimeThisFrame: "<<elapsedTimeThisFrame<<"\nTerminating Residual: "<<terminatingResidual<<"\nTolerance: "<<tolerance<<"\n";
    cudaStreamSynchronize(stream);
    gpuErrchk(cudaPeekAtLastError());
}

void Particles::updateVoxelVelocities(){
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    cudaVelocityUpdate(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, p, voxelsUx, voxelsUy, voxelsUz, dt/(0.014*voxelSize*voxelSize), stream);
    gpuErrchk(cudaPeekAtLastError());
}

//G2P: a block per node and a thread per particle. The block copies the new and old face velocities of every voxel the node stores into shared memory,
//from their owners, and turns the faces inside walls into their mirror images (the wall's normal component reversed, so it's 0 on the wall, and the
//tangential ones as they are, so walls don't drag). Each particle then reads its 27 faces per component from there; the weights sum to 1, so no
//normalizing. PIC takes the new grid velocity, FLIP adds its change to the particle's own, flipRatio blends them, and the particle moves with the
//grid's. 6 floats per slot: 12 KB for an 8^3 block
__global__ void gatherVoxelVelsToParticles(uint numParticleNodes, uint numParticles, uint numVoxels1D, const uint* gridNodeIndicesToFirstParticleIndex, const uint* gridPosition,
                                            double* px, double* py, double* pz, float* vx, float* vy, float* vz,
                                            const uint* numVoxelsEachNode, const uint* voxelIDs, const uint* voxelOwners, const char* solids,
                                            const float* ux, const float* uy, const float* uz, const float* oldUx, const float* oldUy, const float* oldUz,
                                            float flipRatio, float dt, double radius, Grid grid, uint refinementLevel){
    extern __shared__ float blockVelocities[];  //per slot: new x, y, z, then old x, y, z
    int voxels1D = numVoxels1D;
    int voxels3D = voxels1D*voxels1D*voxels1D;
    int apronCells = floor(radius);
    uint firstParticle = gridNodeIndicesToFirstParticleIndex[blockIdx.x];
    uint lastParticle = blockIdx.x == numParticleNodes - 1 ? numParticles : gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1];
    uint startVoxel = blockIdx.x == 0 ? 0 : numVoxelsEachNode[blockIdx.x - 1];
    uint endVoxel = numVoxelsEachNode[blockIdx.x];
    const float* velocities[6] = {ux, uy, uz, oldUx, oldUy, oldUz};
    for(int i = threadIdx.x; i < 6*voxels3D; i += blockDim.x){    //every face a particle reaches is stored, but never leave garbage to read
        blockVelocities[i] = 0.0f;
    }
    __syncthreads();
    for(uint i = startVoxel + threadIdx.x; i < endVoxel; i += blockDim.x){  //coalesced over the node's voxels
        int slot = voxelIDs[i];
        uint owner = voxelOwners[i];
        #pragma unroll
        for(int v = 0; v < 6; ++v){
            blockVelocities[v*voxels3D + slot] = velocities[v][owner];
        }
    }
    __syncthreads();
    uint cell = gridPosition[firstParticle];
    int interiorWidth = voxels1D - 2*apronCells;
    int3 origin = make_int3((int)(cell % grid.sizeX)*interiorWidth - apronCells, (int)(cell / grid.sizeX % grid.sizeY)*interiorWidth - apronCells, (int)(cell / (grid.sizeX*grid.sizeY))*interiorWidth - apronCells);
    int3 domainVoxels = make_int3(grid.sizeX*interiorWidth, grid.sizeY*interiorWidth, grid.sizeZ*interiorWidth);
    for(uint i = startVoxel + threadIdx.x; i < endVoxel; i += blockDim.x){  //a wall face's mirror image is always inside the domain, so never another wall face
        if(solids[i]){
            int slot = voxelIDs[i];
            int3 voxel = make_int3(slot % voxels1D, slot / voxels1D % voxels1D, slot / (voxels1D*voxels1D));
            #pragma unroll
            for(int dim = 0; dim < 3; ++dim){
                float sign = 1.0f;
                int source = mirroredSlot(voxel, dim, origin, domainVoxels, voxels1D, sign);
                blockVelocities[dim*voxels3D + slot] = sign*blockVelocities[dim*voxels3D + source];
                blockVelocities[(3 + dim)*voxels3D + slot] = sign*blockVelocities[(3 + dim)*voxels3D + source];
            }
        }
    }
    __syncthreads();
    double* positions[3] = {px, py, pz};
    float* particleVelocities[3] = {vx, vy, vz};
    for(uint index = firstParticle + threadIdx.x; index < lastParticle; index += blockDim.x){    //consecutive threads, consecutive particles
        float3 pos = positionInNodeBlock(index, gridPosition, px, py, pz, grid, refinementLevel, apronCells);
        FaceStencil stencil(pos);
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            const Spline3& x = stencil.axis(dim, 0);
            const Spline3& y = stencil.axis(dim, 1);
            const Spline3& z = stencil.axis(dim, 2);
            float newVelocity = 0.0f;
            float oldVelocity = 0.0f;
            #pragma unroll
            for(int k = 0; k < 3; ++k){
                #pragma unroll
                for(int j = 0; j < 3; ++j){
                    int rowStart = x.base + (y.base + j)*voxels1D + (z.base + k)*voxels1D*voxels1D;
                    float yz = y.w[j]*z.w[k];
                    #pragma unroll
                    for(int i = 0; i < 3; ++i){
                        float weight = x.w[i]*yz;
                        newVelocity += weight*blockVelocities[dim*voxels3D + rowStart + i];
                        oldVelocity += weight*blockVelocities[(3 + dim)*voxels3D + rowStart + i];
                    }
                }
            }
            particleVelocities[dim][index] = newVelocity + flipRatio*(particleVelocities[dim][index] - oldVelocity);
            positions[dim][index] += dt*newVelocity;
        }
    }
}

__global__ void advectParticlePositions(uint numParticles, double dt, double* position, const float* v, Grid grid){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        position[index] += dt*v[index];
    }
}

//cheap counter-based random number in [0, 1), instead of initialising a curand state for every thread
__device__ inline double hashToUniform(uint a, uint b){
    uint hash = a*0x9E3779B1u ^ (b + 0x7F4A7C15u)*0x85EBCA77u;
    hash ^= hash >> 15;
    hash *= 0x2C1B3C6Du;
    hash ^= hash >> 12;
    hash *= 0x297A2D39u;
    hash ^= hash >> 15;
    return (hash >> 8) * (1.0/16777216.0);
}

__global__ void moveSolids(uint numUsedGridNodes, uint numParticles, uint numVoxelsPerNode, uint numVoxels1D, uint* gridNodeIndicesToFirstParticleIndex, uint* gridPosition, double* px, double* py, double* pz, uint* numVoxelsEachNode, uint* voxelIDs, char* solids, double radius, Grid grid, uint refinementLevel, uint seed){
    extern __shared__ char sharedSolids[];
    __shared__ uint* processedVoxels;
    __shared__ uint lastParticleIndex;
    __shared__ uint xySize;
    __shared__ uint startVoxelIndex;
    if(threadIdx.x == 0){
        xySize = grid.sizeX*grid.sizeY;
        processedVoxels = (uint*)(sharedSolids + numVoxelsPerNode);
        if(blockIdx.x == numUsedGridNodes - 1){
            lastParticleIndex = numParticles;
        }
        else{
            lastParticleIndex = gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1];
        }
        if(blockIdx.x == 0){
            startVoxelIndex = 0;
        }
        else{
            startVoxelIndex = numVoxelsEachNode[blockIdx.x - 1];
        }
    }
    __syncthreads();
    for(int i = threadIdx.x; i < numVoxelsPerNode; i += blockDim.x){
        sharedSolids[i] = 0;
        processedVoxels[i] = numVoxelsPerNode;
    }
    __syncthreads();

    for(int i = startVoxelIndex + threadIdx.x; i < numVoxelsEachNode[blockIdx.x]; i += blockDim.x){
        sharedSolids[voxelIDs[i]] = solids[i];
        processedVoxels[i - startVoxelIndex] = voxelIDs[i];
    }
    __syncthreads();
    for(int index = gridNodeIndicesToFirstParticleIndex[blockIdx.x] + threadIdx.x; index < lastParticleIndex; index += blockDim.x){
        uint moduloWRTxySize = gridPosition[index] % xySize;
        uint gridIDz = gridPosition[index] / (xySize);
        uint gridIDy = moduloWRTxySize / grid.sizeX;
        uint gridIDx = moduloWRTxySize % grid.sizeX;
        
        int apronCells = floor(radius);
        double subCellWidth = calcSubCellWidth(refinementLevel, grid);
        double pxInGridCell = (px[index] - grid.negX - gridIDx*grid.cellSize + apronCells*subCellWidth);
        double pyInGridCell = (py[index] - grid.negY - gridIDy*grid.cellSize + apronCells*subCellWidth);
        double pzInGridCell = (pz[index] - grid.negZ - gridIDz*grid.cellSize + apronCells*subCellWidth);

        uint subCellPositionX = floor(pxInGridCell/subCellWidth);
        uint subCellPositionY = floor(pyInGridCell/subCellWidth);
        uint subCellPositionZ = floor(pzInGridCell/subCellWidth);
        
        if(subCellPositionX >= numVoxels1D || subCellPositionY >= numVoxels1D || subCellPositionZ >= numVoxels1D){
            //particle left its node's block (shouldn't happen under the CFL limit): move it to a fluid voxel. Particles in wall voxels are left to rootCell, which reflects them back inside
            int i = 0;
            for(; i < numVoxelsEachNode[blockIdx.x] - startVoxelIndex && sharedSolids[processedVoxels[i]]; ++i){  //stop at this node's first fluid voxel
            }
            int newVoxelX = processedVoxels[i] % numVoxels1D;
            int newVoxelY = (processedVoxels[i] % (numVoxels1D*numVoxels1D)) / numVoxels1D;
            int newVoxelZ = processedVoxels[i] / (numVoxels1D*numVoxels1D);

            px[index] = grid.negX + gridIDx*grid.cellSize + (newVoxelX - apronCells)*subCellWidth + hashToUniform(index, 3*seed)*subCellWidth;
            py[index] = grid.negY + gridIDy*grid.cellSize + (newVoxelY - apronCells)*subCellWidth + hashToUniform(index, 3*seed + 1)*subCellWidth;
            pz[index] = grid.negZ + gridIDz*grid.cellSize + (newVoxelZ - apronCells)*subCellWidth + hashToUniform(index, 3*seed + 2)*subCellWidth;
        }
    }
}

void Particles::voxelVelsToParticles(){
    gatherVoxelVelsToParticles<<<numParticleNodes, NODE_THREADS, 6*sizeof(float)*numVoxelsPerNode, stream>>>(numParticleNodes, size, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr(),
        nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelOwners.devPtr(), solids.devPtr(), voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr(), voxelsUxOld.devPtr(), voxelsUyOld.devPtr(), voxelsUzOld.devPtr(), flipRatio, dt, radius, grid, refinementLevel);
    gpuErrchk(cudaPeekAtLastError());
}

void Particles::advectParticles(){
    advectParticlePositions<<<size / WORKSIZE + 1, WORKSIZE, 0, stream>>>(size, dt, px.devPtr(), vx.devPtr(), grid);
    gpuErrchk(cudaPeekAtLastError());
    advectParticlePositions<<<size / WORKSIZE + 1, WORKSIZE, 0, stream>>>(size, dt, py.devPtr(), vy.devPtr(), grid);
    gpuErrchk(cudaPeekAtLastError());
    advectParticlePositions<<<size / WORKSIZE + 1, WORKSIZE, 0, stream>>>(size, dt, pz.devPtr(), vz.devPtr(), grid);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaStreamSynchronize(stream));
    gpuErrchk(cudaPeekAtLastError());
}

void Particles::moveSolidParticles(){
    //move particles that move to solid cell/out of bounds back to a fluid voxel in that grid
    static uint substep = 0;    //varies moveSolids' random offsets from substep to substep
    moveSolids<<<numParticleNodes, 32, sizeof(char)*numVoxelsPerNode + sizeof(uint)*numVoxelsPerNode, stream>>>(numParticleNodes, size, numVoxelsPerNode, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), solids.devPtr(), radius, grid, refinementLevel, substep++);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaStreamSynchronize(stream));
}

void Particles::setupCudaDevices(){
    int numDevices;
    gpuErrchk(cudaGetDeviceCount(&numDevices));
    deviceProp.resize(numDevices);
    for(int i = 0; i < numDevices; ++i){
        gpuErrchk( cudaGetDeviceProperties(deviceProp.data() + i, i) );
    }

    for(int i = 0; i < deviceProp.size(); ++i){
        std::cout<<"Device "<<i<<": "<<deviceProp[i].name<<" with "<<deviceProp[i].totalGlobalMem / (1<<20)<<"MB VRAM available"<<std::endl;
    }
}

void Particles::solveFrame(double fps){
    frameDt = 1.0f/fps;
    dt = frameDt / 10.0f;
    elapsedTimeThisFrame = 0.0f;
    while(elapsedTimeThisFrame < frameDt){
        particleVelToVoxels();
        cudaStreamSynchronize(stream);
        pressureSolve();
        cudaStreamSynchronize(stream);
        updateVoxelVelocities();
        cudaStreamSynchronize(stream);
        voxelVelsToParticles();   //also moves the particles
        cudaStreamSynchronize(stream);
        moveSolidParticles();
        cudaStreamSynchronize(stream);
        initialize();
    }
}

void Particles::initialize(){
        alignParticlesToGrid();
        cudaStreamSynchronize(stream);
        sortParticles();
        cudaStreamSynchronize(stream);
        generateVoxels();
        cudaStreamSynchronize(stream);
}

#include <iostream>
#include <fstream>

__global__ void packPositions(uint numParticles, const double* px, const double* py, const double* pz, float* xyz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        xyz[3*index] = px[index];
        xyz[3*index + 1] = py[index];
        xyz[3*index + 2] = pz[index];
    }
}

//binary frame: float32 x, y, z per particle, written to disk on frameWriter's thread while the simulation carries on
void Particles::writePositionsToFile(const std::string& fileName){
    packPositions<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, px.devPtr(), py.devPtr(), pz.devPtr(), frameWriter.deviceFrame());
    frameWriter.write(fileName, stream);
}

Particles::~Particles(){
    gpuErrchk( cudaStreamDestroy(stream) );
}