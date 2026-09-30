//Copyright 2023 Aberrant Behavior LLC

#include "particles.hu"

#include "algorithms/radixSortKernels.hu"
#include "algorithms/particleToGridFunctions.hu"
#include "algorithms/perVoxelParticleListFunctions.hu"
#include "algorithms/voxelSolveFunctions.hu"
#include "algorithms/conjugateGradientFunctions.hu"
#include "algorithms/parallelPrefixSumKernels.hu"
#include "localExchange.hu"

#include "typedefs.h"
#include <cmath>
#include <iostream>
#include <sstream>
#include <type_traits>

Particles::Particles(uint size)
: size(size)
, radius(2)
, refinementLevel(1)
{
    gpuErrchk(cudaGetDevice(&cudaDevice));     //everything this partition makes lives on the GPU that's current now
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

void Particles::setPartition(int rank, int numRanks, LocalExchange* exchange, PartitionContext* context, const std::vector<uint>& planes){
    this->rank = rank;
    this->numRanks = numRanks;
    this->exchange = exchange;
    this->context = context;
    partitionPlanes = planes;
}

void Particles::setDomain(double nx, double ny, double nz, uint x, uint y, uint z, double cellSize){
    grid.setNegativeCorner(nx, ny, nz);
    grid.setSize(x, y, z, cellSize);
    if(partitionPlanes.empty()){    //on its own, the partition is the whole domain
        partitionPlanes = {0, z};
    }
    boxLo = partitionPlanes[rank];
    boxHi = partitionPlanes[rank + 1];
    if(numRanks > 1 && (unsigned long long)x*y*z > 1ull << 29){   //orderNodes keys each node by its group, in 3 bits above its cell
        std::cerr<<"setDomain: "<<x<<"x"<<y<<"x"<<z<<" nodes is too many to split; up to 2^29 can be\n";
        exit(1);
    }
    cellToNode.resizeAsync(x*y*z, stream);
    cellToNode.zeroDeviceAsync(stream);   //holds node indices, stale or not, but never CLAIMED_NODE
    }

void Particles::alignParticlesToGrid(){
    cudaFindGridCell(px.devPtr(), py.devPtr(), pz.devPtr(), size, grid, gridCell.devPtr(), stream);
}

void Particles::sortParticles(){
    if(size == 0){
        return;
    }
    uint* tempGridCell = gridCell.devPtr();
    uint* tempSortedIndices = reorderedGridIndices.devPtr();
    cudaSortParticlesByGridNode(size, tempGridCell, tempSortedIndices, grid.sizeX*grid.sizeY*grid.sizeZ, stream);
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
//neighbours without particles: the thread that swaps a neighbour's stale cellToNode entry for CLAIMED_NODE appends it, and mapCellsToNodes then registers it.
//Only cells in [firstClaimCell, endClaimCell) are claimed: a partition's own node planes and the ones next to them, where its ghost nodes are
__global__ void claimEmptyNeighborNodes(uint numParticleNodes, uint* nodeCells, uint* cellToNode, uint* numNodes, Grid grid, uint firstClaimCell, uint endClaimCell){
    uint cell = neighborCell(nodeCells[blockIdx.x], threadIdx.x, grid);
    if(threadIdx.x < 27 && cell >= firstClaimCell && cell < endClaimCell){
        uint entry = cellToNode[cell];
        bool particleNode = entry < numParticleNodes && nodeCells[entry] == cell;
        if(!particleNode && entry != CLAIMED_NODE && atomicCAS(cellToNode + cell, entry, CLAIMED_NODE) == entry){
            nodeCells[atomicAdd(numNodes, 1)] = cell;
        }
    }
}

//marking pass 1, per node holding particles: every voxel of its block with a face in one of its particles' stencils. voxels1D is 8, so a block row
//starts on a multiple of 8 and the stencil's 3 voxels of it are 3 bits of one word. One mask per particle node, maskStride words apart: a lone partition
//writes them straight into their slots in usedVoxelMasks; split, they're packed for the neighbours to read, and orderNodes puts them in their slots
__global__ void markReachedVoxels(uint numParticleNodes, uint numParticles, uint numVoxels1D, const uint* gridNodeIndicesToFirstParticleIndex, const uint* gridPosition, const double* px, const double* py, const double* pz,
                                    uint* reachedMasks, uint maskStride, double radius, Grid grid, uint refinementLevel){
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
        reachedMasks[blockIdx.x*maskStride + word] = reached[word];
    }
}

//marking pass 2, per node: its fluid is every interior voxel any particle reaches, its own or a neighbour's, so the node owning a voxel solves for it
//wherever the particles reaching it live. It stores its fluid, the voxels above each fluid voxel in x, y and z (which hold its upper faces), the voxels
//its own particles reach (to find their owners) and, if it has particles, its block's wall voxels. Write the stored mask, then the fluid mask, and count the stored voxels.
//Nodes hold particles here or, for ghost nodes and the nodes 2 planes out, in the partition that owns them: nodeHasParticles says which
__global__ void markUsedVoxels(uint numUsedGridNodes, const char* nodeHasParticles, uint numVoxels1D, const uint* nodeCells, const uint* cellToNode, uint* usedVoxelMasks, uint* numVoxelsEachNode, double radius, Grid grid, uint refinementLevel){
    extern __shared__ uint mask[];
    __shared__ uint neighborNodes[27];
    int voxels1D = numVoxels1D;
    int voxels3D = voxels1D*voxels1D*voxels1D;
    int maskWords = (voxels3D + 31) / 32;
    uint* fluid = mask + maskWords;
    int apronCells = floor(radius);
    int interiorWidth = voxels1D - 2*apronCells;
    uint cell = nodeCells[blockIdx.x];
    bool hasParticles = nodeHasParticles[blockIdx.x];
    loadNeighborNodes(neighborNodes, cell, numUsedGridNodes, nodeCells, cellToNode, grid);
    if(threadIdx.x < 27 && neighborNodes[threadIdx.x] < numUsedGridNodes && !nodeHasParticles[neighborNodes[threadIdx.x]]){
        neighborNodes[threadIdx.x] = numUsedGridNodes;  //only nodes holding particles have reached masks: settle which once, not per voxel
    }
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
            if(node < numUsedGridNodes && neighborX >= 0 && neighborY >= 0 && neighborZ >= 0 && neighborX < voxels1D && neighborY < voxels1D && neighborZ < voxels1D
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
//Interior fluid voxels are the pressure solve's unknowns: also record their red/black colour and the owners of their six face neighbours (NO_VOXEL for air, WALL_VOXEL outside the domain).
//An air voxel above an unknown stores the face between them, and updates it itself from the unknown's values, so each unknown also records itself as that
//air voxel's lower neighbour in the same arrays (they're otherwise unused for voxels that aren't unknowns, and start as NO_VOXEL).
//Ghost nodes (numOwnNodes on) get codes 3 red / 4 black: they're unknowns to the stencils, but only their owners solve for them
__global__ void buildVoxelTopology(uint numUsedGridNodes, uint numOwnNodes, uint numVoxels1D, double radius, const uint* nodeCells, const uint* cellToNode, const uint* numVoxelsEachNode, const uint* voxelIDs, uint* voxelOwners,
                                    uint* neighborNx, uint* neighborPx, uint* neighborNy, uint* neighborPy, uint* neighborNz, uint* neighborPz, char* solveCodes, uint* coarseCells, uint* interiorVoxels,
                                    Grid grid, uint refinementLevel){
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
        solveCodes[i] = unknown ? (blockIdx.x < numOwnNodes ? 1 : 3) + (x + y + z) % 2 : 0;    //1 red, 2 black; 3 and 4 for ghosts
        if(x >= apronCells && y >= apronCells && z >= apronCells && x < voxels1D - apronCells && y < voxels1D - apronCells && z < voxels1D - apronCells){
            interiorVoxels[blockIdx.x*interiorWidth*interiorWidth*interiorWidth + (x - apronCells) + (y - apronCells)*interiorWidth + (z - apronCells)*interiorWidth*interiorWidth] = i;
        }
        if(unknown){
            uint* neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
            #pragma unroll
            for(int face = 0; face < 6; ++face){
                int neighborSlot = slot + (face % 2 ? 1 : -1)*(face < 2 ? 1 : face < 4 ? voxels1D : voxels1D*voxels1D);
                neighbors[face][i] = isWallVoxel(cell, neighborSlot, voxels1D, apronCells, grid, refinementLevel) ? WALL_VOXEL : sharedOwners[neighborSlot];
            }
            #pragma unroll
            for(int axis = 0; axis < 3; ++axis){
                uint upper = neighbors[2*axis + 1][i];
                if(upper < WALL_VOXEL && !solveCodes[upper]){   //only its lower neighbour writes here. solveCodes is being rewritten meanwhile, but only from fluid (1) to a colour
                    neighbors[2*axis][upper] = i;
                }
            }
            uint globalX = (cell % grid.sizeX)*interiorWidth + x - apronCells;  //the 2x2x2 block of the domain's voxels holding it
            uint globalY = (cell / grid.sizeX % grid.sizeY)*interiorWidth + y - apronCells;
            uint globalZ = cell / (grid.sizeX*grid.sizeY)*interiorWidth + z - apronCells;
            coarseCells[i] = globalX/2 + globalY/2*(grid.sizeX*interiorWidth/2) + globalZ/2*(grid.sizeX*interiorWidth/2)*(grid.sizeY*interiorWidth/2);
        }
    }
}

//how many of the first count sorted cells (or node order keys) come before firstCell: a binary search, in one thread
__global__ void countCellsBefore(const uint* sortedCells, uint count, uint firstCell, uint* result){
    uint low = 0;
    uint high = count;
    while(low < high){
        uint middle = low + (high - low) / 2;
        if(sortedCells[middle] < firstCell){
            low = middle + 1;
        }
        else{
            high = middle;
        }
    }
    *result = low;
}

//which of the first numNodes nodes hold particles: this partition's particle nodes, the first numParticleNodes. Split, orderNodes also notes where each came from
__global__ void markParticleNodes(uint numParticleNodes, uint numNodes, uint* nodeOrigins, char* nodeHasParticles){
    uint node = threadIdx.x + blockIdx.x*blockDim.x;
    if(node < numNodes){
        nodeHasParticles[node] = node < numParticleNodes;
        if(nodeOrigins != nullptr){
            nodeOrigins[node] = node;
        }
    }
}

//orderNodes' sort key for each node after this partition's particle nodes: its group, in the 3 bits above the cellBits of its cell, then its cell. The
//groups: 0 this partition's empty nodes (the only nodes in its own planes besides its particle nodes); then its ghost nodes, in the planes next to its
//own: 1 below holding particles, 2 below empty, 3 above holding particles, 4 above empty, the way their owners keep them; 5 the neighbours' particle
//nodes 2 planes out. Nodes before particleNodes hold particles
__global__ void nodeOrderKeys(uint firstNode, uint numNodes, uint particleNodes, const uint* nodeCells, uint firstOwnCell, uint endOwnCell, uint cellsPerPlane,
                              uint cellBits, uint* keys, uint* origins){
    uint node = firstNode + threadIdx.x + blockIdx.x*blockDim.x;
    if(node < numNodes){
        uint cell = nodeCells[node];
        uint empty = node < particleNodes ? 0 : 1;
        uint group = cell >= firstOwnCell && cell < endOwnCell ? 0
                   : cell + cellsPerPlane >= firstOwnCell && cell < firstOwnCell ? 1 + empty
                   : cell >= endOwnCell && cell < endOwnCell + cellsPerPlane ? 3 + empty
                   : 5;
        keys[node - firstNode] = group << cellBits | cell;
        origins[node - firstNode] = node;
    }
}

//puts the sorted nodes in their places, noting where each came from and whether it holds particles (every node before numParticleNodes did)
__global__ void placeOrderedNodes(uint firstNode, uint count, const uint* keys, const uint* origins, uint numParticleNodes, uint cellMask, uint* nodeCells, uint* nodeOrigins, char* nodeHasParticles){
    uint i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < count){
        nodeCells[firstNode + i] = keys[i] & cellMask;
        nodeOrigins[firstNode + i] = origins[i];
        nodeHasParticles[firstNode + i] = origins[i] < numParticleNodes;
    }
}

//each particle node's reached mask into its slot of usedVoxelMasks: this partition's own (numOwnParticleNodes of them came first) or a neighbour's
__global__ void placeReachedMasks(uint maskWords, uint numOwnParticleNodes, const uint* nodeOrigins, const char* nodeHasParticles, const uint* ownMasks, const uint* foreignMasks, uint* usedVoxelMasks){
    uint node = blockIdx.x;
    if(nodeHasParticles[node]){
        uint origin = nodeOrigins[node];
        const uint* mask = origin < numOwnParticleNodes ? ownMasks + origin*maskWords : foreignMasks + (origin - numOwnParticleNodes)*maskWords;
        for(uint word = threadIdx.x; word < maskWords; word += blockDim.x){
            usedVoxelMasks[(3*node + 2)*maskWords + word] = mask[word];
        }
    }
}

//the first voxel of each listed node: the prefix sums of the nodes' voxel counts end each node, so node numNodes's is the voxel count
struct NodeIndices{
    uint index[18];
    uint count;
};

__global__ void voxelStartsOf(NodeIndices nodes, const uint* voxelEnds, uint* starts){
    uint i = threadIdx.x;
    if(i < nodes.count){
        starts[i] = nodes.index[i] == 0 ? 0 : voxelEnds[nodes.index[i] - 1];
    }
}

//After the sort, the particles that left this partition's planes are a run at each end, as the cells run along z slowest: those now below it first,
//those above it last. The first time, every partition holds all the particles and keeps just its own. After that, each takes its neighbours' leavers,
//from below first, then its own, then from above: the order one partition would have them in, which the stable sort that follows keeps
void Particles::exchangeParticles(){
    if(numRanks == 1){
        return;
    }
    uint cellsPerPlane = grid.sizeX*grid.sizeY;
    uint ends[2] = {0, 0};
    uint firstAndLast[2] = {0, 0};
    if(size > 0){
        uint* found;
        gpuErrchk(cudaMallocAsync((void**)&found, 2*sizeof(uint), stream));
        countCellsBefore<<<1, 1, 0, stream>>>(gridCell.devPtr(), size, boxLo*cellsPerPlane, found);
        countCellsBefore<<<1, 1, 0, stream>>>(gridCell.devPtr(), size, boxHi*cellsPerPlane, found + 1);
        cudaMemcpyAsync(ends, found, 2*sizeof(uint), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(firstAndLast, gridCell.devPtr(), sizeof(uint), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(firstAndLast + 1, gridCell.devPtr() + size - 1, sizeof(uint), cudaMemcpyDeviceToHost, stream);
        cudaFreeAsync(found, stream);
        gpuErrchk(cudaStreamSynchronize(stream));
    }
    particlesBelow = ends[0];
    particlesAbove = size - ends[1];
    if(distributed){    //a particle can only move into the next partition over: each is at least 2 node planes deep, and a particle moves at most one a substep
        bool tooFarBelow = particlesBelow > 0 && (rank == 0 || firstAndLast[0] < partitionPlanes[rank - 1]*cellsPerPlane);
        bool tooFarAbove = particlesAbove > 0 && (rank == numRanks - 1 || firstAndLast[1] >= partitionPlanes[rank + 2]*cellsPerPlane);
        if(tooFarBelow || tooFarAbove){
            std::cerr<<"partition "<<rank<<": a particle moved past the next partition in one substep\n";
            exit(1);
        }
    }
    std::vector<int> neighbours = exchange->neighbours(rank);
    exchange->startReads(rank, neighbours);     //every partition has sorted and counted its leavers
    Particles* below = rank > 0 ? &exchange->partition(rank - 1) : nullptr;
    Particles* above = rank < numRanks - 1 ? &exchange->partition(rank + 1) : nullptr;
    uint fromBelow = distributed && below != nullptr ? below->particlesAbove : 0;
    uint fromAbove = distributed && above != nullptr ? above->particlesBelow : 0;
    uint kept = size - particlesBelow - particlesAbove;
    uint newSize = fromBelow + kept + fromAbove;
    auto rebuild = [&](auto member){    //a per-particle array's new contents: from below, kept, from above
        auto& mine = this->*member;
        using T = std::remove_reference_t<decltype(*mine.devPtr())>;
        T* fresh = nullptr;
        if(newSize > 0){
            gpuErrchk(cudaMallocAsync((void**)&fresh, sizeof(T)*newSize, stream));
        }
        if(fromBelow > 0){
            exchange->copyFrom(rank, fresh, rank - 1, (below->*member).devPtr() + below->size - fromBelow, sizeof(T)*fromBelow);
        }
        if(kept > 0){
            cudaMemcpyAsync(fresh + fromBelow, mine.devPtr() + particlesBelow, sizeof(T)*kept, cudaMemcpyDeviceToDevice, stream);
        }
        if(fromAbove > 0){
            exchange->copyFrom(rank, fresh + fromBelow + kept, rank + 1, (above->*member).devPtr(), sizeof(T)*fromAbove);
        }
        return fresh;
    };
    double* newPx = rebuild(&Particles::px);
    double* newPy = rebuild(&Particles::py);
    double* newPz = rebuild(&Particles::pz);
    float* newVx = rebuild(&Particles::vx);
    float* newVy = rebuild(&Particles::vy);
    float* newVz = rebuild(&Particles::vz);
    uint* newCells = rebuild(&Particles::gridCell);
    gpuErrchk(cudaPeekAtLastError());
    exchange->finishReads(rank, neighbours);    //the neighbours' copies from this partition are done before it frees what they copied
    px.adoptAsync(newPx, newSize, stream);
    py.adoptAsync(newPy, newSize, stream);
    pz.adoptAsync(newPz, newSize, stream);
    vx.adoptAsync(newVx, newSize, stream);
    vy.adoptAsync(newVy, newSize, stream);
    vz.adoptAsync(newVz, newSize, stream);
    gridCell.adoptAsync(newCells, newSize, stream);
    size = newSize;
    reorderedGridIndices.resizeAsync(size, stream);
    uniqueGridNodeIndices.resizeAsync(size, stream);
    bool arrivals = fromBelow + fromAbove > 0;
    distributed = true;
    if(arrivals){
        sortParticles();
    }
}

//The neighbouring partitions' particle nodes within 2 planes of this partition's, with the voxels their particles reach. With them, a partition decides
//which of its ghost nodes exist and what they store exactly as their owners do: a ghost node is next to this partition's planes, so its neighbours are
//within 2. They go after this partition's own particle nodes, the partition below's first, so in cell order, with their masks in foreignParticleNodeMasks
void Particles::receiveParticleNodes(uint maskWords){
    numForeignParticleNodes = 0;
    if(numRanks == 1){
        return;
    }
    uint cellsPerPlane = grid.sizeX*grid.sizeY;
    uint ends[2] = {0, 0};  //this partition's particle nodes before its third plane, and before its last 2: the runs its neighbours below and above take
    if(numParticleNodes > 0){
        uint* found;
        gpuErrchk(cudaMallocAsync((void**)&found, 2*sizeof(uint), stream));
        countCellsBefore<<<1, 1, 0, stream>>>(nodeCells.devPtr(), numParticleNodes, (boxLo + 2)*cellsPerPlane, found);
        countCellsBefore<<<1, 1, 0, stream>>>(nodeCells.devPtr(), numParticleNodes, (boxHi - 2)*cellsPerPlane, found + 1);
        cudaMemcpyAsync(ends, found, 2*sizeof(uint), cudaMemcpyDeviceToHost, stream);
        cudaFreeAsync(found, stream);
        gpuErrchk(cudaStreamSynchronize(stream));
    }
    lowerBoundaryParticleNodes = ends[0];
    upperBoundaryParticleNodes = numParticleNodes - ends[1];
    std::vector<int> neighbours = exchange->neighbours(rank);
    exchange->startReads(rank, neighbours);     //every partition has its particle nodes, their masks and these counts
    Particles* below = rank > 0 ? &exchange->partition(rank - 1) : nullptr;
    Particles* above = rank < numRanks - 1 ? &exchange->partition(rank + 1) : nullptr;
    uint fromBelow = below != nullptr ? below->upperBoundaryParticleNodes : 0;
    uint fromAbove = above != nullptr ? above->lowerBoundaryParticleNodes : 0;
    numForeignParticleNodes = fromBelow + fromAbove;
    foreignParticleNodeMasks.resizeAsync(numForeignParticleNodes*maskWords, stream);
    if(fromBelow > 0){
        uint first = below->numParticleNodes - fromBelow;
        exchange->copyFrom(rank, nodeCells.devPtr() + numParticleNodes, rank - 1, below->nodeCells.devPtr() + first, sizeof(uint)*fromBelow);
        exchange->copyFrom(rank, foreignParticleNodeMasks.devPtr(), rank - 1, below->particleNodeMasks.devPtr() + first*maskWords, sizeof(uint)*fromBelow*maskWords);
    }
    if(fromAbove > 0){
        exchange->copyFrom(rank, nodeCells.devPtr() + numParticleNodes + fromBelow, rank + 1, above->nodeCells.devPtr(), sizeof(uint)*fromAbove);
        exchange->copyFrom(rank, foreignParticleNodeMasks.devPtr() + fromBelow*maskWords, rank + 1, above->particleNodeMasks.devPtr(), sizeof(uint)*fromAbove*maskWords);
    }
    gpuErrchk(cudaPeekAtLastError());
    exchange->finishReads(rank, neighbours);    //the neighbours' copies are done before anyone changes what they copied
    if(numForeignParticleNodes > 0){    //registered in cellToNode, cells already known, and no particles here
        mapCellsToNodes<<<numForeignParticleNodes / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticleNodes, numParticleNodes + numForeignParticleNodes, numParticleNodes, size, gridCell.devPtr(),
            gridNodeIndicesToFirstParticleIndex.devPtr(), nodeCells.devPtr(), cellToNode.devPtr());
    }
}

//Puts the nodes after this partition's particle nodes in their order: its own empty nodes; its ghost nodes below, holding particles first, then above,
//the same way (as their owners keep them); the neighbours' particle nodes 2 planes out; each group by cell. Registers them in cellToNode, notes which hold
//particles and, split, puts each particle node's reached mask in its slot of usedVoxelMasks and finds the runs of nodes neighbours exchange
void Particles::orderNodes(uint maskWords){
    uint cellsPerPlane = grid.sizeX*grid.sizeY;
    uint numCells = cellsPerPlane*grid.sizeZ;
    uint numOthers = numUsedGridNodes - numParticleNodes;
    nodeHasParticles.resizeAsync(numUsedGridNodes, stream);
    usedVoxelMasks.resizeAsync(3*numUsedGridNodes*maskWords, stream);
    if(numRanks == 1){  //alone: its particle nodes, then its empty ones in cell order, and generateVoxels writes the reached masks into their slots itself
        cudaSortUints(numOthers, nodeCells.devPtr() + numParticleNodes, keyBits(numCells - 1), stream);
        if(numUsedGridNodes > 0){
            markParticleNodes<<<numUsedGridNodes / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticleNodes, numUsedGridNodes, nullptr, nodeHasParticles.devPtr());
        }
        if(numOthers > 0){
            mapCellsToNodes<<<numOthers / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticleNodes, numUsedGridNodes, numParticleNodes, size, gridCell.devPtr(), gridNodeIndicesToFirstParticleIndex.devPtr(),
                nodeCells.devPtr(), cellToNode.devPtr());
        }
        numOwnNodes = numUsedGridNodes;
        numStoredNodes = numUsedGridNodes;
        gpuErrchk(cudaPeekAtLastError());
        return;
    }
    uint cellBits = keyBits(numCells - 1);      //setDomain made sure the group's 3 bits fit above them
    uint particleNodes = numParticleNodes + numForeignParticleNodes;
    nodeOrigins.resizeAsync(numUsedGridNodes, stream);
    //where the runs start, found on the GPU and read back together: in the sorted keys, the starts of groups 1 to 5, and where this partition's empty
    //nodes leave its first plane and reach its last (group 0's keys are just cells); then the same among its particle nodes
    uint* found;
    gpuErrchk(cudaMallocAsync((void**)&found, 9*sizeof(uint), stream));
    gpuErrchk(cudaMemsetAsync(found, 0, 9*sizeof(uint), stream));
    if(numParticleNodes > 0){
        markParticleNodes<<<numParticleNodes / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticleNodes, numParticleNodes, nodeOrigins.devPtr(), nodeHasParticles.devPtr());
        countCellsBefore<<<1, 1, 0, stream>>>(nodeCells.devPtr(), numParticleNodes, (boxLo + 1)*cellsPerPlane, found + 7);
        countCellsBefore<<<1, 1, 0, stream>>>(nodeCells.devPtr(), numParticleNodes, (boxHi - 1)*cellsPerPlane, found + 8);
    }
    uint* keys = nullptr;
    uint* origins = nullptr;
    if(numOthers > 0){
        gpuErrchk(cudaMallocAsync((void**)&keys, sizeof(uint)*numOthers, stream));
        gpuErrchk(cudaMallocAsync((void**)&origins, sizeof(uint)*numOthers, stream));
        nodeOrderKeys<<<numOthers / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticleNodes, numUsedGridNodes, particleNodes, nodeCells.devPtr(), boxLo*cellsPerPlane, boxHi*cellsPerPlane, cellsPerPlane,
            cellBits, keys, origins);
        cudaSortNodeOrder(numOthers, keys, origins, cellBits + 3, stream);
        placeOrderedNodes<<<numOthers / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticleNodes, numOthers, keys, origins, particleNodes, (1u << cellBits) - 1, nodeCells.devPtr(), nodeOrigins.devPtr(),
            nodeHasParticles.devPtr());
        for(uint group = 1; group <= 5; ++group){   //the keys are sorted, so each group is a run
            countCellsBefore<<<1, 1, 0, stream>>>(keys, numOthers, group << cellBits, found + group - 1);
        }
        countCellsBefore<<<1, 1, 0, stream>>>(keys, numOthers, (boxLo + 1)*cellsPerPlane, found + 5);
        countCellsBefore<<<1, 1, 0, stream>>>(keys, numOthers, (boxHi - 1)*cellsPerPlane, found + 6);
        mapCellsToNodes<<<numOthers / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numParticleNodes, numUsedGridNodes, numParticleNodes, size, gridCell.devPtr(), gridNodeIndicesToFirstParticleIndex.devPtr(),
            nodeCells.devPtr(), cellToNode.devPtr());
    }
    uint counts[9];
    cudaMemcpyAsync(counts, found, 9*sizeof(uint), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(found, stream);
    if(numOthers > 0){
        cudaFreeAsync(keys, stream);
        cudaFreeAsync(origins, stream);
    }
    gpuErrchk(cudaStreamSynchronize(stream));
    uint numOwnEmpty = counts[0];
    numOwnNodes = numParticleNodes + numOwnEmpty;
    numStoredNodes = numParticleNodes + counts[4];     //all but the neighbours' particle nodes 2 planes out
    //the node runs at its edges and its ghosts either side, whose voxels generateVoxels finds
    edgeParticles[0] = {0, counts[7]};
    edgeParticles[1] = {counts[8], numParticleNodes - counts[8]};
    edgeEmpty[0] = {numParticleNodes, counts[5]};
    edgeEmpty[1] = {numParticleNodes + counts[6], numOwnEmpty - counts[6]};
    ghostParticles[0] = {numParticleNodes + counts[0], counts[1] - counts[0]};
    ghostEmpty[0] = {numParticleNodes + counts[1], counts[2] - counts[1]};
    ghostParticles[1] = {numParticleNodes + counts[2], counts[3] - counts[2]};
    ghostEmpty[1] = {numParticleNodes + counts[3], counts[4] - counts[3]};
    if(numUsedGridNodes > 0){
        placeReachedMasks<<<numUsedGridNodes, 32, 0, stream>>>(maskWords, numParticleNodes, nodeOrigins.devPtr(), nodeHasParticles.devPtr(), particleNodeMasks.devPtr(), foreignParticleNodeMasks.devPtr(),
            usedVoxelMasks.devPtr());
    }
    gpuErrchk(cudaPeekAtLastError());
}

//each ghost node's voxels here and in its owner, for filling and reducing; and the check that both partitions built it alike
//A partition's ghost runs have to be its neighbours' edge runs, node for node and voxel for voxel. Both sides build them from the same particles, so they
//always should be; if they ever differ, filling ghosts would mix up voxels, so stop
void Particles::checkGhostRuns(){
    if(numRanks == 1){
        return;
    }
    exchange->barrier();    //every partition's runs are known, and stay put until after the next substep's exchangeParticles
    for(int side = 0; side < 2; ++side){
        int neighbour = side == 0 ? rank - 1 : rank + 1;
        if(neighbour < 0 || neighbour >= numRanks){
            continue;
        }
        Particles& other = exchange->partition(neighbour);
        const VoxelRun* mine[2] = {&ghostParticles[side], &ghostEmpty[side]};
        const VoxelRun* theirs[2] = {&other.edgeParticles[1 - side], &other.edgeEmpty[1 - side]};
        for(int run = 0; run < 2; ++run){
            if(mine[run]->nodes != theirs[run]->nodes || mine[run]->count != theirs[run]->count){
                std::cerr<<"partition "<<rank<<": its ghost copies of partition "<<neighbour<<"'s "<<(run == 0 ? "nodes holding particles" : "empty nodes")<<" are "
                         <<mine[run]->nodes<<" nodes with "<<mine[run]->count<<" voxels, but the owner has "<<theirs[run]->nodes<<" with "<<theirs[run]->count<<"\n";
                exit(1);
            }
        }
    }
}

void Particles::generateVoxels(){
    numParticleNodes = size > 0 ? cudaMarkUniqueGridCellsAndCount(size, gridCell.devPtr(), uniqueGridNodeIndices.devPtr(), stream) : 0;

    //the used nodes: the ones holding particles, then their neighbours without any, which the particles reach into. Split between partitions, the
    //neighbours' particle nodes within 2 planes come too, and the nodes in the planes next to this partition's are its ghost nodes
    uint cellsPerPlane = grid.sizeX*grid.sizeY;
    uint numCells = grid.sizeX*grid.sizeY*grid.sizeZ;
    uint mostParticleNodes = numParticleNodes + (numRanks > 1 ? 4*cellsPerPlane : 0);  //with the neighbours' 2 planes either side
    uint maxNodes = 27*mostParticleNodes < numCells ? 27*mostParticleNodes : numCells;
    gridNodeIndicesToFirstParticleIndex.resizeAsync(maxNodes, stream);
    nodeCells.resizeAsync(maxNodes, stream);
    uint maskWords = (numVoxelsPerNode + 31) / 32;
    if(numParticleNodes > 0){
        cudaMapNodeIndicesToParticles(size, uniqueGridNodeIndices.devPtr(), gridNodeIndicesToFirstParticleIndex.devPtr(), stream);
        mapCellsToNodes<<<numParticleNodes / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(0, numParticleNodes, numParticleNodes, size, gridCell.devPtr(), gridNodeIndicesToFirstParticleIndex.devPtr(), nodeCells.devPtr(), cellToNode.devPtr());
    }
    if(numRanks > 1){   //split, the neighbours take the reached masks of the particle nodes near them before there's a node table to put them in
        particleNodeMasks.resizeAsync(numParticleNodes*maskWords, stream);
        if(numParticleNodes > 0){
            markReachedVoxels<<<numParticleNodes, NODE_THREADS, sizeof(uint)*maskWords, stream>>>(numParticleNodes, size, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(),
                px.devPtr(), py.devPtr(), pz.devPtr(), particleNodeMasks.devPtr(), maskWords, radius, grid, refinementLevel);
        }
    }
    receiveParticleNodes(maskWords);
    uint particleNodes = numParticleNodes + numForeignParticleNodes;
    cudaMemcpyAsync(nodeCount.devPtr(), &particleNodes, sizeof(uint), cudaMemcpyHostToDevice, stream);
    uint firstClaimCell = (boxLo > 0 ? boxLo - 1 : 0)*cellsPerPlane;     //empty nodes worth having: in this partition's planes, and the ghosts next to them
    uint endClaimCell = (boxHi < grid.sizeZ ? boxHi + 1 : grid.sizeZ)*cellsPerPlane;
    if(particleNodes > 0){
        claimEmptyNeighborNodes<<<particleNodes, 32, 0, stream>>>(particleNodes, nodeCells.devPtr(), cellToNode.devPtr(), nodeCount.devPtr(), grid, firstClaimCell, endClaimCell);
    }
    cudaMemcpyAsync(&numUsedGridNodes, nodeCount.devPtr(), sizeof(uint), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    //the empty nodes were appended in whatever order their claims landed: put everything after the particle nodes in a fixed order, so every run numbers the nodes, and so the voxels, the same
    orderNodes(maskWords);
    if(numRanks == 1 && numParticleNodes > 0){  //alone, the particle nodes come first in the node table, in the same order: each mask goes straight into its slot
        markReachedVoxels<<<numParticleNodes, NODE_THREADS, sizeof(uint)*maskWords, stream>>>(numParticleNodes, size, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(),
            px.devPtr(), py.devPtr(), pz.devPtr(), usedVoxelMasks.devPtr() + 2*maskWords, 3*maskWords, radius, grid, refinementLevel);
    }

    nodeIndexUsedVoxels.resizeAsync(numUsedGridNodes, stream);
    nodeIndexUsedVoxels.zeroDeviceAsync(stream);    //the nodes 2 planes out store no voxels
    if(numStoredNodes > 0){
        markUsedVoxels<<<numStoredNodes, 64, 2*sizeof(uint)*maskWords, stream>>>(numUsedGridNodes, nodeHasParticles.devPtr(), numVoxels1D, nodeCells.devPtr(), cellToNode.devPtr(), usedVoxelMasks.devPtr(),
            nodeIndexUsedVoxels.devPtr(), radius, grid, refinementLevel);
    }
    cudaParallelPrefixSum(numUsedGridNodes, nodeIndexUsedVoxels.devPtr(), stream);

    uint numUsedVoxels = 0;
    numOwnVoxels = 0;
    if(numRanks == 1){
        if(numUsedGridNodes > 0){
            cudaMemcpyAsync(&numUsedVoxels, nodeIndexUsedVoxels.devPtr() + numUsedGridNodes - 1, sizeof(uint), cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);
        numOwnVoxels = numUsedVoxels;
    }
    else{   //the edge and ghost runs' voxels too, read back with the counts
        VoxelRun* runs[8] = {&edgeParticles[0], &edgeParticles[1], &edgeEmpty[0], &edgeEmpty[1], &ghostParticles[0], &ghostParticles[1], &ghostEmpty[0], &ghostEmpty[1]};
        NodeIndices nodes;
        nodes.count = 18;
        for(int run = 0; run < 8; ++run){
            nodes.index[2*run] = runs[run]->firstNode;
            nodes.index[2*run + 1] = runs[run]->firstNode + runs[run]->nodes;
        }
        nodes.index[16] = numOwnNodes;
        nodes.index[17] = numUsedGridNodes;
        uint* starts;
        uint found[18];
        gpuErrchk(cudaMallocAsync((void**)&starts, sizeof(found), stream));
        voxelStartsOf<<<1, 32, 0, stream>>>(nodes, nodeIndexUsedVoxels.devPtr(), starts);
        cudaMemcpyAsync(found, starts, sizeof(found), cudaMemcpyDeviceToHost, stream);
        cudaFreeAsync(starts, stream);
        gpuErrchk(cudaStreamSynchronize(stream));
        for(int run = 0; run < 8; ++run){
            runs[run]->start = found[2*run];
            runs[run]->count = found[2*run + 1] - found[2*run];
        }
        numOwnVoxels = found[16];
        numUsedVoxels = found[17];
    }
    voxelIDsUsed.resizeAsync(numUsedVoxels, stream);
    voxelOwners.resizeAsync(numUsedVoxels, stream);
    solids.resizeAsync(numUsedVoxels, stream);
    solveCodes.resizeAsync(numUsedVoxels, stream);
    footprintDepth.resizeAsync(numUsedVoxels, stream);
    for(CudaVec<uint>* neighbor : {&neighborNx, &neighborPx, &neighborNy, &neighborPy, &neighborNz, &neighborPz, &coarseCells}){
        neighbor->resizeAsync(numUsedVoxels, stream);
    }
    uint interiorWidth = numVoxels1D - 2*(uint)std::floor(radius);
    nodeInteriorVoxels.resizeAsync(numUsedGridNodes*interiorWidth*interiorWidth*interiorWidth, stream);
    cudaMemsetAsync(nodeInteriorVoxels.devPtr(), 0xFF, sizeof(uint)*nodeInteriorVoxels.size(), stream);   //NO_VOXEL until buildVoxelTopology finds one
    for(CudaVec<float>* voxelData : {&voxelsUx, &voxelsUy, &voxelsUz, &voxelsUxOld, &voxelsUyOld, &voxelsUzOld, &voxelWeightsX, &voxelWeightsY, &voxelWeightsZ, &particleCounts, &divU, &p, &residuals, &Anx, &Apx, &Any, &Apy, &Anz, &Apz, &Adiag}){
        voxelData->resizeAsync(numUsedVoxels, stream);
    }
    p.zeroDeviceAsync(stream);

    for(CudaVec<uint>* lower : {&neighborNx, &neighborNy, &neighborNz}){   //air voxels' lower neighbours: NO_VOXEL unless an unknown below claims them
        cudaMemsetAsync(lower->devPtr(), 0xFF, sizeof(uint)*numUsedVoxels, stream);
    }
    if(numStoredNodes > 0){     //the nodes 2 planes out have no voxels to build
        writeUsedVoxelIDs<<<numStoredNodes, 32, 0, stream>>>(numVoxels1D, usedVoxelMasks.devPtr(), nodeIndexUsedVoxels.devPtr(), nodeCells.devPtr(), voxelIDsUsed.devPtr(), solids.devPtr(), solveCodes.devPtr(), radius, grid, refinementLevel);
        buildVoxelTopology<<<numStoredNodes, 64, sizeof(uint)*numVoxelsPerNode, stream>>>(numUsedGridNodes, numOwnNodes, numVoxels1D, radius, nodeCells.devPtr(), cellToNode.devPtr(), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(),
            voxelOwners.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(), solveCodes.devPtr(), coarseCells.devPtr(), nodeInteriorVoxels.devPtr(),
            grid, refinementLevel);
    }
    gpuErrchk(cudaPeekAtLastError());
    checkGhostRuns();

    if(numRanks == 1){
        std::cout<<"Nodes: "<<numParticleNodes<<" with particles, "<<numUsedGridNodes<<" in all, holding "<<numUsedVoxels<<" voxels\n";
    }
    else{
        std::ostringstream line;    //written whole, so the partitions' threads can't interleave their lines
        line<<"Partition "<<rank<<": "<<size<<" particles, nodes: "<<numParticleNodes<<" with particles, "<<numOwnNodes<<" its own, "<<numStoredNodes - numOwnNodes<<" ghosts, "
            <<numUsedGridNodes - numStoredNodes<<" beyond; "<<numOwnVoxels<<" of its "<<numUsedVoxels<<" voxels its own\n";
        std::cout<<line.str();
    }
    if(verbose()){      //written whole too: the other partitions print their lines meanwhile
        std::cout<<"Using " + std::to_string((CudaVec<uint>::GPU_MEMORY_ALLOCATED + CudaVec<float>::GPU_MEMORY_ALLOCATED + CudaVec<double>::GPU_MEMORY_ALLOCATED + CudaVec<char>::GPU_MEMORY_ALLOCATED) / (1<<20)) + " MB on GPU\n";
    }
}

//P2G: a block per node and a thread per particle. Each particle adds its momentum and weight on the 27 faces of each component around it into the
//node's voxel block in shared memory, and 1 to the count of the voxel holding it; then each voxel the node stores adds the block's sums into its owner.
//It sums in 32-bit fixed point, as sm_86 has native shared integer atomics but not float ones. 7 ints per slot: 14 KB for an 8^3 block. The owners'
//sums stay fixed point too, in the float arrays' storage: integers add up the same in any order, so the result doesn't depend on which node gets there
//first. normalizeVoxelVelocities turns them into floats
__global__ void scatterParticleVelsToVoxels(uint numParticleNodes, uint numParticles, uint numVoxels1D, const uint* gridNodeIndicesToFirstParticleIndex, const uint* gridPosition,
                                            const double* px, const double* py, const double* pz, const float* vx, const float* vy, const float* vz,
                                            const uint* numVoxelsEachNode, const uint* voxelIDs, const uint* voxelOwners,
                                            int* ux, int* uy, int* uz, int* weightsX, int* weightsY, int* weightsZ, int* particleCounts, float momentumScale, float weightScale, double radius, Grid grid, uint refinementLevel){
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
    int* accumulators[7] = {ux, uy, uz, weightsX, weightsY, weightsZ, particleCounts};
    for(uint i = (blockIdx.x == 0 ? 0 : numVoxelsEachNode[blockIdx.x - 1]) + threadIdx.x; i < numVoxelsEachNode[blockIdx.x]; i += blockDim.x){    //coalesced over the node's voxels
        int slot = voxelIDs[i];
        uint owner = voxelOwners[i];    //itself for interior voxels, so these atomics are mostly consecutive
        #pragma unroll
        for(int sum = 0; sum < 7; ++sum){
            int value = fixedSums[sum*voxels3D + slot];
            if(value != 0){
                atomicAdd(accumulators[sum] + owner, value);
            }
        }
    }
}

#include "algorithms/reductionKernels.hu"
#include <cmath>

double Particles::getCourantDt(){    //every substep: the fastest particle moves at most cfl voxels. P2G finds the fastest
    double voxelSize = (grid.cellSize / (numVoxels1D - 2*std::floor(radius)));
    if(verbose()){
        std::cout<<"maxVel: "<<maxVelocity<<" voxelSize: "<<voxelSize<<"\n";
    }
    return cfl * (voxelSize / maxVelocity + 0.0001);
}

void Particles::particleVelToVoxels(){
    for(CudaVec<float>* accumulator : {&voxelsUx, &voxelsUy, &voxelsUz, &voxelWeightsX, &voxelWeightsY, &voxelWeightsZ, &particleCounts}){
        accumulator->zeroDeviceAsync(stream);
    }
    maxVelocity = largestMagnitude({&vx, &vy, &vz}, stream);   //the fastest component, which also sets the CFL timestep
    maxVelocity = context->maxOverPartitions(maxVelocity);     //every partition's fixed point and timestep have to agree
    //P2G sums in fixed point: a face's weight sum is about the particles per voxel, so budget 128 (15x rest) and keep every sum under 2^30
    float weightScale = (1 << 30) / 128.0f;
    float momentumScale = weightScale / fmax(maxVelocity, 1e-6);
    if(numParticleNodes > 0){
        scatterParticleVelsToVoxels<<<numParticleNodes, NODE_THREADS, 7*sizeof(int)*numVoxelsPerNode, stream>>>(numParticleNodes, size, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr(),
            nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelOwners.devPtr(), (int*)voxelsUx.devPtr(), (int*)voxelsUy.devPtr(), (int*)voxelsUz.devPtr(),
            (int*)voxelWeightsX.devPtr(), (int*)voxelWeightsY.devPtr(), (int*)voxelWeightsZ.devPtr(), (int*)particleCounts.devPtr(), momentumScale, weightScale, radius, grid, refinementLevel);
    }
    //particles near this partition's edge reach into ghost voxels: their owners add those sums in, which in fixed point come out the same in any order
    for(CudaVec<float>* accumulator : {&voxelsUx, &voxelsUy, &voxelsUz, &voxelWeightsX, &voxelWeightsY, &voxelWeightsZ, &particleCounts}){
        context->reduceGhosts((int*)accumulator->devPtr(), stream);
    }
    cudaNormalizeVoxelVelocities(solids, voxelWeightsX, voxelWeightsY, voxelWeightsZ, voxelsUx, voxelsUy, voxelsUz, particleCounts, momentumScale, weightScale, stream);
    for(CudaVec<float>* field : {&voxelsUx, &voxelsUy, &voxelsUz, &voxelWeightsX, &voxelWeightsY, &voxelWeightsZ, &particleCounts}){   //and the ghosts take the totals
        context->fillGhosts(field->devPtr(), stream);
    }
    cudaExtrapolateUnreachedFaces(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, voxelWeightsX, voxelWeightsY, voxelWeightsZ, voxelsUx, voxelsUy, voxelsUz, *context, stream);
    cudaFindFootprintDepth(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, footprintDepth, freeSurface, *context, stream);
    for(auto [velocity, before] : {std::pair{&voxelsUx, &voxelsUxOld}, {&voxelsUy, &voxelsUyOld}, {&voxelsUz, &voxelsUzOld}}){    //for FLIP's velocity change over the solve
        cudaMemcpyAsync(before->devPtr(), velocity->devPtr(), sizeof(float)*velocity->size(), cudaMemcpyDeviceToDevice, stream);
    }
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
    hasFreeSurface = context->anyOverPartitions(hasFreeSurface != 0);
    //divergence per unit of relative density error. Without air the fluid can't change volume, and the solve has nowhere to take a net divergence, so it's off
    double correctionRate = hasFreeSurface && densityCorrectionTime > 0.0 ? voxelSize / densityCorrectionTime : 0.0;
    //@TODO: need to use courant number for dt from max voxel u and voxel dimensions
    dt = getCourantDt();
    if(verbose()){
        std::cout<<"initial dt: "<<dt<<std::endl;
    }
    if(frameDt - elapsedTimeThisFrame < dt){
        dt = frameDt - elapsedTimeThisFrame;
    }
    applyGravity(solids, voxelsUy, dt, stream);
    cudaCalcDivU(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, voxelsUx, voxelsUy, voxelsUz, particleCounts, footprintDepth, restParticlesPerVoxel, correctionRate, divU, stream);
    cudaGetA(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag, dt/(density*voxelSize*voxelSize), stream);
    gpuErrchk(cudaPeekAtLastError());
    uint interiorWidth = numVoxels1D - 2*(uint)std::floor(radius);
    uint3 domainVoxels = make_uint3(grid.sizeX*interiorWidth, grid.sizeY*interiorWidth, grid.sizeZ*interiorWidth);
    VoxelLayout layout = {nodeCells.devPtr(), nodeInteriorVoxels.devPtr(), coarseCells.devPtr(), numOwnNodes, interiorWidth, domainVoxels};   //the nodes this partition solves for
    auto solve = [&](){
        switch(pressureSolver){
            case PressureSolver::cg:
                return cudaConjugateGradient(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag, divU, p, residuals, tolerance, maxIterations,
                                             layout, dotProductSums, numOwnVoxels, *context, stream);
            case PressureSolver::jacobi:
                return cudaJacobiConjugateGradient(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag, divU, p, residuals, tolerance, maxIterations,
                                                   layout, dotProductSums, numOwnVoxels, *context, stream);
            case PressureSolver::multigrid:
                return cudaMultigridConjugateGradient(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag, divU, p, residuals, tolerance, maxIterations,
                                                      layout, dt/(density*voxelSize*voxelSize), dotProductSums, numOwnVoxels, *context, stream);
            default:
                return cudaGSiteration(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag, divU, p, residuals, tolerance, maxIterations,
                                       numOwnVoxels, *context, stream);
        }
    };
    while(previousTerminatingResidual - (terminatingResidual = solve()) > 0.0){    //while residual getting smaller
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
    if(verbose()){
        std::cout<<"dt: "<<dt<<" ElapsedTime: "<<elapsedTime<<" elapsedTimeThisFrame: "<<elapsedTimeThisFrame<<"\nTerminating Residual: "<<terminatingResidual<<"\nTolerance: "<<tolerance<<"\n";
    }
    gpuErrchk(cudaPeekAtLastError());
}

void Particles::updateVoxelVelocities(){
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    cudaVelocityUpdate(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, p, voxelsUx, voxelsUy, voxelsUz, dt/(0.014*voxelSize*voxelSize), stream);
    gpuErrchk(cudaPeekAtLastError());
    for(CudaVec<float>* velocity : {&voxelsUx, &voxelsUy, &voxelsUz}){   //G2P and advection read ghost faces
        context->fillGhosts(velocity->devPtr(), stream);
    }
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
                                            float flipRatio, double radius, Grid grid, uint refinementLevel){
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
        }
    }
}

__global__ void advectParticlePositions(uint numParticles, double dt, double* position, const float* v, Grid grid){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        position[index] += dt*v[index];
    }
}

//a MAC velocity component at a point of an advection tile (in voxels), trilinear between the 8 faces around it. Component dim's faces sit on voxel
//boundaries along dim and mid-voxel along the other axes. NaN if a face it leans on has no velocity
__device__ inline float sampleTile(const float* tile, int tileWidth, int dim, float3 point){
    float coordinates[3] = {point.x, point.y, point.z};
    int base[3];
    float fraction[3];
    #pragma unroll
    for(int axis = 0; axis < 3; ++axis){
        float c = coordinates[axis] - (axis == dim ? 0.0f : 0.5f);
        base[axis] = min(max((int)floorf(c), 0), tileWidth - 2);   //never off the tile, even past CFL 4
        fraction[axis] = fminf(fmaxf(c - base[axis], 0.0f), 1.0f);
    }
    float sum = 0.0f;
    #pragma unroll
    for(int corner = 0; corner < 8; ++corner){
        int dx = corner & 1, dy = corner >> 1 & 1, dz = corner >> 2;
        float weight = (dx ? fraction[0] : 1.0f - fraction[0])*(dy ? fraction[1] : 1.0f - fraction[1])*(dz ? fraction[2] : 1.0f - fraction[2]);
        if(weight != 0.0f){     //a face with no say can't spoil it
            sum += weight*tile[base[0] + dx + (base[1] + dy)*tileWidth + (base[2] + dz)*tileWidth*tileWidth];
        }
    }
    return sum;
}

//the grid's velocity at a point of the tile, any component with no velocity there taking fallback's. Each component's point is first kept inside the
//walls: a wall's own faces hold its no-flow condition, and beside a wall the tangential velocity carries on to it unchanged, like G2P's mirrored ghosts
__device__ inline float3 sampleVelocity(const float* tile, int tileWidth, float3 point, int3 tileOrigin, int3 domainVoxels, float3 fallback){
    int origin[3] = {tileOrigin.x, tileOrigin.y, tileOrigin.z};
    int size[3] = {domainVoxels.x, domainVoxels.y, domainVoxels.z};
    float fallbacks[3] = {fallback.x, fallback.y, fallback.z};
    float velocity[3];
    #pragma unroll
    for(int dim = 0; dim < 3; ++dim){
        float coordinates[3] = {point.x, point.y, point.z};
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            float inset = axis == dim ? 0.0f : 0.5f;
            coordinates[axis] = fminf(fmaxf(coordinates[axis], inset - origin[axis]), size[axis] - inset - origin[axis]);
        }
        float v = sampleTile(tile + dim*tileWidth*tileWidth*tileWidth, tileWidth, dim, make_float3(coordinates[0], coordinates[1], coordinates[2]));
        velocity[dim] = isnan(v) ? fallbacks[dim] : v;
    }
    return make_float3(velocity[0], velocity[1], velocity[2]);
}

//moves each particle through the grid's new velocity field for the whole step: with Ralston's RK3, which samples the field 3 times along the way, or
//straight along it (forward Euler). A straight step squeezes particles together where the flow stretches and spreads them where it turns, by an amount
//growing with dt^2, which clumps them at large CFL; RK3 leaves that at dt^4. A block per node with particles: its interior and its 26 neighbours' go
//into shared memory, a tile 3 nodes wide, which holds every trilinear sample RK3 takes up to CFL 4. Where nothing reached (spray leaving the fluid), a
//stage reuses the stage before, so the particle flies straight
__global__ void advectThroughGrid(uint numParticleNodes, uint numParticles, const uint* gridNodeIndicesToFirstParticleIndex, const uint* gridPosition,
                                  double* px, double* py, double* pz, const float* vx, const float* vy, const float* vz,
                                  uint numUsedGridNodes, const uint* nodeCells, const uint* cellToNode, const uint* interiorVoxels,
                                  const float* ux, const float* uy, const float* uz, const float* weightsX, const float* weightsY, const float* weightsZ,
                                  float dt, bool rungeKutta3, Grid grid, uint refinementLevel){
    extern __shared__ float tile[];     //per component, the tile's voxels' negative faces
    __shared__ uint neighborNodes[27];
    int interiorWidth = 2<<refinementLevel;
    int tileWidth = 3*interiorWidth;
    int tileVoxels = tileWidth*tileWidth*tileWidth;
    uint firstParticle = gridNodeIndicesToFirstParticleIndex[blockIdx.x];
    uint lastParticle = blockIdx.x == numParticleNodes - 1 ? numParticles : gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1];
    uint cell = gridPosition[firstParticle];
    loadNeighborNodes(neighborNodes, cell, numUsedGridNodes, nodeCells, cellToNode, grid);
    int3 domainVoxels = make_int3(grid.sizeX*interiorWidth, grid.sizeY*interiorWidth, grid.sizeZ*interiorWidth);
    int3 tileOrigin = make_int3(((int)(cell % grid.sizeX) - 1)*interiorWidth, ((int)(cell / grid.sizeX % grid.sizeY) - 1)*interiorWidth, ((int)(cell / (grid.sizeX*grid.sizeY)) - 1)*interiorWidth);
    __syncthreads();
    const float* velocities[3] = {ux, uy, uz};
    const float* weights[3] = {weightsX, weightsY, weightsZ};
    for(int t = threadIdx.x; t < tileVoxels; t += blockDim.x){
        int x = t % tileWidth, y = t / tileWidth % tileWidth, z = t / (tileWidth*tileWidth);
        int3 global = make_int3(tileOrigin.x + x, tileOrigin.y + y, tileOrigin.z + z);
        bool inside = global.x >= 0 && global.y >= 0 && global.z >= 0 && global.x < domainVoxels.x && global.y < domainVoxels.y && global.z < domainVoxels.z;
        uint node = neighborNodes[x / interiorWidth + 3*(y / interiorWidth) + 9*(z / interiorWidth)];
        uint voxel = inside && node < numUsedGridNodes ?
            interiorVoxels[node*interiorWidth*interiorWidth*interiorWidth + x % interiorWidth + (y % interiorWidth)*interiorWidth + (z % interiorWidth)*interiorWidth*interiorWidth] : NO_VOXEL;
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){   //past the walls, their faces at rest; inside, where nothing is stored or no particle reached, no velocity
            tile[dim*tileVoxels + t] = !inside ? 0.0f : voxel != NO_VOXEL && weights[dim][voxel] > 0.0f ? velocities[dim][voxel] : nanf("");
        }
    }
    __syncthreads();
    float voxelSize = grid.cellSize / interiorWidth;
    float step = dt / voxelSize;    //turns a velocity into voxels moved this step
    for(uint index = firstParticle + threadIdx.x; index < lastParticle; index += blockDim.x){
        float3 point = make_float3((px[index] - grid.negX)/voxelSize - tileOrigin.x, (py[index] - grid.negY)/voxelSize - tileOrigin.y, (pz[index] - grid.negZ)/voxelSize - tileOrigin.z);
        float3 k1 = sampleVelocity(tile, tileWidth, point, tileOrigin, domainVoxels, make_float3(vx[index], vy[index], vz[index]));
        float3 velocity = k1;
        if(rungeKutta3){
            float3 k2 = sampleVelocity(tile, tileWidth, make_float3(point.x + 0.5f*step*k1.x, point.y + 0.5f*step*k1.y, point.z + 0.5f*step*k1.z), tileOrigin, domainVoxels, k1);
            float3 k3 = sampleVelocity(tile, tileWidth, make_float3(point.x + 0.75f*step*k2.x, point.y + 0.75f*step*k2.y, point.z + 0.75f*step*k2.z), tileOrigin, domainVoxels, k2);
            velocity = make_float3((2.0f*k1.x + 3.0f*k2.x + 4.0f*k3.x)/9.0f, (2.0f*k1.y + 3.0f*k2.y + 4.0f*k3.y)/9.0f, (2.0f*k1.z + 3.0f*k2.z + 4.0f*k3.z)/9.0f);
        }
        px[index] += dt*velocity.x;
        py[index] += dt*velocity.y;
        pz[index] += dt*velocity.z;
    }
}

void Particles::voxelVelsToParticles(){
    if(numParticleNodes == 0){  //no particles in this partition
        return;
    }
    gatherVoxelVelsToParticles<<<numParticleNodes, NODE_THREADS, 6*sizeof(float)*numVoxelsPerNode, stream>>>(numParticleNodes, size, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr(),
        nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelOwners.devPtr(), solids.devPtr(), voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr(), voxelsUxOld.devPtr(), voxelsUyOld.devPtr(), voxelsUzOld.devPtr(), flipRatio, radius, grid, refinementLevel);
    gpuErrchk(cudaPeekAtLastError());
    uint tileWidth = 3*(numVoxels1D - 2*(uint)std::floor(radius));
    advectThroughGrid<<<numParticleNodes, NODE_THREADS, 3*sizeof(float)*tileWidth*tileWidth*tileWidth, stream>>>(numParticleNodes, size, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(),
        px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr(), numUsedGridNodes, nodeCells.devPtr(), cellToNode.devPtr(), nodeInteriorVoxels.devPtr(),
        voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr(), voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), dt, rungeKutta3, grid, refinementLevel);
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

    //cudaMallocAsync's pool keeps the memory it's given. By default it hands everything free back to the driver at every sync, and each substep's
    //allocations then map it all again: milliseconds of idle GPU a substep at 10M particles
    int device;
    gpuErrchk(cudaGetDevice(&device));
    cudaMemPool_t pool;
    gpuErrchk(cudaDeviceGetDefaultMemPool(&pool, device));
    unsigned long long keepEverything = ~0ull;  //a 64-bit threshold
    gpuErrchk(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keepEverything));
}

void Particles::solveFrame(double fps){
    frameDt = 1.0f/fps;
    dt = frameDt / 10.0f;
    elapsedTimeThisFrame = 0.0f;
    while(elapsedTimeThisFrame < frameDt){    //every step is queued on this partition's one stream, in order: the host only waits where it reads something back
        particleVelToVoxels();
        pressureSolve();
        updateVoxelVelocities();
        voxelVelsToParticles();   //also moves the particles; initialize re-bins them wherever they landed, reflecting any that crossed a wall
        initialize();
    }
}

void Particles::initialize(){
        alignParticlesToGrid();
        sortParticles();
        exchangeParticles();    //particles that crossed into another partition's planes move there
        generateVoxels();
}

#include <iostream>
#include <fstream>

__global__ void packPositionsToFloats(uint numParticles, const double* px, const double* py, const double* pz, float* xyz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        xyz[3*index] = px[index];
        xyz[3*index + 1] = py[index];
        xyz[3*index + 2] = pz[index];
    }
}

//a frame's float32 x, y, z per particle, this partition's run of it, into pinned host memory at xyz. Packed on this partition's stream, then copied out on
//frameStream, so the simulation's next kernels run alongside the copy; copied is recorded once it's in. The next frame's pack waits for the copy to finish
//reading framePositions
void Particles::copyPositionsToHost(float* xyz, cudaEvent_t copied){
    if(frameStream == nullptr){     //on this partition's GPU, which the caller has made current
        gpuErrchk(cudaStreamCreateWithFlags(&frameStream, cudaStreamNonBlocking));
        gpuErrchk(cudaEventCreateWithFlags(&framePacked, cudaEventDisableTiming));
        gpuErrchk(cudaEventCreateWithFlags(&frameCopied, cudaEventDisableTiming));
    }
    gpuErrchk(cudaStreamWaitEvent(stream, frameCopied, 0));    //never recorded yet: no wait
    if(size > 0){
        framePositions.resizeAsync(3*size, stream);
        packPositionsToFloats<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, px.devPtr(), py.devPtr(), pz.devPtr(), framePositions.devPtr());
        gpuErrchk(cudaPeekAtLastError());
    }
    gpuErrchk(cudaEventRecord(framePacked, stream));
    gpuErrchk(cudaStreamWaitEvent(frameStream, framePacked, 0));
    if(size > 0){
        gpuErrchk(cudaMemcpyAsync(xyz, framePositions.devPtr(), sizeof(float)*3*size, cudaMemcpyDeviceToHost, frameStream));
    }
    gpuErrchk(cudaEventRecord(frameCopied, frameStream));
    gpuErrchk(cudaEventRecord(copied, frameStream));
}

Particles::~Particles(){     //with its GPU the current one
    if(frameStream != nullptr){
        cudaStreamSynchronize(frameStream);
        cudaStreamDestroy(frameStream);
        cudaEventDestroy(framePacked);
        cudaEventDestroy(frameCopied);
    }
    gpuErrchk( cudaStreamDestroy(stream) );
}