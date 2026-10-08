//Copyright 2023 Aberrant Behavior LLC

#include "particles.hu"

#include "algorithms/radixSortKernels.hu"
#include "algorithms/particleToGridFunctions.hu"
#include "algorithms/perVoxelParticleListFunctions.hu"
#include "algorithms/voxelSolveFunctions.hu"
#include "algorithms/conjugateGradientFunctions.hu"
#include "algorithms/parallelPrefixSumKernels.hu"
#include "transport.hu"
#include "gridSampling.hu"

#include "typedefs.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <type_traits>

//ids first, first + 1, ... for count particles
__global__ void numberParticles(uint count, uint first, uint* ids){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < count){
        ids[index] = first + index;
    }
}

//the particles numbered from first on are air
__global__ void markAir(uint count, uint first, uint* ids){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < count && ids[index] >= first){
        ids[index] |= AIR_PARTICLE;
    }
}

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

    particleIds.resizeAsync(size, stream);      //numbered in the order they're given in, born at the start; setIdentities says otherwise
    particleBirths.resizeAsync(size, stream);
    particleBirths.zeroDeviceAsync(stream);
    if(size > 0){
        numberParticles<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, 0, particleIds.devPtr());
    }
    nextParticleId = size;

    reorderedGridIndices.resize(size);

    uniqueGridNodeIndices.resize(size);
    nodeCount.resize(1);
    freeSurface.resize(1);

    numVoxels1D = 2*(uint)(std::floor(radius)) + (2<<refinementLevel);
    numVoxelsPerNode = numVoxels1D*numVoxels1D*numVoxels1D;
    frameDt = 1.0/24.0;
    prevDt = 0.0;
    accelerationBits.resizeAsync(1, stream);
    accelerationBits.zeroDeviceAsync(stream);   //no substep yet: nothing measured
}

void Particles::setForceFields(const std::vector<ForceField>& fields, const std::vector<std::shared_ptr<const SceneField>>& volumes){
    for(int field = 0; field < forces.numFields; ++field){     //any there were
        freeForceVolume(forces.fields[field].volume);
    }
    forces.numFields = 0;
    fieldAcceleration = 0.0;
    fieldDrag = 0.0;
    for(size_t index = 0; index < fields.size(); ++index){
        if(forces.numFields == MAX_FORCE_FIELDS){
            std::cerr<<"Particles: only the first "<<MAX_FORCE_FIELDS<<" force fields act\n";
            break;
        }
        ForceField field = fields[index];
        field.volume = {};
        float longest = 0.0f;
        if(field.kind == FORCE_VOLUME){
            if(index >= volumes.size() || !volumes[index]){
                continue;   //a volume with nothing in it does nothing
            }
            field.volume = uploadForceVolume(*volumes[index], stream);
            longest = volumes[index]->fastest;
        }
        addFieldLimits(field, longest, fieldAcceleration, fieldDrag);   //for the timestep
        forces.fields[forces.numFields++] = field;
    }
}

void Particles::setPartition(int rank, int numRanks, Transport* transport, PartitionContext* context, const std::vector<uint>& planes){
    this->rank = rank;
    this->numRanks = numRanks;
    this->transport = transport;
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
    cudaSortParticlesByGridNode(size, tempGridCell, tempSortedIndices, grid.sizeX*grid.sizeY*grid.sizeZ + (removing() ? 1 : 0), stream);  //removed particles' cell is one past the last
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
    for(CudaVec<float>* data : particleFloats()){
        reorder(data);
    }
    reorder(&particleIds);
    reorder(&particleBirths);
    gpuErrchk(cudaPeekAtLastError());
}

std::vector<CudaVec<float>*> Particles::particleFloats(){
    std::vector<CudaVec<float>*> arrays = {&vx, &vy, &vz};
    if(apic){
        for(CudaVec<float>& gradient : affine){
            arrays.push_back(&gradient);
        }
    }
    return arrays;
}

void Particles::setApic(bool on){
    apic = on;
    for(CudaVec<float>& gradient : affine){
        if(on){
            gradient.resizeAsync(size, stream);
            gradient.zeroDeviceAsync(stream);   //a particle starts out with none
        }
        else if(gradient.size() > 0){
            gradient.clear();
        }
    }
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
static constexpr int MIRRORED_PER_THREAD = 4;   //G2P's wall voxels per thread at most: a node can store NODE_THREADS times as many voxels (voxelVelsToParticles)

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
//wall's normal component, so that's 0 on the wall itself, and times that wall's along when tangential (WallStick::along, in the neighbours' order): 1,
//as is, so the wall doesn't drag, down to -1 for one that holds what's on it, which puts 0 on the wall for those too. origin is the block's first
//voxel in domain coordinates. across says whether the face is in or past a wall along dim itself: -1 the lower one, 1 the upper, 0 neither
__device__ inline int mirroredSlot(int3 voxel, int dim, int3 origin, int3 domainVoxels, int voxels1D, const float* along, float& sign, int& across){
    int local[3] = {voxel.x, voxel.y, voxel.z};
    int start[3] = {origin.x, origin.y, origin.z};
    int size[3] = {domainVoxels.x, domainVoxels.y, domainVoxels.z};
    across = 0;
    #pragma unroll
    for(int axis = 0; axis < 3; ++axis){
        int global = start[axis] + local[axis];
        int tangential = axis != dim;   //a dim face sits on a voxel boundary along dim, and mid-voxel along the other axes
        if(global < 0 || global >= size[axis]){
            local[axis] += (global < 0 ? -2*global : 2*(size[axis] - global)) - tangential;
            sign = tangential ? along[2*axis + (global >= size[axis])]*sign : -sign;
            across = tangential ? across : global < 0 ? -1 : 1;
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
    uint leaving[2] = {distributed ? particlesBelow : 0u, distributed ? particlesAbove : 0u};  //the first time, nobody sends: each keeps its own
    std::vector<uint> allLeaving(2*numRanks);
    transport->allGatherHost(leaving, allLeaving.data(), sizeof(leaving));
    uint fromBelow = rank > 0 ? allLeaving[2*(rank - 1) + 1] : 0;
    uint fromAbove = rank < numRanks - 1 ? allLeaving[2*(rank + 1)] : 0;
    uint kept = size - particlesBelow - particlesAbove;
    uint newSize = fromBelow + kept + fromAbove;
    std::vector<TransportSend> sends;
    std::vector<TransportReceive> receives;
    auto rebuild = [&](auto& mine){     //a per-particle array's new contents: from below, kept, from above. The same order of arrays on every partition
        using T = std::remove_reference_t<decltype(*mine.devPtr())>;
        T* fresh = nullptr;
        if(newSize > 0){
            gpuErrchk(cudaMallocAsync((void**)&fresh, sizeof(T)*newSize, stream));
        }
        if(kept > 0){
            cudaMemcpyAsync(fresh + fromBelow, mine.devPtr() + particlesBelow, sizeof(T)*kept, cudaMemcpyDeviceToDevice, stream);
        }
        if(rank > 0){
            sends.push_back({rank - 1, mine.devPtr(), sizeof(T)*leaving[0]});
            receives.push_back({rank - 1, fresh, sizeof(T)*fromBelow});
        }
        if(rank < numRanks - 1){
            sends.push_back({rank + 1, mine.devPtr() + size - leaving[1], sizeof(T)*leaving[1]});
            receives.push_back({rank + 1, fresh + fromBelow + kept, sizeof(T)*fromAbove});
        }
        return fresh;
    };
    double* newPx = rebuild(px);
    double* newPy = rebuild(py);
    double* newPz = rebuild(pz);
    std::vector<CudaVec<float>*> floats = particleFloats();
    std::vector<float*> newFloats;
    for(CudaVec<float>* data : floats){
        newFloats.push_back(rebuild(*data));
    }
    uint* newCells = rebuild(gridCell);
    uint* newIds = rebuild(particleIds);
    float* newBirths = rebuild(particleBirths);
    gpuErrchk(cudaPeekAtLastError());
    transport->exchange(sends, receives, stream);   //the old arrays are freed after it, in stream order
    px.adoptAsync(newPx, newSize, stream);
    py.adoptAsync(newPy, newSize, stream);
    pz.adoptAsync(newPz, newSize, stream);
    for(size_t array = 0; array < floats.size(); ++array){
        floats[array]->adoptAsync(newFloats[array], newSize, stream);
    }
    gridCell.adoptAsync(newCells, newSize, stream);
    particleIds.adoptAsync(newIds, newSize, stream);
    particleBirths.adoptAsync(newBirths, newSize, stream);
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
    uint boundary[2] = {lowerBoundaryParticleNodes, upperBoundaryParticleNodes};
    std::vector<uint> allBoundaries(2*numRanks);
    transport->allGatherHost(boundary, allBoundaries.data(), sizeof(boundary));
    uint fromBelow = rank > 0 ? allBoundaries[2*(rank - 1) + 1] : 0;
    uint fromAbove = rank < numRanks - 1 ? allBoundaries[2*(rank + 1)] : 0;
    numForeignParticleNodes = fromBelow + fromAbove;
    foreignParticleNodeMasks.resizeAsync(numForeignParticleNodes*maskWords, stream);
    std::vector<TransportSend> sends;
    std::vector<TransportReceive> receives;
    if(rank > 0){   //cells, then masks, each way
        sends.push_back({rank - 1, nodeCells.devPtr(), sizeof(uint)*lowerBoundaryParticleNodes});
        sends.push_back({rank - 1, particleNodeMasks.devPtr(), sizeof(uint)*lowerBoundaryParticleNodes*maskWords});
        receives.push_back({rank - 1, nodeCells.devPtr() + numParticleNodes, sizeof(uint)*fromBelow});
        receives.push_back({rank - 1, foreignParticleNodeMasks.devPtr(), sizeof(uint)*fromBelow*maskWords});
    }
    if(rank < numRanks - 1){
        uint first = numParticleNodes - upperBoundaryParticleNodes;
        sends.push_back({rank + 1, nodeCells.devPtr() + first, sizeof(uint)*upperBoundaryParticleNodes});
        sends.push_back({rank + 1, particleNodeMasks.devPtr() + first*maskWords, sizeof(uint)*upperBoundaryParticleNodes*maskWords});
        receives.push_back({rank + 1, nodeCells.devPtr() + numParticleNodes + fromBelow, sizeof(uint)*fromAbove});
        receives.push_back({rank + 1, foreignParticleNodeMasks.devPtr() + fromBelow*maskWords, sizeof(uint)*fromAbove*maskWords});
    }
    transport->exchange(sends, receives, stream);
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
    uint edges[8] = {edgeParticles[0].nodes, edgeParticles[0].count, edgeEmpty[0].nodes, edgeEmpty[0].count, edgeParticles[1].nodes, edgeParticles[1].count, edgeEmpty[1].nodes, edgeEmpty[1].count};
    std::vector<uint> allEdges(8*numRanks);
    transport->allGatherHost(edges, allEdges.data(), sizeof(edges));
    for(int side = 0; side < 2; ++side){
        int neighbour = side == 0 ? rank - 1 : rank + 1;
        if(neighbour < 0 || neighbour >= numRanks){
            continue;
        }
        const uint* theirs = allEdges.data() + 8*neighbour + 4*(1 - side);  //its plane facing this partition: nodes and voxels holding particles, then empty
        const VoxelRun* mine[2] = {&ghostParticles[side], &ghostEmpty[side]};
        for(int run = 0; run < 2; ++run){
            if(mine[run]->nodes != theirs[2*run] || mine[run]->count != theirs[2*run + 1]){
                std::cerr<<"partition "<<rank<<": its ghost copies of partition "<<neighbour<<"'s "<<(run == 0 ? "nodes holding particles" : "empty nodes")<<" are "
                         <<mine[run]->nodes<<" nodes with "<<mine[run]->count<<" voxels, but the owner has "<<theirs[2*run]<<" with "<<theirs[2*run + 1]<<"\n";
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
    if(twoPhase.on){    //each face's density, and with air particles the liquid's weights that go into it (twophase.cu)
        for(CudaVec<float>& lightness : faceLightness){
            lightness.resizeAsync(numUsedVoxels, stream);
        }
        if(twoPhase.on){
            for(CudaVec<float>& liquid : liquidWeights){
                liquid.resizeAsync(numUsedVoxels, stream);
            }
        }
    }
    if(twoPhase.escaping() || whitewater.on){   //what escaping particles and the whitewater are judged by
        liquidShare.resizeAsync(numUsedVoxels, stream);
    }
    if(lettingGo()){    //which voxels boundaries have let go of, a solve at a time (boundaries.cu)
        letGo.resizeAsync(numUsedVoxels, stream);
        heldCodes.resizeAsync(numUsedVoxels, stream);
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
    markObstacleSolids();   //the voxels inside obstacles stop being unknowns, and their neighbours see walls
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

//APIC's per-particle velocity gradients, for the kernels: component c's along axis a at c[3*c + a], in 1/s
struct AffineVelocities{
    float* c[9];
};

static AffineVelocities affineVelocities(CudaVec<float> (&affine)[9]){
    AffineVelocities out;
    for(int i = 0; i < 9; ++i){
        out.c[i] = affine[i].devPtr();
    }
    return out;
}

//P2G: a block per node and a thread per particle. Each particle adds its weight and its momentum on the 27 faces of each component around it into
//the node's voxel block in shared memory, and 1 to the count of the voxel holding it; then each voxel the node stores adds the block's sums into its
//owner. It sums in fixed point, as sm_86 has native shared integer atomics but not float ones, and integers add up the same in any order, so the
//result doesn't depend on which node gets there first.
//A particle's share of a face's weight is a whole number of units (weightScale to a weight of 1), and its share of the face's momentum is that many
//times its velocity there, in whole units too (velocityScale to 1 m/s: 2^MOMENTUM_DIGIT to the fastest there is). So a face's momentum over its
//weight is the mean of its particles' velocities, each by its share, exactly, however little of any of them reaches it; and a share too small to
//be a unit adds to neither. (Each was rounded on its own once, the momentum from the weight unrounded. A face at the edge of a particle's reach
//then held a few units of each, and their ratio was anything from nothing to the fastest velocity there is: round a ball moving all at one
//velocity, faces were off by a third of it, the pressure solve took that for divergence, and the ball lost a thousandth of its speed a substep.)
//The products take 64 bits, and the shared atomics that are native are 32: so each sum is two words, the product's low half added to the low
//one, and its high half to the high one with the carry, which the add that wrapped the low word knows it made, by what it found there. Per slot,
//3 such sums, and 4 ints, the weights and the count: 20 KB for an 8^3 block. The owners' sums are 32-bit, in the float arrays' storage: the
//momentum's as two digits, the low MOMENTUM_DIGIT bits of each block's sum (low) and the rest of it (ux, uy, uz). No more than 8 blocks reach a
//voxel, so neither digit can run over. normalizeVoxelVelocities turns them into floats.
//With APIC, each face takes the particle's velocity carried out to it along its gradient, v + c.(x_face - x). With two PHASES (TwoPhase,
//particles.hu), a particle weighs as much as its fluid, air's being airMass of the liquid's: the momentum and the weights are then each face's
//momentum and mass, so the velocity they give is the two fluids' together, and three more sums a slot keep the liquid's weights alone: what's left
//of a face's mass is its air's, so how much of the face is liquid is exact, and so is whether either fluid reaches it at all (26 KB)
template<bool APIC, bool PHASES>
__global__ void scatterParticleVelsToVoxels(uint numParticleNodes, uint numParticles, uint numVoxels1D, const uint* gridNodeIndicesToFirstParticleIndex, const uint* gridPosition,
                                            const double* px, const double* py, const double* pz, const float* vx, const float* vy, const float* vz,
                                            AffineVelocities affine, float voxelSize,
                                            const uint* numVoxelsEachNode, const uint* voxelIDs, const uint* voxelOwners,
                                            int* ux, int* uy, int* uz, unsigned int* lowX, unsigned int* lowY, unsigned int* lowZ, int* weightsX, int* weightsY, int* weightsZ, int* particleCounts,
                                            float velocityScale, float weightScale, double radius, Grid grid, uint refinementLevel,
                                            const uint* ids, float airMass, int* liquidX, int* liquidY, int* liquidZ){
    extern __shared__ long long blockSums[];    //per slot: x, y, z momentum; then, as ints, x, y, z weight and the particle count, and with PHASES the liquid's x, y, z weight
    const int SUMS = PHASES ? 7 : 4;
    int voxels1D = numVoxels1D;
    int voxels3D = voxels1D*voxels1D*voxels1D;
    unsigned int* momentum = (unsigned int*)blockSums;      //each sum's low word, then its high one
    int* fixedSums = (int*)(blockSums + 3*voxels3D);
    int apronCells = floor(radius);
    uint firstParticle = gridNodeIndicesToFirstParticleIndex[blockIdx.x];
    uint lastParticle = blockIdx.x == numParticleNodes - 1 ? numParticles : gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1];
    for(int i = threadIdx.x; i < (6 + SUMS)*voxels3D; i += blockDim.x){
        ((int*)blockSums)[i] = 0;
    }
    __syncthreads();
    const float* velocities[3] = {vx, vy, vz};
    for(uint index = firstParticle + threadIdx.x; index < lastParticle; index += blockDim.x){    //consecutive threads, consecutive particles
        float3 pos = positionInNodeBlock(index, gridPosition, px, py, pz, grid, refinementLevel, apronCells);
        FaceStencil stencil(pos);
        float mass = 1.0f;
        bool liquid = true;
        if constexpr(PHASES){
            if((ids[index] & (AIR_PARTICLE | ESCAPED_PARTICLE)) == ESCAPED_PARTICLE){
                continue;   //a droplet: off the grid. (A bubble stays, with its weight and its room)
            }
            liquid = !(ids[index] & AIR_PARTICLE);
            mass = liquid ? 1.0f : airMass;
        }
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            const Spline3& x = stencil.axis(dim, 0);
            const Spline3& y = stencil.axis(dim, 1);
            const Spline3& z = stencil.axis(dim, 2);
            float velocity = velocities[dim][index];
            int units = __float2int_rn(velocity*velocityScale);     //the velocity it gives a face, in whole units: with APIC, each face's own
            float slope[3], start[3];   //APIC: the component's change per voxel along each axis, and the stencil's first face from the particle, in voxels
            if constexpr(APIC){
                float q[3] = {pos.x, pos.y, pos.z};
                #pragma unroll
                for(int a = 0; a < 3; ++a){
                    slope[a] = voxelSize*affine.c[3*dim + a][index];
                    start[a] = stencil.axis(dim, a).base + (a == dim ? 0.0f : 0.5f) - q[a];
                }
            }
            #pragma unroll
            for(int k = 0; k < 3; ++k){
                #pragma unroll
                for(int j = 0; j < 3; ++j){
                    int rowStart = x.base + (y.base + j)*voxels1D + (z.base + k)*voxels1D*voxels1D;
                    float yz = y.w[j]*z.w[k];
                    float rowVelocity = velocity;
                    if constexpr(APIC){
                        rowVelocity += slope[1]*(start[1] + j) + slope[2]*(start[2] + k);
                    }
                    #pragma unroll
                    for(int i = 0; i < 3; ++i){
                        int share = __float2int_rn(x.w[i]*yz*mass*weightScale);
                        if(share != 0){
                            if constexpr(APIC){
                                units = __float2int_rn((rowVelocity + slope[0]*(start[0] + i))*velocityScale);
                            }
                            atomicAdd(fixedSums + dim*voxels3D + rowStart + i, share);
                            long long product = (long long)share*units;
                            unsigned int low = (unsigned int)product;
                            unsigned int* words = momentum + 2*(dim*voxels3D + rowStart + i);
                            int high = (int)(product >> 32) + (atomicAdd(words, low) + low < low);     //and the carry, if this wrapped the low word
                            if(high != 0){
                                atomicAdd((int*)words + 1, high);
                            }
                            if constexpr(PHASES){
                                if(liquid){     //its mass is 1, so its share of the liquid's weight is its share of the mass
                                    atomicAdd(fixedSums + (4 + dim)*voxels3D + rowStart + i, share);
                                }
                            }
                        }
                    }
                }
            }
        }
        atomicAdd(fixedSums + 3*voxels3D + (int)pos.x + (int)pos.y*voxels1D + (int)pos.z*voxels1D*voxels1D, 1);
    }
    __syncthreads();
    int* accumulators[7] = {weightsX, weightsY, weightsZ, particleCounts, liquidX, liquidY, liquidZ};
    int* high[3] = {ux, uy, uz};
    unsigned int* low[3] = {lowX, lowY, lowZ};
    for(uint i = (blockIdx.x == 0 ? 0 : numVoxelsEachNode[blockIdx.x - 1]) + threadIdx.x; i < numVoxelsEachNode[blockIdx.x]; i += blockDim.x){    //coalesced over the node's voxels
        int slot = voxelIDs[i];
        uint owner = voxelOwners[i];    //itself for interior voxels, so these atomics are mostly consecutive
        #pragma unroll
        for(int sum = 0; sum < SUMS; ++sum){
            int value = fixedSums[sum*voxels3D + slot];
            if(value != 0){
                atomicAdd(accumulators[sum] + owner, value);
            }
        }
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            long long value = blockSums[dim*voxels3D + slot];
            if(value != 0){     //as two digits: value is the high one times 2^MOMENTUM_DIGIT, plus the low one, which is never negative
                atomicAdd(low[dim] + owner, (unsigned int)(value & ((1LL << MOMENTUM_DIGIT) - 1)));
                atomicAdd(high[dim] + owner, (int)(value >> MOMENTUM_DIGIT));
            }
        }
    }
}

#include "algorithms/reductionKernels.hu"
#include <cmath>

//With APIC, P2G sums each particle's velocity carried out along its gradient to faces up to 1.5 voxels away on each axis. Per particle, the most that
//comes to (largest[1]), and its largest velocity component (largest[0]), as float bits, which order like the floats for values >= 0, so atomicMax
//takes each exactly, whatever order the blocks land in
__global__ void largestApicVelocities(uint numParticles, const float* vx, const float* vy, const float* vz, AffineVelocities affine, float reach, unsigned int* largest){
    __shared__ unsigned int warpLargest[2][BLOCKSIZE / 32];
    unsigned int bits[2] = {0, 0};
    const float* velocities[3] = {vx, vy, vz};
    for(uint i = threadIdx.x + blockIdx.x*blockDim.x; i < numParticles; i += blockDim.x*gridDim.x){
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            float speed = fabsf(velocities[dim][i]);
            float carried = speed + reach*(fabsf(affine.c[3*dim][i]) + fabsf(affine.c[3*dim + 1][i]) + fabsf(affine.c[3*dim + 2][i]));
            bits[0] = max(bits[0], __float_as_uint(speed));
            bits[1] = max(bits[1], __float_as_uint(carried));
        }
    }
    #pragma unroll
    for(int which = 0; which < 2; ++which){
        for(int offset = 16; offset > 0; offset /= 2){
            bits[which] = max(bits[which], __shfl_down_sync(0xffffffffu, bits[which], offset));
        }
        if(threadIdx.x % 32 == 0){
            warpLargest[which][threadIdx.x / 32] = bits[which];
        }
    }
    __syncthreads();
    if(threadIdx.x < 32){
        #pragma unroll
        for(int which = 0; which < 2; ++which){
            bits[which] = threadIdx.x < blockDim.x / 32 ? warpLargest[which][threadIdx.x] : 0;
            for(int offset = 16; offset > 0; offset /= 2){
                bits[which] = max(bits[which], __shfl_down_sync(0xffffffffu, bits[which], offset));
            }
            if(threadIdx.x == 0){
                atomicMax(largest + which, bits[which]);
            }
        }
    }
}

//every substep: the fastest particle moves at most cfl voxels (P2G finds it), and so does the fastest point of a moving obstacle's surface, so a wall
//can't sweep past a voxel's worth of fluid in one substep. Every partition places the obstacles alike, so they agree on it.
//A particle moves with the velocity the substep leaves it with, so with what it gains during the substep too: at least gravity's largest component
//and every field's strongest pull (setForceFields), and as much as the substep before changed any particle's velocity by, per second
//(maxAcceleration: G2P's update, which is what the pressure made of the forces then, and an escaped particle's flight), the best that's known of what
//this one will: the toe of a collapsing dam, a splash against a wall, air a wave drives, a bubble let go, all go well past g. From v, gaining a a
//second, it goes (v + a dt) dt, which is cfl voxels at
//dt = 2 cfl h / (v + sqrt(v^2 + 4 a cfl h)): cfl h / v where the forces are nothing to its speed, and sqrt(cfl h / a) from rest. Without that, what
//starts at rest falls g dt^2 in its first substep however long the frame lets that be, and a strong field moves the fluid as far as it likes. The
//substep an impact begins in has only the substep before to go by, and can still go further
double Particles::getCourantDt(){
    double voxelSize = (grid.cellSize / (numVoxels1D - 2*std::floor(radius)));
    double fastest = std::max(maxVelocity, (double)obstacles.fastest());
    double strongest = std::max({std::abs(forces.gravity.x), std::abs(forces.gravity.y), std::abs(forces.gravity.z)}) + fieldAcceleration + fieldDrag*maxVelocity;
    strongest = std::max(strongest, maxAcceleration);
    if(verbose()){
        std::cout<<"maxVel: "<<maxVelocity<<" fastest obstacle: "<<obstacles.fastest()<<" strongest force: "<<strongest<<" (the last substep's acceleration "<<maxAcceleration<<") voxelSize: "<<voxelSize<<"\n";
    }
    return cfl * (2.0*voxelSize / (fastest + std::sqrt(fastest*fastest + 4.0*strongest*cfl*voxelSize)) + 0.0001);
}

void Particles::particleVelToVoxels(){
    //the voxels' sums: the momentum's in two arrays a component, its high digits where the velocities will be and its low ones where FLIP's old
    //velocities go once there are any (scatterParticleVelsToVoxels)
    for(CudaVec<float>* accumulator : {&voxelsUx, &voxelsUy, &voxelsUz, &voxelsUxOld, &voxelsUyOld, &voxelsUzOld, &voxelWeightsX, &voxelWeightsY, &voxelWeightsZ, &particleCounts}){
        accumulator->zeroDeviceAsync(stream);
    }
    float voxelSize = grid.cellSize / (2<<refinementLevel);
    double transferred;     //the largest velocity P2G sums on a face: the largest component, or with APIC, the most a particle carries out to one
    if(apic){
        unsigned int* largest;
        unsigned int bits[2];
        gpuErrchk(cudaMallocAsync((void**)&largest, 2*sizeof(unsigned int), stream));
        gpuErrchk(cudaMemsetAsync(largest, 0, 2*sizeof(unsigned int), stream));
        largestApicVelocities<<<std::min(size / BLOCKSIZE + 1, 1024u), BLOCKSIZE, 0, stream>>>(size, vx.devPtr(), vy.devPtr(), vz.devPtr(), affineVelocities(affine), 1.5f*voxelSize, largest);
        gpuErrchk(cudaMemcpyAsync(bits, largest, 2*sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));   //to pageable memory: back once it's landed
        gpuErrchk(cudaFreeAsync(largest, stream));
        float values[2];
        memcpy(values, bits, sizeof(values));
        maxVelocity = context->maxOverPartitions(values[0]);
        transferred = context->maxOverPartitions(values[1]);
    }
    else{
        maxVelocity = largestMagnitude({&vx, &vy, &vz}, stream);   //the fastest component, which also sets the CFL timestep
        maxVelocity = context->maxOverPartitions(maxVelocity);     //every partition's fixed point and timestep have to agree
        transferred = maxVelocity;
    }
    unsigned int accelerationFound;     //and the most the last substep changed a particle's velocity by, per second (voxelVelsToParticles), for the timestep too
    gpuErrchk(cudaMemcpyAsync(&accelerationFound, accelerationBits.devPtr(), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));    //to pageable memory: back once it's landed
    float accelerated;
    memcpy(&accelerated, &accelerationFound, sizeof(accelerated));
    maxAcceleration = context->maxOverPartitions(accelerated);
    //P2G sums in fixed point: a face's weight sum is about the particles per voxel, so budget 128 (15x rest) and keep every sum under 2^30. And
    //velocities in units of the fastest there is over 2^MOMENTUM_DIGIT: a face's momentum, the weights' units times those, is then under 2^58
    float weightScale = (1 << 30) / 128.0f;
    float velocityScale = (float)((1 << MOMENTUM_DIGIT) / fmax(transferred, 1e-6));
    bool phases = twoPhase.on;     //air and liquid: each particle weighs as its fluid does, and the liquid's weights are kept beside the masses
    if(phases){
        liquidWeightUnit = 1.0f / weightScale;
        for(CudaVec<float>& liquid : liquidWeights){
            liquid.zeroDeviceAsync(stream);
        }
    }
    if(numParticleNodes > 0){
        auto scatter = phases ? (apic ? scatterParticleVelsToVoxels<true, true> : scatterParticleVelsToVoxels<false, true>)
                              : (apic ? scatterParticleVelsToVoxels<true, false> : scatterParticleVelsToVoxels<false, false>);
        scatter<<<numParticleNodes, NODE_THREADS, (3*sizeof(long long) + (phases ? 7 : 4)*sizeof(int))*numVoxelsPerNode, stream>>>(numParticleNodes, size, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr(),
            affineVelocities(affine), voxelSize, nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelOwners.devPtr(), (int*)voxelsUx.devPtr(), (int*)voxelsUy.devPtr(), (int*)voxelsUz.devPtr(),
            (unsigned int*)voxelsUxOld.devPtr(), (unsigned int*)voxelsUyOld.devPtr(), (unsigned int*)voxelsUzOld.devPtr(),
            (int*)voxelWeightsX.devPtr(), (int*)voxelWeightsY.devPtr(), (int*)voxelWeightsZ.devPtr(), (int*)particleCounts.devPtr(), velocityScale, weightScale, radius, grid, refinementLevel,
            particleIds.devPtr(), 1.0f / twoPhase.densityRatio, (int*)liquidWeights[0].devPtr(), (int*)liquidWeights[1].devPtr(), (int*)liquidWeights[2].devPtr());
    }
    //particles near this partition's edge reach into ghost voxels: their owners add those sums in, which in fixed point come out the same in any order
    for(CudaVec<float>* accumulator : {&voxelsUx, &voxelsUy, &voxelsUz, &voxelsUxOld, &voxelsUyOld, &voxelsUzOld, &voxelWeightsX, &voxelWeightsY, &voxelWeightsZ, &particleCounts}){
        context->reduceGhosts((int*)accumulator->devPtr(), stream);
    }
    if(phases){
        for(CudaVec<float>& liquid : liquidWeights){
            context->reduceGhosts((int*)liquid.devPtr(), stream);
            context->fillGhosts(liquid.devPtr(), stream);   //as bits: they stay fixed point
        }
    }
    cudaNormalizeVoxelVelocities(solids, voxelWeightsX, voxelWeightsY, voxelWeightsZ, voxelsUx, voxelsUy, voxelsUz, voxelsUxOld, voxelsUyOld, voxelsUzOld, particleCounts, velocityScale, weightScale, stream);
    for(CudaVec<float>* field : {&voxelsUx, &voxelsUy, &voxelsUz, &voxelWeightsX, &voxelWeightsY, &voxelWeightsZ, &particleCounts}){   //and the ghosts take the totals
        context->fillGhosts(field->devPtr(), stream);
    }
    creditObstacleVolume();     //voxels obstacles partly cover hold fewer particles at rest: the density correction mustn't read them as thin
    cudaExtrapolateUnreachedFaces(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, voxelWeightsX, voxelWeightsY, voxelWeightsZ, voxelsUx, voxelsUy, voxelsUz, *context, stream);
    cudaFindFootprintDepth(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, footprintDepth, freeSurface, *context, stream);
    if(needsLevelSet()){    //the liquid's surface, where viscosity and surface tension act, and the sharp free surface's liquid (freesurface.cu)
        buildLevelSet();
    }
    retireDryUnknowns();    //sharp, the solve's liquid is the unknowns whose centres are in it
    findFaceDensities();    //two phases: how light each face is, for the pressure equations and the velocity update
    obstacleGhostVelocities(voxelsUx, voxelsUy, voxelsUz);      //the faces in and beside obstacles, so FLIP's change over the solve is measured from the same kind of value there
    for(auto [velocity, before] : {std::pair{&voxelsUx, &voxelsUxOld}, {&voxelsUy, &voxelsUyOld}, {&voxelsUz, &voxelsUzOld}}){    //for FLIP's velocity change over the solve
        cudaMemcpyAsync(before->devPtr(), velocity->devPtr(), sizeof(float)*velocity->size(), cudaMemcpyDeviceToDevice, stream);
    }
    //and FLIP's old faces that obstacles cut as what crosses them all told, as the new ones will be, with the faces inside them from those. The solve's own
    //take the open part alone, as it needs
    mixObstacleFaces(voxelsUxOld, voxelsUyOld, voxelsUzOld);
    obstacleGhostVelocities(voxelsUxOld, voxelsUyOld, voxelsUzOld);
}

void Particles::pressureSolve(){
    double density = 0.014;
    double tolerance = 0.001;   //largest residual, relative to the largest divergence
    //a V-cycle an iteration, the multigrid's solve takes 6 to 20 of them; the other solvers can need thousands on a large grid
    uint maxIterations = pressureSolver == PressureSolver::multigrid ? 256 : 16384;
    double terminatingResidual;
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    uint hasFreeSurface;
    findSealedPockets();    //queued ahead of the wait for the free surface, which brings back whether any fluid can't see air
    cudaMemcpyAsync(&hasFreeSurface, freeSurface.devPtr(), sizeof(uint), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    hasFreeSurface = context->anyOverPartitions(hasFreeSurface != 0);
    settleSealedPockets();
    //divergence per unit of relative density error. Without air the fluid can't change volume, and the solve has nowhere to take a net divergence, so it's off.
    //With air particles it stays on though they fill a closed box: both fluids' particles bunch up, and the sealed pockets' balance takes the net out
    double correctionRate = (hasFreeSurface || twoPhase.on) && densityCorrectionTime > 0.0 ? voxelSize / densityCorrectionTime : 0.0;
    //@TODO: need to use courant number for dt from max voxel u and voxel dimensions
    substepStart = elapsedTime;
    dt = getCourantDt();
    if(surfaceTension > 0.0){   //explicit surface tension holds only up to the capillary limit
        dt = std::min(dt, capillaryDt());
    }
    if(viscosity > 0.0 && viscousCfl > 0.0){    //and thick liquid's threads only coil with viscosity spreading a few voxels a substep (setViscousCfl)
        dt = std::min(dt, viscousDt());
    }
    if(verbose()){
        std::cout<<"initial dt: "<<dt<<std::endl;
    }
    if(frameDt - elapsedTimeThisFrame < dt){
        dt = frameDt - elapsedTimeThisFrame;
    }
    findWhitewater();   //what the surface makes this substep, now its length is known: read from the grid as P2G left it, and nothing of the fluid's changes by it
    applyForces(forces, dt, elapsedTime, forceVoxels(), stream);
    floatInAir(dt);     //an air band: less the still air's weight
    if(surfaceTension > 0.0){
        applySurfaceTension();
    }
    cudaCalcDivU(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, voxelsUx, voxelsUy, voxelsUz, particleCounts, footprintDepth, restParticlesPerVoxel, correctionRate, divU, stream);
    addObstacleFlux();      //what obstacles' surfaces make of the flow through the faces they cut or close
    addSurfaceDivergence(correctionRate);   //sharp: the surface tension's pressure, and spreading what's packed near the surface
    balanceSealedPockets(); //and fluid no air reaches can't change its volume
    auto assemble = [&](){
        cudaGetA(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag, dt/(density*voxelSize*voxelSize), stream);
        weighCutCells(dt/(density*voxelSize*voxelSize));      //the faces obstacles cut weigh as much as they're open
        weighSurfaceFaces(dt/(density*voxelSize*voxelSize));  //sharp, the liquid's faces to air as near as the surface is
        weighDensityFaces(dt/(density*voxelSize*voxelSize));  //two phases, every face as light as what's on it
    };
    assemble();
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
                                                      layout, dotProductSums, numOwnVoxels, *context, stream);
            default:
                return cudaGSiteration(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag, divU, p, residuals, tolerance, maxIterations,
                                       numOwnVoxels, *context, stream);
        }
    };
    //A solve either comes to the tolerance or there's no pressure to move the fluid by: the solver stalled, or ran out of iterations. Nothing about the
    //substep would change that (a shorter dt only scales every equation alike), so it isn't tried again: the substep stops here, with nothing moved and
    //the time where it was, and solveFailure says what happened (solveFrame stops on it, and flip2 bake ends there with the frames before it kept)
    auto solved = [&](){
        terminatingResidual = solve();
        gpuErrchk(cudaPeekAtLastError());
        if(terminatingResidual < tolerance){
            return true;
        }
        std::ostringstream message;
        message<<"the pressure solve ended at a residual of "<<terminatingResidual<<", over its tolerance of "<<tolerance<<", in substep "<<substepIndex + 1
               <<" (at "<<elapsedTime<<" s)";
        solveFailure = message.str();
        return false;
    };
    //With boundaries that let go of the liquid (boundaries.cu), a solve is done once none of their voxels is left pulled on, or let go and squeezed.
    //The voxels the solve leaves pulled on, that air can reach, are let go; sweeps of relaxation hold again the ones that letting go of the others
    //leaves pressed on, and the rest are solved for. If that leaves any of those let go squeezed they're held, and it's swept and solved again,
    //until a solve leaves none. Without such boundaries, or where none pulls on anything, it's the one solve
    auto settled = [&](){
        holdEverything();
        if(!solved()){
            return false;
        }
        if(!lettingGo() || !letGoOfSuction()){
            return true;
        }
        for(bool more = true; more;){
            relaxLetGo();
            followLetGo();
            if(!solved()){
                takeHoldAgain();
                return false;
            }
            more = holdSqueezed();
            if(more){   //the equations whole again, for the sweeps
                unknownsAgain();
                assemble();
            }
        }
        return true;
    };
    //Viscosity goes between two solves: the first gives the flow the forces drive once the pressure has answered them, the viscous step acts on that,
    //and the solve below takes out what divergence it leaves. Straight after the forces it would see gravity but not the pressure gradient that turns
    //gravity into a puddle spreading, which would then go on unresisted, as if the floor were slippery
    if(viscosity > 0.0){
        if(!settled()){
            return;
        }
        cudaVelocityUpdate(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, p, voxelsUx, voxelsUy, voxelsUz, dt/(density*voxelSize*voxelSize), stream);
        if(takeHoldAgain()){    //the equations are whole again for the solve below
            assemble();
        }
        correctSurfaceFaces();
        lightenFaceUpdates(dt/(density*voxelSize*voxelSize));   //two phases, as updateVoxelVelocities has it: the viscous step acts on the flow the first solve leaves
        p.zeroDeviceAsync(stream);
        applyViscosity();
        cudaCalcDivU(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, voxelsUx, voxelsUy, voxelsUz, particleCounts, footprintDepth, restParticlesPerVoxel, correctionRate, divU, stream);
        addObstacleFlux();
        addSurfaceDivergence(correctionRate);
        balanceSealedPockets();
    }
    if(!settled()){
        return;
    }
    elapsedTime += dt;
    elapsedTimeThisFrame += dt;
    prevDt = dt;
    if(verbose()){
        std::cout<<"dt: "<<dt<<" ElapsedTime: "<<elapsedTime<<" elapsedTimeThisFrame: "<<elapsedTimeThisFrame<<"\nTerminating Residual: "<<terminatingResidual<<"\nTolerance: "<<tolerance<<"\n";
    }
    gpuErrchk(cudaPeekAtLastError());
}

//where the forces act: every stored voxel, own and ghost, so the ghosts' faces get what their owners' do
ForceVoxels Particles::forceVoxels(){
    return {voxelsUy.size(), numStoredNodes, nodeIndexUsedVoxels.devPtr(), nodeCells.devPtr(), voxelIDsUsed.devPtr(), solids.devPtr(), footprintDepth.devPtr(),
            {voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr()}, {voxelsUxOld.devPtr(), voxelsUyOld.devPtr(), voxelsUzOld.devPtr()}, grid, refinementLevel, (int)std::floor(radius)};
}

void Particles::updateVoxelVelocities(){
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    cudaVelocityUpdate(solveCodes, neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, p, voxelsUx, voxelsUy, voxelsUz, dt/(0.014*voxelSize*voxelSize), stream);
    gpuErrchk(cudaPeekAtLastError());
    bool released = takeHoldAgain();    //the voxels boundaries let go of are the liquid's again
    correctSurfaceFaces();      //sharp, the liquid's faces to air with the ghost pressure past them
    lightenFaceUpdates(dt/(0.014*voxelSize*voxelSize));     //two phases, each face pushed as its own density lets the pressure push it
    extendLiquidVelocities();   //and the faces around the liquid that particles read carry its velocity on
    pinEmitterVelocities();     //an emitter's fluid leaves at its velocity, whatever the solve made of it
    mixObstacleFaces(voxelsUx, voxelsUy, voxelsUz);         //the faces obstacles cut carry what crosses them all told, not the open part's alone
    obstacleGhostVelocities(voxelsUx, voxelsUy, voxelsUz);  //the faces between fluid and obstacles take the obstacles' velocity across them; the faces inside continue the fluid's
    if(released && obstacles.lettingGo()){  //but the liquid's, where an obstacle let go of it; and in FLIP's old velocities, set before the solve said where
        carryLetGoFaces(voxelsUx, voxelsUy, voxelsUz);
        carryLetGoFaces(voxelsUxOld, voxelsUyOld, voxelsUzOld);
        for(CudaVec<float>* velocity : {&voxelsUxOld, &voxelsUyOld, &voxelsUzOld}){
            context->fillGhosts(velocity->devPtr(), stream);
        }
    }
    for(CudaVec<float>* velocity : {&voxelsUx, &voxelsUy, &voxelsUz}){   //G2P and advection read ghost faces
        context->fillGhosts(velocity->devPtr(), stream);
    }
}

//the largest of a non-negative float's bits over the block, into largest with one atomicMax: non-negative floats order like their bits, so the largest
//is exact whatever order the blocks land in (as largestMagnitudeBits, cudaVec.cu). Every thread of the block calls it, with warpLargest a slot a warp
__device__ inline void largestInBlock(unsigned int bits, unsigned int* warpLargest, unsigned int* largest){
    for(int offset = 16; offset > 0; offset /= 2){
        bits = max(bits, __shfl_down_sync(0xffffffffu, bits, offset));
    }
    if(threadIdx.x % 32 == 0){
        warpLargest[threadIdx.x / 32] = bits;
    }
    __syncthreads();
    if(threadIdx.x < 32){
        bits = threadIdx.x < blockDim.x / 32 ? warpLargest[threadIdx.x] : 0;
        for(int offset = 16; offset > 0; offset /= 2){
            bits = max(bits, __shfl_down_sync(0xffffffffu, bits, offset));
        }
        if(threadIdx.x == 0){
            atomicMax(largest, bits);
        }
    }
}

//G2P: a block per node and a thread per particle. The block copies the new and old face velocities of every voxel the node stores into shared memory,
//from their owners, and turns the faces inside walls into their mirror images (the wall's normal component reversed, so it's 0 on the wall, and the
//tangential ones as they are, so walls don't drag). Each particle then reads its 27 faces per component from there; the weights sum to 1, so no
//normalizing. PIC takes the new grid velocity, FLIP adds its change to the particle's own, flipRatio blends them, and the particle moves with the
//grid's. 6 floats per slot: 12 KB for an 8^3 block. With APIC, the particle also takes the new velocity's gradient around it: c = B D^-1, where
//B = sum of w u (x_face - x) and D = dx^2/4 on every axis for quadratic B-splines (Jiang et al. 2015)
//With two PHASES (TwoPhase, particles.hu) the air's particles blend at their own ratio, and with escaping particles (share not nullptr) each is judged
//here first, by the liquid's share where it is (shareAround, from the voxels' shares, which the block holds with the faces): a particle on the wrong
//side of the interface escapes, or one back on the right side is the grid's again. An escaped one, as it's judged, keeps the velocity it has: the
//grid's change isn't its own, and its flight through the substep is flyEscaped's. And the walls' hold is each fluid's own (stick the liquid's,
//airStick the air's: a viscous liquid holds to them and air, with no viscosity here, only as far as their friction holds it): the faces inside the
//walls mirror what's in the domain as it is, and each particle weighs the ones past a wall, along an axis they're tangential to, by what its own
//fluid's mirror would have carried there. That's what that mirror would have made of them: its factor is one per axis, as a face's weight is. A
//template, so that without them the kernel is the one it always was.
//Where a wall has let go of a voxel this substep (letGo, boundaries.cu; nullptr with no such walls) the wall isn't holding the liquid there, and
//its faces don't either: the wall's own face and the ones past it carry on the liquid's velocity across it, the last face inside's, as the faces
//past the liquid's surface do, in place of the image reversed. Reversed, a particle a quarter of a voxel from a ceiling that had let go took a
//quarter of what the grid gave the liquid under it, and the liquid's top layer left a third of a second after the rest
template<bool APIC, bool PHASES>
__global__ void gatherVoxelVelsToParticles(uint numParticleNodes, uint numParticles, uint numVoxels1D, const uint* gridNodeIndicesToFirstParticleIndex, const uint* gridPosition,
                                            double* px, double* py, double* pz, float* vx, float* vy, float* vz, AffineVelocities affine, float voxelSize,
                                            const uint* numVoxelsEachNode, const uint* voxelIDs, const uint* voxelOwners, const char* solids,
                                            const float* ux, const float* uy, const float* uz, const float* oldUx, const float* oldUy, const float* oldUz,
                                            float flipRatio, WallStick stick, WallStick airStick, double radius, Grid grid, uint refinementLevel,
                                            uint* ids, float airFlipRatio, const float* share,      //two phases; else ids is null
                                            const float* births, float bornSince, float perSecond, unsigned int* largestChange,     //the most the velocity of any particle
                                            //from before this substep changes by, per second, as a float's bits: one born in it (an emitter's, the band's air)
                                            //starts at its own velocity and takes the grid's, which isn't an acceleration
                                            const char* letGo){
    extern __shared__ float blockVelocities[];  //per slot: new x, y, z, then old x, y, z; with PHASES, then the liquid's share of the voxel; then, a byte a slot, whether a wall let go of it
    __shared__ unsigned int warpLargest[NODE_THREADS / 32];
    const int PLANES = PHASES ? 7 : 6;
    int voxels1D = numVoxels1D;
    int voxels3D = voxels1D*voxels1D*voxels1D;
    char* gone = (char*)(blockVelocities + PLANES*voxels3D);
    int apronCells = floor(radius);
    uint firstParticle = gridNodeIndicesToFirstParticleIndex[blockIdx.x];
    uint lastParticle = blockIdx.x == numParticleNodes - 1 ? numParticles : gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1];
    uint startVoxel = blockIdx.x == 0 ? 0 : numVoxelsEachNode[blockIdx.x - 1];
    uint endVoxel = numVoxelsEachNode[blockIdx.x];
    const float* velocities[6] = {ux, uy, uz, oldUx, oldUy, oldUz};
    for(int i = threadIdx.x; i < PLANES*voxels3D; i += blockDim.x){   //every face a particle reaches is stored, but never leave garbage to read; a share
        blockVelocities[i] = PHASES && i >= 6*voxels3D ? nanf("") : 0.0f;   //no voxel holds is unknown (shareAround), not air
    }
    if(letGo != nullptr){
        for(int i = threadIdx.x; i < voxels3D; i += blockDim.x){
            gone[i] = 0;
        }
    }
    __syncthreads();
    for(uint i = startVoxel + threadIdx.x; i < endVoxel; i += blockDim.x){  //coalesced over the node's voxels
        int slot = voxelIDs[i];
        uint owner = voxelOwners[i];
        #pragma unroll
        for(int v = 0; v < 6; ++v){
            blockVelocities[v*voxels3D + slot] = velocities[v][owner];
        }
        if constexpr(PHASES){
            if(share != nullptr){
                blockVelocities[6*voxels3D + slot] = share[owner];
            }
        }
        if(letGo != nullptr){
            gone[slot] = letGo[owner];
        }
    }
    __syncthreads();
    uint cell = gridPosition[firstParticle];
    int interiorWidth = voxels1D - 2*apronCells;
    int3 origin = make_int3((int)(cell % grid.sizeX)*interiorWidth - apronCells, (int)(cell / grid.sizeX % grid.sizeY)*interiorWidth - apronCells, (int)(cell / (grid.sizeX*grid.sizeY))*interiorWidth - apronCells);
    int3 domainVoxels = make_int3(grid.sizeX*interiorWidth, grid.sizeY*interiorWidth, grid.sizeZ*interiorWidth);
    const int strides[3] = {1, voxels1D, voxels1D*voxels1D};
    const int first[3] = {origin.x, origin.y, origin.z}, size[3] = {domainVoxels.x, domainVoxels.y, domainVoxels.z};
    float along[6];     //what the images past each wall carry of the faces inside it, along it
    #pragma unroll
    for(int wall = 0; wall < 6; ++wall){
        along[wall] = PHASES ? 1.0f : stick.along(wall);
    }
    if(letGo != nullptr){   //a voxel a lower wall has let go of: its own face there is the wall's, and takes the face's across it from it
        for(uint i = startVoxel + threadIdx.x; i < endVoxel; i += blockDim.x){
            int slot = voxelIDs[i];
            if(gone[slot] && !solids[i]){
                #pragma unroll
                for(int dim = 0; dim < 3; ++dim){
                    if(first[dim] + slot / strides[dim] % voxels1D == 0){
                        blockVelocities[dim*voxels3D + slot] = blockVelocities[dim*voxels3D + slot + strides[dim]];
                        blockVelocities[(3 + dim)*voxels3D + slot] = blockVelocities[(3 + dim)*voxels3D + slot + strides[dim]];
                    }
                }
            }
        }
        __syncthreads();
    }
    //Every wall face's mirror image is read before any is written: a face in a wall's upper plane is its own mirror image, and a corner voxel's mirror
    //image can be one of those, which another thread may be writing meanwhile. (P2G leaves walls' faces at rest, so that face is 0 either way, but
    //read as it goes it would come out +0 or -0 by timing, and the frames' bits with it)
    float images[MIRRORED_PER_THREAD][6];
    #pragma unroll
    for(int k = 0; k < MIRRORED_PER_THREAD; ++k){
        uint i = startVoxel + threadIdx.x + k*blockDim.x;
        if(i < endVoxel && solids[i]){
            int slot = voxelIDs[i];
            int3 voxel = make_int3(slot % voxels1D, slot / voxels1D % voxels1D, slot / (voxels1D*voxels1D));
            #pragma unroll
            for(int dim = 0; dim < 3; ++dim){
                float sign = 1.0f;
                int across;
                int source = mirroredSlot(voxel, dim, origin, domainVoxels, voxels1D, along, sign, across);
                if(letGo != nullptr && across != 0){    //in or past a wall along its own component: whether the wall has let go of the voxel on it there
                    int beside = source + ((across < 0 ? 0 : size[dim] - 1) - first[dim] - source / strides[dim] % voxels1D)*strides[dim];
                    if(gone[beside]){
                        sign = -sign;       //the liquid's velocity carried on, not reversed
                        source = source == slot ? beside : source;  //and the upper wall's own plane, its own image, takes the last face inside
                    }
                }
                images[k][dim] = sign*blockVelocities[dim*voxels3D + source];
                images[k][3 + dim] = sign*blockVelocities[(3 + dim)*voxels3D + source];
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for(int k = 0; k < MIRRORED_PER_THREAD; ++k){
        uint i = startVoxel + threadIdx.x + k*blockDim.x;
        if(i < endVoxel && solids[i]){
            int slot = voxelIDs[i];
            #pragma unroll
            for(int dim = 0; dim < 3; ++dim){
                blockVelocities[dim*voxels3D + slot] = images[k][dim];
                blockVelocities[(3 + dim)*voxels3D + slot] = images[k][3 + dim];
            }
        }
    }
    __syncthreads();
    float* particleVelocities[3] = {vx, vy, vz};
    float change = 0.0f;    //the most this thread's particles' velocities change by along an axis
    for(uint index = firstParticle + threadIdx.x; index < lastParticle; index += blockDim.x){    //consecutive threads, consecutive particles
        float3 pos = positionInNodeBlock(index, gridPosition, px, py, pz, grid, refinementLevel, apronCells);
        FaceStencil stencil(pos);
        float blend = ids != nullptr && (ids[index] & AIR_PARTICLE) ? airFlipRatio : flipRatio;
        bool escaped = false;   //PHASES: a droplet or a bubble through this substep
        if constexpr(PHASES){
            if(share != nullptr){
                uint id = ids[index];
                float liquid = shareAround(blockVelocities + 6*voxels3D, voxels1D, pos, origin, domainVoxels);
                float own = id & AIR_PARTICLE ? 1.0f - liquid : liquid;     //its own fluid's share where it is (nan where nothing's known: no change)
                if(id & ESCAPED_PARTICLE){
                    if(own > REJOIN_SHARE){
                        id &= ~ESCAPED_PARTICLE;    //back in its own fluid: the grid's again, with the velocity it arrives with
                    }
                }
                else if(own < ESCAPE_SHARE){
                    id |= ESCAPED_PARTICLE;         //a droplet, or a bubble, from here on
                }
                ids[index] = id;
                escaped = id & ESCAPED_PARTICLE;
            }
            const WallStick& held = ids[index] & AIR_PARTICLE ? airStick : stick;   //the walls' hold on its fluid, on the faces past them that this particle reads
            #pragma unroll
            for(int a = 0; a < 3; ++a){
                #pragma unroll
                for(int i = 0; i < 3; ++i){
                    int voxel = first[a] + stencil.onCentres[a].base + i;
                    if(voxel < 0 || voxel >= size[a]){
                        stencil.onCentres[a].w[i] *= held.along(2*a + (voxel >= size[a]));
                    }
                }
            }
        }
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            const Spline3& x = stencil.axis(dim, 0);
            const Spline3& y = stencil.axis(dim, 1);
            const Spline3& z = stencil.axis(dim, 2);
            float newVelocity = 0.0f;
            float oldVelocity = 0.0f;
            float moment[3] = {0.0f, 0.0f, 0.0f}, start[3];   //APIC: B, in voxels, and the stencil's first face from the particle
            if constexpr(APIC){
                float q[3] = {pos.x, pos.y, pos.z};
                #pragma unroll
                for(int a = 0; a < 3; ++a){
                    start[a] = stencil.axis(dim, a).base + (a == dim ? 0.0f : 0.5f) - q[a];
                }
            }
            #pragma unroll
            for(int k = 0; k < 3; ++k){
                #pragma unroll
                for(int j = 0; j < 3; ++j){
                    int rowStart = x.base + (y.base + j)*voxels1D + (z.base + k)*voxels1D*voxels1D;
                    float yz = y.w[j]*z.w[k];
                    #pragma unroll
                    for(int i = 0; i < 3; ++i){
                        float weight = x.w[i]*yz;
                        float face = blockVelocities[dim*voxels3D + rowStart + i];
                        newVelocity += weight*face;
                        oldVelocity += weight*blockVelocities[(3 + dim)*voxels3D + rowStart + i];
                        if constexpr(APIC){
                            float share = weight*face;
                            moment[0] += share*(start[0] + i);
                            moment[1] += share*(start[1] + j);
                            moment[2] += share*(start[2] + k);
                        }
                    }
                }
            }
            float updated = newVelocity + blend*(particleVelocities[dim][index] - oldVelocity);
            if constexpr(PHASES){
                if(escaped){
                    updated = particleVelocities[dim][index];
                }
            }
            if(births[index] < bornSince){
                change = fmaxf(change, fabsf(updated - particleVelocities[dim][index]));
            }
            particleVelocities[dim][index] = updated;
            if constexpr(APIC){
                #pragma unroll
                for(int a = 0; a < 3; ++a){
                    affine.c[3*dim + a][index] = 4.0f/voxelSize*moment[a];  //B's offsets are in voxels: B D^-1 = (4/dx^2) dx moment
                }
            }
        }
    }
    largestInBlock(__float_as_uint(change*perSecond), warpLargest, largestChange);
}

__global__ void advectParticlePositions(uint numParticles, double dt, double* position, const float* v, Grid grid){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        position[index] += dt*v[index];
    }
}

//An advection tile, into a block's shared memory: the interior of a node and its 26 neighbours', 3 nodes wide. Per component, each voxel's negative
//face: past the walls, their faces at rest; inside, where nothing is stored or no particle reached, no velocity. Then, with share, each voxel's
//share of liquid, and none (not 0: what isn't liquid is air) where no voxel is stored. And where a wall has let go of a voxel this substep (letGo,
//or nullptr), the wall's own face there takes the face's across the voxel from it, as G2P has it: the liquid's velocity carried on to the wall, in
//place of rest, which a particle beside the wall would otherwise be moved by, the nearer the slower. Every thread of the block calls it, between
//two __syncthreads: the one that neighborNodes is whole by, and the one the tile is
__device__ inline void loadTile(float* tile, int interiorWidth, int3 tileOrigin, int3 domainVoxels, const uint* neighborNodes, uint numUsedGridNodes, const uint* interiorVoxels,
                                const float* ux, const float* uy, const float* uz, const float* weightsX, const float* weightsY, const float* weightsZ, const float* share,
                                const char* letGo){
    int tileWidth = 3*interiorWidth;
    int tileVoxels = tileWidth*tileWidth*tileWidth;
    const float* velocities[3] = {ux, uy, uz};
    const float* weights[3] = {weightsX, weightsY, weightsZ};
    auto stored = [&](int t, int3& global){
        int x = t % tileWidth, y = t / tileWidth % tileWidth, z = t / (tileWidth*tileWidth);
        global = make_int3(tileOrigin.x + x, tileOrigin.y + y, tileOrigin.z + z);
        bool inside = global.x >= 0 && global.y >= 0 && global.z >= 0 && global.x < domainVoxels.x && global.y < domainVoxels.y && global.z < domainVoxels.z;
        uint node = neighborNodes[x / interiorWidth + 3*(y / interiorWidth) + 9*(z / interiorWidth)];
        return inside && node < numUsedGridNodes ?
            interiorVoxels[node*interiorWidth*interiorWidth*interiorWidth + x % interiorWidth + (y % interiorWidth)*interiorWidth + (z % interiorWidth)*interiorWidth*interiorWidth] : NO_VOXEL;
    };
    for(int t = threadIdx.x; t < tileVoxels; t += blockDim.x){
        int3 global;
        uint voxel = stored(t, global);
        bool inside = global.x >= 0 && global.y >= 0 && global.z >= 0 && global.x < domainVoxels.x && global.y < domainVoxels.y && global.z < domainVoxels.z;
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            tile[dim*tileVoxels + t] = !inside ? 0.0f : voxel != NO_VOXEL && weights[dim][voxel] > 0.0f ? velocities[dim][voxel] : nanf("");
        }
        if(share != nullptr){
            tile[3*tileVoxels + t] = voxel != NO_VOXEL ? share[voxel] : nanf("");
        }
    }
    if(letGo != nullptr){
        __syncthreads();
        const int strides[3] = {1, tileWidth, tileWidth*tileWidth};
        for(int t = threadIdx.x; t < tileVoxels; t += blockDim.x){
            int3 global;
            uint voxel = stored(t, global);
            if(voxel == NO_VOXEL || !letGo[voxel]){
                continue;
            }
            const int at[3] = {global.x, global.y, global.z}, size[3] = {domainVoxels.x, domainVoxels.y, domainVoxels.z};
            #pragma unroll
            for(int dim = 0; dim < 3; ++dim){
                int along = t / strides[dim] % tileWidth;
                if(at[dim] == 0 && along + 1 < tileWidth){          //on the lower wall: its own face is the wall's
                    tile[dim*tileVoxels + t] = tile[dim*tileVoxels + t + strides[dim]];
                }
                if(at[dim] == size[dim] - 1 && along + 1 < tileWidth){  //on the upper one: the wall's is the next slot's
                    tile[dim*tileVoxels + t + strides[dim]] = tile[dim*tileVoxels + t];
                }
            }
        }
    }
}

//moves each particle through the grid's new velocity field for the whole step: with Ralston's RK3 or straight along it (velocityThrough). A straight
//step squeezes particles together where the flow stretches and spreads them where it turns, by an amount growing with dt^2, which clumps them at large
//CFL; RK3 leaves that at dt^4. A block per node with particles: its interior and its 26 neighbours' go into shared memory, a tile 3 nodes wide, which
//holds every trilinear sample RK3 takes up to CFL 4.
//With two PHASES (ids says which fluid each particle is) the walls' hold is each fluid's own, stick the liquid's and airStick the air's, as in G2P;
//and with escaping particles an escaped one isn't the grid's to move: it's left where it is, for flyEscaped. A template, so that without them the
//kernel is the one it always was
template<bool PHASES>
__global__ void advectThroughGrid(uint numParticleNodes, uint numParticles, const uint* gridNodeIndicesToFirstParticleIndex, const uint* gridPosition,
                                  double* px, double* py, double* pz, float* vx, float* vy, float* vz, const uint* ids, bool escaping,
                                  uint numUsedGridNodes, const uint* nodeCells, const uint* cellToNode, const uint* interiorVoxels,
                                  const float* ux, const float* uy, const float* uz, const float* weightsX, const float* weightsY, const float* weightsZ,
                                  float dt, bool rungeKutta3, WallStick stick, WallStick airStick, Grid grid, uint refinementLevel, const char* letGo){
    extern __shared__ float tile[];     //per component, the tile's voxels' negative faces
    __shared__ uint neighborNodes[27];
    int interiorWidth = 2<<refinementLevel;
    int tileWidth = 3*interiorWidth;
    uint firstParticle = gridNodeIndicesToFirstParticleIndex[blockIdx.x];
    uint lastParticle = blockIdx.x == numParticleNodes - 1 ? numParticles : gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1];
    uint cell = gridPosition[firstParticle];
    loadNeighborNodes(neighborNodes, cell, numUsedGridNodes, nodeCells, cellToNode, grid);
    int3 domainVoxels = make_int3(grid.sizeX*interiorWidth, grid.sizeY*interiorWidth, grid.sizeZ*interiorWidth);
    int3 tileOrigin = make_int3(((int)(cell % grid.sizeX) - 1)*interiorWidth, ((int)(cell / grid.sizeX % grid.sizeY) - 1)*interiorWidth, ((int)(cell / (grid.sizeX*grid.sizeY)) - 1)*interiorWidth);
    __syncthreads();
    loadTile(tile, interiorWidth, tileOrigin, domainVoxels, neighborNodes, numUsedGridNodes, interiorVoxels, ux, uy, uz, weightsX, weightsY, weightsZ, nullptr, letGo);
    __syncthreads();
    float voxelSize = grid.cellSize / interiorWidth;
    float step = dt / voxelSize;    //turns a velocity into voxels moved this step
    for(uint index = firstParticle + threadIdx.x; index < lastParticle; index += blockDim.x){
        bool air = false;
        if constexpr(PHASES){
            if(escaping && (ids[index] & ESCAPED_PARTICLE)){
                continue;
            }
            air = ids[index] & AIR_PARTICLE;
        }
        float3 point = make_float3((px[index] - grid.negX)/voxelSize - tileOrigin.x, (py[index] - grid.negY)/voxelSize - tileOrigin.y, (pz[index] - grid.negZ)/voxelSize - tileOrigin.z);
        float3 velocity = velocityThrough(tile, tileWidth, point, tileOrigin, domainVoxels, make_float3(vx[index], vy[index], vz[index]), step, rungeKutta3, air ? airStick : stick);
        px[index] += dt*velocity.x;
        py[index] += dt*velocity.y;
        pz[index] += dt*velocity.z;
    }
}

//Whether a straight piece of an escaped particle's flight enters its own fluid: way, in voxels from point (both in the tile's voxels), reaching
//where its fluid's share, as G2P judges it (shareAround), is over REJOIN_SHARE: the liquid's for a droplet, what's left for a bubble (air). The
//share is asked where the piece crosses each voxel face along it (Amanatides and Woo's traversal) and where it ends; if it's over at one of those,
//found is how far along the piece that is and before the last point along it where it wasn't (0 for the piece's start, which G2P judged the
//particle by or the piece before came to). Where nothing is stored, or past the tile's edge, nothing is known, so it flies on
__device__ inline bool entersOwnFluid(const float* share, int tileWidth, int3 tileOrigin, int3 domainVoxels, bool air, float3 point, float3 way, float& before, float& found){
    float from[3] = {point.x, point.y, point.z}, along[3] = {way.x, way.y, way.z};
    int voxel[3], towards[3];
    float next[3], apart[3];    //how far along the piece the next face across each axis is, and how far apart that axis' faces are
    #pragma unroll
    for(int axis = 0; axis < 3; ++axis){
        voxel[axis] = (int)floorf(from[axis]);
        if(voxel[axis] < 0 || voxel[axis] >= tileWidth){
            return false;
        }
        towards[axis] = along[axis] > 0.0f ? 1 : -1;
        apart[axis] = along[axis] != 0.0f ? fabsf(1.0f / along[axis]) : INFINITY;
        next[axis] = along[axis] > 0.0f ? (voxel[axis] + 1 - from[axis]) / along[axis] : along[axis] < 0.0f ? (voxel[axis] - from[axis]) / along[axis] : INFINITY;
    }
    before = 0.0f;
    for(int crossing = 0; crossing <= 3*tileWidth; ++crossing){
        int axis = next[0] <= next[1] ? (next[0] <= next[2] ? 0 : 2) : (next[1] <= next[2] ? 1 : 2);
        bool ends = !(next[axis] < 1.0f);   //the piece ends in this voxel: its end is the last point asked
        float at = ends ? 1.0f : next[axis];
        float liquid = shareAround(share, tileWidth, make_float3(from[0] + at*along[0], from[1] + at*along[1], from[2] + at*along[2]), tileOrigin, domainVoxels);
        if((air ? 1.0f - liquid : liquid) > REJOIN_SHARE){
            found = at;
            return true;
        }
        if(ends){
            break;
        }
        before = at;
        voxel[axis] += towards[axis];
        if(voxel[axis] < 0 || voxel[axis] >= tileWidth){
            break;
        }
        next[axis] += apart[axis];
    }
    return false;
}

//Flies the escaped particle at index through a substep (flyEscaped): a droplet, or with air a bubble, through the grid's velocity where the substep
//finds it and what that changed by over the substep, the acceleration of the fluid around it (EscapedPath; at rest where the grid has none). The
//tile holds the new velocities, each voxel's share of liquid, and then the old velocities. It returns how long the particle flew. That's all of dt,
//unless its path enters its own fluid first, where that fluid's share, as G2P judges it (shareAround), is over REJOIN_SHARE: its flight ends there,
//with the velocity it has there. The path is followed in straight pieces about a voxel long, and the time it reaches that share is then found on
//the flight itself. (Judged only where a substep left it, a droplet was the liquid's again as far into the liquid as one substep's flight took it,
//up to cfl voxels under the surface of a pool; and given the whole substep's gravity before it flew, as it was, one that fell 0.07 of a voxel onto
//still water landed at 0.37 m/s, not 0.075. Landing on the face of the first voxel over the share, it landed from 0.15 voxel under to 0.85 over
//the surface by where the surface sat in the voxels, and in two alignments of three on a face bordering a voxel G2P then judged it a droplet by.)
//A wall of the domain takes the part of its velocity into it. It's left past the wall for rootCell to bounce back in as far as it overshot, as every
//particle is: held on the wall's plane instead, it would stay there once it's the grid's again, since the grid's velocity into a wall is 0 on it,
//and the liquid stacks up along the walls and their edges
__device__ inline float flyEscapedParticle(uint index, double* px, double* py, double* pz, float* vx, float* vy, float* vz, bool air, const float* tile, int tileWidth,
                                           int3 tileOrigin, int3 domainVoxels, Grid grid, float voxelSize, float dt, const WallStick& stick, const EscapedFlight& flight){
    double* position[3] = {px, py, pz};
    float* own[3] = {vx, vy, vz};
    double low[3] = {grid.negX, grid.negY, grid.negZ};
    uint cells[3] = {grid.sizeX, grid.sizeY, grid.sizeZ};
    int tileVoxels = tileWidth*tileWidth*tileWidth;
    float step = dt / voxelSize;
    float3 from = make_float3((px[index] - grid.negX)/voxelSize - tileOrigin.x, (py[index] - grid.negY)/voxelSize - tileOrigin.y, (pz[index] - grid.negZ)/voxelSize - tileOrigin.z);
    float3 rest = make_float3(0.0f, 0.0f, 0.0f);
    float3 now = sampleVelocity(tile, tileWidth, from, tileOrigin, domainVoxels, rest, stick);
    float3 before = sampleVelocity(tile + 4*tileVoxels, tileWidth, from, tileOrigin, domainVoxels, rest, stick);
    EscapedPath path(make_float3(vx[index], vy[index], vz[index]), now, make_float3((now.x - before.x)/dt, (now.y - before.y)/dt, (now.z - before.z)/dt), flight, dt);
    float began[3] = {from.x, from.y, from.z}, last[3] = {from.x, from.y, from.z};
    float furthest = 0.0f;  //the most it can move along an axis, in voxels
    #pragma unroll
    for(int axis = 0; axis < 3; ++axis){
        furthest = fmaxf(furthest, (fabsf(path.around[axis]) + fabsf(path.start[axis] - path.around[axis]))*step + 0.5f*fabsf(path.pull[axis])*dt*step);
    }
    int pieces = 1 + (int)fminf(furthest, (float)tileWidth);
    double flown = dt;
    double moved[3];
    float arrives[3];
    bool lands = false;
    for(int piece = 0; piece < pieces && !lands; ++piece){
        double sooner = (double)dt*piece / pieces, later = (double)dt*(piece + 1) / pieces;
        float next[3], before, found;
        path.at(later, moved, arrives);
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            next[axis] = began[axis] + (float)(moved[axis] / voxelSize);
        }
        lands = entersOwnFluid(tile + 3*tileVoxels, tileWidth, tileOrigin, domainVoxels, air, make_float3(last[0], last[1], last[2]), make_float3(next[0] - last[0], next[1] - last[1], next[2] - last[2]), before, found);
        if(lands){  //when it reaches that share: the time between the last point it wasn't over it at and the one it is, halved, down to a float's 24 bits of it
            double low = sooner + (later - sooner)*before, high = sooner + (later - sooner)*found;
            for(int halving = 0; halving < 24; ++halving){
                double middle = 0.5*(low + high);
                path.at(middle, moved, arrives);
                float liquid = shareAround(tile + 3*tileVoxels, tileWidth, make_float3(began[0] + (float)(moved[0] / voxelSize), began[1] + (float)(moved[1] / voxelSize),
                                           began[2] + (float)(moved[2] / voxelSize)), tileOrigin, domainVoxels);
                if((air ? 1.0f - liquid : liquid) > REJOIN_SHARE){
                    high = middle;
                }
                else{
                    low = middle;
                }
            }
            flown = high;
        }
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            last[axis] = next[axis];
        }
    }
    path.at(flown, moved, arrives);
    #pragma unroll
    for(int axis = 0; axis < 3; ++axis){
        double reached = position[axis][index] + moved[axis];
        own[axis][index] = reached < low[axis] || reached > low[axis] + cells[axis]*grid.cellSize ? 0.0f : arrives[axis];
        position[axis][index] = reached;
    }
    return (float)flown;
}

//The escaped particles' part of advection, after advectThroughGrid, which leaves them where they are: a block per node with particles, as it has,
//with its tile, each voxel's share of liquid after the velocities, and the old velocities after that. Each droplet and each bubble flies
//(flyEscapedParticle). Where its flight ends in its own fluid it's the grid's again, with the velocity it arrives with, and the grid moves it for
//what's left of the substep. A block is done as soon as it knows its node has none. A kernel of its own: with the flight in advectThroughGrid's
//body, that took 1.8 times as long for every particle (0.80 ms against 0.45 for 1.3M particles, none of them escaped)
__global__ void flyEscaped(uint numParticleNodes, uint numParticles, const uint* gridNodeIndicesToFirstParticleIndex, const uint* gridPosition,
                           double* px, double* py, double* pz, float* vx, float* vy, float* vz, uint* ids, const float* share,
                           uint numUsedGridNodes, const uint* nodeCells, const uint* cellToNode, const uint* interiorVoxels,
                           const float* ux, const float* uy, const float* uz, const float* oldUx, const float* oldUy, const float* oldUz,
                           const float* weightsX, const float* weightsY, const float* weightsZ,
                           float dt, bool rungeKutta3, WallStick stick, WallStick airStick, Grid grid, uint refinementLevel, EscapedFlight droplet, EscapedFlight bubble,
                           unsigned int* largestChange){    //the most any particle's velocity changes by, per second (G2P's measure, carried on)
    extern __shared__ float tile[];
    __shared__ uint neighborNodes[27];
    __shared__ unsigned int warpLargest[NODE_THREADS / 32];
    __shared__ uint escaped;    //whether the node has any
    uint firstParticle = gridNodeIndicesToFirstParticleIndex[blockIdx.x];
    uint lastParticle = blockIdx.x == numParticleNodes - 1 ? numParticles : gridNodeIndicesToFirstParticleIndex[blockIdx.x + 1];
    if(threadIdx.x == 0){
        escaped = 0;
    }
    __syncthreads();
    for(uint index = firstParticle + threadIdx.x; index < lastParticle; index += blockDim.x){
        if(ids[index] & ESCAPED_PARTICLE){
            escaped = 1;
        }
    }
    __syncthreads();
    if(!escaped){
        return;
    }
    int interiorWidth = 2<<refinementLevel;
    int tileWidth = 3*interiorWidth;
    int tileVoxels = tileWidth*tileWidth*tileWidth;
    uint cell = gridPosition[firstParticle];
    loadNeighborNodes(neighborNodes, cell, numUsedGridNodes, nodeCells, cellToNode, grid);
    int3 domainVoxels = make_int3(grid.sizeX*interiorWidth, grid.sizeY*interiorWidth, grid.sizeZ*interiorWidth);
    int3 tileOrigin = make_int3(((int)(cell % grid.sizeX) - 1)*interiorWidth, ((int)(cell / grid.sizeX % grid.sizeY) - 1)*interiorWidth, ((int)(cell / (grid.sizeX*grid.sizeY)) - 1)*interiorWidth);
    __syncthreads();
    loadTile(tile, interiorWidth, tileOrigin, domainVoxels, neighborNodes, numUsedGridNodes, interiorVoxels, ux, uy, uz, weightsX, weightsY, weightsZ, share, nullptr);
    loadTile(tile + 4*tileVoxels, interiorWidth, tileOrigin, domainVoxels, neighborNodes, numUsedGridNodes, interiorVoxels, oldUx, oldUy, oldUz, weightsX, weightsY, weightsZ, nullptr, nullptr);
    __syncthreads();
    float voxelSize = grid.cellSize / interiorWidth;
    float change = 0.0f;
    for(uint index = firstParticle + threadIdx.x; index < lastParticle; index += blockDim.x){
        if(!(ids[index] & ESCAPED_PARTICLE)){
            continue;
        }
        bool air = ids[index] & AIR_PARTICLE;
        float3 before = make_float3(vx[index], vy[index], vz[index]);
        //the walls' hold on the fluid around it as it flies, the liquid for a bubble and the air for a droplet, and then on its own once it has landed
        float flown = flyEscapedParticle(index, px, py, pz, vx, vy, vz, air, tile, tileWidth, tileOrigin, domainVoxels, grid, voxelSize, dt, air ? stick : airStick, air ? bubble : droplet);
        change = fmaxf(change, fmaxf(fabsf(vx[index] - before.x), fmaxf(fabsf(vy[index] - before.y), fabsf(vz[index] - before.z))));
        if(!(flown < dt)){
            continue;   //escaped still
        }
        ids[index] &= ~ESCAPED_PARTICLE;
        float left = dt - flown;
        float3 point = make_float3((px[index] - grid.negX)/voxelSize - tileOrigin.x, (py[index] - grid.negY)/voxelSize - tileOrigin.y, (pz[index] - grid.negZ)/voxelSize - tileOrigin.z);
        float3 velocity = velocityThrough(tile, tileWidth, point, tileOrigin, domainVoxels, make_float3(vx[index], vy[index], vz[index]), left / voxelSize, rungeKutta3, air ? airStick : stick);
        px[index] += left*velocity.x;
        py[index] += left*velocity.y;
        pz[index] += left*velocity.z;
    }
    largestInBlock(__float_as_uint(change / dt), warpLargest, largestChange);
}

//each wall's hold on what moves along it (WallStick, gridSampling.hu): the liquid's, or the air's, which only the walls' friction holds
WallStick Particles::wallSticks(bool liquid) const{
    WallStick sticks;
    for(int wall = 0; wall < 6; ++wall){
        sticks.stick[wall] = (float)std::max(liquid ? wallStick() : 0.0, 0.5*wallFriction[wall]);
        sticks.any = sticks.any || sticks.stick[wall] > 0.0f;
    }
    return sticks;
}

void Particles::voxelVelsToParticles(){
    if(numParticleNodes == 0){  //no particles in this partition
        return;
    }
    if(numVoxelsPerNode > NODE_THREADS*MIRRORED_PER_THREAD){
        std::cerr<<"Particles: a node of "<<numVoxelsPerNode<<" voxels is more than G2P mirrors walls for ("<<NODE_THREADS*MIRRORED_PER_THREAD<<"): raise MIRRORED_PER_THREAD\n";
        exit(1);
    }
    WallStick stick = wallSticks(true);     //how far each wall holds the liquid's particles: a viscous liquid's as its faces are held in the viscous solve
    WallStick airStick = wallSticks(false); //and the air's
    bool phases = twoPhase.on;
    accelerationBits.zeroDeviceAsync(stream);   //the most any particle's velocity changes by this substep, per second: G2P's update and the escaped ones' flights, for the next substep's timestep (getCourantDt)
    auto gather = phases ? (apic ? gatherVoxelVelsToParticles<true, true> : gatherVoxelVelsToParticles<false, true>)
                         : (apic ? gatherVoxelVelsToParticles<true, false> : gatherVoxelVelsToParticles<false, false>);
    const char* gone = lettingGo() ? letGo.devPtr() : nullptr;  //the voxels boundaries let go of in this substep's solve: their walls' faces carry the liquid's velocity on
    gather<<<numParticleNodes, NODE_THREADS, ((phases ? 7 : 6)*sizeof(float) + 1)*numVoxelsPerNode, stream>>>(numParticleNodes, size, numVoxels1D, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(), px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr(),
        affineVelocities(affine), (float)(grid.cellSize / (2<<refinementLevel)), nodeIndexUsedVoxels.devPtr(), voxelIDsUsed.devPtr(), voxelOwners.devPtr(), solids.devPtr(), voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr(), voxelsUxOld.devPtr(), voxelsUyOld.devPtr(), voxelsUzOld.devPtr(), flipRatio, stick, airStick, radius, grid, refinementLevel,
        phases ? particleIds.devPtr() : nullptr, twoPhase.airFlipRatio, twoPhase.escaping() ? liquidShare.devPtr() : nullptr, particleBirths.devPtr(), (float)substepStart, (float)(1.0 / dt), accelerationBits.devPtr(),
        gone);
    gpuErrchk(cudaPeekAtLastError());
    uint tileWidth = 3*(numVoxels1D - 2*(uint)std::floor(radius));
    auto advect = phases ? advectThroughGrid<true> : advectThroughGrid<false>;
    advect<<<numParticleNodes, NODE_THREADS, 3*sizeof(float)*tileWidth*tileWidth*tileWidth, stream>>>(numParticleNodes, size, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(),
        px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr(), phases ? particleIds.devPtr() : nullptr, twoPhase.escaping(), numUsedGridNodes, nodeCells.devPtr(), cellToNode.devPtr(), nodeInteriorVoxels.devPtr(),
        voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr(), voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), dt, rungeKutta3, stick, airStick, grid, refinementLevel, gone);
    gpuErrchk(cudaPeekAtLastError());
    if(twoPhase.escaping()){    //then the escaped ones, which that left where they were
        double voxelSize = grid.cellSize / (2<<refinementLevel);
        //a droplet's or a bubble's radius, unless the scene gives one: a ball of the fluid one particle stands for
        double ball = voxelSize*std::cbrt(3.0 / (4.0*3.14159265358979323846*restParticlesPerVoxel));
        double dropletRadius = twoPhase.dropletRadius > 0.0f ? twoPhase.dropletRadius : ball;
        double bubbleRadius = twoPhase.bubbleRadius > 0.0f ? twoPhase.bubbleRadius : ball;
        //A bubble flattens as it rises, the more the larger it is against the surface tension that rounds it: by its Eotvos number, its buoyancy over
        //that. The drag coefficient of one that has levels at 8/3 Eo / (Eo + 4) (Tomiyama et al. 1998, for water that isn't pure, as Schiller and
        //Naumann's ball is), which is what holds bubbles from 2 mm to 2 cm across all to 0.2 to 0.3 m/s in water
        double pulled = std::sqrt((double)forces.gravity.x*forces.gravity.x + (double)forces.gravity.y*forces.gravity.y + (double)forces.gravity.z*forces.gravity.z);
        double eotvos = pulled*(1.0 - 1.0 / twoPhase.densityRatio)*4.0*bubbleRadius*bubbleRadius / twoPhase.surfaceTension;
        EscapedFlight droplet = {forces.gravity, (float)dropletRadius, twoPhase.airViscosity, twoPhase.densityRatio, 0.44f};
        EscapedFlight bubble = {forces.gravity, (float)bubbleRadius, twoPhase.liquidViscosity, 1.0f / twoPhase.densityRatio, (float)(8.0/3.0*eotvos / (eotvos + 4.0))};
        flyEscaped<<<numParticleNodes, NODE_THREADS, 7*sizeof(float)*tileWidth*tileWidth*tileWidth, stream>>>(numParticleNodes, size, gridNodeIndicesToFirstParticleIndex.devPtr(), gridCell.devPtr(),
            px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr(), particleIds.devPtr(), liquidShare.devPtr(),
            numUsedGridNodes, nodeCells.devPtr(), cellToNode.devPtr(), nodeInteriorVoxels.devPtr(),
            voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr(), voxelsUxOld.devPtr(), voxelsUyOld.devPtr(), voxelsUzOld.devPtr(),
            voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), dt, rungeKutta3, stick, airStick, grid, refinementLevel, droplet, bubble, accelerationBits.devPtr());
        gpuErrchk(cudaPeekAtLastError());
    }
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
    substepsThisFrame = 0;
    while(elapsedTimeThisFrame < frameDt && solveFailure.empty()){    //every step is queued on this partition's one stream, in order: the host only waits where it reads something back
        particleVelToVoxels();
        pressureSolve();
        if(!solveFailure.empty()){  //no pressure to move anything by: the frame stops where it is, in every partition alike
            break;
        }
        smallestDtThisFrame = substepsThisFrame == 0 ? dt : std::fmin(smallestDtThisFrame, dt);
        largestDtThisFrame = substepsThisFrame == 0 ? dt : std::fmax(largestDtThisFrame, dt);
        ++substepsThisFrame;
        updateVoxelVelocities();
        voxelVelsToParticles();   //also moves the particles; initialize re-bins them wherever they landed, reflecting any that crossed a wall
        stepWhitewater();         //the whitewater flies through the same grid, and what the surface made joins it: nothing of the fluid's changes by it
        ++substepIndex;
        initialize();
    }
}

void Particles::initialize(){
        updateObstacles();          //where the obstacles are now
        pushOutParticles();         //particles that went into one come back out
        markRemovedParticles();     //sinks and open faces, and at the start fluid inside obstacles, before rootCell reflects anything back into the domain
        alignParticlesToGrid();
        markBeyondBand();           //two phases with an air band: the air past it, by the rings the last initialize found, goes too
        killRemovedParticles();     //the removed particles' cell is one past the last, so the sort puts them at the end
        sortParticles();
        dropRemovedParticles();
        exchangeParticles();    //particles that crossed into another partition's planes move there
        emitParticles();        //then emitters see every particle in their cells
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

void Particles::makeFrameStream(){
    if(frameStream == nullptr){     //on this partition's GPU, which the caller has made current
        gpuErrchk(cudaStreamCreateWithFlags(&frameStream, cudaStreamNonBlocking));
        gpuErrchk(cudaEventCreateWithFlags(&framePacked, cudaEventDisableTiming));
        gpuErrchk(cudaEventCreateWithFlags(&frameCopied, cudaEventDisableTiming));
        gpuErrchk(cudaEventCreateWithFlags(&columnsCopied, cudaEventDisableTiming));
    }
}

//A frame's float32 x, y, z per particle, packed into framePositions on this partition's stream once the last frame's copy out of it is done
const float* Particles::packPositions(){
    makeFrameStream();
    gpuErrchk(cudaStreamWaitEvent(stream, frameCopied, 0));    //never recorded yet: no wait
    if(size > 0){
        framePositions.resizeAsync(3*size, stream);
        packPositionsToFloats<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, px.devPtr(), py.devPtr(), pz.devPtr(), framePositions.devPtr());
        gpuErrchk(cudaPeekAtLastError());
    }
    return framePositions.devPtr();
}

//this partition's run of a frame, into pinned host memory at xyz: packed on its stream, then copied out on frameStream, so the simulation's next kernels
//run alongside the copy; copied is recorded once it's in
void Particles::copyPositionsToHost(float* xyz, cudaEvent_t copied){
    packPositions();
    gpuErrchk(cudaEventRecord(framePacked, stream));
    gpuErrchk(cudaStreamWaitEvent(frameStream, framePacked, 0));
    if(size > 0){
        gpuErrchk(cudaMemcpyAsync(xyz, framePositions.devPtr(), sizeof(float)*3*size, cudaMemcpyDeviceToHost, frameStream));
    }
    gpuErrchk(cudaEventRecord(frameCopied, frameStream));
    gpuErrchk(cudaEventRecord(copied, frameStream));
}

//P and v, then where they're asked for (not nullptr) the ids, 64 bits each in two floats' place, and the ages, now less each particle's birth. With air
//(ends not nullptr) the liquid's particles alone, closed up in the order they're in: ends is how many of them there are up to each particle, and packed
//how many in all. An id is the particle's number, without the bits that say what it is now (numbers: idNumbers)
__global__ void packFrameColumns(uint numParticles, const double* px, const double* py, const double* pz, const float* vx, const float* vy, const float* vz,
                                 const uint* ids, uint numbers, const float* births, double now, const uint* ends, uint packed, float* planes){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        size_t n = numParticles;
        size_t at = index;
        if(ends != nullptr){
            if(ends[index] == (index == 0 ? 0 : ends[index - 1])){
                return;     //air
            }
            n = packed;
            at = ends[index] - 1;
        }
        planes[at] = px[index];
        planes[n + at] = py[index];
        planes[2*n + at] = pz[index];
        planes[3*n + at] = vx[index];
        planes[4*n + at] = vy[index];
        planes[5*n + at] = vz[index];
        size_t next = 6*n;
        if(ids != nullptr){     //6n floats in, so on an 8-byte boundary
            ((unsigned long long*)(planes + next))[at] = ids[index] & numbers;
            next += 2*n;
        }
        if(births != nullptr){     //never negative: a birth time is a float, which can round to just after now
            planes[next + at] = fmaxf((float)(now - births[index]), 0.0f);
        }
    }
}

__global__ void flagLiquid(uint numParticles, const uint* ids, uint* liquid){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        liquid[index] = !(ids[index] & AIR_PARTICLE);
    }
}

uint Particles::countFrameParticles(){
    frameParticles = size;
    if(twoPhase.on && size > 0){
        frameEnds.resizeAsync(size, stream);
        flagLiquid<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, particleIds.devPtr(), frameEnds.devPtr());
        gpuErrchk(cudaPeekAtLastError());
        cudaParallelPrefixSum(size, frameEnds.devPtr(), stream);
        gpuErrchk(cudaMemcpyAsync(&frameParticles, frameEnds.devPtr() + size - 1, sizeof(uint), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
    }
    return frameParticles;
}

//this partition's part of a cache frame, the particles countFrameParticles just counted, into pinned host memory: packed on its stream (once the last
//frame's copy out of frameColumns is done), then copied out on frameStream, so the simulation's next kernels run alongside the copy; copied is recorded
//once it's in
void Particles::copyFrameColumnsToHost(float* planes, bool ids, bool ages, cudaEvent_t copied){
    makeFrameStream();
    gpuErrchk(cudaStreamWaitEvent(stream, columnsCopied, 0));
    size_t columns = 6 + (ids ? 2 : 0) + (ages ? 1 : 0);    //in floats
    if(frameParticles > 0){
        frameColumns.resizeAsync(columns*frameParticles, stream);
        packFrameColumns<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr(),
            ids ? particleIds.devPtr() : nullptr, idNumbers(), ages ? particleBirths.devPtr() : nullptr, elapsedTime, twoPhase.on ? frameEnds.devPtr() : nullptr,
            frameParticles, frameColumns.devPtr());
        gpuErrchk(cudaPeekAtLastError());
    }
    gpuErrchk(cudaEventRecord(framePacked, stream));
    gpuErrchk(cudaStreamWaitEvent(frameStream, framePacked, 0));
    if(frameParticles > 0){
        gpuErrchk(cudaMemcpyAsync(planes, frameColumns.devPtr(), sizeof(float)*columns*(size_t)frameParticles, cudaMemcpyDeviceToHost, frameStream));
    }
    gpuErrchk(cudaEventRecord(columnsCopied, frameStream));
    gpuErrchk(cudaEventRecord(copied, frameStream));
}

//each particle's id as the cache has it, in 64 bits
__global__ void widenIds(uint numParticles, const uint* ids, unsigned long long* wide){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles){
        wide[index] = ids[index];
    }
}

//positions (double) then velocities, and with APIC their gradients (particleFloats' order), then the ids (64 bits) and the birth times, each a plane of
//every particle's value: the layout of a checkpoint's state shard. Copied on this partition's own stream, so they're the state at this point, whatever
//the next substep does to them
void Particles::copyCheckpointToHost(char* host, cudaEvent_t copied){
    size_t count = size;
    char* at = host;
    for(CudaVec<double>* position : {&px, &py, &pz}){
        if(count > 0){
            gpuErrchk(cudaMemcpyAsync(at, position->devPtr(), sizeof(double)*count, cudaMemcpyDeviceToHost, stream));
        }
        at += sizeof(double)*count;
    }
    for(CudaVec<float>* data : particleFloats()){
        if(count > 0){
            gpuErrchk(cudaMemcpyAsync(at, data->devPtr(), sizeof(float)*count, cudaMemcpyDeviceToHost, stream));
        }
        at += sizeof(float)*count;
    }
    if(count > 0){
        unsigned long long* wide;
        gpuErrchk(cudaMallocAsync((void**)&wide, sizeof(unsigned long long)*count, stream));
        widenIds<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, particleIds.devPtr(), wide);
        gpuErrchk(cudaMemcpyAsync(at, wide, sizeof(unsigned long long)*count, cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaFreeAsync(wide, stream));
        gpuErrchk(cudaMemcpyAsync(at + sizeof(unsigned long long)*count, particleBirths.devPtr(), sizeof(float)*count, cudaMemcpyDeviceToHost, stream));
    }
    gpuErrchk(cudaEventRecord(copied, stream));
}

void Particles::markAirParticles(uint first){
    if(size > 0){
        markAir<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, first, particleIds.devPtr());
        gpuErrchk(cudaPeekAtLastError());
    }
    nextParticleId = first;
}

void Particles::setIdentities(const uint* ids, const float* births, unsigned long long nextId){
    if(size > 0){   //from pageable memory, so they're copied before this returns
        gpuErrchk(cudaMemcpyAsync(particleIds.devPtr(), ids, sizeof(uint)*size, cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(particleBirths.devPtr(), births, sizeof(float)*size, cudaMemcpyHostToDevice, stream));
    }
    nextParticleId = nextId;
}

//A checkpoint holds the particles as the end of its frame left them: its last substep had pushed them out of obstacles, deleted those in sinks, emitted,
//exchanged and sorted them. So carrying on rebuilds only what follows from them: the obstacles where they are then, the particles' cells, this partition's
//own particles (the first exchange keeps just those, as at the start; they're already in its planes, in order), and the grid. Running initialize again
//would emit the last substep's fluid a second time
double Particles::acceleration(){
    unsigned int bits;
    gpuErrchk(cudaMemcpyAsync(&bits, accelerationBits.devPtr(), sizeof(bits), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    float measured;
    memcpy(&measured, &bits, sizeof(measured));
    return measured;
}

void Particles::resume(double time, unsigned long long substep, double acceleration){
    elapsedTime = time;
    substepIndex = substep;
    float measured = (float)acceleration;   //what the checkpoint's last substep accelerated the fluid at, where P2G reads it from
    unsigned int bits;
    memcpy(&bits, &measured, sizeof(bits));
    gpuErrchk(cudaMemcpy(accelerationBits.devPtr(), &bits, sizeof(bits), cudaMemcpyHostToDevice));
    updateObstacles();
    alignParticlesToGrid();
    sortParticles();
    exchangeParticles();
    findBandRings();    //an air band's rings, as the initialize this carries on from left them: from the same particles
    generateVoxels();
}

Particles::~Particles(){     //with its GPU the current one
    if(frameStream != nullptr){
        cudaStreamSynchronize(frameStream);
        cudaStreamDestroy(frameStream);
        cudaEventDestroy(framePacked);
        cudaEventDestroy(frameCopied);
        cudaEventDestroy(columnsCopied);
    }
    if(pocketSeen != nullptr){
        cudaFreeHost(pocketSeen);
    }
    if(viscousSeen != nullptr){
        cudaFreeHost(viscousSeen);
    }
    for(int field = 0; field < forces.numFields; ++field){
        freeForceVolume(forces.fields[field].volume);
    }
    whitewaterPool.release(stream);
    gpuErrchk( cudaStreamDestroy(stream) );
}