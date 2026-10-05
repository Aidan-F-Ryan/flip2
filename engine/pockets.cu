//Copyright 2023 Aberrant Behavior LLC

//Sealed pockets: fluid no air reaches, shut in by obstacles and the domain's walls. Nothing in a pocket's pressure equations is held at 0, so they fix its
//pressure only up to a constant, and they have a solution only if the fluid's net flow out of the pocket is 0: fluid that can't be compressed can't leave
//a closed space or arrive in one. An obstacle closing in on a pocket asks for exactly that, as can a deforming one's growth or the density correction, and
//then the solve can't converge, and there's no pressure to step by (pressureSolve stops the bake there). So each pocket's divergence
//has its mean taken out before the solve: the pocket keeps its volume, and its pressure balances the rest. Fluid air reaches doesn't change, and nothing
//but the divergence the solvers are handed does.
//
//findSealedPockets labels each substep's unknowns once their topology is built (generateVoxels, markObstacleSolids). Union-find over the faces between
//unknowns finds each partition's pieces of fluid, and each piece takes the smallest label among its unknowns: 0 for an unknown with a face to air,
//otherwise 1 + its index among the domain's voxels. Partitions then swap labels with their ghosts until every piece agrees, so fluid air reaches ends up at
//0 and a pocket at its smallest voxel's label, the same on every partition however the domain is split. Each pocket gets a place in a table every
//partition holds alike, from the partition that owns its smallest voxel (registerPockets). balanceSealedPockets then takes each pocket's mean divergence
//out, added up exactly as the CG's dot products are, so that doesn't depend on the split either.
//
//Without pockets, only the labelling costs anything, and the host learns there are none along with the free surface pressureSolve already waits for.
//Partitions agreeing on a piece of fluid that crosses between them takes a round of ghost labels per partition boundary it crosses, and the host waits on
//each round to see whether another is needed: those waits only happen while some partition has fluid it can't see air from.

#include "particles.hu"
#include "algorithms/voxelSolveFunctions.hu"   //NO_VOXEL, WALL_VOXEL
#include <cmath>
#include <iostream>

static constexpr uint MOST_POCKETS = 256;   //the most pockets there can be in all: past it none are balanced, whichever partitions hold them
static constexpr int POCKET_DIGITS = 9;
static constexpr uint POCKET_WORDS = POCKET_DIGITS + 2;     //per pocket: its divergence summed exactly, how many values weren't finite, and its unknowns
static constexpr uint NO_POCKET = 0xFFFFFFFFu;
static constexpr uint POCKET_THREADS = 128;
static constexpr uint FULL_WARP = 0xFFFFFFFFu;

//the pockets' bookkeeping, in 64-bit words (Particles::pocketWords): two counts added up over the partitions; the table of pockets, numRanks*MOST_POCKETS
//labels with each partition's from rank*MOST_POCKETS on, then how many pockets the partitions registered in all; each table entry's sum; and its mean
struct PocketBook{
    long long* summed;      //[0] the unknowns no air reaches, [1] the labels the last round changed
    long long* table;
    long long* sums;
    double* means;
    uint entries;
};

static PocketBook pocketBook(CudaVec<double>& words, int numRanks, cudaStream_t stream){
    uint entries = numRanks*MOST_POCKETS;
    size_t size = 2 + (entries + 1) + (size_t)entries*POCKET_WORDS + entries;
    if(words.size() != size){
        words.resizeAsync((uint)size, stream);
    }
    long long* base = (long long*)words.devPtr();
    return {base, base + 2, base + 3 + entries, (double*)(base + 3 + entries + (size_t)entries*POCKET_WORDS), entries};
}

// ---- labels ----

//every unknown its own piece, or the piece of its first unknown neighbour if that comes before it, and its first label: 0 with a face to air (no voxel
//there, or a stored one that isn't an unknown, as the pressure equations see it), otherwise 1 + its index among the domain's voxels. A block per stored
//node. A ghost's missing neighbours past the ghost plane would look like air, but ghosts take their owners' labels (fillGhosts) before any are read
__global__ void startPockets(VoxelPlaces places, uint3 domainVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy,
                             const uint* neighborPy, const uint* neighborNz, const uint* neighborPz, uint* parents, uint* labels){
    uint node = blockIdx.x;
    uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
    uint last = places.nodeVoxelEnds[node];
    uint cell = places.nodeCells[node];
    const uint* neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        if(!solveCodes[index]){
            parents[index] = NO_VOXEL;
            labels[index] = 0;
            continue;
        }
        uint parent = index;
        bool air = false;
        for(int face = 0; face < 6; ++face){
            uint neighbor = neighbors[face][index];
            bool unknown = neighbor < WALL_VOXEL && solveCodes[neighbor];
            air = air || neighbor == NO_VOXEL || (neighbor < WALL_VOXEL && !unknown);
            parent = unknown ? min(parent, neighbor) : parent;
        }
        parents[index] = parent;
        int3 voxel = places.voxelOf(cell, places.voxelSlots[index]);
        labels[index] = air ? 0 : 1 + (uint)voxel.x + domainVoxels.x*((uint)voxel.y + domainVoxels.y*(uint)voxel.z);
    }
}

//the root of an unknown's piece: parents point to smaller indices, and a root to itself. Halves the path on the way, which only ever points an unknown
//further up its own tree
__device__ inline uint pocketRoot(uint index, uint* parents){
    uint current = parents[index];
    if(current != index){
        uint previous = index;
        uint next;
        while(current > (next = parents[current])){
            parents[previous] = next;
            previous = current;
            current = next;
        }
    }
    return current;
}

//joins the pieces either side of every face between two unknowns, the larger root under the smaller (Jaiganesh and Burtscher's ECL-CC), so each piece's
//root ends up its first unknown whichever joins happen first. A thread per voxel; both of a face's unknowns join across it, which costs only a look
__global__ void joinPockets(uint numVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy,
                            const uint* neighborNz, const uint* neighborPz, uint* parents){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index >= numVoxels || !solveCodes[index]){
        return;
    }
    const uint* neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
    for(int face = 0; face < 6; ++face){
        uint neighbor = neighbors[face][index];
        if(neighbor >= WALL_VOXEL || !solveCodes[neighbor]){
            continue;
        }
        uint mine = pocketRoot(index, parents);
        uint theirs = pocketRoot(neighbor, parents);
        while(mine != theirs){
            uint larger = max(mine, theirs);
            uint smaller = min(mine, theirs);
            uint was = atomicCAS(&parents[larger], larger, smaller);
            if(was == larger){
                break;
            }
            mine = was;     //larger was joined to something meanwhile: carry on from there
            theirs = smaller;
        }
    }
}

//every unknown pointed straight at its root. The walks only read, and every value they can meet points up the right tree, so they all end at the root
__global__ void flattenPockets(uint numVoxels, const char* solveCodes, uint* parents){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels && solveCodes[index]){
        uint root = parents[index];
        uint up;
        while((up = parents[root]) != root){
            root = up;
        }
        parents[index] = root;
    }
}

//each piece's smallest label, at its root. Most of a warp's unknowns are in one piece, so they agree on its smallest first and send it once
__global__ void pocketMinimum(uint numVoxels, const char* solveCodes, const uint* parents, const uint* labels, uint* minima){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    bool unknown = index < numVoxels && solveCodes[index];
    uint root = unknown ? parents[index] : NO_VOXEL;
    uint label = unknown ? labels[index] : NO_VOXEL;
    uint unknowns = __ballot_sync(FULL_WARP, unknown);
    if(unknowns == 0){
        return;
    }
    int leader = __ffs(unknowns) - 1;
    uint leaderRoot = __shfl_sync(FULL_WARP, root, leader);
    if(__all_sync(FULL_WARP, !unknown || root == leaderRoot)){
        for(int offset = 16; offset > 0; offset /= 2){
            label = min(label, __shfl_xor_sync(FULL_WARP, label, offset));
        }
        if((int)(threadIdx.x % 32) == leader){
            atomicMin(&minima[leaderRoot], label);
        }
    }
    else if(unknown){
        atomicMin(&minima[root], label);
    }
}

//each own unknown takes its piece's smallest label. Counts the own unknowns no air reaches, so far as this round knows, and the labels that changed
__global__ void pocketRelabel(uint numOwnVoxels, const char* solveCodes, const uint* parents, const uint* minima, uint* labels, long long* summed){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    bool sealed = false;
    bool changed = false;
    if(index < numOwnVoxels && solveCodes[index]){
        uint label = minima[parents[index]];
        changed = label != labels[index];
        sealed = label != 0;
        labels[index] = label;
    }
    uint sealedLanes = __ballot_sync(FULL_WARP, sealed);
    uint changedLanes = __ballot_sync(FULL_WARP, changed);
    if(threadIdx.x % 32 == 0){
        if(sealedLanes != 0){
            atomicAdd((unsigned long long*)&summed[0], (unsigned long long)__popc(sealedLanes));
        }
        if(changedLanes != 0){
            atomicAdd((unsigned long long*)&summed[1], (unsigned long long)__popc(changedLanes));
        }
    }
}

// ---- the table of pockets ----

//a pocket's smallest voxel registers it in its partition's part of the table. A block per own node. The count goes on past the table's room, so once
//it's added up over the partitions, every partition can tell there were too many
__global__ void registerPockets(VoxelPlaces places, uint3 domainVoxels, uint rank, const char* solveCodes, const uint* labels, long long* table, long long* registered){
    uint node = blockIdx.x;
    uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
    uint last = places.nodeVoxelEnds[node];
    uint cell = places.nodeCells[node];
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        if(!solveCodes[index] || labels[index] == 0){
            continue;
        }
        int3 voxel = places.voxelOf(cell, places.voxelSlots[index]);
        if(labels[index] != 1 + (uint)voxel.x + domainVoxels.x*((uint)voxel.y + domainVoxels.y*(uint)voxel.z)){
            continue;
        }
        unsigned long long place = atomicAdd((unsigned long long*)registered, 1ull);
        if(place < MOST_POCKETS){
            table[rank*MOST_POCKETS + place] = labels[index];
        }
    }
}

//each own piece of a pocket finds its pocket in the table, by its label, and its root keeps where (in slots, which held the pieces' smallest labels
//until now). Only roots look, a handful, so a scan of the table does
__global__ void findPocketSlots(uint numOwnVoxels, uint entries, const char* solveCodes, const uint* parents, const uint* labels, const long long* table, uint* slots){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index >= numOwnVoxels || !solveCodes[index] || parents[index] != index || labels[index] == 0){
        return;
    }
    uint slot = NO_POCKET;
    if(table[entries] <= MOST_POCKETS){
        for(uint entry = 0; entry < entries && slot == NO_POCKET; ++entry){
            slot = table[entry] == (long long)labels[index] ? entry : NO_POCKET;
        }
    }
    slots[index] = slot;
}

// ---- balancing the divergence ----

//adds a float to an exact sum in fixed point, as the CG adds its dot products (conjugateGradientFunctions.cu): digit k of sum counts units of
//2^(32k - 149), in a 64-bit integer with room for 2^31 additions' carries, so the digits come out the same in any order, and add up across partitions
//word by word. sum[POCKET_DIGITS] counts infs and NaNs
__device__ inline void addPocketExactly(float value, long long* sum){
    uint bits = __float_as_uint(value);
    int exponent = bits >> 23 & 0xFF;
    if(value == 0.0f){
        return;
    }
    if(exponent == 0xFF){
        atomicAdd((unsigned long long*)&sum[POCKET_DIGITS], 1ull);
        return;
    }
    unsigned long long significand = bits & 0x7FFFFF;
    if(exponent != 0){
        significand |= 0x800000;
    }
    else{
        exponent = 1;
    }
    int position = exponent - 1;
    significand <<= position % 32;
    long long sign = bits >> 31 ? -1 : 1;
    atomicAdd((unsigned long long*)sum + position/32, (unsigned long long)(sign*(long long)(significand & 0xFFFFFFFFull)));
    atomicAdd((unsigned long long*)sum + position/32 + 1, (unsigned long long)(sign*(long long)(significand >> 32)));
}

//each pocket's divergence over its own unknowns, and how many. A block's unknowns in pockets are mostly in one: those add into the block's share of it,
//which goes in once, and the rest straight into theirs
__global__ void sumPockets(uint numOwnVoxels, const char* solveCodes, const uint* parents, const uint* labels, const uint* slots, const float* divU, long long* sums){
    __shared__ long long share[POCKET_WORDS];
    __shared__ uint blockSlot;
    if(threadIdx.x < POCKET_WORDS){
        share[threadIdx.x] = 0;
    }
    if(threadIdx.x == 0){
        blockSlot = NO_POCKET;
    }
    __syncthreads();
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    uint slot = NO_POCKET;
    if(index < numOwnVoxels && solveCodes[index] && labels[index] != 0){
        slot = slots[parents[index]];
    }
    if(slot != NO_POCKET){
        atomicMin(&blockSlot, slot);
    }
    __syncthreads();
    if(slot != NO_POCKET){
        long long* sum = slot == blockSlot ? share : sums + (size_t)slot*POCKET_WORDS;
        addPocketExactly(divU[index], sum);
        atomicAdd((unsigned long long*)&sum[POCKET_DIGITS + 1], 1ull);
    }
    __syncthreads();
    if(blockSlot != NO_POCKET && threadIdx.x < POCKET_WORDS && share[threadIdx.x] != 0){
        atomicAdd((unsigned long long*)(sums + (size_t)blockSlot*POCKET_WORDS + threadIdx.x), (unsigned long long)share[threadIdx.x]);
    }
}

//each pocket's mean divergence, from its exact sum (finishExactSum's conversion, conjugateGradientFunctions.cu). One that isn't finite is left alone:
//balancing can't help that solve
__global__ void finishPockets(uint entries, const long long* sums, double* means){
    uint entry = threadIdx.x + blockIdx.x*blockDim.x;
    if(entry >= entries){
        return;
    }
    const long long* sum = sums + (size_t)entry*POCKET_WORDS;
    long long unknowns = sum[POCKET_DIGITS + 1];
    if(unknowns == 0 || sum[POCKET_DIGITS] != 0){
        means[entry] = 0.0;
        return;
    }
    unsigned long long digits[POCKET_DIGITS];
    long long carry = 0;
    for(int k = 0; k < POCKET_DIGITS; ++k){
        long long value = sum[k] + carry;
        digits[k] = (unsigned long long)value & 0xFFFFFFFFull;
        carry = value >> 32;
    }
    bool negative = carry < 0;
    if(negative){
        unsigned long long add = 1;
        for(int k = 0; k < POCKET_DIGITS; ++k){
            unsigned long long value = (~digits[k] & 0xFFFFFFFFull) + add;
            digits[k] = value & 0xFFFFFFFFull;
            add = value >> 32;
        }
        carry = ~carry + (long long)add;
    }
    double magnitude = (double)carry*scalbn(1.0, 32*POCKET_DIGITS - 149);
    for(int k = POCKET_DIGITS - 1; k >= 0; --k){
        magnitude += (double)digits[k]*scalbn(1.0, 32*k - 149);
    }
    means[entry] = (negative ? -magnitude : magnitude) / (double)unknowns;
}

//each own unknown in a pocket gives up its pocket's mean divergence
__global__ void balancePockets(uint numOwnVoxels, const char* solveCodes, const uint* parents, const uint* labels, const uint* slots, const double* means, float* divU){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numOwnVoxels && solveCodes[index] && labels[index] != 0){
        uint slot = slots[parents[index]];
        if(slot != NO_POCKET){
            divU[index] = (float)((double)divU[index] - means[slot]);
        }
    }
}

// ---- the steps pressureSolve takes ----

//the domain's voxels along each axis, or 0s if there are too many to label with 32 bits
static uint3 pocketDomain(const Grid& grid, uint interiorWidth){
    unsigned long long x = (unsigned long long)grid.sizeX*interiorWidth, y = (unsigned long long)grid.sizeY*interiorWidth, z = (unsigned long long)grid.sizeZ*interiorWidth;
    return x*y*z < 0xFFFFFFFFull ? make_uint3((uint)x, (uint)y, (uint)z) : make_uint3(0, 0, 0);
}

//a round of labels: each piece's smallest, which each own unknown takes, counting into summed
void Particles::pocketRound(long long* summed){
    uint numVoxels = pocketLabels.size();
    gpuErrchk(cudaMemsetAsync(pocketMinima.devPtr(), 0xFF, sizeof(uint)*numVoxels, stream));
    if(numVoxels > 0){
        pocketMinimum<<<numVoxels / POCKET_THREADS + 1, POCKET_THREADS, 0, stream>>>(numVoxels, solveCodes.devPtr(), pocketParents.devPtr(), pocketLabels.devPtr(), pocketMinima.devPtr());
    }
    if(numOwnVoxels > 0){
        pocketRelabel<<<numOwnVoxels / POCKET_THREADS + 1, POCKET_THREADS, 0, stream>>>(numOwnVoxels, solveCodes.devPtr(), pocketParents.devPtr(), pocketMinima.devPtr(),
            pocketLabels.devPtr(), summed);
    }
}

//labels this substep's unknowns, and queues the counts of the ones no air reaches for the host, which pressureSolve waits for along with the free surface.
//Every partition makes the same exchanges in the same order, whether it has voxels or not
void Particles::findSealedPockets(){
    uint interiorWidth = numVoxels1D - 2*(uint)std::floor(radius);
    uint3 domainVoxels = pocketDomain(grid, interiorWidth);
    if(pocketSeen == nullptr){
        gpuErrchk(cudaMallocHost((void**)&pocketSeen, 2*sizeof(long long)));
        if(domainVoxels.x == 0 && verbose()){
            std::cerr<<"The domain has too many voxels to look for sealed pockets of fluid in: their divergence won't be balanced\n";
        }
    }
    pocketSeen[0] = pocketSeen[1] = 0;
    if(domainVoxels.x == 0){
        return;
    }
    uint numVoxels = voxelIDsUsed.size();
    for(CudaVec<uint>* perVoxel : {&pocketParents, &pocketLabels, &pocketMinima}){
        perVoxel->resizeAsync(numVoxels, stream);
    }
    PocketBook book = pocketBook(pocketWords, numRanks, stream);
    gpuErrchk(cudaMemsetAsync(book.summed, 0, 2*sizeof(long long), stream));
    if(numStoredNodes > 0 && numVoxels > 0){
        startPockets<<<numStoredNodes, POCKET_THREADS, 0, stream>>>(voxelPlaces(), domainVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(),
            neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(), pocketParents.devPtr(), pocketLabels.devPtr());
        joinPockets<<<numVoxels / POCKET_THREADS + 1, POCKET_THREADS, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(),
            neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(), pocketParents.devPtr());
        flattenPockets<<<numVoxels / POCKET_THREADS + 1, POCKET_THREADS, 0, stream>>>(numVoxels, solveCodes.devPtr(), pocketParents.devPtr());
    }
    context->fillGhosts((float*)pocketLabels.devPtr(), stream);    //labels go across bit for bit
    pocketRound(book.summed);
    context->sumOverPartitions(book.summed, 2, stream);
    gpuErrchk(cudaMemcpyAsync(pocketSeen, book.summed, 2*sizeof(long long), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaPeekAtLastError());
}

//once the host has the counts: partitions sharing fluid that might be sealed go on swapping labels until none changes, waiting on each round; then each
//pocket takes its place in the table
void Particles::settleSealedPockets(){
    long long unknowns = pocketSeen[0];
    long long changed = pocketSeen[1];
    PocketBook book = pocketBook(pocketWords, numRanks, stream);
    while(numRanks > 1 && unknowns > 0 && changed > 0){    //a lone partition's first round is its last
        gpuErrchk(cudaMemsetAsync(book.summed, 0, 2*sizeof(long long), stream));
        context->fillGhosts((float*)pocketLabels.devPtr(), stream);
        pocketRound(book.summed);
        context->sumOverPartitions(book.summed, 2, stream);
        gpuErrchk(cudaMemcpyAsync(pocketSeen, book.summed, 2*sizeof(long long), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        unknowns = pocketSeen[0];
        changed = pocketSeen[1];
    }
    pocketUnknowns = unknowns;
    if(unknowns == 0){
        return;
    }
    uint interiorWidth = numVoxels1D - 2*(uint)std::floor(radius);
    gpuErrchk(cudaMemsetAsync(book.table, 0, sizeof(long long)*(book.entries + 1), stream));
    if(numOwnNodes > 0){
        registerPockets<<<numOwnNodes, POCKET_THREADS, 0, stream>>>(voxelPlaces(), pocketDomain(grid, interiorWidth), rank, solveCodes.devPtr(), pocketLabels.devPtr(), book.table,
            book.table + book.entries);
    }
    context->sumOverPartitions(book.table, book.entries + 1, stream);
    if(numOwnVoxels > 0){
        findPocketSlots<<<numOwnVoxels / POCKET_THREADS + 1, POCKET_THREADS, 0, stream>>>(numOwnVoxels, book.entries, solveCodes.devPtr(), pocketParents.devPtr(), pocketLabels.devPtr(),
            book.table, pocketMinima.devPtr());
    }
    gpuErrchk(cudaPeekAtLastError());
    if(verbose()){
        std::cout<<"Sealed pockets: "<<unknowns<<" unknowns no air reaches; each pocket's divergence is balanced (at most "<<MOST_POCKETS<<" pockets)\n";
    }
}

//takes each pocket's mean out of the divergence just found, after each cudaCalcDivU and addObstacleFlux; the ghosts take their owners' results
void Particles::balanceSealedPockets(){
    if(pocketUnknowns == 0){
        return;
    }
    PocketBook book = pocketBook(pocketWords, numRanks, stream);
    gpuErrchk(cudaMemsetAsync(book.sums, 0, sizeof(long long)*book.entries*POCKET_WORDS, stream));
    if(numOwnVoxels > 0){
        sumPockets<<<numOwnVoxels / POCKET_THREADS + 1, POCKET_THREADS, 0, stream>>>(numOwnVoxels, solveCodes.devPtr(), pocketParents.devPtr(), pocketLabels.devPtr(), pocketMinima.devPtr(),
            divU.devPtr(), book.sums);
    }
    context->sumOverPartitions(book.sums, book.entries*POCKET_WORDS, stream);
    finishPockets<<<book.entries / POCKET_THREADS + 1, POCKET_THREADS, 0, stream>>>(book.entries, book.sums, book.means);
    if(numOwnVoxels > 0){
        balancePockets<<<numOwnVoxels / POCKET_THREADS + 1, POCKET_THREADS, 0, stream>>>(numOwnVoxels, solveCodes.devPtr(), pocketParents.devPtr(), pocketLabels.devPtr(), pocketMinima.devPtr(),
            book.means, divU.devPtr());
    }
    context->fillGhosts(divU.devPtr(), stream);
    gpuErrchk(cudaPeekAtLastError());
}
