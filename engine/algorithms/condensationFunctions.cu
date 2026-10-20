//Copyright 2023 Aberrant Behavior LLC

//The multigrid's preconditioner where a solve's term outweighs the stencil: the unknowns it does are eliminated from the V-cycle and solved for
//exactly, with everything the equations tie them to (block elimination, or static condensation).
//
//The bodies' coupling (bodies.cu) adds T = s B M~^-1 B^T to the pressure equations' A. The V-cycle approximates A^-1, and CG makes up what it misses of T
//in a few iterations while T is small beside A: in open water a body's share of the own coefficient of an unknown on its surface is a few percent, and
//the solve takes the 4 to 8 iterations it takes without bodies. Where bodies touch, though, a voxel between them holds a sliver of liquid that its
//faces all but shut in: open by a few parts in 65535, where a corner sits a hair outside both bodies. The stencil's own coefficient there is 10^5 times
//smaller than in open water, while what the bodies' motion asks of the sliver isn't, and T outweighs A there by thousands. Seeing only A, the V-cycle
//takes such a voxel for all but cut off and answers a residual there thousands of times too strongly: in a crowd of 16 cubes the preconditioned
//equations' eigenvalues ran from 0.9 to 6e4, CG took 70 iterations where it had taken 4, and its residual rose 4.5 times over before it fell, which the
//stopping rule took for a stall.
//
//So those unknowns, S, the ones T's own coefficient is over A's at (the trapped), come out of the V-cycle, and the preconditioner solves for them
//exactly. With F the rest, the equations K = A + T in blocks are
//    K = [I 0; K_FS K_SS^-1 I] [K_SS 0; 0 Sigma] [I K_SS^-1 K_SF; 0 I],    Sigma = K_FF - K_FS K_SS^-1 K_SF
//and M is K with the V-cycle's G in place of Sigma^-1, so M^-1 r is
//    1. y = K_SS^-1 r_S
//    2. z_F = G (r - K [y; 0])_F
//    3. z_S = K_SS^-1 (r_S - (K [0; z_F])_S)
//The V-cycle's grids are F's, the trapped held at 0 as the bounded solve holds its active unknowns (Multigrid's some). M is symmetric and positive
//definite as G is, and M^-1 K's eigenvalues are 1 for S and G's on Sigma for the rest. Sigma lies between A_FF less what A couples F to S by (next to
//nothing: those faces are all but shut) and K_FF = A_FF + T_FF, where T is nowhere over A, so a body's coupling to what's left in F adds no more than
//the sum of those unknowns' shares of T over A, about its added mass over its own mass as in open water.
//
//K_SS is small, a few hundred unknowns where a pile of bodies is under water, dense where bodies tie it together, and its inverse is made once a solve in
//double: T's part from W, its factor (SolveTerm::factorRows, T = W W^T), A's from each unknown's coefficients. Every partition gathers every partition's
//trapped unknowns and puts them in one order, by where they are in the domain, so the block, its inverse and all they make are the same bits however
//the domain is split; the dense products are sums in a fixed order, fused only where written so (__fma_rn), which every GPU rounds alike. The inverse
//comes from tile kernels (Cholesky, the factor's inverse, its transpose times it), about 5 launches per 32 unknowns and n^3 work; a product with it is
//n^2, two a preconditioning. A pile with thousands of trapped unknowns would want a sparse factor instead.
//
//How many there are reaches the host by the wait the solve starts with (its largest divergence): finding them adds no wait of its own.

#include "condensationFunctions.hu"
#include "voxelSolveFunctions.hu"     //NO_VOXEL
#include <cub/cub.cuh>

static constexpr uint TILE = 32;                    //the dense block's tiles are TILE square
static constexpr int RECORD_WORDS = 24 + TERM_ROW_ENTRIES;  //a trapped unknown's record, in 64-bit words (packRecords)

// ---- finding them ----

//where an own node's interior slot is in the domain, in voxels
__device__ inline int3 placeOf(const VoxelLayout& layout, uint node, uint slot){
    uint w = layout.interiorWidth;
    uint3 nodes = make_uint3(layout.domainVoxels.x / w, layout.domainVoxels.y / w, layout.domainVoxels.z / w);
    uint cell = layout.nodeCells[node];
    return make_int3((int)(cell % nodes.x*w + slot % w), (int)(cell / nodes.x % nodes.y*w + slot / w % w), (int)(cell / (nodes.x*nodes.y)*w + slot / (w*w)));
}

//A thread per interior slot of each own node: an unknown whose term's own coefficient is over the stencil's is trapped. Listed as they come, room
//allowing (all are counted), and put in one order with every partition's later
__global__ void markTrapped(VoxelLayout layout, const char* solveCodes, const float* termDiagonal, const float* Adiag, uint capacity, char* trapped, uint* own,
                            int3* places, uint* count){
    uint perNode = layout.interiorWidth*layout.interiorWidth*layout.interiorWidth;
    uint thread = threadIdx.x + blockIdx.x*blockDim.x;
    uint node = thread / perNode;
    if(node >= layout.numNodes){
        return;
    }
    uint slot = thread % perNode;
    uint voxel = layout.interiorVoxels[(size_t)node*perNode + slot];
    if(voxel == NO_VOXEL || !solveCodes[voxel] || !(termDiagonal[voxel] > Adiag[voxel])){
        return;
    }
    trapped[voxel] = 1;
    uint place = atomicAdd(count, 1u);
    if(place < capacity){
        own[place] = voxel;
        places[place] = placeOf(layout, node, slot);
    }
}

//the same list again, from the flags, once there's room for all of them
__global__ void listTrapped(VoxelLayout layout, const char* trapped, uint* own, int3* places, uint* count){
    uint perNode = layout.interiorWidth*layout.interiorWidth*layout.interiorWidth;
    uint thread = threadIdx.x + blockIdx.x*blockDim.x;
    uint node = thread / perNode;
    if(node >= layout.numNodes){
        return;
    }
    uint slot = thread % perNode;
    uint voxel = layout.interiorVoxels[(size_t)node*perNode + slot];
    if(voxel != NO_VOXEL && trapped[voxel]){
        uint place = atomicAdd(count, 1u);
        own[place] = voxel;
        places[place] = placeOf(layout, node, slot);
    }
}

//the V-cycle's unknowns: those codes names (the solve's, or the bounded solve's free ones) that aren't trapped
__global__ void maskColors(uint numUsedVoxels, const char* codes, const char* trapped, char* colors){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        colors[index] = trapped[index] ? 0 : codes[index];
    }
}

__global__ void shareCount(const uint* count, int rank, long long* counts){
    counts[rank] = *count;
}

void findTrapped(Condensation& condensation, Stencil A, const char* solveCodes, const VoxelLayout& layout, uint numUsedVoxels, SolveTerm& term,
                 PartitionContext& context, cudaStream_t stream){
    Condensation& c = condensation;
    int ranks = context.numRanks();
    if(c.seenRanks < ranks){
        if(c.seen != nullptr){
            gpuErrchk(cudaFreeHost(c.seen));
        }
        gpuErrchk(cudaMallocHost((void**)&c.seen, sizeof(long long)*ranks));
        c.seenRanks = ranks;
    }
    c.ready = false;
    c.numTrapped = c.numOwn = 0;
    float* termDiagonal;
    uint slots = layout.numNodes*layout.interiorWidth*layout.interiorWidth*layout.interiorWidth;
    gpuErrchk(cudaMallocAsync((void**)&termDiagonal, sizeof(float)*numUsedVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&c.trapped, numUsedVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&c.colors, numUsedVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&c.own, sizeof(uint)*c.ownCapacity, stream));
    gpuErrchk(cudaMallocAsync((void**)&c.places, sizeof(int3)*c.ownCapacity, stream));
    gpuErrchk(cudaMallocAsync((void**)&c.ownCount, sizeof(uint), stream));
    gpuErrchk(cudaMallocAsync((void**)&c.counts, sizeof(long long)*ranks, stream));
    gpuErrchk(cudaMemsetAsync(termDiagonal, 0, sizeof(float)*numUsedVoxels, stream));
    gpuErrchk(cudaMemsetAsync(c.trapped, 0, numUsedVoxels, stream));
    gpuErrchk(cudaMemsetAsync(c.ownCount, 0, sizeof(uint), stream));
    gpuErrchk(cudaMemsetAsync(c.counts, 0, sizeof(long long)*ranks, stream));
    term.addDiagonal(termDiagonal, stream);
    markTrapped<<<slots / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(layout, solveCodes, termDiagonal, A.Adiag, c.ownCapacity, c.trapped, c.own, c.places, c.ownCount);
    context.fillGhosts(c.trapped, stream);
    maskColors<<<numUsedVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes, c.trapped, c.colors);
    shareCount<<<1, 1, 0, stream>>>(c.ownCount, context.rank(), c.counts);
    context.sumOverPartitions(c.counts, (uint)ranks, stream);
    gpuErrchk(cudaMemcpyAsync(c.seen, c.counts, sizeof(long long)*ranks, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaFreeAsync(termDiagonal, stream));
    gpuErrchk(cudaPeekAtLastError());
}

// ---- gathering them: a record each, in one order ----

//a record: where it is in the domain (its index there, x fastest), whose it is (rank, then its place in its owner's list), its own coefficient in A,
//its six couplings in A (the stencil's order, two to a word), then its row of W: the columns, two to a word, and the values
__device__ inline unsigned long long recordKey(const long long* record){
    return (unsigned long long)record[0];
}
__device__ inline float recordDiagonal(const long long* record){
    return __uint_as_float((uint)record[2]);
}
__device__ inline float recordCoupling(const long long* record, int face){
    unsigned long long word = (unsigned long long)record[3 + face/2];
    return __uint_as_float(face % 2 ? (uint)(word >> 32) : (uint)word);
}
__device__ inline uint recordColumn(const long long* record, int entry){
    unsigned long long word = (unsigned long long)record[6 + entry/2];
    return entry % 2 ? (uint)(word >> 32) : (uint)word;
}
__device__ inline double recordValue(const long long* record, int entry){
    return __longlong_as_double(record[24 + entry]);
}

__global__ void packRecords(uint count, uint offset, int rank, const uint* own, const int3* places, Stencil A, uint3 domainVoxels, const uint* columns,
                            const double* values, long long* records){
    uint k = threadIdx.x + blockIdx.x*blockDim.x;
    if(k >= count){
        return;
    }
    uint voxel = own[k];
    int3 at = places[k];
    long long* record = records + (size_t)(offset + k)*RECORD_WORDS;
    record[0] = (long long)((unsigned long long)at.x + (unsigned long long)domainVoxels.x*((unsigned long long)at.y + (unsigned long long)domainVoxels.y*(unsigned long long)at.z));
    record[1] = (long long)((unsigned long long)rank << 32 | k);
    record[2] = (long long)__float_as_uint(A.Adiag[voxel]);
    for(int pair = 0; pair < 3; ++pair){
        record[3 + pair] = (long long)((unsigned long long)__float_as_uint(A.A[2*pair + 1][voxel]) << 32 | __float_as_uint(A.A[2*pair][voxel]));
    }
    const uint* rowColumns = columns + (size_t)k*TERM_ROW_ENTRIES;
    const double* rowValues = values + (size_t)k*TERM_ROW_ENTRIES;
    for(int entry = 0; entry < TERM_ROW_ENTRIES; entry += 2){
        record[6 + entry/2] = (long long)((unsigned long long)rowColumns[entry + 1] << 32 | rowColumns[entry]);
    }
    for(int entry = 0; entry < TERM_ROW_ENTRIES; ++entry){
        record[24 + entry] = __double_as_longlong(rowValues[entry]);
    }
}

__global__ void keyRecords(uint count, const long long* records, unsigned long long* keys, uint* positions){
    uint k = threadIdx.x + blockIdx.x*blockDim.x;
    if(k < count){
        keys[k] = recordKey(records + (size_t)k*RECORD_WORDS);
        positions[k] = k;
    }
}

//each of this partition's trapped unknowns' place in the order
__global__ void findSlots(uint count, int rank, const long long* records, const uint* order, uint* slots){
    uint place = threadIdx.x + blockIdx.x*blockDim.x;
    if(place < count){
        unsigned long long owner = (unsigned long long)records[(size_t)order[place]*RECORD_WORDS + 1];
        if((int)(owner >> 32) == rank){
            slots[(uint)owner] = place;
        }
    }
}

//whether each is free, a word a place, from its owner
__global__ void shareFree(uint count, const uint* own, const uint* slots, const char* free, long long* words){
    uint k = threadIdx.x + blockIdx.x*blockDim.x;
    if(k < count){
        words[slots[k]] = free[own[k]] ? 1 : 0;
    }
}

// ---- their block of the equations ----

//K_SS, a thread an entry over the whole padded block: the stencil's coefficient (an unknown's own, or its coupling across the face two of them share,
//as the one nearer the domain's origin has it), plus T's, their rows of W times each other over the columns they share, in column order. Both ways
//round an entry is the same sum, so the block is symmetric to the bit. A place that isn't free, and the padding, take a row and column of the identity
__global__ void assembleBlock(uint count, uint padded, const long long* records, const uint* order, const long long* freeWords, uint3 domainVoxels, double* block){
    uint i = threadIdx.y + blockIdx.y*blockDim.y;
    uint j = threadIdx.x + blockIdx.x*blockDim.x;
    if(i >= padded || j >= padded){
        return;
    }
    double value = i == j ? 1.0 : 0.0;
    if(i < count && j < count && (freeWords == nullptr || (freeWords[i] != 0 && freeWords[j] != 0))){
        const long long* a = records + (size_t)order[i]*RECORD_WORDS;
        const long long* b = records + (size_t)order[j]*RECORD_WORDS;
        double coupled = 0.0;
        for(int ea = 0, eb = 0; ea < TERM_ROW_ENTRIES && eb < TERM_ROW_ENTRIES;){
            uint ca = recordColumn(a, ea), cb = recordColumn(b, eb);
            if(ca == NO_TERM_COLUMN || cb == NO_TERM_COLUMN){
                break;
            }
            if(ca == cb){
                coupled = __fma_rn(recordValue(a, ea), recordValue(b, eb), coupled);
                ++ea;
                ++eb;
            }
            else if(ca < cb){
                ++ea;
            }
            else{
                ++eb;
            }
        }
        double stencil = 0.0;
        if(i == j){
            stencil = recordDiagonal(a);
        }
        else{
            unsigned long long ka = recordKey(a), kb = recordKey(b);
            const long long* lower = ka < kb ? a : b;
            unsigned long long low = ka < kb ? ka : kb, high = ka < kb ? kb : ka;
            unsigned long long strides[3] = {1ull, (unsigned long long)domainVoxels.x, (unsigned long long)domainVoxels.x*domainVoxels.y};
            for(int axis = 0; axis < 3; ++axis){
                if(high == low + strides[axis]){
                    stencil = recordCoupling(lower, 2*axis + 1);    //its face toward +axis: 0 if that's the domain's edge
                }
            }
        }
        value = __dadd_rn(stencil, coupled);
    }
    block[(size_t)i*padded + j] = value;
}

// ---- its inverse: tiles of TILE square, in double ----

__device__ inline void pairOfTiles(uint pair, uint& a, uint& b){    //the pair-th (a, b) with b <= a, row after row
    a = (uint)((sqrt(8.0*pair + 1.0) - 1.0)/2.0);
    while((a + 1)*(a + 2)/2 <= pair){
        ++a;
    }
    while(a*(a + 1)/2 > pair){
        --a;
    }
    b = pair - a*(a + 1)/2;
}

//Cholesky, a column of tiles at a time: the diagonal tile's own factor, a thread a row (its strict upper part left 0)
__global__ void factorDiagonalTile(double* M, uint padded, uint K){
    __shared__ double tile[TILE][TILE + 1];
    uint t = threadIdx.x;
    double* base = M + (size_t)K*TILE*padded + K*TILE;
    for(uint c = 0; c < TILE; ++c){
        tile[t][c] = base[(size_t)t*padded + c];
    }
    __syncthreads();
    for(uint k = 0; k < TILE; ++k){
        if(t == k){
            tile[k][k] = __dsqrt_rn(tile[k][k]);
        }
        __syncthreads();
        if(t > k){
            tile[t][k] = __ddiv_rn(tile[t][k], tile[k][k]);
        }
        __syncthreads();
        if(t > k){
            for(uint c = k + 1; c <= t; ++c){
                tile[t][c] = __fma_rn(-tile[t][k], tile[c][k], tile[t][c]);
            }
        }
        __syncthreads();
    }
    for(uint c = 0; c < TILE; ++c){
        base[(size_t)t*padded + c] = c <= t ? tile[t][c] : 0.0;
    }
}

//the tiles below it, L_IK = A_IK L_KK^-T: a block a tile, a thread a row
__global__ void solveTilesBelow(double* M, uint padded, uint K){
    __shared__ double L[TILE][TILE + 1];
    __shared__ double X[TILE][TILE + 1];
    uint t = threadIdx.x;
    uint I = K + 1 + blockIdx.x;
    const double* diagonal = M + (size_t)K*TILE*padded + K*TILE;
    double* below = M + (size_t)I*TILE*padded + K*TILE;
    for(uint r = 0; r < TILE; ++r){
        L[r][t] = diagonal[(size_t)r*padded + t];
        X[r][t] = below[(size_t)r*padded + t];
    }
    __syncthreads();
    for(uint c = 0; c < TILE; ++c){
        double sum = X[t][c];
        for(uint m = 0; m < c; ++m){
            sum = __fma_rn(-X[t][m], L[c][m], sum);
        }
        X[t][c] = __ddiv_rn(sum, L[c][c]);
    }
    __syncthreads();
    for(uint r = 0; r < TILE; ++r){
        below[(size_t)r*padded + t] = X[r][t];
    }
}

//and what's left below and right of them, A_IJ -= L_IK L_JK^T for K < J <= I: a block a tile, a thread an entry
__global__ void updateTilesLeft(double* M, uint padded, uint K){
    __shared__ double Li[TILE][TILE + 1];
    __shared__ double Lj[TILE][TILE + 1];
    uint a, b;
    pairOfTiles(blockIdx.x, a, b);
    uint I = K + 1 + a, J = K + 1 + b;
    uint tx = threadIdx.x, ty = threadIdx.y;
    Li[ty][tx] = M[(size_t)(I*TILE + ty)*padded + K*TILE + tx];
    Lj[ty][tx] = M[(size_t)(J*TILE + ty)*padded + K*TILE + tx];
    __syncthreads();
    double* entry = M + (size_t)(I*TILE + ty)*padded + J*TILE + tx;
    double sum = *entry;
    for(uint m = 0; m < TILE; ++m){
        sum = __fma_rn(-Li[ty][m], Lj[tx][m], sum);
    }
    *entry = sum;
}

//the factor's inverse X = L^-1, lower too: each diagonal tile's own, a thread a column
__global__ void invertDiagonalTiles(const double* L, double* X, uint padded){
    __shared__ double l[TILE][TILE + 1];
    __shared__ double x[TILE][TILE + 1];
    uint t = threadIdx.x, I = blockIdx.x;
    for(uint r = 0; r < TILE; ++r){
        l[r][t] = L[(size_t)(I*TILE + r)*padded + I*TILE + t];
    }
    __syncthreads();
    for(uint r = 0; r < TILE; ++r){
        double sum = r == t ? 1.0 : 0.0;
        for(uint m = t; m < r; ++m){
            sum = __fma_rn(-l[r][m], x[m][t], sum);
        }
        x[r][t] = r < t ? 0.0 : __ddiv_rn(sum, l[r][r]);
    }
    __syncthreads();
    for(uint r = 0; r < TILE; ++r){
        X[(size_t)(I*TILE + r)*padded + I*TILE + t] = x[r][t];
    }
}

//then the tiles distance tiles below the diagonal, from those nearer it: X_IJ = -X_II sum over J <= K < I of L_IK X_KJ, a block a tile
__global__ void invertTilesBelow(const double* L, double* X, uint padded, uint distance){
    __shared__ double a[TILE][TILE + 1];
    __shared__ double b[TILE][TILE + 1];
    uint J = blockIdx.x, I = J + distance;
    uint tx = threadIdx.x, ty = threadIdx.y;
    double sum = 0.0;
    for(uint K = J; K < I; ++K){
        a[ty][tx] = L[(size_t)(I*TILE + ty)*padded + K*TILE + tx];
        b[ty][tx] = X[(size_t)(K*TILE + ty)*padded + J*TILE + tx];
        __syncthreads();
        for(uint m = 0; m < TILE; ++m){
            sum = __fma_rn(a[ty][m], b[m][tx], sum);
        }
        __syncthreads();
    }
    a[ty][tx] = X[(size_t)(I*TILE + ty)*padded + I*TILE + tx];
    b[ty][tx] = sum;
    __syncthreads();
    double product = 0.0;
    for(uint m = 0; m <= ty; ++m){
        product = __fma_rn(-a[ty][m], b[m][tx], product);
    }
    X[(size_t)(I*TILE + ty)*padded + J*TILE + tx] = product;
}

//the block's inverse, X^T X, both halves: tile (I, J) for J <= I is the sum over K >= I of X_KI^T X_KJ, and its mirror the same sums. A block a tile
__global__ void multiplyInverse(const double* X, double* R, uint padded, uint tiles){
    __shared__ double a[TILE][TILE + 1];
    __shared__ double b[TILE][TILE + 1];
    uint I, J;
    pairOfTiles(blockIdx.x, I, J);
    uint tx = threadIdx.x, ty = threadIdx.y;
    double sum = 0.0;
    for(uint K = I; K < tiles; ++K){
        a[ty][tx] = X[(size_t)(K*TILE + ty)*padded + I*TILE + tx];
        b[ty][tx] = X[(size_t)(K*TILE + ty)*padded + J*TILE + tx];
        __syncthreads();
        for(uint m = 0; m < TILE; ++m){
            sum = __fma_rn(a[m][ty], b[m][tx], sum);
        }
        __syncthreads();
    }
    R[(size_t)(I*TILE + ty)*padded + J*TILE + tx] = sum;
    if(I != J){
        R[(size_t)(J*TILE + tx)*padded + I*TILE + ty] = sum;
    }
}

static void invertBlock(Condensation& c, cudaStream_t stream){
    uint padded = c.numPadded;
    uint tiles = padded / TILE;
    for(uint K = 0; K < tiles; ++K){
        factorDiagonalTile<<<1, TILE, 0, stream>>>(c.block, padded, K);
        uint below = tiles - K - 1;
        if(below > 0){
            solveTilesBelow<<<below, TILE, 0, stream>>>(c.block, padded, K);
            updateTilesLeft<<<below*(below + 1)/2, dim3(TILE, TILE), 0, stream>>>(c.block, padded, K);
        }
    }
    gpuErrchk(cudaMemsetAsync(c.work, 0, sizeof(double)*padded*padded, stream));
    invertDiagonalTiles<<<tiles, TILE, 0, stream>>>(c.block, c.work, padded);
    for(uint distance = 1; distance < tiles; ++distance){
        invertTilesBelow<<<tiles - distance, dim3(TILE, TILE), 0, stream>>>(c.block, c.work, padded, distance);
    }
    multiplyInverse<<<tiles*(tiles + 1)/2, dim3(TILE, TILE), 0, stream>>>(c.work, c.block, padded, tiles);
    gpuErrchk(cudaPeekAtLastError());
}

void condense(Condensation& condensation, Stencil A, const VoxelLayout& layout, uint numUsedVoxels, const char* free, SolveTerm& term,
              PartitionContext& context, cudaStream_t stream){
    Condensation& c = condensation;
    int rank = context.rank();
    if(!c.ready){   //the first time: the solve has waited since findTrapped queued the counts' copy, so they're here
        c.ready = true;
        uint offset = 0;
        for(int other = 0; other < context.numRanks(); ++other){
            offset += other < rank ? (uint)c.seen[other] : 0;
            c.numTrapped += (uint)c.seen[other];
        }
        c.numOwn = (uint)c.seen[rank];
        if(c.numTrapped > 0){
            uint n = c.numTrapped;
            uint ownBlocks = c.numOwn / BLOCKSIZE + 1;
            if(c.numOwn > c.ownCapacity){   //the list ran out of room: make enough, and list them again
                while(c.ownCapacity < c.numOwn){
                    c.ownCapacity *= 2;
                }
                gpuErrchk(cudaFreeAsync(c.own, stream));
                gpuErrchk(cudaFreeAsync(c.places, stream));
                gpuErrchk(cudaMallocAsync((void**)&c.own, sizeof(uint)*c.ownCapacity, stream));
                gpuErrchk(cudaMallocAsync((void**)&c.places, sizeof(int3)*c.ownCapacity, stream));
                gpuErrchk(cudaMemsetAsync(c.ownCount, 0, sizeof(uint), stream));
                uint slots = layout.numNodes*layout.interiorWidth*layout.interiorWidth*layout.interiorWidth;
                listTrapped<<<slots / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(layout, c.trapped, c.own, c.places, c.ownCount);
            }
            //each one's record, every partition's gathered
            uint* columns;
            double* values;
            gpuErrchk(cudaMallocAsync((void**)&c.records, sizeof(long long)*n*RECORD_WORDS, stream));
            gpuErrchk(cudaMemsetAsync(c.records, 0, sizeof(long long)*n*RECORD_WORDS, stream));
            gpuErrchk(cudaMallocAsync((void**)&columns, sizeof(uint)*(c.numOwn + 1)*TERM_ROW_ENTRIES, stream));
            gpuErrchk(cudaMallocAsync((void**)&values, sizeof(double)*(c.numOwn + 1)*TERM_ROW_ENTRIES, stream));
            if(c.numOwn > 0){
                term.factorRows(c.numOwn, c.own, c.places, columns, values, stream);
                packRecords<<<ownBlocks, BLOCKSIZE, 0, stream>>>(c.numOwn, offset, rank, c.own, c.places, A, layout.domainVoxels, columns, values, c.records);
            }
            context.sumOverPartitions(c.records, n*RECORD_WORDS, stream);
            gpuErrchk(cudaFreeAsync(columns, stream));
            gpuErrchk(cudaFreeAsync(values, stream));
            //in the one order, by where they are in the domain
            unsigned long long* keys;
            unsigned long long* sortedKeys;
            uint* positions;
            gpuErrchk(cudaMallocAsync((void**)&keys, sizeof(unsigned long long)*n, stream));
            gpuErrchk(cudaMallocAsync((void**)&sortedKeys, sizeof(unsigned long long)*n, stream));
            gpuErrchk(cudaMallocAsync((void**)&positions, sizeof(uint)*n, stream));
            gpuErrchk(cudaMallocAsync((void**)&c.order, sizeof(uint)*n, stream));
            keyRecords<<<n / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(n, c.records, keys, positions);
            size_t bytes = 0;
            gpuErrchk(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, sortedKeys, positions, c.order, (int)n, 0, 64, stream));
            void* scratch;
            gpuErrchk(cudaMallocAsync(&scratch, bytes, stream));
            gpuErrchk(cub::DeviceRadixSort::SortPairs(scratch, bytes, keys, sortedKeys, positions, c.order, (int)n, 0, 64, stream));
            for(void* used : {(void*)keys, (void*)sortedKeys, (void*)positions, scratch}){
                gpuErrchk(cudaFreeAsync(used, stream));
            }
            gpuErrchk(cudaMallocAsync((void**)&c.slots, sizeof(uint)*(c.numOwn + 1), stream));
            findSlots<<<n / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(n, rank, c.records, c.order, c.slots);
            //and the rest a solve needs
            c.numPadded = (n + TILE - 1) / TILE * TILE;
            gpuErrchk(cudaMallocAsync((void**)&c.block, sizeof(double)*c.numPadded*c.numPadded, stream));
            gpuErrchk(cudaMallocAsync((void**)&c.work, sizeof(double)*c.numPadded*c.numPadded, stream));
            gpuErrchk(cudaMallocAsync((void**)&c.freeWords, sizeof(long long)*n, stream));
            gpuErrchk(cudaMallocAsync((void**)&c.gathered, sizeof(long long)*n, stream));
            gpuErrchk(cudaMallocAsync((void**)&c.solved, sizeof(double)*n, stream));
            for(float** perVoxel : {&c.lifted, &c.reduced, &c.product}){
                gpuErrchk(cudaMallocAsync((void**)perVoxel, sizeof(float)*numUsedVoxels, stream));
            }
        }
    }
    if(free != nullptr){    //the bounded solve's colours follow its free unknowns
        maskColors<<<numUsedVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numUsedVoxels, free, c.trapped, c.colors);
    }
    if(c.numTrapped == 0){
        return;
    }
    uint n = c.numTrapped;
    if(free != nullptr){
        gpuErrchk(cudaMemsetAsync(c.freeWords, 0, sizeof(long long)*n, stream));
        shareFree<<<c.numOwn / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(c.numOwn, c.own, c.slots, free, c.freeWords);
        context.sumOverPartitions(c.freeWords, n, stream);
    }
    dim3 entries(TILE, 8);
    assembleBlock<<<dim3(c.numPadded / TILE, c.numPadded / 8), entries, 0, stream>>>(n, c.numPadded, c.records, c.order, free != nullptr ? c.freeWords : nullptr,
                                                                                     layout.domainVoxels, c.block);
    invertBlock(c, stream);
}

// ---- preconditioning ----

//r at each own trapped unknown, at its place (0 where the bounded solve has it held)
__global__ void shareResidual(uint count, const uint* own, const uint* slots, const char* free, const float* r, long long* words){
    uint k = threadIdx.x + blockIdx.x*blockDim.x;
    if(k < count){
        uint voxel = own[k];
        words[slots[k]] = __double_as_longlong(free == nullptr || free[voxel] ? (double)r[voxel] : 0.0);
    }
}

//y = the inverse times x over the first count places: a warp a row, its lanes taking every 32nd column, then pairing off
__global__ void multiplyBlock(uint count, uint padded, const double* inverse, const long long* x, double* y){
    uint row = (threadIdx.x + blockIdx.x*blockDim.x) / 32;
    uint lane = threadIdx.x % 32;
    if(row >= count){   //the whole warp
        return;
    }
    const double* entries = inverse + (size_t)row*padded;
    double sum = 0.0;
    for(uint column = lane; column < count; column += 32){
        sum = __fma_rn(entries[column], __longlong_as_double(x[column]), sum);
    }
    for(int lanes = 16; lanes > 0; lanes >>= 1){
        sum = __dadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, lanes));
    }
    if(lane == 0){
        y[row] = sum;
    }
}

//this partition's places' values at their voxels
__global__ void placeSolved(uint count, const uint* own, const uint* slots, const double* solved, float* values){
    uint k = threadIdx.x + blockIdx.x*blockDim.x;
    if(k < count){
        values[own[k]] = (float)solved[slots[k]];
    }
}

//the residual less the stencil's part of what x makes of it, at every unknown (startDownhill's form)
__global__ void subtractStencil(uint numUsedVoxels, const char* solveCodes, Stencil A, const float* r, const float* x, float* reduced){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels){
        reduced[index] = solveCodes[index] ? r[index] - A.rowTimes(index, x) : 0.0f;
    }
}

//at each own trapped unknown, r less what the V-cycle's correction makes of it there, (A + T) z, its place's value (0 where held)
__global__ void shareRemainder(uint count, const uint* own, const uint* slots, const char* free, Stencil A, const float* r, const float* z, const float* termTimesZ,
                               long long* words){
    uint k = threadIdx.x + blockIdx.x*blockDim.x;
    if(k < count){
        uint voxel = own[k];
        double remainder = free == nullptr || free[voxel] ? (double)r[voxel] - (double)A.rowTimes(voxel, z) - (double)termTimesZ[voxel] : 0.0;
        words[slots[k]] = __double_as_longlong(remainder);
    }
}

void condensedVCycle(Condensation& condensation, const Multigrid& multigrid, Stencil A, const char* solveCodes, const char* free, SolveTerm& term,
                     const float* r, float* z, uint numUsedVoxels, PartitionContext& context, cudaStream_t stream){
    Condensation& c = condensation;
    uint n = c.numTrapped;
    uint voxelBlocks = numUsedVoxels / BLOCKSIZE + 1;
    uint ownBlocks = c.numOwn / BLOCKSIZE + 1;
    uint rowBlocks = (n*32 + BLOCKSIZE - 1) / BLOCKSIZE;
    //1. y = K_SS^-1 r_S
    gpuErrchk(cudaMemsetAsync(c.gathered, 0, sizeof(long long)*n, stream));
    shareResidual<<<ownBlocks, BLOCKSIZE, 0, stream>>>(c.numOwn, c.own, c.slots, free, r, c.gathered);
    context.sumOverPartitions(c.gathered, n, stream);
    multiplyBlock<<<rowBlocks, BLOCKSIZE, 0, stream>>>(n, c.numPadded, c.block, c.gathered, c.solved);
    //2. z_F = G (r - K [y; 0])_F, the V-cycle leaving z 0 at the trapped
    gpuErrchk(cudaMemsetAsync(c.lifted, 0, sizeof(float)*numUsedVoxels, stream));
    placeSolved<<<ownBlocks, BLOCKSIZE, 0, stream>>>(c.numOwn, c.own, c.slots, c.solved, c.lifted);
    context.fillGhosts(c.lifted, stream);
    subtractStencil<<<voxelBlocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes, A, r, c.lifted, c.reduced);
    term.addTimes(c.lifted, c.reduced, -1.0f, nullptr, stream);
    vCycle(multigrid, c.reduced, z, stream);
    //3. z_S = K_SS^-1 (r_S - (K [0; z_F])_S)
    gpuErrchk(cudaMemsetAsync(c.product, 0, sizeof(float)*numUsedVoxels, stream));
    term.addTimes(z, c.product, 1.0f, nullptr, stream);
    gpuErrchk(cudaMemsetAsync(c.gathered, 0, sizeof(long long)*n, stream));
    shareRemainder<<<ownBlocks, BLOCKSIZE, 0, stream>>>(c.numOwn, c.own, c.slots, free, A, r, z, c.product, c.gathered);
    context.sumOverPartitions(c.gathered, n, stream);
    multiplyBlock<<<rowBlocks, BLOCKSIZE, 0, stream>>>(n, c.numPadded, c.block, c.gathered, c.solved);
    placeSolved<<<ownBlocks, BLOCKSIZE, 0, stream>>>(c.numOwn, c.own, c.slots, c.solved, z);
    gpuErrchk(cudaPeekAtLastError());
}

void releaseTrapped(Condensation& condensation, cudaStream_t stream){
    Condensation& c = condensation;
    for(void** buffer : {(void**)&c.trapped, (void**)&c.colors, (void**)&c.own, (void**)&c.places, (void**)&c.ownCount, (void**)&c.counts, (void**)&c.slots,
                         (void**)&c.records, (void**)&c.order, (void**)&c.freeWords, (void**)&c.block, (void**)&c.work, (void**)&c.gathered, (void**)&c.solved,
                         (void**)&c.lifted, (void**)&c.reduced, (void**)&c.product}){
        if(*buffer != nullptr){
            gpuErrchk(cudaFreeAsync(*buffer, stream));
            *buffer = nullptr;
        }
    }
    c.ready = false;
    c.numTrapped = c.numOwn = c.numPadded = 0;
}

void freeCondensation(Condensation& condensation){
    if(condensation.seen != nullptr){
        cudaFreeHost(condensation.seen);
        condensation.seen = nullptr;
        condensation.seenRanks = 0;
    }
}
