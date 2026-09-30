//Copyright 2023 Aberrant Behavior LLC

//The V-cycle that preconditions CG, z = M^-1*r: an approximate solve of A*z = r.
//
//Relaxation (Gauss-Seidel, as in the SOR) wipes out error that's jagged at its own grid's scale in a few sweeps, but barely touches smooth error. Smooth
//error on one grid is jagged on a grid with twice the spacing, though. So the V-cycle relaxes a couple of sweeps, hands what's left (the residual) down to
//a grid with cells twice the size, relaxes there, and so on down to a grid small enough for one block to solve outright. Then, on the way back up, each
//grid adds the coarser one's correction and relaxes again. Every wavelength gets fixed on the grid where it's jagged, so a cycle cuts the error by about
//the same factor at any resolution, for about 8/7 of the work of relaxing the finest grid.
//
//The finest grid is the pressure unknowns themselves. The coarser ones are dense grids over the whole domain, each cell 2x2x2 of the one above, and a cell
//is an unknown only if all its children are; otherwise it's air, pinned at 0 (McAdams, Sifakis and Teran 2010). Walls are the domain's edges throughout.
//Handing a residual down sums a cell's children; a correction comes back up to every child unchanged, the transpose; and the sweeps on the way up run in
//the opposite colour order to those on the way down. That makes the whole cycle symmetric, as CG needs of a preconditioner

#include "multigridFunctions.hu"

static const int SWEEPS = 2;                    //Gauss-Seidel sweeps on each grid, each way
static const uint MAX_COARSEST_CELLS = 1024;    //one block solves the coarsest grid
static const int COARSEST_SWEEPS = 32;          //symmetric sweeps there

// ---- the finest grid: the pressure unknowns ----

//Gauss-Seidel on A*z = r for the unknowns of one colour, whose neighbours are all the other colour: each takes the value that balances its equation.
//No over-relaxing, as the SOR does: that makes a poor smoother
__global__ void relaxVoxels(uint numUsedVoxels, const char* colors, char color, Stencil A, const float* r, float* z){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels && colors[index] == color){
        z[index] = (r[index] - A.offDiagonalTimes(index, z)) / A.Adiag[index];
    }
}

//hands the unknowns' leftover residual, r - A*z, down to the first coarse grid: each cell sums its children's
__global__ void restrictFromVoxels(uint numUsedVoxels, const char* colors, Stencil A, const uint* coarseCells, const float* r, const float* z, float* coarseB){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels && colors[index]){
        atomicAdd(coarseB + coarseCells[index], r[index] - A.rowTimes(index, z));
    }
}

//brings the first coarse grid's correction back up: each unknown adds its cell's
__global__ void prolongToVoxels(uint numUsedVoxels, const char* colors, const uint* coarseCells, const float* coarseZ, float* z){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels && colors[index]){
        z[index] += coarseZ[coarseCells[index]];
    }
}

__global__ void countUnknownChildren(uint numUsedVoxels, const char* colors, const uint* coarseCells, uint* unknownChildren){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels && colors[index]){
        atomicAdd(unknownChildren + coarseCells[index], 1u);
    }
}

__global__ void markFirstLevel(uint numCells, const uint* unknownChildren, char* fluid){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numCells){
        fluid[index] = unknownChildren[index] == 8;
    }
}

// ---- the coarse grids: dense ----

__host__ __device__ inline uint numCellsOf(const CoarseLevel& level){
    return level.cells.x*level.cells.y*level.cells.z;
}

__device__ inline uint3 cellOf(uint index, uint3 cells){
    return make_uint3(index % cells.x, index / cells.x % cells.y, index / (cells.x*cells.y));
}

__device__ inline uint childOf(uint3 cell, int child, uint3 fineCells){    //child 0-7 of a cell, in the grid below
    return (2*cell.x + child % 2) + (2*cell.y + child / 2 % 2)*fineCells.x + (2*cell.z + child / 4)*fineCells.x*fineCells.y;
}

//a cell's neighbours inside the domain (air ones, pinned at 0, still count) and the sum of its unknown neighbours' z, so (A*z) = coupling*(inDomain*z - unknownSum)
__device__ inline void neighborSums(uint index, uint3 cell, uint3 cells, const char* fluid, const float* z, int& inDomain, float& unknownSum){
    uint coordinates[3] = {cell.x, cell.y, cell.z};
    uint sizes[3] = {cells.x, cells.y, cells.z};
    uint strides[3] = {1, cells.x, cells.x*cells.y};
    inDomain = 0;
    unknownSum = 0.0f;
    #pragma unroll
    for(int axis = 0; axis < 3; ++axis){
        if(coordinates[axis] > 0){
            ++inDomain;
            unknownSum += fluid[index - strides[axis]] ? z[index - strides[axis]] : 0.0f;
        }
        if(coordinates[axis] + 1 < sizes[axis]){
            ++inDomain;
            unknownSum += fluid[index + strides[axis]] ? z[index + strides[axis]] : 0.0f;
        }
    }
}

//Gauss-Seidel on one colour of a coarse grid's unknowns
__global__ void relaxCells(CoarseLevel level, int color){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    uint3 cell = cellOf(index, level.cells);
    if(index < numCellsOf(level) && level.fluid[index] && (cell.x + cell.y + cell.z) % 2 == color){
        int inDomain;
        float unknownSum;
        neighborSums(index, cell, level.cells, level.fluid, level.z, inDomain, unknownSum);
        if(inDomain > 0){
            level.z[index] = (level.b[index] + level.coupling*unknownSum) / (level.coupling*inDomain);
        }
    }
}

//hands a grid's leftover residual, b - A*z, down to the next: each coarser unknown sums its 8 children's, which are all unknowns. Air cells get 0
__global__ void restrictCells(CoarseLevel fine, CoarseLevel coarse){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numCellsOf(coarse)){
        float sum = 0.0f;
        if(coarse.fluid[index]){
            uint3 cell = cellOf(index, coarse.cells);
            for(int child = 0; child < 8; ++child){
                uint fineIndex = childOf(cell, child, fine.cells);
                int inDomain;
                float unknownSum;
                neighborSums(fineIndex, cellOf(fineIndex, fine.cells), fine.cells, fine.fluid, fine.z, inDomain, unknownSum);
                sum += fine.b[fineIndex] - fine.coupling*(inDomain*fine.z[fineIndex] - unknownSum);
            }
        }
        coarse.b[index] = sum;
    }
}

//brings a grid's correction back up a level: each finer unknown adds its parent's (an air parent's is 0)
__global__ void prolongCells(CoarseLevel coarse, CoarseLevel fine){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numCellsOf(fine) && fine.fluid[index]){
        uint3 cell = cellOf(index, fine.cells);
        uint3 parent = make_uint3(cell.x / 2, cell.y / 2, cell.z / 2);
        if(parent.x < coarse.cells.x && parent.y < coarse.cells.y && parent.z < coarse.cells.z){
            fine.z[index] += coarse.z[parent.x + parent.y*coarse.cells.x + parent.z*coarse.cells.x*coarse.cells.y];
        }
    }
}

//a coarser grid's unknowns: the cells whose 8 children all are
__global__ void markCoarserLevel(CoarseLevel fine, CoarseLevel coarse){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numCellsOf(coarse)){
        uint3 cell = cellOf(index, coarse.cells);
        bool all = true;
        for(int child = 0; child < 8; ++child){
            all = all && fine.fluid[childOf(cell, child, fine.cells)];
        }
        coarse.fluid[index] = all;
    }
}

//the coarsest grid, solved by one block in shared memory: symmetric Gauss-Seidel sweeps (red then black, then black then red), over and over
__global__ void solveCoarsest(CoarseLevel level){
    extern __shared__ float shared[];
    uint numCells = numCellsOf(level);
    float* z = shared;
    float* b = shared + numCells;
    char* fluid = (char*)(b + numCells);
    uint index = threadIdx.x;
    if(index < numCells){
        z[index] = 0.0f;
        b[index] = level.b[index];
        fluid[index] = level.fluid[index];
    }
    __syncthreads();
    uint3 cell = cellOf(index, level.cells);
    int colors[4] = {0, 1, 1, 0};
    for(int sweep = 0; sweep < COARSEST_SWEEPS; ++sweep){
        for(int half = 0; half < 4; ++half){
            if(index < numCells && fluid[index] && (cell.x + cell.y + cell.z) % 2 == colors[half]){
                int inDomain;
                float unknownSum;
                neighborSums(index, cell, level.cells, fluid, z, inDomain, unknownSum);
                if(inDomain > 0){
                    z[index] = (b[index] + level.coupling*unknownSum) / (level.coupling*inDomain);
                }
            }
            __syncthreads();
        }
    }
    if(index < numCells){
        level.z[index] = z[index];
    }
}

Multigrid buildMultigrid(Stencil A, const char* solveCodes, const uint* coarseCells, uint numUsedVoxels, uint3 domainVoxels, float scale, cudaStream_t stream){
    Multigrid multigrid = {A, solveCodes, coarseCells, numUsedVoxels, {}, 0};
    uint3 cells = make_uint3(domainVoxels.x / 2, domainVoxels.y / 2, domainVoxels.z / 2);
    float coupling = 2.0f*scale;
    while(true){    //halve until the grid fits one block, or can't halve further
        CoarseLevel& level = multigrid.levels[multigrid.numLevels++];
        level.cells = cells;
        level.coupling = coupling;
        uint numCells = numCellsOf(level);
        gpuErrchk(cudaMallocAsync((void**)&level.fluid, numCells, stream));
        gpuErrchk(cudaMallocAsync((void**)&level.b, sizeof(float)*numCells, stream));
        gpuErrchk(cudaMallocAsync((void**)&level.z, sizeof(float)*numCells, stream));
        if(numCells <= MAX_COARSEST_CELLS || cells.x < 2 || cells.y < 2 || cells.z < 2 || multigrid.numLevels == 16){
            break;
        }
        cells = make_uint3(cells.x / 2, cells.y / 2, cells.z / 2);
        coupling *= 2.0f;
    }
    //which cells are unknowns: the first grid's from the voxels, each coarser one's from the one above
    CoarseLevel& first = multigrid.levels[0];
    uint* unknownChildren;
    gpuErrchk(cudaMallocAsync((void**)&unknownChildren, sizeof(uint)*numCellsOf(first), stream));
    cudaMemsetAsync(unknownChildren, 0, sizeof(uint)*numCellsOf(first), stream);
    countUnknownChildren<<<numUsedVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numUsedVoxels, solveCodes, coarseCells, unknownChildren);
    markFirstLevel<<<numCellsOf(first) / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numCellsOf(first), unknownChildren, first.fluid);
    cudaFreeAsync(unknownChildren, stream);
    for(int level = 1; level < multigrid.numLevels; ++level){
        markCoarserLevel<<<numCellsOf(multigrid.levels[level]) / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(multigrid.levels[level - 1], multigrid.levels[level]);
    }
    return multigrid;
}

static void relaxLevel(const CoarseLevel& level, int firstColor, cudaStream_t stream){
    for(int sweep = 0; sweep < SWEEPS; ++sweep){
        relaxCells<<<numCellsOf(level) / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(level, firstColor);
        relaxCells<<<numCellsOf(level) / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(level, 1 - firstColor);
    }
}

void vCycle(const Multigrid& multigrid, const float* r, float* z, cudaStream_t stream){
    uint numUsedVoxels = multigrid.numUsedVoxels;
    uint voxelBlocks = numUsedVoxels / BLOCKSIZE + 1;
    const CoarseLevel* levels = multigrid.levels;
    int coarsest = multigrid.numLevels - 1;

    //down: relax each grid, then hand its leftover residual to the next coarser one
    cudaMemsetAsync(z, 0, sizeof(float)*numUsedVoxels, stream);
    for(int sweep = 0; sweep < SWEEPS; ++sweep){
        for(char color = 1; color <= 2; ++color){   //red, then black
            relaxVoxels<<<voxelBlocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, multigrid.colors, color, multigrid.A, r, z);
        }
    }
    cudaMemsetAsync(levels[0].b, 0, sizeof(float)*numCellsOf(levels[0]), stream);
    restrictFromVoxels<<<voxelBlocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, multigrid.colors, multigrid.A, multigrid.coarseCells, r, z, levels[0].b);
    for(int level = 0; level < coarsest; ++level){
        cudaMemsetAsync(levels[level].z, 0, sizeof(float)*numCellsOf(levels[level]), stream);
        relaxLevel(levels[level], 0, stream);
        restrictCells<<<numCellsOf(levels[level + 1]) / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(levels[level], levels[level + 1]);
    }

    //the coarsest grid, solved outright
    uint coarsestCells = numCellsOf(levels[coarsest]);
    if(coarsestCells <= MAX_COARSEST_CELLS){
        solveCoarsest<<<1, (coarsestCells + 31) / 32 * 32, (2*sizeof(float) + 1)*coarsestCells, stream>>>(levels[coarsest]);
    }
    else{   //too big for a block (a very flat domain): many symmetric sweeps instead
        cudaMemsetAsync(levels[coarsest].z, 0, sizeof(float)*coarsestCells, stream);
        for(int sweep = 0; sweep < COARSEST_SWEEPS; ++sweep){
            relaxLevel(levels[coarsest], 0, stream);
            relaxLevel(levels[coarsest], 1, stream);
        }
    }

    //up: each grid adds the coarser one's correction, then relaxes in the opposite colour order to the way down
    for(int level = coarsest - 1; level >= 0; --level){
        prolongCells<<<numCellsOf(levels[level]) / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(levels[level + 1], levels[level]);
        relaxLevel(levels[level], 1, stream);
    }
    prolongToVoxels<<<voxelBlocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, multigrid.colors, multigrid.coarseCells, levels[0].z, z);
    for(int sweep = 0; sweep < SWEEPS; ++sweep){
        for(char color = 2; color >= 1; --color){   //black, then red
            relaxVoxels<<<voxelBlocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, multigrid.colors, color, multigrid.A, r, z);
        }
    }
}

void freeMultigrid(Multigrid& multigrid, cudaStream_t stream){
    for(int level = 0; level < multigrid.numLevels; ++level){
        cudaFreeAsync(multigrid.levels[level].fluid, stream);
        cudaFreeAsync(multigrid.levels[level].b, stream);
        cudaFreeAsync(multigrid.levels[level].z, stream);
    }
    multigrid.numLevels = 0;
}
