//Copyright 2023 Aberrant Behavior LLC

//The V-cycle that preconditions CG, z = M^-1*r: an approximate solve of A*z = r.
//
//Relaxation (Gauss-Seidel, as in the SOR) wipes out error that's jagged at its own grid's scale in a few sweeps, but barely touches smooth error. Smooth
//error on one grid is jagged on a grid with twice the spacing, though. So the V-cycle relaxes a couple of sweeps, hands what's left (the residual) down to
//a grid with cells twice the size, relaxes there, and so on down to a grid small enough for one block to solve outright. Then, on the way back up, each
//grid adds the coarser one's correction and relaxes again. Every wavelength gets fixed on the grid where it's jagged, so a cycle cuts the error by about
//the same factor at any resolution, for about 8/7 of the work of relaxing the finest grid.
//
//The finest grid is the pressure unknowns themselves. The coarser ones are dense grids over the whole domain, each cell 2x2x2 of the one above, and a
//cell's equation is the sum of its children's (Galerkin coarsening: the coarse matrix is P^T A P for the sum and the copy below), so whatever weighs the
//unknowns' faces weighs the coarse ones too: obstacles' cut faces, a sharp surface's, two fluids' densities. Giving every coarse face the same weight
//instead, with a cell an unknown only if all its children are (McAdams, Sifakis and Teran 2010), needs no coefficients on the coarse grids, but takes 15
//iterations where this takes 6 once obstacles shape the fluid, and stops converging between two fluids. Walls are the domain's edges throughout.
//Handing a residual down sums a cell's children; a correction comes back up to every child unchanged, the transpose; and the sweeps on the way up run in
//the opposite colour order to those on the way down. That makes the whole cycle symmetric, as CG needs of a preconditioner

#include "multigridFunctions.hu"
#include "voxelSolveFunctions.hu"     //NO_VOXEL

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

//the unknowns' leftover residual, r - A*z, for restrictFromVoxels to hand down
__global__ void findResidual(uint numUsedVoxels, const char* colors, Stencil A, const float* r, const float* z, float* residual){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels && colors[index]){
        residual[index] = r[index] - A.rowTimes(index, z);
    }
}

//one of the first coarse grid's cells inside a node, from a thread index running over every node's: its index in that grid, and its 8 voxels' indices
//among the stored voxels (x fastest; NO_VOXEL where nothing's stored). False past the last node
__device__ inline bool firstLevelCell(uint thread, const VoxelLayout& layout, uint3 cells, uint& index, uint children[8]){
    uint perAxis = layout.interiorWidth / 2;
    uint perNode = perAxis*perAxis*perAxis;
    uint node = thread / perNode;
    if(node >= layout.numNodes){
        return false;
    }
    uint local = thread % perNode;
    uint3 nodes = make_uint3(layout.domainVoxels.x / layout.interiorWidth, layout.domainVoxels.y / layout.interiorWidth, layout.domainVoxels.z / layout.interiorWidth);
    uint nodeCell = layout.nodeCells[node];
    uint3 local3 = make_uint3(local % perAxis, local / perAxis % perAxis, local / (perAxis*perAxis));
    uint3 cell = make_uint3(nodeCell % nodes.x*perAxis + local3.x, nodeCell / nodes.x % nodes.y*perAxis + local3.y, nodeCell / (nodes.x*nodes.y)*perAxis + local3.z);
    index = cell.x + cell.y*cells.x + cell.z*cells.x*cells.y;
    const uint* interior = layout.interiorVoxels + node*layout.interiorWidth*layout.interiorWidth*layout.interiorWidth;
    for(int child = 0; child < 8; ++child){
        uint x = 2*local3.x + child % 2, y = 2*local3.y + child / 2 % 2, z = 2*local3.z + child / 4;
        children[child] = interior[x + y*layout.interiorWidth + z*layout.interiorWidth*layout.interiorWidth];
    }
    return true;
}

//hands the unknowns' leftover residual down to the first coarse grid: each of its unknowns sums its children's that are unknowns (the rest are air, or
//aren't stored at all), always in the same order, so the result doesn't depend on thread timing the way atomics' would. Cells with no equation get
//nothing; nothing reads them
__global__ void restrictFromVoxels(VoxelLayout layout, const char* colors, const float* residual, CoarseLevel first){
    uint index;
    uint children[8];
    if(firstLevelCell(threadIdx.x + blockIdx.x*blockDim.x, layout, first.cells, index, children) && first.fluid[index]){
        float sum = 0.0f;
        for(int child = 0; child < 8; ++child){
            if(children[child] != NO_VOXEL && colors[children[child]]){
                sum += residual[children[child]];
            }
        }
        first.b[index] = sum;
    }
}

//brings the first coarse grid's correction back up: each unknown adds its cell's
__global__ void prolongToVoxels(uint numUsedVoxels, const char* colors, const uint* coarseCells, const float* coarseZ, float* z){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numUsedVoxels && colors[index]){
        z[index] += coarseZ[coarseCells[index]];
    }
}

// ---- the coarse grids: dense, their equations summed from the unknowns' ----

__host__ __device__ inline uint numCellsOf(const CoarseLevel& level){
    return level.cells.x*level.cells.y*level.cells.z;
}

__device__ inline uint3 cellOf(uint index, uint3 cells){
    return make_uint3(index % cells.x, index / cells.x % cells.y, index / (cells.x*cells.y));
}

__device__ inline uint childOf(uint3 cell, int child, uint3 fineCells){    //child 0-7 of a cell, in the grid below
    return (2*cell.x + child % 2) + (2*cell.y + child / 2 % 2)*fineCells.x + (2*cell.z + child / 4)*fineCells.x*fineCells.y;
}

//The first coarse grid's equations, from its cells' 8 voxels' (a thread per cell inside each node). With the sum for restriction and the copy for
//prolongation, P^T A P's row for a cell is: on the diagonal, its children's own coefficients less twice the couplings between them (each counted from
//both ends); and to the next cell along an axis, the couplings of its children on that side, whose upper neighbours are that cell's. All of it halved:
//P^T A P weighs a coarse face as its 4 finer ones, but the pressures either side of it are twice as far apart. A cell with no unknown child has no
//equation.
//
//The diagonal is summed from what's left when the couplings between the children are gone, never by taking them off: the children's couplings to other
//cells' unknowns, and their anchors, what each one's own coefficient holds past all its couplings (its faces to air: Stencil::air). Between air and water the
//couplings inside a pocket of air are a thousand times the ones that hold it to the water around it, and the difference of the two big sums came out
//wrong by more than the small one is worth, which leaves a coarse grid that isn't positive definite, and CG then diverges. Summed this way a cell whose
//children couple only to each other, sealed in it, comes to exactly 0, and that alone is what has no equation
__global__ void coarsenFromVoxels(VoxelLayout layout, Stencil A, const char* colors, CoarseLevel first){
    uint index;
    uint children[8];
    if(firstLevelCell(threadIdx.x + blockIdx.x*blockDim.x, layout, first.cells, index, children)){
        float outside = 0.0f, anchor = 0.0f;
        float across[3] = {0.0f, 0.0f, 0.0f};
        for(int child = 0; child < 8; ++child){
            uint voxel = children[child];
            if(voxel != NO_VOXEL && colors[voxel]){
                #pragma unroll
                for(int axis = 0; axis < 3; ++axis){    //its couplings to its neighbours along axis: 0 where one's no unknown
                    if(child >> axis & 1){      //its lower neighbour is the child beside it, its upper one the next cell's
                        across[axis] -= A.A[2*axis + 1][voxel];
                        outside -= A.A[2*axis + 1][voxel];
                    }
                    else{
                        outside -= A.A[2*axis][voxel];
                    }
                }
                anchor += A.air(voxel);
            }
        }
        float own = 0.5f*(outside + anchor);
        first.diagonal[index] = own;
        first.anchor[index] = 0.5f*anchor;
        first.toX[index] = 0.5f*across[0];
        first.toY[index] = 0.5f*across[1];
        first.toZ[index] = 0.5f*across[2];
        first.fluid[index] = own > 0.0f;
    }
}

//a coarser grid's equations from the one above's, the same way
__global__ void coarsenLevel(CoarseLevel fine, CoarseLevel coarse){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numCellsOf(coarse)){
        uint3 cell = cellOf(index, coarse.cells);
        const float* fineTo[3] = {fine.toX, fine.toY, fine.toZ};
        uint coordinates[3] = {cell.x, cell.y, cell.z};
        uint strides[3] = {1, fine.cells.x, fine.cells.x*fine.cells.y};
        float outside = 0.0f, anchor = 0.0f;
        float across[3] = {0.0f, 0.0f, 0.0f};
        for(int child = 0; child < 8; ++child){
            uint fineIndex = childOf(cell, child, fine.cells);
            anchor += fine.anchor[fineIndex];
            #pragma unroll
            for(int axis = 0; axis < 3; ++axis){
                if(child >> axis & 1){
                    across[axis] += fineTo[axis][fineIndex];
                    outside += fineTo[axis][fineIndex];
                }
                else if(coordinates[axis] > 0){
                    outside += fineTo[axis][fineIndex - strides[axis]];     //the finer cell below it's coupling up to it
                }
            }
        }
        //a cell on the grid's upper edge has no next cell, even if the finer grid had an odd cell more for its children to couple to: that coupling is
        //then to nothing, and stays on the diagonal. A closed box's last cell holds everything, coupled to nothing: 0, as coarsenFromVoxels' sealed cells
        float own = 0.5f*(outside + anchor);
        coarse.diagonal[index] = own;
        coarse.anchor[index] = 0.5f*anchor;
        coarse.toX[index] = cell.x + 1 < coarse.cells.x ? 0.5f*across[0] : 0.0f;
        coarse.toY[index] = cell.y + 1 < coarse.cells.y ? 0.5f*across[1] : 0.0f;
        coarse.toZ[index] = cell.z + 1 < coarse.cells.z ? 0.5f*across[2] : 0.0f;
        coarse.fluid[index] = own > 0.0f;
    }
}

//the sum over a cell's neighbours of its coupling to each times how far that neighbour's z is above over. A coupling is stored by the lower cell. With
//over 0 it's what the neighbours add to the cell's own equation, (A*z) = diagonal*z - this, which relaxing divides out. With over the cell's own z,
//(A*z) = anchor*z - this: the row as differences across its faces, as Stencil::rowTimes takes the unknowns'
__device__ inline float coupledSum(uint index, uint3 cell, uint3 cells, const float* toX, const float* toY, const float* toZ, const float* z, float over){
    uint coordinates[3] = {cell.x, cell.y, cell.z};
    uint sizes[3] = {cells.x, cells.y, cells.z};
    uint strides[3] = {1, cells.x, cells.x*cells.y};
    const float* to[3] = {toX, toY, toZ};
    float sum = 0.0f;
    #pragma unroll
    for(int axis = 0; axis < 3; ++axis){
        if(coordinates[axis] > 0){
            sum += to[axis][index - strides[axis]]*(z[index - strides[axis]] - over);
        }
        if(coordinates[axis] + 1 < sizes[axis]){
            sum += to[axis][index]*(z[index + strides[axis]] - over);
        }
    }
    return sum;
}

//Gauss-Seidel on one colour of a coarse grid's unknowns
__global__ void relaxCells(CoarseLevel level, int color){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    uint3 cell = cellOf(index, level.cells);
    if(index < numCellsOf(level) && level.fluid[index] && (cell.x + cell.y + cell.z) % 2 == color){
        level.z[index] = (level.b[index] + coupledSum(index, cell, level.cells, level.toX, level.toY, level.toZ, level.z, 0.0f)) / level.diagonal[index];
    }
}

//hands a grid's leftover residual, b - A*z, down to the next: each coarser unknown sums its children's that are unknowns. Cells with no equation get 0
__global__ void restrictCells(CoarseLevel fine, CoarseLevel coarse){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numCellsOf(coarse)){
        float sum = 0.0f;
        if(coarse.fluid[index]){
            uint3 cell = cellOf(index, coarse.cells);
            for(int child = 0; child < 8; ++child){
                uint fineIndex = childOf(cell, child, fine.cells);
                if(fine.fluid[fineIndex]){
                    float here = fine.z[fineIndex];
                    sum += fine.b[fineIndex] - (fine.anchor[fineIndex]*here
                                               - coupledSum(fineIndex, cellOf(fineIndex, fine.cells), fine.cells, fine.toX, fine.toY, fine.toZ, fine.z, here));
                }
            }
        }
        coarse.b[index] = sum;
    }
}

//brings a grid's correction back up a level: each finer unknown adds its parent's (0 for a parent with no equation)
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

//the coarsest grid, solved by one block in shared memory: symmetric Gauss-Seidel sweeps (red then black, then black then red), over and over
__global__ void solveCoarsest(CoarseLevel level){
    extern __shared__ float shared[];
    uint numCells = numCellsOf(level);
    float* z = shared;
    float* b = shared + numCells;
    float* diagonal = b + numCells;
    float* toX = diagonal + numCells;
    float* toY = toX + numCells;
    float* toZ = toY + numCells;
    uint index = threadIdx.x;
    if(index < numCells){
        z[index] = 0.0f;
        b[index] = level.b[index];
        diagonal[index] = level.diagonal[index];
        toX[index] = level.toX[index];
        toY[index] = level.toY[index];
        toZ[index] = level.toZ[index];
    }
    __syncthreads();
    uint3 cell = cellOf(index, level.cells);
    int colors[4] = {0, 1, 1, 0};
    for(int sweep = 0; sweep < COARSEST_SWEEPS; ++sweep){
        for(int half = 0; half < 4; ++half){
            if(index < numCells && diagonal[index] > 0.0f && (cell.x + cell.y + cell.z) % 2 == colors[half]){
                z[index] = (b[index] + coupledSum(index, cell, level.cells, toX, toY, toZ, z, 0.0f)) / diagonal[index];
            }
            __syncthreads();
        }
    }
    if(index < numCells){
        level.z[index] = z[index];
    }
}

static uint firstLevelThreads(const VoxelLayout& layout){  //a thread per first-level cell inside each node
    uint perAxis = layout.interiorWidth / 2;
    return layout.numNodes*perAxis*perAxis*perAxis;
}

Multigrid buildMultigrid(Stencil A, const char* solveCodes, uint numUsedVoxels, const VoxelLayout& layout, PartitionContext& context, cudaStream_t stream){
    Multigrid multigrid = {A, solveCodes, layout, numUsedVoxels, nullptr, {}, 0, &context};
    gpuErrchk(cudaMallocAsync((void**)&multigrid.residual, sizeof(float)*numUsedVoxels, stream));
    uint3 cells = make_uint3(layout.domainVoxels.x / 2, layout.domainVoxels.y / 2, layout.domainVoxels.z / 2);
    while(true){    //halve until the grid fits one block, or can't halve further
        CoarseLevel& level = multigrid.levels[multigrid.numLevels++];
        level.cells = cells;
        uint numCells = numCellsOf(level);
        gpuErrchk(cudaMallocAsync((void**)&level.fluid, numCells, stream));
        for(float** values : {&level.b, &level.z, &level.diagonal, &level.toX, &level.toY, &level.toZ, &level.anchor}){
            gpuErrchk(cudaMallocAsync((void**)values, sizeof(float)*numCells, stream));
        }
        if(numCells <= MAX_COARSEST_CELLS || cells.x < 2 || cells.y < 2 || cells.z < 2 || multigrid.numLevels == 16){
            break;
        }
        cells = make_uint3(cells.x / 2, cells.y / 2, cells.z / 2);
    }
    //the grids' equations: the first one's from the unknowns' (this partition's own nodes' cells, then the other partitions', so every partition has the
    //whole grid), each coarser one's from the one above's
    CoarseLevel& first = multigrid.levels[0];
    cudaMemsetAsync(first.fluid, 0, numCellsOf(first), stream);
    for(float* coefficients : {first.diagonal, first.toX, first.toY, first.toZ, first.anchor}){
        cudaMemsetAsync(coefficients, 0, sizeof(float)*numCellsOf(first), stream);
    }
    coarsenFromVoxels<<<firstLevelThreads(layout) / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(layout, A, solveCodes, first);
    context.gatherFirstLevel(first.fluid, sizeof(char), stream);
    for(float* coefficients : {first.diagonal, first.toX, first.toY, first.toZ, first.anchor}){
        context.gatherFirstLevel(coefficients, sizeof(float), stream);
    }
    for(int level = 1; level < multigrid.numLevels; ++level){
        coarsenLevel<<<numCellsOf(multigrid.levels[level]) / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(multigrid.levels[level - 1], multigrid.levels[level]);
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

    //down: relax each grid, then hand its leftover residual to the next coarser one. The half-sweeps update this partition's own unknowns, then the
    //ghosts take their owners' values for the next to read
    PartitionContext& context = *multigrid.context;
    cudaMemsetAsync(z, 0, sizeof(float)*numUsedVoxels, stream);
    for(int sweep = 0; sweep < SWEEPS; ++sweep){
        for(char color = 1; color <= 2; ++color){   //red, then black
            relaxVoxels<<<voxelBlocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, multigrid.colors, color, multigrid.A, r, z);
            context.fillGhosts(z, stream);
        }
    }
    findResidual<<<voxelBlocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, multigrid.colors, multigrid.A, r, z, multigrid.residual);
    restrictFromVoxels<<<firstLevelThreads(multigrid.layout) / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(multigrid.layout, multigrid.colors, multigrid.residual, levels[0]);
    context.gatherFirstLevel(levels[0].b, sizeof(float), stream);   //the coarse grids from here on come out the same in every partition
    for(int level = 0; level < coarsest; ++level){
        cudaMemsetAsync(levels[level].z, 0, sizeof(float)*numCellsOf(levels[level]), stream);
        relaxLevel(levels[level], 0, stream);
        restrictCells<<<numCellsOf(levels[level + 1]) / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(levels[level], levels[level + 1]);
    }

    //the coarsest grid, solved outright
    uint coarsestCells = numCellsOf(levels[coarsest]);
    if(coarsestCells <= MAX_COARSEST_CELLS){
        solveCoarsest<<<1, (coarsestCells + 31) / 32 * 32, 6*sizeof(float)*coarsestCells, stream>>>(levels[coarsest]);
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
    prolongToVoxels<<<voxelBlocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, multigrid.colors, multigrid.layout.coarseCells, levels[0].z, z);
    for(int sweep = 0; sweep < SWEEPS; ++sweep){
        for(char color = 2; color >= 1; --color){   //black, then red
            relaxVoxels<<<voxelBlocks, BLOCKSIZE, 0, stream>>>(numUsedVoxels, multigrid.colors, color, multigrid.A, r, z);
            context.fillGhosts(z, stream);
        }
    }
}

void freeMultigrid(Multigrid& multigrid, cudaStream_t stream){
    cudaFreeAsync(multigrid.residual, stream);
    for(int level = 0; level < multigrid.numLevels; ++level){
        const CoarseLevel& grid = multigrid.levels[level];
        cudaFreeAsync(grid.fluid, stream);
        for(float* values : {grid.b, grid.z, grid.diagonal, grid.toX, grid.toY, grid.toZ, grid.anchor}){
            cudaFreeAsync(values, stream);
        }
    }
    multigrid.numLevels = 0;
}
