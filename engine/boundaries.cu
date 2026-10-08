//Copyright 2023 Aberrant Behavior LLC

//Boundaries that let go of the liquid.
//
//The pressure solve allows no flow through a solid face, a wall of the domain's or an obstacle's, either way. So a boundary pushes on liquid that
//presses on it, as it should, and pulls on liquid that would leave it, with a pressure under the air's: a splash that reaches the ceiling hangs there
//for seconds, since with no air simulated there is nothing to get in behind it. That's what a sealed container does, or a pipe, a syringe, a siphon. It
//isn't what anything standing in open air does. A boundary that lets go pushes and never pulls. Each wall says which it is (setWallsLetGo), and each
//obstacle (SceneObstacle::hold); with the air simulated, the air gets behind the liquid itself, and every boundary holds.
//
//A liquid voxel on a boundary that lets go is then in one of two states (Batty, Bertails and Bridson 2007, whose cut cells the obstacles' are: the
//pressure there is the boundary's push). Held, it keeps its volume as every unknown does, and its pressure isn't under the air's. Let go, its pressure
//is the air's, and it's opening: more flows out of it than in, with the boundary's own faces still closed, which is the liquid leaving the boundary
//with the gap inside the voxel. The states that satisfy both everywhere are found in two solves at most (Particles::pressureSolve):
//  1. with every voxel held, which is the solve there always was. Voxels the boundary would have to pull on come out under the air's pressure. If
//     there are none, that's the answer, to the bit;
//  2. those of them that air can reach (below) are solved for again with the rest, from those pressures, by a solver that keeps them from going
//     under the air's pressure: each ends at it, let go, or over it, held after all (conjugateGradientFunctions.cu's boundedConjugateGradient, or
//     the SOR with its sweeps stopped there). Letting go of a voxel that was pulling on its neighbours leaves some of them pressed on the boundary
//     instead, and which are which is part of what's solved for.
//The first solve can't be spared: where air gets in is judged on it. The second is about another solve's work, the pressures everywhere changing
//with what's let go. On a dam break with every wall and its pillar letting go: 12.0 V-cycles a substep against 6.7 with everything held. (Until the
//solvers could keep to a bound it was done as whole solves, each followed by taking hold again of the voxels it left squeezed, 16.2 V-cycles and
//sweeps of relaxation besides.)
//
//A voxel that's let go is an unknown like any other afterwards, at the air's pressure: the velocity update gives its faces what an air voxel's
//would get, and its faces on the boundary stay closed.
//
//A voxel may be let go if it's on something solid and everything solid it's on lets go: one in the corner between a wall that holds and one that
//doesn't is held. And only if air can get to it: letting go is air coming in behind the liquid, and under water, with no way in for it, a boundary
//holds whatever pulls (short of the liquid boiling, 10 m of water's worth, which nothing here reaches). So of the voxels step 1 leaves pulled on,
//the ones let go are those with a face to air, and those joined to one of them through others, face to face along the boundary, as the air would
//come: a splash on the ceiling peels from its edges, in one substep if all of it is pulled, and the low pressure in the middle of an eddy against a
//wall, deep in the liquid, lets nothing go.
//
//What the particles read. The grid's faces on a boundary are the boundary's, at rest on a wall and moving with an obstacle, and every particle within
//a voxel and a half of one reads them into its velocity (G2P, advection, the whitewater's flight). That's right where the boundary holds the liquid,
//and where it has let go it would keep the liquid's top layer back, and the rest with it. There the liquid's velocity across the boundary's face is
//the liquid's own, carried on from the face across the voxel. The domain's walls have no faces stored past them, so the kernels that read make them
//so, from letGo (gatherVoxelVelsToParticles, loadTile); obstacles' faces are stored, and are set after the velocity update, in the new velocities
//and in FLIP's old (carryLetGoFaces, obstacles.cu).

#include "particles.hu"
#include "algorithms/multigridFunctions.hu"
#include "algorithms/voxelSolveFunctions.hu"

static const uint LET_GO_THREADS = 64;      //per node block in findPulled

//what tells whether a voxel is on a boundary that lets go
struct Boundaries{
    const uint* neighbors[6];
    const char* near;       //obstacleNear, or nullptr with no obstacles
    const uint* opens;      //obstacleOpen
    unsigned int walls;     //the domain's walls that let go, a bit each in the neighbours' order
};

//whether the solve may let an unknown go: it's on something solid, and everything solid it's on lets go. The domain's wall, where the voxel is the
//last one that way and its face there is a wall's; an obstacle, where one covers part of it or of any face of it
__device__ inline bool mayLetGo(const Boundaries& b, uint index, int3 voxel, int3 domainVoxels){
    int at[3] = {voxel.x, voxel.y, voxel.z};
    int size[3] = {domainVoxels.x, domainVoxels.y, domainVoxels.z};
    bool solid = false, held = false;
    #pragma unroll
    for(int face = 0; face < 6; ++face){
        if(at[face/2] == (face % 2 ? size[face/2] - 1 : 0) && b.neighbors[face][index] == WALL_VOXEL
           && !(b.near != nullptr && (b.near[index] & CLOSED_FACE << face))){     //the wall's own face, not one an obstacle against the wall has closed
            solid = true;
            held = held || !(b.walls >> face & 1);
        }
    }
    if(b.near != nullptr && (b.near[index] & NEAR_SURFACE)){
        const uint* opens = b.opens + 4*(size_t)index;
        bool closed = b.near[index] & 63*CLOSED_FACE;   //a face that's closed, or leads into a solid voxel (wallOffSolidNeighbors)
        if(closed || opens[0] != 0xFFFFFFFFu || opens[1] != 0xFFFFFFFFu || opens[2] != 0xFFFFFFFFu || opens[3] != 0){   //or one not all open, or some of it covered
            solid = true;
            held = held || !(b.near[index] & LETS_GO);
        }
    }
    return solid && !held;
}

//Whether air is beside an unknown on an obstacle with none of its own faces showing it. An empty voxel with its centre in an obstacle is solid to the
//solve (findSolidVoxels) and the face to it closed (wallOffSolidNeighbors), though part of it is outside the obstacle, with no liquid in it: the
//face's open fraction is still what the surface leaves of it. Under water that part is too thin for a particle, and there's no air in it. At the
//liquid's edge it's where the air is: liquid under a flat obstacle that isn't on the voxels' planes has its whole top layer in such voxels, and
//every voxel along its edge has them beside it and liquid below. So air is there if the unknown next to this one across another axis has air
//across the same face: that air is against the empty voxel too
__device__ inline bool airPastSolid(const Boundaries& b, const char* solveCodes, uint index){
    if(b.near == nullptr || !(b.near[index] & NEAR_SURFACE)){
        return false;
    }
    for(int face = 0; face < 6; ++face){
        uint word = b.opens[4*(size_t)index + face/2];
        if(!(b.near[index] & CLOSED_FACE << face) || (face % 2 ? word >> 16 : word & 0xFFFFu) == 0){
            continue;   //open, or all of it in the obstacle
        }
        for(int other = 0; other < 6; ++other){
            uint beside = b.neighbors[other][index];
            if(other/2 != face/2 && beside < WALL_VOXEL && solveCodes[beside]){
                uint past = b.neighbors[face][beside];
                if(past == NO_VOXEL || (past < WALL_VOXEL && !solveCodes[past])){
                    return true;
                }
            }
        }
    }
    return false;
}

//letGo's values while the voxels to let go are being found
static constexpr char HELD = 0;
static constexpr char LET_GO = 1;
static constexpr char PULLED = 2;   //pulled on, and no air has got to it yet

//After the solve with every voxel held, a block per node of this partition's: the unknowns on boundaries that let go that it left under the air's
//pressure, the ones with a face to air (which their own coefficients hold: Stencil::air) or air beside them past a voxel that's solid only to the
//solve (airPastSolid) let go, the rest pulled on, until air reaches them or doesn't. found counts each
__global__ void findPulled(VoxelPlaces places, Boundaries b, const char* solveCodes, Stencil A, const float* p, char* letGo, uint* found){
    uint node = blockIdx.x;
    uint start = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
    uint end = places.nodeVoxelEnds[node];
    uint cell = places.nodeCells[node];
    int3 domainVoxels = make_int3(places.grid.sizeX*places.interiorWidth, places.grid.sizeY*places.interiorWidth, places.grid.sizeZ*places.interiorWidth);
    uint mine[2] = {0, 0};      //let go, and pulled on
    for(uint index = start + threadIdx.x; index < end; index += blockDim.x){
        char state = HELD;
        if(solveCodes[index] && p[index] < 0.0f && mayLetGo(b, index, places.voxelOf(cell, places.voxelSlots[index]), domainVoxels)){
            state = A.air(index) > 0.0f || airPastSolid(b, solveCodes, index) ? LET_GO : PULLED;
            ++mine[state == PULLED];
        }
        letGo[index] = state;
    }
    if(mine[0] > 0){
        atomicAdd(found, mine[0]);
    }
    if(mine[1] > 0){
        atomicAdd(found + 1, mine[1]);
    }
}

//The air's way in: a voxel pulled on with a neighbour that's let go is let go. Run until none is, each thread writing its own voxel's state and
//reading its neighbours' as they stand, so how far the air gets in one pass depends on the order the threads run in, and where it ends up doesn't
__global__ void spreadLetGo(uint numVoxels, Stencil A, char* letGo, uint* spread){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels && letGo[index] == PULLED){
        #pragma unroll
        for(int face = 0; face < 6; ++face){
            if(A.A[face][index] != 0.0f && letGo[A.neighbors[face][index]] == LET_GO){     //coupled, so an unknown beside it
                letGo[index] = LET_GO;
                atomicAdd(spread, 1u);
                break;
            }
        }
    }
}

//the voxels still only pulled on, which no air reached, are held
__global__ void holdPulled(uint numVoxels, char* letGo){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels && letGo[index] == PULLED){
        letGo[index] = HELD;
    }
}

//after the solve that keeps the voxels air reached from going under its pressure: the ones it left at it are let go, the ones over it held
__global__ void keepLetGo(uint numVoxels, const float* p, char* letGo){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels && letGo[index] && p[index] != 0.0f){
        letGo[index] = HELD;
    }
}

bool Particles::lettingGo() const{
    return !twoPhase.on && (wallsLetGo != 0 || obstacles.lettingGo());
}

//before a solve: every voxel is held
void Particles::holdEverything(){
    anyLetGo = false;
    if(lettingGo() && letGo.size() > 0){
        letGo.zeroDeviceAsync(stream);
    }
}

//the stencil the kernels here read the equations by
static Stencil stencilOf(const CudaVec<uint>& neighborNx, const CudaVec<uint>& neighborPx, const CudaVec<uint>& neighborNy, const CudaVec<uint>& neighborPy, const CudaVec<uint>& neighborNz,
                         const CudaVec<uint>& neighborPz, const CudaVec<float>& Anx, const CudaVec<float>& Apx, const CudaVec<float>& Any, const CudaVec<float>& Apy, const CudaVec<float>& Anz,
                         const CudaVec<float>& Apz, const CudaVec<float>& Adiag){
    return {{neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr()},
            {Anx.devPtr(), Apx.devPtr(), Any.devPtr(), Apy.devPtr(), Anz.devPtr(), Apz.devPtr()}, Adiag.devPtr()};
}

//After the solve with every voxel held: lets go of the voxels it left pulled on that air can reach, and says whether there are any, in any partition:
//if so the rest are solved for again. It waits for the GPU for that answer, and again for every SPREAD_PASSES voxels the air has to go along a
//boundary from where it gets in: only with boundaries that let go
bool Particles::letGoOfSuction(){
    static const int SPREAD_PASSES = 8;
    uint numVoxels = voxelIDsUsed.size();
    Stencil A = stencilOf(neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag);
    uint* counts = nullptr;     //on the device: the voxels let go for having a face to air and the rest that are pulled on, or how many each pass of the air's spread reaches
    uint seen[SPREAD_PASSES] = {};
    static_assert(SPREAD_PASSES >= 2, "counts holds findPulled's two");
    gpuErrchk(cudaMallocAsync((void**)&counts, sizeof(seen), stream));
    gpuErrchk(cudaMemsetAsync(counts, 0, sizeof(seen), stream));
    if(numOwnNodes > 0 && numVoxels > 0){
        bool beside = obstacles.count() > 0 && numStoredNodes > 0;
        Boundaries b = {{neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr()},
                        beside ? obstacleNear.devPtr() : nullptr, beside ? obstacleOpen.devPtr() : nullptr, wallsLetGo};
        findPulled<<<numOwnNodes, LET_GO_THREADS, 0, stream>>>(voxelPlaces(), b, solveCodes.devPtr(), A, p.devPtr(), letGo.devPtr(), counts);
    }
    gpuErrchk(cudaMemcpyAsync(seen, counts, sizeof(seen), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    bool any = context->anyOverPartitions(seen[0] > 0);
    bool spreading = context->anyOverPartitions(seen[1] > 0) && any;    //some are pulled on that air hasn't reached, and it has got in somewhere
    while(spreading){   //until a pass reaches no more, anywhere
        gpuErrchk(cudaMemsetAsync(counts, 0, sizeof(seen), stream));
        for(int pass = 0; pass < SPREAD_PASSES; ++pass){
            context->fillGhosts(letGo.devPtr(), stream);
            if(numVoxels > 0){
                spreadLetGo<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, A, letGo.devPtr(), counts + pass);
            }
        }
        gpuErrchk(cudaMemcpyAsync(seen, counts, sizeof(seen), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        spreading = context->anyOverPartitions(seen[SPREAD_PASSES - 1] > 0);
    }
    gpuErrchk(cudaFreeAsync(counts, stream));
    if(numVoxels > 0){
        holdPulled<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, letGo.devPtr());     //which no air reached
        gpuErrchk(cudaPeekAtLastError());
    }
    if(any){
        context->fillGhosts(letGo.devPtr(), stream);
        anyLetGo = true;
    }
    return any;
}

//Once the solve that keeps them from going under the air's pressure is done: of the voxels it was given, the ones still at the air's pressure are
//the ones let go, and the rest are held again. The ghosts' come out as their owners', from their owners' pressures
void Particles::keepLetGo(){
    uint numVoxels = voxelIDsUsed.size();
    if(numVoxels > 0){
        ::keepLetGo<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, p.devPtr(), letGo.devPtr());
        gpuErrchk(cudaPeekAtLastError());
    }
}
