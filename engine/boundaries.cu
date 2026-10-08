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
//with the gap inside the voxel. The states that satisfy both everywhere are found by solving, and solving again (Particles::pressureSolve):
//  1. with every voxel held, which is the solve there always was. Voxels the boundary would have to pull on come out under the air's pressure. If
//     there are none, that's the answer, to the bit;
//  2. those of them that air can reach are let go: their pressure is the air's, as an air voxel's is, and the rest are solved for again;
//  3. a voxel let go that this leaves being squeezed, more flowing in than out, is held again; then back to 2, until none is.
//Why that ends, and at the answer: the matrix's off-diagonal coefficients are all negative, so its inverse has no negative entry, and raising any
//pressure that's given (letting go of a voxel under the air's pressure) or holding a voxel that's being squeezed (whose pressure then rises from
//the air's) can only raise every other pressure. So nothing held ever needs letting go after step 1, a voxel squeezed stays squeezed however much
//the pressures rise from there, the voxels let go only dwindle, and where they stop no voxel is pulled on and none let go is squeezed.
//
//Done as written that's five solves a substep in a dam break with every boundary letting go: letting go of a voxel that was pulling on its
//neighbours leaves some of them squeezed, holding those squeezes others, and each solve finds one round of it. But the same argument makes the
//pressures at any point in it a lower bound on the answer's, and relaxation from a lower bound (every unknown in turn given the pressure its own
//equation asks for, and a voxel let go given it only if that's over the air's) raises them towards the answer and never past it. A voxel that
//gets a pressure over the air's from a lower bound has one in the answer: it's held. So before each solve a few sweeps of that (relaxLetGo) hold
//the voxels a voxel or two of liquid decides, and the solve is for the rest, and for the pressures: 3.25 solves a substep on that dam break, 1.24
//with its ceiling alone letting go. More sweeps find more and cost as much as the solves they save. What's left is the method's own. The first
//solve can't be spared: where air gets in is judged on it (below), and where nothing is pulled on it's the answer. Each solve after it starts from
//the last one's pressures and takes four V-cycles where the first takes seven.
//
//A voxel that is let go is no unknown while the solves and the velocity update run: the solvers and the update take it for air, with no change to
//either. What they need for that is the couplings to it gone from its neighbours' equations, which then have the air's pressure across those faces
//as they have across any face to air (followLetGo). Its own equation stays where it is, unused by the solve: it's what says whether it's being
//squeezed. The sweeps want every equation whole, so after a solve that leaves any squeezed the equations are made again as they were with every
//voxel held (Particles::pressureSolve), and afterwards every voxel is an unknown again for everything else in the substep (takeHoldAgain).
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

static const uint LET_GO_THREADS = 64;      //per node block in findPulled and holdSqueezed
static const int LET_GO_SWEEPS = 8;         //of relaxation before each solve with voxels let go: more find more of the voxels to hold, fewer leave them to another solve

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

//Once a solve with voxels let go is done, a block per node of this partition's: a voxel let go whose own equation says more flows into it than
//out, with the pressures the solve found around it and the air's in it, is held again
__global__ void holdSqueezed(VoxelPlaces places, Stencil A, const float* divU, const float* p, char* letGo, uint* changed){
    uint node = blockIdx.x;
    uint start = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
    uint end = places.nodeVoxelEnds[node];
    uint mine = 0;
    for(uint index = start + threadIdx.x; index < end; index += blockDim.x){
        if(letGo[index] && divU[index] + A.rowTimes(index, p) < 0.0f){
            letGo[index] = HELD;
            ++mine;
        }
    }
    if(mine > 0){
        atomicAdd(changed, mine);
    }
}

//A half sweep of relaxation, with every voxel's equation whole: each unknown of one colour takes the pressure its own equation asks for, from its
//neighbours', which are all the other colour's. One that's let go takes it only if it's over the air's, and otherwise stays at the air's
__global__ void relaxLetGo(uint numVoxels, const char* solveCodes, char color, const char* letGo, Stencil A, const float* divU, float* p){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels && solveCodes[index] == color){
        float asked = (-divU[index] - A.offDiagonalTimes(index, p)) / A.Adiag[index];
        p[index] = letGo[index] && !(asked > 0.0f) ? 0.0f : asked;
    }
}

//after the sweeps: a voxel let go that they've given a pressure over the air's is held
__global__ void holdPressed(uint numVoxels, const float* p, char* letGo){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels && letGo[index] && p[index] > 0.0f){
        letGo[index] = HELD;
    }
}

//The codes, the pressures and the equations as the voxels let go have them, from the equations whole: one let go is no unknown, with the air's
//pressure, and its neighbours' couplings to it are gone, which leaves their own coefficients holding those faces as faces to air. Each thread writes
//its own voxel's
__global__ void followLetGo(uint numVoxels, const char* letGo, char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy,
                            const uint* neighborNz, const uint* neighborPz, float* Anx, float* Apx, float* Any, float* Apy, float* Anz, float* Apz, float* p){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        if(letGo[index]){
            solveCodes[index] = 0;
            p[index] = 0.0f;
        }
        else if(solveCodes[index]){
            const uint* neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
            float* A[6] = {Anx, Apx, Any, Apy, Anz, Apz};
            #pragma unroll
            for(int face = 0; face < 6; ++face){
                uint neighbor = neighbors[face][index];
                if(neighbor < WALL_VOXEL && letGo[neighbor]){
                    A[face][index] = 0.0f;
                }
            }
        }
    }
}

//after the velocity update, which took the voxels let go for air: their faces on walls carry no flow, as an unknown's are left
__global__ void closeLetGoFaces(uint numVoxels, const char* letGo, const uint* neighborNx, const uint* neighborNy, const uint* neighborNz, float* ux, float* uy, float* uz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels && letGo[index]){
        if(neighborNx[index] == WALL_VOXEL){
            ux[index] = 0.0f;
        }
        if(neighborNy[index] == WALL_VOXEL){
            uy[index] = 0.0f;
        }
        if(neighborNz[index] == WALL_VOXEL){
            uz[index] = 0.0f;
        }
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
        if(numVoxels > 0){      //every voxel's code as it is with all of them held, to give back
            gpuErrchk(cudaMemcpyAsync(heldCodes.devPtr(), solveCodes.devPtr(), numVoxels, cudaMemcpyDeviceToDevice, stream));
        }
        context->fillGhosts(letGo.devPtr(), stream);
        anyLetGo = true;
    }
    return any;
}

//Once a solve with voxels let go is done: takes hold again of those it left squeezed, and says whether there were any, in any partition. It waits for
//the GPU for that answer
bool Particles::holdSqueezed(){
    uint numVoxels = voxelIDsUsed.size();
    uint count = 0;
    if(numOwnNodes > 0 && numVoxels > 0){
        uint* changed;
        gpuErrchk(cudaMallocAsync((void**)&changed, sizeof(uint), stream));
        gpuErrchk(cudaMemsetAsync(changed, 0, sizeof(uint), stream));
        ::holdSqueezed<<<numOwnNodes, LET_GO_THREADS, 0, stream>>>(voxelPlaces(), stencilOf(neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag),
                                                               divU.devPtr(), p.devPtr(), letGo.devPtr(), changed);
        gpuErrchk(cudaMemcpyAsync(&count, changed, sizeof(uint), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaFreeAsync(changed, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
    }
    bool any = context->anyOverPartitions(count > 0);
    if(any){
        context->fillGhosts(letGo.devPtr(), stream);
    }
    return any;
}

//Before a solve with voxels let go, with every equation whole: sweeps of relaxation over the unknowns, red then black, the ghosts taking their
//owners' pressures after each colour as the SOR has them, the voxels let go kept from going under the air's pressure. From pressures that are too
//low or right everywhere, which they are after the solve with every voxel held and after any solve since, each sweep raises them towards the
//answer and not past it, so a voxel let go that comes out of them over the air's pressure is held
void Particles::relaxLetGo(){
    uint numVoxels = voxelIDsUsed.size();
    if(numVoxels == 0){
        return;
    }
    Stencil A = stencilOf(neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz, Anx, Apx, Any, Apy, Anz, Apz, Adiag);
    for(int sweep = 0; sweep < LET_GO_SWEEPS; ++sweep){
        for(char color = 1; color <= 2; ++color){
            ::relaxLetGo<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), color, letGo.devPtr(), A, divU.devPtr(), p.devPtr());
            context->fillGhosts(p.devPtr(), stream);
        }
    }
    holdPressed<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, p.devPtr(), letGo.devPtr());   //the ghosts too, from their owners' pressures
    gpuErrchk(cudaPeekAtLastError());
}

//then, for the solve: the voxels still let go are no unknowns, and their neighbours' equations have the air there (followLetGo), in the ghosts too
void Particles::followLetGo(){
    uint numVoxels = voxelIDsUsed.size();
    if(numVoxels > 0){
        ::followLetGo<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, letGo.devPtr(), solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(),
            neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(), Anx.devPtr(), Apx.devPtr(), Any.devPtr(), Apy.devPtr(), Anz.devPtr(), Apz.devPtr(), p.devPtr());
        gpuErrchk(cudaPeekAtLastError());
    }
}

//after a solve that left voxels let go squeezed: every voxel is an unknown again, for the equations to be made whole
void Particles::unknownsAgain(){
    uint numVoxels = voxelIDsUsed.size();
    if(anyLetGo && numVoxels > 0){
        gpuErrchk(cudaMemcpyAsync(solveCodes.devPtr(), heldCodes.devPtr(), numVoxels, cudaMemcpyDeviceToDevice, stream));
    }
}

//After the velocity update: the voxels let go are unknowns again, for everything the rest of the substep does with the liquid's voxels, and their
//faces on walls are closed. Returns whether there were any: their neighbours' equations are short of them until they're assembled again
bool Particles::takeHoldAgain(){
    uint numVoxels = voxelIDsUsed.size();
    if(!anyLetGo){
        return false;
    }
    unknownsAgain();
    if(numVoxels > 0){
        closeLetGoFaces<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, letGo.devPtr(), neighborNx.devPtr(), neighborNy.devPtr(), neighborNz.devPtr(), voxelsUx.devPtr(),
            voxelsUy.devPtr(), voxelsUz.devPtr());
        gpuErrchk(cudaPeekAtLastError());
    }
    anyLetGo = false;
    return true;
}
