//Copyright 2023 Aberrant Behavior LLC

//The sharp free surface (FreeSurface::sharp, particles.hu): the pressure solve's liquid is what the level set finds (levelset.cu), and its pressure is 0
//at the surface itself, or sigma kappa with surface tension, rather than at the centres of the air voxels past the particles' footprint.
//
//With the footprint, the default, every voxel the particles' stencils reach is an unknown, and the pressure is 0 at the centre of each air voxel past
//them: the solve's surface sits a voxel or two outside the liquid's. Liquid near the surface then moves as if it were that much deeper: a drop 16 voxels
//in radius oscillates 9% slow (2% sharp), and a pool's sloshing dies away in a few seconds where sharp it carries on.
//
//Sharp, each substep, from the level set smoothed by SURFACE_PASSES passes of blur (Particles::surfaceLevel, liquidTile.hu). The particles' unevenness
//moves the level set's own surface up and down by a tenth of a voxel or so from one column of voxels to the next, and the solve turns that into
//pressure, which kicks the liquid sideways by about g dt times as much each substep: unsmoothed, a still tank of randomly placed particles never settles.
//The blur also closes the gaps between particles deep in the liquid, which the level set can put outside it: as air, each would be a bubble at 0
//pressure under the liquid's weight, and collapse at metres a second.
//1. The unknowns whose centres are outside the liquid stop being unknowns (retireDryUnknowns): the solve's liquid is the voxels whose centres are in it.
//   Nothing's added, as the level set's liquid lies inside the footprint. Particles outside it move with the velocities step 5 gives them, with no
//   pressure to hold them apart, so an unknown packed with particles stays one whatever the level set says (dryUnknown).
//2. Each liquid unknown's face to air weighs 1/theta in its pressure equation (weighSurfaceFaces), theta being how far from its centre towards the air
//   voxel's the level set crosses 0: the pressure past the face is the one a line through the unknown's pressure and the surface's reaches there, the
//   ghost fluid method (Gibou, Fedkiw, Cheng and Kang 2002; Enright, Nguyen, Gibou and Fedkiw 2003). Only the diagonal changes, and it only grows, so
//   the equations stay symmetric and positive definite, and every solver takes them as they are. The multigrid builds its coarse grids from which voxels
//   are unknowns, as ever, and relaxes its finest grid with the equations' own diagonal, so it stays a symmetric preconditioner.
//3. With surface tension, the surface's pressure goes into the divergence through the same faces (addSurfaceDivergence), in place of the continuum
//   surface force: a drop at rest has the pressure sigma kappa throughout, and nothing moves. And the density correction spreads voxels near the surface
//   that are packed denser than rest, which the footprint never lets happen (see surfaceDivergenceKernel).
//4. The velocity update takes the same pressures past those faces (correctSurfaceFaces, after cudaVelocityUpdate), so the velocities are the ones the
//   equations solved for.
//5. The faces that particles read but no pressure reaches, around the liquid, take the liquid's velocity carried out from its faces, EXTEND_LAYERS voxels
//   deep (extendLiquidVelocities), after the solve. FLIP's old velocities stay as P2G left them there, the particles' own, so the particles just outside
//   the liquid end up with the liquid's velocity plus what each differed by from the ones around it: they move with the surface. (With the old
//   velocities carried out too, they kept their own and took only the liquid's change, and a film crept out along the floor from a drop's edge with
//   nothing to bring it back.) Further out, spray's faces keep what P2G gave them and the forces, so it flies on its own.
//
//It costs the level set every substep, which the footprint only needs with viscosity or surface tension, and a few passes over the voxels: about a quarter
//more a substep (11.5 ms against 9.2 at 1.3M particles). The liquid is damped less, so it's faster for longer, which takes more substeps (half as many
//again in a dam break); and with surface tension the capillary limit on the timestep is 1/sqrt(2) of the footprint's (Particles::capillaryDt). Every
//voxel is worked out from values its partition holds alike, so it's the same bit for bit however the nodes are split.
//
//Each part is there because the runs fail without it (2026-10-03, a still tank seeded at random and evenly, a dam break over 10 s, a drop oscillating and
//drops on the floor in zero gravity): no blur, or an odd number of passes, and the tank churns; no packed-voxel rule and the dam break's particles end up
//in a few voxels; no spreading and its pool loses a fifth of its volume to a packed top layer; no velocities carried out and everything is rough. What
//didn't earn a place: moving the solve's surface outward so that more particles are inside it, as solvers that build their level set as an envelope
//round the particles have it, only replaces the spreading at half a voxel, where the drop is 5.8% slow rather than 1.9% and the randomly seeded tank
//four times as lively, and never the packed-voxel rule; and counting every voxel that holds a particle as liquid makes each stray one a bump in the
//solve's surface, which at these timesteps throws the liquid about.
//
//Known limits. Still liquid is livelier than the footprint leaves it: a tank of randomly placed particles settles to about 2 cm/s at its fastest, against
//3 to 5 mm/s (seeded evenly, 6 mm/s against 1).

#include "particles.hu"
#include "liquidTile.hu"
#include "algorithms/voxelSolveFunctions.hu"   //NO_VOXEL, WALL_VOXEL
#include <cmath>

static constexpr float DENSE_SHARE = 2.0f;          //an unknown holding this many times the rest count of particles is liquid, whatever the level set says:
                                                    //anything from 1.5 to 4 did as well
static constexpr float NEAREST_SURFACE = 0.01f;     //theta at least this: a surface nearer a liquid voxel's centre is taken to be this share of the way out,
                                                    //so a face weighs at most 100 times what it did. Larger was no steadier: 0.5 shed more from a drop on a
                                                    //wall than 0.1
static constexpr int EXTEND_LAYERS = 3;             //voxels the liquid's velocity is carried out: a particle's stencil reaches 1.5 voxels round it, and a
                                                    //particle can be as much as a voxel outside the level set's liquid. Also the tile's halo. 2 did as
                                                    //well in every run here; with 1, a drop on a wall lost a few more particles
static constexpr int EXTEND_WIDTH = 4 + 2*EXTEND_LAYERS;    //a tile's side: a node's 4 voxels and EXTEND_LAYERS either side
static constexpr int EXTEND_SLOTS = EXTEND_WIDTH*EXTEND_WIDTH*EXTEND_WIDTH;
static constexpr uint EXTEND_THREADS = 256;
static constexpr unsigned char FACE_SOLVED = 0;     //a tile face's state: the solve gave it its velocity, the liquid's own
static constexpr unsigned char FACE_OPEN = 0x7F;    //it's to take the liquid's velocity once a face beside it has it, and then holds the layer it took it in
static constexpr unsigned char FACE_KEPT = 0xFF;    //it takes none: nothing's stored there, it's on a wall, or it's an obstacle's (obstacleGhostVelocities)

// ---- 1: the liquid's unknowns ----

//whether an unknown's centre is outside the liquid (surface, the smoothed level set), which retires it. Not one holding particles packed denser than
//any surface's: the level set goes by where the particles within reach of a centre are, not by how many, and a clump of them to one side of it can read
//as outside. As air, nothing would stop more particles falling in, and the clump would grow without end
__device__ inline bool dryUnknown(uint index, const char* solveCodes, const float* surface, const float* counts, float dense){
    return solveCodes[index] && !(surface[index] < 0.0f) && counts[index] < dense;
}

//A dry unknown becomes air, which the pressure update reaches through the lower neighbours (pressureToAcceleration): as an air voxel it holds the face
//above a liquid unknown below it, if one is there across a face obstacles don't close, and names it; and an air voxel above it that named it no longer
//does. Its upper neighbours aren't read once it isn't an unknown, and the unknowns beside it see it as air once its code is 0 (dropDryUnknowns). Only this
//voxel writes either place, and nothing it reads changes until dropDryUnknowns. Ghosts work out the same as their owners, from the same level set
__global__ void unlinkDryUnknowns(uint numVoxels, const char* solveCodes, const float* surface, const float* counts, float dense, uint* neighborNx, const uint* neighborPx,
                                  uint* neighborNy, const uint* neighborPy, uint* neighborNz, const uint* neighborPz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index >= numVoxels || !dryUnknown(index, solveCodes, surface, counts, dense)){
        return;
    }
    uint* lower[3] = {neighborNx, neighborNy, neighborNz};
    const uint* upper[3] = {neighborPx, neighborPy, neighborPz};
    #pragma unroll
    for(int axis = 0; axis < 3; ++axis){
        uint above = upper[axis][index];
        if(above < WALL_VOXEL && !solveCodes[above] && lower[axis][above] == index){
            lower[axis][above] = NO_VOXEL;
        }
        uint below = lower[axis][index];
        lower[axis][index] = below < WALL_VOXEL && solveCodes[below] && !dryUnknown(below, solveCodes, surface, counts, dense) ? below : NO_VOXEL;
    }
}

__global__ void dropDryUnknowns(uint numVoxels, const float* surface, const float* counts, float dense, char* solveCodes){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels && dryUnknown(index, solveCodes, surface, counts, dense)){
        solveCodes[index] = 0;
    }
}

// ---- 2-4: the pressure equations and the velocity update ----

//how far from a liquid unknown's centre (level here) towards a neighbour that isn't one (level there) the surface is, as a share of the way: where the
//level set crosses 0 between them. Where it doesn't, the surface is at the neighbour's centre, as the footprint has it: the neighbour's in the liquid too
//by the level set, though it's no unknown, or the unknown is outside it, and one only for the particles packed in it (dryUnknown)
__device__ inline float surfaceShare(float here, float there){
    float share = here < 0.0f && there >= 0.0f ? here / (here - there) : 1.0f;
    return fminf(fmaxf(share, NEAREST_SURFACE), 1.0f);
}

//the smoothed level set at a liquid unknown's neighbour; where nothing's stored, a voxel further out than the unknown
__device__ inline float levelBeyond(uint neighbor, float here, const float* surface){
    return neighbor < WALL_VOXEL ? surface[neighbor] : here + 1.0f;
}

//the surface's curvature where it crosses between a liquid unknown and a neighbour that isn't one, share of the way: between the two voxels' curvatures,
//or the one's that's worked out (within CURVED_WITHIN of the surface, by the level set itself)
__device__ inline float surfaceCurvature(uint index, uint neighbor, float share, const float* level, const float* curvature){
    bool atHere = fabsf(level[index]) < CURVED_WITHIN;
    bool atThere = neighbor < WALL_VOXEL && fabsf(level[neighbor]) < CURVED_WITHIN;
    if(atHere && atThere){
        return curvature[index] + share*(curvature[neighbor] - curvature[index]);
    }
    return atHere ? curvature[index] : atThere ? curvature[neighbor] : 0.0f;
}

//whether a liquid unknown's face is to air: not a wall's (the domain's, or one obstacles close) and not to another unknown
__device__ inline bool airFace(uint neighbor, const char* solveCodes){
    return neighbor != WALL_VOXEL && (neighbor == NO_VOXEL || !solveCodes[neighbor]);
}

//Each liquid unknown's faces to air in its pressure equation, a thread per voxel: cudaGetA put scale on the diagonal for each, and weighCutCells scaled
//that by how open each is, the pressure past them being 0 at the air voxel's centre. The ghost pressure takes 1/theta of that
__global__ void weighSurfaceFacesKernel(uint numVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy,
                                        const uint* neighborNz, const uint* neighborPz, const float* surface, const char* near, const uint* opens, float scale, float* Adiag){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index >= numVoxels || !solveCodes[index]){
        return;
    }
    const uint* neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
    bool cut = near != nullptr && (near[index] & NEAR_SURFACE);
    CutFaces faces;
    if(cut){
        faces = cachedCut(opens, index);
    }
    float here = surface[index];
    float more = 0.0f;
    #pragma unroll
    for(int face = 0; face < 6; ++face){
        uint neighbor = neighbors[face][index];
        if(airFace(neighbor, solveCodes)){
            float share = surfaceShare(here, levelBeyond(neighbor, here, surface));
            more += (cut ? faces.open[face] : 1.0f)*(1.0f/share - 1.0f);
        }
    }
    if(more > 0.0f){
        Adiag[index] += scale*more;
    }
}

//What the liquid near the surface adds to its divergence, a thread per voxel, once cudaCalcDivU and the obstacles' flux are in:
//- With surface tension the surface's pressure is sigma kappa, which each liquid unknown's faces to air carry in as the ghost pressure past them takes
//  it: tension kappa/theta per face, by how open it is, tension being dt (sigma/rho)/dx^2 with kappa in 1/voxels.
//- The density correction, where cudaCalcDivU leaves it out (within CORRECTION_DEPTH of the footprint's edge), for voxels packed denser than rest only:
//  particles falling through the air above the liquid, which nothing holds apart there, land in its top voxels, and with nothing to spread them those
//  pack ever tighter (the footprint holds them as they come). Spreading only, measured as cudaCalcDivU measures it, so the partly filled voxels at the
//  surface aren't taken for sparse and filled. Not in the voxels obstacles cut, whose correction obstacleFaceFlux scales
__global__ void surfaceDivergenceKernel(uint numVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy,
                                        const uint* neighborNz, const uint* neighborPz, const float* surface, const float* level, const float* curvature, const char* near,
                                        const uint* opens, float tension, const float* particleCounts, const char* footprintDepth, float restParticlesPerVoxel,
                                        float correctionRate, float* divU){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index >= numVoxels || !solveCodes[index]){
        return;
    }
    const uint* neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
    bool cut = near != nullptr && (near[index] & NEAR_SURFACE);
    float change = 0.0f;
    if(curvature != nullptr){
        CutFaces faces;
        if(cut){
            faces = cachedCut(opens, index);
        }
        float here = surface[index];
        float sum = 0.0f;
        #pragma unroll
        for(int face = 0; face < 6; ++face){
            uint neighbor = neighbors[face][index];
            if(airFace(neighbor, solveCodes)){
                float share = surfaceShare(here, levelBeyond(neighbor, here, surface));
                sum += (cut ? faces.open[face] : 1.0f)*surfaceCurvature(index, neighbor, share, level, curvature)/share;
            }
        }
        change -= tension*sum;
    }
    if(correctionRate > 0.0f && footprintDepth[index] < CORRECTION_DEPTH && !cut){
        float count = 6.0f*particleCounts[index];
        float weight = 6.0f;
        #pragma unroll
        for(int face = 0; face < 6; ++face){
            uint neighbor = neighbors[face][index];
            if(neighbor < WALL_VOXEL){
                count += particleCounts[neighbor];
                weight += 1.0f;
            }
        }
        float packed = count / (weight*restParticlesPerVoxel) - 1.0f;
        if(packed > 0.0f){
            change -= packed*correctionRate;
        }
    }
    if(change != 0.0f){
        divU[index] += change;
    }
}

//The velocity update's faces between liquid and air, with the ghost pressure past them where cudaVelocityUpdate took 0: scale p (1/theta - 1) more
//towards the air, less tension kappa/theta with surface tension (tension as surfaceDivergenceKernel's). Each voxel corrects the faces it stores, its lower
//ones, as cudaVelocityUpdate updates them: a liquid unknown's with air below it, and an air voxel's above a liquid unknown, which only those name
__global__ void correctSurfaceFacesKernel(uint numVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborNy, const uint* neighborNz, const float* surface,
                                          const float* level, const float* curvature, const float* p, float scale, float tension, float* ux, float* uy, float* uz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index >= numVoxels){
        return;
    }
    const uint* lower[3] = {neighborNx, neighborNy, neighborNz};
    float* velocities[3] = {ux, uy, uz};
    bool unknown = solveCodes[index];
    #pragma unroll
    for(int dim = 0; dim < 3; ++dim){
        uint below = lower[dim][index];
        uint liquid, other;     //the face's liquid unknown, and the voxel across the face from it
        if(unknown){
            if(!airFace(below, solveCodes)){
                continue;
            }
            liquid = index;
            other = below;
        }
        else{
            if(below >= WALL_VOXEL || !solveCodes[below]){
                continue;
            }
            liquid = below;
            other = index;
        }
        float here = surface[liquid];
        float share = surfaceShare(here, levelBeyond(other, here, surface));
        float kappa = curvature != nullptr ? surfaceCurvature(liquid, other, share, level, curvature) : 0.0f;
        float outwards = scale*p[liquid]*(1.0f/share - 1.0f) - tension*kappa/share;
        velocities[dim][index] += unknown ? -outwards : outwards;   //the air is below this voxel, or this voxel is the air above the liquid
    }
}

// ---- 5: the liquid's velocity carried out ----

//The faces around the liquid, a block per node of this partition's own on a tile of its voxels and EXTEND_LAYERS more on every side (liquidTile.hu), from
//before, the face velocities as they were (other blocks write their own nodes' meanwhile). The faces the solve gives a velocity are the liquid's: a liquid
//unknown's that aren't a wall's, and an air voxel's above a liquid unknown. Each layer, a face that isn't takes the mean of those of its six neighbours of
//the same component that are, or that took one in an earlier layer, and EXTEND_LAYERS layers leave the node's own faces as passes over the whole grid
//would. An air voxel's face on one of the domain's walls is set at rest, as cudaVelocityUpdate leaves the unknowns', and the obstacles' faces (closed, or
//with a solid voxel either side) are left for the obstacle passes, which set them after this
__global__ void extendVelocities(uint numUsedGridNodes, const uint* nodeCells, const uint* cellToNode, const uint* interiorVoxels, Grid grid, const char* solveCodes,
                                 const uint* neighborNx, const uint* neighborNy, const uint* neighborNz, const char* solid, const char* near, const float* before, uint numVoxels,
                                 float* ux, float* uy, float* uz){
    __shared__ float values[3][EXTEND_SLOTS];
    __shared__ unsigned char states[3][EXTEND_SLOTS];
    __shared__ char solidSlots[EXTEND_SLOTS];
    __shared__ uint nodes[27];
    uint cell = nodeCells[blockIdx.x];
    loadTileNodes(nodes, cell, numUsedGridNodes, nodeCells, cellToNode, grid);
    __syncthreads();
    Tile tile(cell, grid, EXTEND_WIDTH - 2*EXTEND_LAYERS, EXTEND_LAYERS);
    const uint* lower[3] = {neighborNx, neighborNy, neighborNz};
    for(int slot = threadIdx.x; slot < EXTEND_SLOTS; slot += blockDim.x){
        int3 t = tile.at(slot);
        uint voxel = tile.inDomain(t) ? tile.voxel(t, nodes, numUsedGridNodes, interiorVoxels) : NO_VOXEL;
        int3 g = tile.global(t);
        int coordinates[3] = {g.x, g.y, g.z};
        bool inSolid = voxel != NO_VOXEL && solid != nullptr && solid[voxel];
        solidSlots[slot] = inSolid;
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            unsigned char state = FACE_KEPT;
            float value = 0.0f;
            if(voxel != NO_VOXEL){
                value = before[dim*(size_t)numVoxels + voxel];
                uint below = lower[dim][voxel];
                bool liquid = solveCodes[voxel];
                if(liquid ? below != WALL_VOXEL : below < WALL_VOXEL){
                    state = FACE_SOLVED;
                }
                else if(!liquid && coordinates[dim] > 0 && !inSolid && !(near != nullptr && (near[voxel] & CLOSED_FACE << 2*dim))){
                    state = FACE_OPEN;
                }
            }
            values[dim][slot] = value;
            states[dim][slot] = state;
        }
    }
    __syncthreads();
    const int strides[3] = {1, EXTEND_WIDTH, EXTEND_WIDTH*EXTEND_WIDTH};
    bool open = false;
    bool solved = false;
    for(int slot = threadIdx.x; slot < EXTEND_SLOTS; slot += blockDim.x){
        int3 t = tile.at(slot);
        int coordinates[3] = {t.x, t.y, t.z};
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            if(states[dim][slot] == FACE_OPEN && coordinates[dim] > 0 && solidSlots[slot - strides[dim]]){    //a solid voxel below it: the obstacle's
                states[dim][slot] = FACE_KEPT;
            }
            open = open || states[dim][slot] == FACE_OPEN;
            solved = solved || states[dim][slot] == FACE_SOLVED;
        }
    }
    open = __syncthreads_or(open);
    solved = __syncthreads_or(solved);
    if(open && solved){     //no faces to fill (deep in the liquid), or nothing to fill them from (spray): it stays as it is
        for(int layer = 1; layer <= EXTEND_LAYERS; ++layer){
            //a face takes a value in this layer only if it had none, and reads only faces that had one before it: nothing read is written meanwhile
            for(int slot = threadIdx.x; slot < EXTEND_SLOTS; slot += blockDim.x){
                int3 t = tile.at(slot);
                int coordinates[3] = {t.x, t.y, t.z};
                #pragma unroll
                for(int dim = 0; dim < 3; ++dim){
                    if(states[dim][slot] != FACE_OPEN){
                        continue;
                    }
                    float sum = 0.0f;
                    int count = 0;
                    #pragma unroll
                    for(int axis = 0; axis < 3; ++axis){
                        #pragma unroll
                        for(int step = -1; step <= 1; step += 2){
                            int moved = coordinates[axis] + step;
                            if(moved < 0 || moved >= EXTEND_WIDTH){
                                continue;
                            }
                            int neighbor = slot + step*strides[axis];
                            if(states[dim][neighbor] < layer){
                                sum += values[dim][neighbor];
                                ++count;
                            }
                        }
                    }
                    if(count > 0){
                        values[dim][slot] = sum / count;
                        states[dim][slot] = (unsigned char)layer;
                    }
                }
            }
            __syncthreads();
        }
    }
    float* velocities[3] = {ux, uy, uz};
    for(int slot = threadIdx.x; slot < EXTEND_SLOTS; slot += blockDim.x){
        int3 t = tile.at(slot);
        if(!tile.interior(t)){
            continue;
        }
        uint voxel = tile.voxel(t, nodes, numUsedGridNodes, interiorVoxels);
        if(voxel == NO_VOXEL){
            continue;
        }
        int3 g = tile.global(t);
        int coordinates[3] = {g.x, g.y, g.z};
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            unsigned char state = states[dim][slot];
            if(state != FACE_SOLVED && state <= EXTEND_LAYERS){
                velocities[dim][voxel] = values[dim][slot];
            }
            else if(coordinates[dim] == 0 && !solveCodes[voxel]){   //an air voxel's face on a wall
                velocities[dim][voxel] = 0.0f;
            }
        }
    }
}

// ---- the steps the substep takes ----

//once the level set is found, before anything reads which voxels are unknowns: the solve's liquid is the unknowns whose centres are in the liquid
void Particles::retireDryUnknowns(){
    uint numVoxels = solveCodes.size();
    if(freeSurfaceMode != FreeSurface::sharp || numVoxels == 0){
        return;
    }
    float dense = (float)(DENSE_SHARE*restParticlesPerVoxel);
    unlinkDryUnknowns<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), surfaceLevel.devPtr(), particleCounts.devPtr(), dense, neighborNx.devPtr(),
        neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr());
    dropDryUnknowns<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, surfaceLevel.devPtr(), particleCounts.devPtr(), dense, solveCodes.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}

//after cudaGetA and weighCutCells, with the same scale
void Particles::weighSurfaceFaces(float scale){
    uint numVoxels = solveCodes.size();
    if(freeSurfaceMode != FreeSurface::sharp || numVoxels == 0){
        return;
    }
    bool cut = obstacles.count() > 0;
    weighSurfaceFacesKernel<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(),
        neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(), surfaceLevel.devPtr(), cut ? obstacleNear.devPtr() : nullptr, cut ? obstacleOpen.devPtr() : nullptr, scale,
        Adiag.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}

//after each cudaCalcDivU and addObstacleFlux, before balanceSealedPockets (a sealed pocket has no face to air), with the density correction's rate
void Particles::addSurfaceDivergence(float correctionRate){
    uint numVoxels = solveCodes.size();
    bool tension = surfaceTension > 0.0;
    if(freeSurfaceMode != FreeSurface::sharp || (!tension && correctionRate <= 0.0f) || numVoxels == 0){
        return;
    }
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    bool cut = obstacles.count() > 0;
    surfaceDivergenceKernel<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(),
        neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(), surfaceLevel.devPtr(), liquidLevel.devPtr(), tension ? liquidCurvature.devPtr() : nullptr,
        cut ? obstacleNear.devPtr() : nullptr, cut ? obstacleOpen.devPtr() : nullptr, (float)(dt*surfaceTension / (voxelSize*voxelSize)), particleCounts.devPtr(),
        footprintDepth.devPtr(), (float)restParticlesPerVoxel, correctionRate, divU.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}

//after each cudaVelocityUpdate, with the same dt; the ghosts take their owners' faces
void Particles::correctSurfaceFaces(){
    if(freeSurfaceMode != FreeSurface::sharp){
        return;
    }
    uint numVoxels = solveCodes.size();
    if(numVoxels > 0){
        double voxelSize = grid.cellSize / (2<<refinementLevel);
        bool tension = surfaceTension > 0.0;
        correctSurfaceFacesKernel<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborNy.devPtr(), neighborNz.devPtr(),
            surfaceLevel.devPtr(), liquidLevel.devPtr(), tension ? liquidCurvature.devPtr() : nullptr, p.devPtr(), (float)(dt/(0.014*voxelSize*voxelSize)),
            tension ? (float)(dt*surfaceTension / (voxelSize*voxelSize)) : 0.0f, voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr());
        gpuErrchk(cudaPeekAtLastError());
    }
    for(CudaVec<float>* velocity : {&voxelsUx, &voxelsUy, &voxelsUz}){
        context->fillGhosts(velocity->devPtr(), stream);
    }
}

//the liquid's velocity carried out into the faces around it, after the velocity update and before the obstacle passes, which set the faces in and
//against obstacles from the ones beside them; the ghosts take their owners' faces
void Particles::extendLiquidVelocities(){
    if(freeSurfaceMode != FreeSurface::sharp){
        return;
    }
    CudaVec<float>& ux = voxelsUx;
    CudaVec<float>& uy = voxelsUy;
    CudaVec<float>& uz = voxelsUz;
    uint numVoxels = ux.size();
    if(numOwnNodes > 0 && numVoxels > 0){
        float* before;
        gpuErrchk(cudaMallocAsync((void**)&before, 3*sizeof(float)*(size_t)numVoxels, stream));
        CudaVec<float>* velocities[3] = {&ux, &uy, &uz};
        for(int dim = 0; dim < 3; ++dim){
            gpuErrchk(cudaMemcpyAsync(before + dim*(size_t)numVoxels, velocities[dim]->devPtr(), sizeof(float)*numVoxels, cudaMemcpyDeviceToDevice, stream));
        }
        bool near = obstacles.count() > 0;
        extendVelocities<<<numOwnNodes, EXTEND_THREADS, 0, stream>>>(numUsedGridNodes, nodeCells.devPtr(), cellToNode.devPtr(), nodeInteriorVoxels.devPtr(), grid, solveCodes.devPtr(),
            neighborNx.devPtr(), neighborNy.devPtr(), neighborNz.devPtr(), near ? obstacleSolids.devPtr() : nullptr, near ? obstacleNear.devPtr() : nullptr, before, numVoxels,
            ux.devPtr(), uy.devPtr(), uz.devPtr());
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaFreeAsync(before, stream));
    }
    for(CudaVec<float>* velocity : {&ux, &uy, &uz}){
        context->fillGhosts(velocity->devPtr(), stream);
    }
}
