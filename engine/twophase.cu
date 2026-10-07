//Copyright 2023 Aberrant Behavior LLC

//Two phases (TwoPhase, particles.hu): air around the liquid, simulated with it. Experimental, after PF-FLIP (Braun, Bender and Thuerey 2025).
//
//The air is particles like the liquid's, each weighing 1/densityRatio of a liquid one. P2G (particles.cu) weighs every particle by its fluid, so a face's
//velocity is its momentum over its mass, the two fluids' together, and it keeps the liquid's own weights too. From those every face gets a lightness
//here: the liquid's density over the face's own, from 1 where it's all liquid to the density ratio where it's all air. Then:
//1. Each face of the pressure equations weighs its lightness times what it did (weighDensityFaces): the equations are the variable-density ones,
//   div((1/rho) grad p) = div u / dt, still symmetric and positive definite, so every solver takes them as they are: the multigrid's coarse grids
//   sum their equations from the unknowns' (multigridFunctions.cu), so they weigh what these faces do.
//2. The velocity update pushes each face by its lightness times what it did (lightenFaceUpdates, after cudaVelocityUpdate), so the velocities are the ones
//   the equations solved for.
//Nothing else changes: gravity accelerates both fluids alike and the pressure's answer to it differs with their densities, which is all buoyancy is.
//The liquid's viscosity takes the same lightness: the viscous step weighs each face's inertia by it (weighFacesAsHeld, viscosity.cu), and where the
//liquid is, for its stretches and shears, is its level set's to say (levelset.cu), which only the liquid's particles on the grid make. Its surface
//tension goes by the particles themselves and the energy of the surface between the two fluids (tensionOnMixture, below).
//
//A face's lightness is both fluids' weight on it over its mass, which is linear in the share of that weight that's the liquid's: the face's density is
//the mass on it over its volume, so the pressure's impulse changes the face's momentum by exactly itself. (Two other sources were built and dropped on
//2026-10-05: PF-FLIP's phase field from the face's mass alone, which stayed noisy at rest, and the share of the segment between the voxels' centres
//inside the liquid's level set, which erased liquid under two voxels thick and left it weightless in the air: a mist that rendered as water.) The
//particles settle the faces that only one fluid's reach (settleFacesOfOneFluid): all liquid, or all air.
//
//Every voxel's values come from what its partition holds alike, so they're the same bit for bit however the nodes are split.

#include "particles.hu"
#include "algorithms/voxelSolveFunctions.hu"   //NO_VOXEL, WALL_VOXEL
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>

//the liquid's density over a face's, when the share theta of the face is liquid and the rest air: 1 to ratio
__device__ inline float lightnessOf(float theta, float ratio){
    return 1.0f / (theta + (1.0f - theta)/ratio);
}

//fractions. Each stored voxel's three lower faces from P2G's sums: a face's mass is its liquid's weight and its air's over the ratio, so both fluids'
//weight there over its mass is the liquid's density over the face's. A face no particle reached takes what its voxel's other two faces come to
//together, and with nothing on those either it's air
__global__ void lightnessFromFractions(uint numVoxels, const float* massX, const float* massY, const float* massZ, const int* liquidX, const int* liquidY, const int* liquidZ,
                                       float liquidUnit, float ratio, float* lightX, float* lightY, float* lightZ){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        const float* mass[3] = {massX, massY, massZ};
        const int* liquid[3] = {liquidX, liquidY, liquidZ};
        float* light[3] = {lightX, lightY, lightZ};
        float masses[3], weights[3];
        float allMass = 0.0f, allWeight = 0.0f;
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            masses[axis] = mass[axis][index];
            float ofLiquid = liquid[axis][index]*liquidUnit;
            weights[axis] = ofLiquid + (masses[axis] - ofLiquid)*ratio;
            allMass += masses[axis];
            allWeight += weights[axis];
        }
        float unreached = allMass > 0.0f ? fminf(fmaxf(allWeight / allMass, 1.0f), ratio) : ratio;
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            light[axis][index] = masses[axis] > 0.0f ? fminf(fmaxf(weights[axis] / masses[axis], 1.0f), ratio) : unreached;
        }
    }
}

//The particles settle the faces that only one fluid's reach: a face with none of the air's weight on it is all liquid, and one with none of the
//liquid's is all air. P2G's sums say which exactly: a liquid particle adds the same number to a face's mass and to the liquid's weight there, so the
//two are equal to the bit until an air particle adds to the mass. Away from obstacles the fractions say as much themselves, both fluids' particles
//reaching a voxel and a half past the surface between them; beside an obstacle a face's mass falls with how much of its reach the obstacle takes,
//and read against a full face's it would be part air where there's only water: sealed in a full box, the water set off along the walls at over 1 m/s
//(0.1 with this)
__global__ void settleFacesOfOneFluid(uint numVoxels, const float* massX, const float* massY, const float* massZ, const int* liquidX, const int* liquidY, const int* liquidZ,
                                      float liquidUnit, float ratio, float* lightX, float* lightY, float* lightZ){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        const float* mass[3] = {massX, massY, massZ};
        const int* liquid[3] = {liquidX, liquidY, liquidZ};
        float* light[3] = {lightX, lightY, lightZ};
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            float ofLiquid = liquid[axis][index]*liquidUnit;
            if(ofLiquid > 0.0f && mass[axis][index] == ofLiquid){
                light[axis][index] = 1.0f;
            }
            else if(ofLiquid == 0.0f && mass[axis][index] > 0.0f){
                light[axis][index] = ratio;
            }
        }
    }
}

//how much of a face's reach along an axis lies past a wall that many half voxels from it: P2G's quadratic B-spline reaches a voxel and a half each way
__device__ inline float pastWall(int halfVoxels){
    return halfVoxels == 0 ? 0.5f : halfVoxels == 1 ? 1.0f/6.0f : halfVoxels == 2 ? 1.0f/48.0f : 0.0f;
}

//For escaping particles (particles.hu): each stored voxel's share of liquid: the liquid's density there over the density at rest, the mean over its
//faces of the liquid's P2G weight on them. An unknown has all six faces, its upper ones its upper neighbours'; anything else goes by the three it
//stores. The weight a face can have at rest is less beside a wall of the domain, where part of its reach has no particles to it: half of it for a face
//on the wall, a sixth across a face half a voxel off, a 48th a voxel off. Each face counts for what's left, or water against a wall would read as
//four fifths water, two thirds along an edge and half in a corner: air there would never be a bubble, and a bubble that came there would stop being
//one. In a voxel an obstacle's surface passes near (near not nullptr), part of each face's reach is inside the obstacle the same way, with no telling
//how much: there it's over what both fluids weigh on the faces instead, the room the obstacle leaves. Against the density at rest, the half voxel of
//water over a paddle's top read as too thin for the grid, and so did the water in the voxels a pillar cuts at the waterline: droplets, falling down its
//side. A voxel obstacles close has no room for either fluid, and no share (closed not nullptr): what isn't liquid there isn't air
__global__ void findLiquidShare(uint numVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy,
                                const uint* neighborNz, const uint* neighborPz, const char* near, const char* closed,
                                const float* massX, const float* massY, const float* massZ, const int* liquidX, const int* liquidY, const int* liquidZ, float liquidUnit,
                                float ratio, float rest, float* share){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        const uint* lower[3] = {neighborNx, neighborNy, neighborNz};
        const uint* upper[3] = {neighborPx, neighborPy, neighborPz};
        const float* mass[3] = {massX, massY, massZ};
        const int* liquid[3] = {liquidX, liquidY, liquidZ};
        float inside[3][3];     //along each axis, how much of the reach of the voxel's lower face, of its centre and of its upper face is inside the walls
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            int low = 2, high = 2;  //the voxels between this one and the wall below it, and above: 2 for any more, which no face's reach crosses
            if(solveCodes[index]){
                uint below = lower[axis][index], above = upper[axis][index];
                low = below == WALL_VOXEL ? 0 : below < WALL_VOXEL && lower[axis][below] == WALL_VOXEL ? 1 : 2;
                high = above == WALL_VOXEL ? 0 : above < WALL_VOXEL && upper[axis][above] == WALL_VOXEL ? 1 : 2;
            }
            inside[axis][0] = 1.0f - pastWall(2*low) - pastWall(2*high + 2);
            inside[axis][1] = 1.0f - pastWall(2*low + 1) - pastWall(2*high + 1);
            inside[axis][2] = 1.0f - pastWall(2*low + 2) - pastWall(2*high);
        }
        float ofLiquid = 0.0f, ofBoth = 0.0f;
        float room = 0.0f;      //what the faces can weigh at rest, in faces away from the walls
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            uint above = upper[axis][index];
            uint ends[2] = {index, solveCodes[index] && above < WALL_VOXEL ? above : NO_VOXEL};
            #pragma unroll
            for(int end = 0; end < 2; ++end){
                if(ends[end] != NO_VOXEL){
                    float weight = liquid[axis][ends[end]]*liquidUnit;
                    ofLiquid += weight;
                    ofBoth += weight + (mass[axis][ends[end]] - weight)*ratio;
                    room += inside[axis][2*end]*inside[(axis + 1) % 3][1]*inside[(axis + 2) % 3][1];
                }
            }
        }
        bool beside = near != nullptr && (near[index] & NEAR_SURFACE);
        share[index] = closed != nullptr && closed[index] ? nanf("") : beside ? (ofBoth > 0.0f ? ofLiquid / ofBoth : 0.0f) : ofLiquid / (room*rest);
    }
}

//Each unknown's pressure equation with every face weighing its lightness too: the row over again, from scale, how open obstacles leave each face
//(weighCutFaces' fractions) and the faces' lightness. A lower face is the unknown's own; an upper one its upper neighbour's, which is always stored
__global__ void weighDensityFacesKernel(uint numVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy,
                                        const uint* neighborNz, const uint* neighborPz, const char* near, const uint* opens, const float* lightX, const float* lightY,
                                        const float* lightZ, float* Anx, float* Apx, float* Any, float* Apy, float* Anz, float* Apz, float* Adiag, float scale){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels && solveCodes[index]){
        const uint* neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
        const float* light[3] = {lightX, lightY, lightZ};
        float* coefficients[6] = {Anx, Apx, Any, Apy, Anz, Apz};
        bool cut = near != nullptr && (near[index] & NEAR_SURFACE);
        CutFaces fractions;
        if(cut){
            fractions = cachedCut(opens, index);
        }
        float diagonal = 0.0f;
        #pragma unroll
        for(int face = 0; face < 6; ++face){
            uint neighbor = neighbors[face][index];
            if(neighbor == WALL_VOXEL){
                continue;   //no flow through it: cudaGetA left it out, and so does this
            }
            float lightness = face % 2 == 0 || neighbor == NO_VOXEL ? light[face/2][index] : light[face/2][neighbor];
            //rounded on its own, never folded into the diagonal's sum as a multiply-add: the diagonal takes the very number the coupling stores, as
            //weighCutFaces' does, so with no face to air it's the couplings' sum to the bit (Stencil::air)
            float weight = __fmul_rn(scale*lightness, cut ? fractions.open[face] : 1.0f);
            coefficients[face][index] = neighbor != NO_VOXEL && solveCodes[neighbor] ? -weight : 0.0f;
            diagonal += weight;
        }
        Adiag[index] = diagonal;
    }
}

//The velocity update over again on every face it moved, by what its lightness adds: pressureToAcceleration gave an unknown's lower face
//-scale (p - the lower neighbour's p), and an air voxel's above an unknown scale (the unknown's p); each should have been that times its lightness
__global__ void lightenFaceUpdatesKernel(uint numVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborNy, const uint* neighborNz, const float* p,
                                         const float* lightX, const float* lightY, const float* lightZ, float* ux, float* uy, float* uz, float scale){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        const uint* lower[3] = {neighborNx, neighborNy, neighborNz};
        const float* light[3] = {lightX, lightY, lightZ};
        float* u[3] = {ux, uy, uz};
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            uint below = lower[axis][index];
            float more = scale*(light[axis][index] - 1.0f);
            if(solveCodes[index]){
                if(below != WALL_VOXEL){
                    u[axis][index] -= more*(p[index] - (below == NO_VOXEL ? 0.0f : p[below]));
                }
            }
            else if(below < WALL_VOXEL){
                u[axis][index] += more*p[below];
            }
        }
    }
}

void Particles::findFaceDensities(){
    if(!twoPhase.on){
        return;
    }
    if(freeSurfaceMode == FreeSurface::sharp){
        std::cerr<<"Particles: two phases don't go with the sharp free surface yet\n";
        exit(1);
    }
    uint numVoxels = voxelIDsUsed.size();   //none in a partition the fluid hasn't reached, which still takes its turn in the ghosts' exchanges below
    uint blocks = numVoxels / BLOCKSIZE + 1;
    float* light[3] = {faceLightness[0].devPtr(), faceLightness[1].devPtr(), faceLightness[2].devPtr()};
    float ratio = twoPhase.densityRatio;
    bool beside = obstacles.count() > 0 && numStoredNodes > 0 && numVoxels > 0;    //the voxels beside obstacles are read with the obstacles in mind
    lightnessFromFractions<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), (const int*)liquidWeights[0].devPtr(),
        (const int*)liquidWeights[1].devPtr(), (const int*)liquidWeights[2].devPtr(), liquidWeightUnit, ratio, light[0], light[1], light[2]);
    settleFacesOfOneFluid<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), (const int*)liquidWeights[0].devPtr(),
        (const int*)liquidWeights[1].devPtr(), (const int*)liquidWeights[2].devPtr(), liquidWeightUnit, ratio, light[0], light[1], light[2]);
    for(float* lightness : light){  //the ghosts take their owners', which read their own neighbours
        context->fillGhosts(lightness, stream);
    }
    if(twoPhase.escaping()){    //from the faces as the ghosts' exchange left them: a voxel's upper faces are its neighbours'
        findLiquidShare<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(),
            neighborNz.devPtr(), neighborPz.devPtr(), beside ? obstacleNear.devPtr() : nullptr, beside ? obstacleSolids.devPtr() : nullptr, voxelWeightsX.devPtr(),
            voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), (const int*)liquidWeights[0].devPtr(), (const int*)liquidWeights[1].devPtr(), (const int*)liquidWeights[2].devPtr(),
            liquidWeightUnit, ratio, (float)restParticlesPerVoxel, liquidShare.devPtr());
        context->fillGhosts(liquidShare.devPtr(), stream);
    }
    gpuErrchk(cudaPeekAtLastError());
}

void Particles::weighDensityFaces(float scale){
    uint numVoxels = voxelIDsUsed.size();
    if(!twoPhase.on || numVoxels == 0){
        return;
    }
    bool cut = obstacles.count() > 0;
    weighDensityFacesKernel<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(),
        neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(), cut ? obstacleNear.devPtr() : nullptr, cut ? obstacleOpen.devPtr() : nullptr, faceLightness[0].devPtr(),
        faceLightness[1].devPtr(), faceLightness[2].devPtr(), Anx.devPtr(), Apx.devPtr(), Any.devPtr(), Apy.devPtr(), Anz.devPtr(), Apz.devPtr(), Adiag.devPtr(), scale);
    gpuErrchk(cudaPeekAtLastError());
}

void Particles::lightenFaceUpdates(float scale){
    uint numVoxels = voxelIDsUsed.size();
    if(!twoPhase.on || numVoxels == 0){
        return;
    }
    lightenFaceUpdatesKernel<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborNy.devPtr(), neighborNz.devPtr(), p.devPtr(),
        faceLightness[0].devPtr(), faceLightness[1].devPtr(), faceLightness[2].devPtr(), voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr(), scale);
    gpuErrchk(cudaPeekAtLastError());
}

//The air band (TwoPhase::band): air only within band voxels of the liquid, in whole nodes, and nothing past it, where the pressure is the open air's, 0,
//as it is past a free surface. Each initialize, once the particles have their cells:
//- findBandRings counts the liquid's particles in every node cell of the domain. A cell with BAND_LIQUID_PARTICLES of them holds liquid (a few stray
//  drops get no air of their own: they fly through nothing, as they do with no air at all), and rings spread out from those cells a node at a time.
//- emitParticles (sources.cu) then makes air, at rest, on the seeding lattice's points in the band's voxels that hold no particle.
//- markBeyondBand, at the next initialize, flags the air in a cell past the band's last ring as removed, and it goes the way a sink's particles do.
//With nothing past the band to hold the air up, gravity would pour it out of the band's foot: floatInAir takes the air's own weight off every face, so
//the pressure solved for is what's over the still air's, and that is 0 past the band at every height.
//Unoptimised: the counts are atomics on a dense array of node cells, and the rings a sweep of that array each

static constexpr uint BAND_LIQUID_PARTICLES = 16;   //two voxels' worth at 8 a voxel

__global__ void countLiquidInCells(uint numParticles, const uint* cells, const uint* ids, uint* counts){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles && !(ids[index] & (AIR_PARTICLE | ESCAPED_PARTICLE))){   //the liquid the grid carries: a droplet takes no air along
        atomicAdd(counts + cells[index], 1u);
    }
}

__global__ void startBandRings(uint numCells, const uint* counts, char* rings){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numCells){
        rings[index] = counts[index] >= BAND_LIQUID_PARTICLES ? 0 : BEYOND_BAND;
    }
}

//one ring more: a cell no ring has reached, beside one that a ring has (any of the 26 around it), is in this one
__global__ void growBandRings(Grid grid, const char* from, char* to, char ring){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < grid.sizeX*grid.sizeY*grid.sizeZ){
        char value = from[index];
        if(value == BEYOND_BAND){
            int x = index % grid.sizeX, y = index / grid.sizeX % grid.sizeY, z = index / (grid.sizeX*grid.sizeY);
            bool reached = false;
            for(int dz = -1; dz <= 1; ++dz){
                for(int dy = -1; dy <= 1; ++dy){
                    for(int dx = -1; dx <= 1; ++dx){
                        int nx = x + dx, ny = y + dy, nz = z + dz;
                        if(nx >= 0 && ny >= 0 && nz >= 0 && nx < (int)grid.sizeX && ny < (int)grid.sizeY && nz < (int)grid.sizeZ){
                            reached = reached || from[nx + grid.sizeX*(ny + grid.sizeY*nz)] != BEYOND_BAND;
                        }
                    }
                }
            }
            value = reached ? ring : BEYOND_BAND;
        }
        to[index] = value;
    }
}

__global__ void markAirBeyondBand(uint numParticles, const uint* cells, const uint* ids, const char* rings, char last, char* removed){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles && (ids[index] & AIR_PARTICLE) && rings[cells[index]] > last){
        removed[index] = 1;
    }
}

//Each node cell's ring, from the liquid every partition holds: once each holds the particles in its own planes and no others (after exchangeParticles,
//and the emitters), so each counts its own cells whole, and takes the other partitions' counts for theirs. The rings then come out the same in every
//partition, and the same however the domain is split. They serve this initialize's fill, and the next one's markBeyondBand
void Particles::findBandRings(){
    if(!airBand()){
        return;
    }
    uint numCells = grid.sizeX*grid.sizeY*grid.sizeZ;
    uint cellBlocks = numCells / BLOCKSIZE + 1;
    bandRings.resizeAsync(numCells, stream);
    uint* counts;
    char* other;
    gpuErrchk(cudaMallocAsync((void**)&counts, sizeof(uint)*numCells, stream));
    gpuErrchk(cudaMallocAsync((void**)&other, numCells, stream));
    gpuErrchk(cudaMemsetAsync(counts, 0, sizeof(uint)*numCells, stream));
    if(size > 0){
        countLiquidInCells<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, gridCell.devPtr(), particleIds.devPtr(), counts);
    }
    context->gatherNodeCells(counts, sizeof(uint), stream);
    startBandRings<<<cellBlocks, BLOCKSIZE, 0, stream>>>(numCells, counts, bandRings.devPtr());
    char* from = bandRings.devPtr();
    char* to = other;
    int last = bandRingCount();
    for(int ring = 1; ring <= last; ++ring){
        growBandRings<<<cellBlocks, BLOCKSIZE, 0, stream>>>(grid, from, to, (char)ring);
        std::swap(from, to);
    }
    if(from != bandRings.devPtr()){
        gpuErrchk(cudaMemcpyAsync(bandRings.devPtr(), from, numCells, cudaMemcpyDeviceToDevice, stream));
    }
    gpuErrchk(cudaFreeAsync(counts, stream));
    gpuErrchk(cudaFreeAsync(other, stream));
    gpuErrchk(cudaPeekAtLastError());
}

//after alignParticlesToGrid, so every particle's cell is where it is now, and markRemovedParticles, which sized the flags: the air in a cell past the
//band, by the rings the last initialize found (a substep old, which a band of whole nodes has room for; none yet at the start)
void Particles::markBeyondBand(){
    if(!airBand() || size == 0 || bandRings.size() != grid.sizeX*grid.sizeY*grid.sizeZ){
        return;
    }
    markAirBeyondBand<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, gridCell.devPtr(), particleIds.devPtr(), bandRings.devPtr(), (char)bandRingCount(),
        removedFlags.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}

//applyForces gave every face gravity's dt g. The still air's pressure answers rho_air g of that per volume, which is the share lightness/ratio of what
//the face's own fluid weighs: all of it on a face of air, a thousandth on one of water. Taken off, each face falls by what
//it weighs over the air
__global__ void floatInAirKernel(uint numVoxels, float3 fall, const char* solids, const float* lightX, const float* lightY, const float* lightZ, float* ux, float* uy, float* uz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels && !solids[index]){
        ux[index] -= fall.x*lightX[index];
        uy[index] -= fall.y*lightY[index];
        uz[index] -= fall.z*lightZ[index];
    }
}

void Particles::floatInAir(float dt){
    uint numVoxels = voxelIDsUsed.size();
    if(!airBand() || numVoxels == 0){
        return;
    }
    float share = dt / twoPhase.densityRatio;
    float3 fall = make_float3(forces.gravity.x*share, forces.gravity.y*share, forces.gravity.z*share);
    floatInAirKernel<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, fall, solids.devPtr(), faceLightness[0].devPtr(), faceLightness[1].devPtr(), faceLightness[2].devPtr(),
        voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr());
    gpuErrchk(cudaPeekAtLastError());
}

// ---- surface tension between the two fluids ----

//Without air the liquid's surface tension is a force on the faces its level set crosses, from that level set's curvature (levelset.cu). With air that
//force gives wrong results: its curvature is smoothed until a particle's place no longer shows in it, so it knows nothing of the surface under five
//or six voxels across and is out by a few percent along the rest, and the two fluids' solve, which doesn't drag on the surface's motion as the
//liquid's own does, lets what it gets wrong turn the two over along the surface: a ball at rest in zero gravity was ragged by 2 s and had shed a
//sixth of itself by 5.
//
//So with air the force comes from an energy, a phase field's (Cahn and Hilliard's). Each unknown has c, the liquid's share of both fluids' weight on
//its six faces (P2G's sums), blurred once (TENSION_BLUR), and over the unknowns
//    E = sigma dx^2 / K ( sum over voxels of W(c)  +  1/2 sum over faces of (c above - c below)^2 )  -  sigma dx^2 cos(angle) sum over wall faces of c^2 (3 - 2 c)
//The first sum is what mixing the two costs, the second what a change in c from voxel to voxel costs, and the last a wall's liking for the liquid
//(Young's): sigma cos(angle) between dry and wet, by a curve that's level at both ends, so that neither liquid deep against a wall nor a trace of it
//on a dry one is pulled at, only the surface where it meets the wall (as Jacqmin 2000 has it; taken straight in c, a 60 degree wall drew a film of
//liquid out along itself). W is the one well for which the sharpest surface the particles can make is at rest: with c(h) the share of a voxel h
//voxels inside a flat surface between all liquid and all air (shareAtDepth), W'(c(h)) = c(h + 1) - 2 c(h) + c(h - 1) (wellSlope). Then for a flat
//surface facing along an axis the energy's slope with every voxel's c, mu, is the same on every voxel that holds both fluids, wherever in the
//voxels the surface sits: nothing pushes along it, and nothing holds it to the voxels. Its energy is K per voxel face (surfaceEnergyPerFace),
//which the sigma dx^2 / K makes sigma times its area; facing other ways it's up to 1.7% more (along a voxel's long diagonal; unblurred, 4.2%). On
//a curved surface what's left of mu is the curvature's, sigma kappa in all: the pressure a drop holds. And a surface the two fluids have spread
//across, a finger of one into the other, a stray particle near it, are all away from that rest, and cost.
//
//The force is the one that energy gives, in the form that leaves the air alone (Jamet, Torres and Brackbill 2002): each of the liquid's particles
//is pushed down mu's slope where it is, -(sigma dx / particles a voxel) grad mu, and the air's not at all. Summed over what's in a voxel that's
//-c grad mu, which differs from the energy's own slope with where the fluid is, mu grad c, by a gradient: the pressure's to take. So the push lands
//on the faces by how much liquid is on them, and a face of nearly all air, a thousand times as easy to move, gets next to none; where mu is level
//there's none at all. Each particle's push goes onto the faces by the weights advection moves it with (tensionOfParticles, particles.cu), as
//momentum, and each face's velocity changes by that over its mass.
//
//Measured, a ball of liquid 16 voxels in radius in air, in zero gravity: at rest it stays at rest for 5 s, its particles' rms speed 0.010 m/s where
//the capillary speed is 0.32 (the liquid's own force with no air leaves the same), the air's 0.009; stretched and let go, it swings at 3% under
//Lamb's frequency and loses 5 to 6% of its swing each half period (with no air, 11% under and 19%). The blur is what that takes: unblurred, the
//particles' own unevenness shows in c, and a swing is spent on it within two periods; blurred twice, the frequency and the loss are no better and
//the liquid at the surface's edge is held more loosely. Half a ball of water 5 mm in radius, thick enough to settle, comes to rest on the floor
//under gravity within 13% of Young and Laplace's height and 4% of its base's radius at 60, 90 and 120 degrees (with no air, 14% and 21%).

static constexpr int TENSION_BLUR = 1;      //passes of the 1 2 1 blur over c, along each axis

//the quadratic B-spline's integral up to x: how much of a face's P2G weight comes from one side of a plane x voxels past it
__host__ __device__ inline float splineBelow(float x){
    return x <= -1.5f ? 0.0f : x <= -0.5f ? (x + 1.5f)*(x + 1.5f)*(x + 1.5f) / 6.0f : x <= 0.5f ? 1.0f/6.0f + 0.75f*(x + 0.5f) - (x*x*x + 0.125f) / 3.0f :
           x <= 1.5f ? 1.0f - (1.5f - x)*(1.5f - x)*(1.5f - x) / 6.0f : 1.0f;
}

//the liquid's share of a voxel whose centre is h voxels inside a flat surface facing along an axis, with all liquid on one side and all air on the
//other: four of its faces at its own depth and one half a voxel either way; then blurred along the axis, passes times over (blurAlong)
__host__ __device__ inline float shareAtDepth(float h, int passes){
    float share = 0.0f;
    float weight = 1.0f;    //the blur's weights along the row: the binomial coefficients of 2 passes, over 4^passes
    #pragma unroll 5
    for(int k = -passes; k <= passes; ++k){
        if(k > -passes){
            weight = weight*(passes - k + 1) / (passes + k);
        }
        float at = h + k;
        share += weight*(4.0f*splineBelow(at) + splineBelow(at - 0.5f) + splineBelow(at + 0.5f)) / 6.0f;
    }
    return share / (float)(1 << 2*passes);
}

//W'(c): the second difference of that profile at the depth where the share is c, found by halving (the profile only rises, from 0 to 1 over the
//2 + passes voxels either side of the surface)
__host__ __device__ inline float wellSlope(float c, int passes){
    float low = -2.0f - passes, high = 2.0f + passes;
    for(int halving = 0; halving < 26; ++halving){
        float middle = 0.5f*(low + high);
        if(shareAtDepth(middle, passes) < c){
            low = middle;
        }
        else{
            high = middle;
        }
    }
    float h = 0.5f*(low + high);
    return shareAtDepth(h + 1.0f, passes) - 2.0f*shareAtDepth(h, passes) + shareAtDepth(h - 1.0f, passes);
}

//K: the energy of a voxel face's worth of flat surface facing along an axis, sum of W(c) + half the squared steps of c along the row of voxels
//through it, with W the integral of W' from 0. The same wherever the surface sits in the voxels (W was made so), so it's taken on a voxel's face
static double surfaceEnergyPerFace(int passes){
    const int STEPS = 4000;     //per voxel, for W: the integral of W'(c(h)) c'(h) dh up to each voxel's depth
    int reach = 3 + passes;
    double well = 0.0, total = 0.0;
    double before = shareAtDepth(-(float)reach, passes);
    for(int step = 1; step <= 2*reach*STEPS; ++step){
        double h = -reach + (double)step / STEPS;
        double share = shareAtDepth((float)h, passes);
        double middle = h - 0.5 / STEPS;
        double slope = (double)shareAtDepth((float)(middle + 1.0), passes) - 2.0*shareAtDepth((float)middle, passes) + shareAtDepth((float)(middle - 1.0), passes);
        well += slope*(share - before);
        before = share;
        if(step % STEPS == 0){      //a voxel's centre
            double next = shareAtDepth((float)(h + 1.0), passes);
            total += well + 0.5*(next - share)*(next - share);
        }
    }
    return total;
}

//Each unknown's share of liquid, c, from both fluids' weight on its six faces: its three lower ones and its upper neighbours' lower ones, where
//those are stored (a wall's and an obstacle's closed face are no one's). And 1 over what those faces weigh at rest, which is less beside a wall,
//where part of each face's reach has no particles to it (as findLiquidShare has it). Anything else stored is past the solve, and air: 0 for both
__global__ void shareOfVoxels(uint numVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy,
                              const uint* neighborNz, const uint* neighborPz, const float* massX, const float* massY, const float* massZ, const int* liquidX,
                              const int* liquidY, const int* liquidZ, float liquidUnit, float ratio, float rest, float* share, float* perRest){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        float c = 0.0f, perWeight = 0.0f;
        if(solveCodes[index]){
            const uint* lower[3] = {neighborNx, neighborNy, neighborNz};
            const uint* upper[3] = {neighborPx, neighborPy, neighborPz};
            const float* mass[3] = {massX, massY, massZ};
            const int* liquid[3] = {liquidX, liquidY, liquidZ};
            float inside[3][3];     //along each axis, how much of the reach of the voxel's lower face, of its centre and of its upper face is inside the walls
            #pragma unroll
            for(int axis = 0; axis < 3; ++axis){
                uint below = lower[axis][index], above = upper[axis][index];
                int low = below == WALL_VOXEL ? 0 : below < WALL_VOXEL && lower[axis][below] == WALL_VOXEL ? 1 : 2;
                int high = above == WALL_VOXEL ? 0 : above < WALL_VOXEL && upper[axis][above] == WALL_VOXEL ? 1 : 2;
                inside[axis][0] = 1.0f - pastWall(2*low) - pastWall(2*high + 2);
                inside[axis][1] = 1.0f - pastWall(2*low + 1) - pastWall(2*high + 1);
                inside[axis][2] = 1.0f - pastWall(2*low + 2) - pastWall(2*high);
            }
            float ofLiquid = 0.0f, ofBoth = 0.0f;
            float room = 0.0f;      //what the faces weigh at rest, in faces away from the walls
            #pragma unroll
            for(int axis = 0; axis < 3; ++axis){
                uint above = upper[axis][index];
                uint ends[2] = {index, above < WALL_VOXEL ? above : NO_VOXEL};
                #pragma unroll
                for(int end = 0; end < 2; ++end){
                    if(ends[end] != NO_VOXEL){
                        float weight = liquid[axis][ends[end]]*liquidUnit;
                        ofLiquid += weight;
                        ofBoth += weight + (mass[axis][ends[end]] - weight)*ratio;
                        room += inside[axis][2*end]*inside[(axis + 1) % 3][1]*inside[(axis + 2) % 3][1];
                    }
                }
            }
            c = ofBoth > 0.0f ? fminf(ofLiquid / ofBoth, 1.0f) : 0.0f;
            perWeight = room > 0.0f ? 1.0f / (room*rest) : 0.0f;
        }
        share[index] = c;
        perRest[index] = perWeight;
    }
}

//One pass of a 1 2 1 blur along an axis, over the unknowns: each takes half its own value and a quarter of each neighbour's along the axis. A
//neighbour past a wall counts as the voxel's own (the field carries on square to the wall), and one that isn't an unknown as outside, whatever a
//field is there. Each pair of unknowns weighs on each other alike, so the pass is its own transpose: what carries a field's values out is also what
//carries an energy's slopes with the blurred values back onto the unblurred ones
__global__ void blurAlong(uint numVoxels, const char* solveCodes, const uint* lower, const uint* upper, const float* from, float outside, float* to){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        float value = outside;
        if(solveCodes[index]){
            float here = from[index];
            uint below = lower[index], above = upper[index];
            float under = below == WALL_VOXEL ? here : below < WALL_VOXEL && solveCodes[below] ? from[below] : outside;
            float over = above == WALL_VOXEL ? here : above < WALL_VOXEL && solveCodes[above] ? from[above] : outside;
            value = 0.5f*here + 0.25f*(under + over);
        }
        to[index] = value;
    }
}

//mu, the energy's slope with each unknown's (blurred) c: W'(c) less the sum of c's steps to its six neighbours, over K; through a wall there's no
//step, and past what the solve holds is air, c = 0. Each of its faces on one of the domain's walls takes off the wall's part: the wetting, the
//cosine of the angle the liquid meets it at, times 6 c (1 - c), the slope of c^2 (3 - 2 c), so only where the surface meets the wall; a face an
//obstacle closes (near, not nullptr with none) is a wall with no angle of its own, square on. Kept as what it's over its value deep in the liquid,
//where W' is atLiquid: only its slope pushes, and deep in the liquid, where nothing should, it's then exactly nothing, however the walls cut into
//a particle's reach. W' at all air and at all liquid are given, as most voxels are one or the other
__global__ void potentialOfVoxels(uint numVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy,
                                  const uint* neighborNz, const uint* neighborPz, const char* near, const float* share, int passes, float atAir, float atLiquid, float perFace,
                                  float wetting, float* potential){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        float value = 0.0f;
        if(solveCodes[index]){
            const uint* neighbors[6] = {neighborNx, neighborPx, neighborNy, neighborPy, neighborNz, neighborPz};
            float c = share[index];
            float steps = 0.0f;
            int walls = 0;
            #pragma unroll
            for(int face = 0; face < 6; ++face){
                uint neighbor = neighbors[face][index];
                if(neighbor == WALL_VOXEL){
                    walls += near == nullptr || !(near[index] & CLOSED_FACE << face);
                    continue;
                }
                steps += (neighbor != NO_VOXEL && solveCodes[neighbor] ? share[neighbor] : 0.0f) - c;
            }
            float well = c >= 1.0f ? atLiquid : c <= 0.0f ? atAir : wellSlope(c, passes);
            value = (well - steps - atLiquid)*perFace - wetting*walls*6.0f*c*(1.0f - c);
        }
        potential[index] = value;
    }
}

//What the liquid's particles take from each stored voxel's three lower faces: mu at the voxel the face is the lower face of and at the unknown below
//it, whose upper face it is, each over what that voxel's faces weigh at rest. Summed over the faces by a particle's weights on them, that's mu
//where the particle is, over the particles a voxel holds
__global__ void potentialOfFaces(uint numVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborNy, const uint* neighborNz, const float* potential,
                                 const float* perRest, float* ofX, float* ofY, float* ofZ){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        const uint* lower[3] = {neighborNx, neighborNy, neighborNz};
        float* ofFaces[3] = {ofX, ofY, ofZ};
        float own = solveCodes[index] ? potential[index]*perRest[index] : 0.0f;
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            uint below = lower[axis][index];
            ofFaces[axis][index] = own + (below < WALL_VOXEL && solveCodes[below] ? potential[below]*perRest[below] : 0.0f);
        }
    }
}

//The particles' forces as they landed on the faces (tensionOfParticles), 64-bit fixed-point sums, as two 32-bit parts the partitions can add up
//(reduceGhosts takes ints): the low 20 bits and the rest. A sum of a few partitions' low parts stays well inside an int
static constexpr int TENSION_LOW_BITS = 20;

__global__ void splitTensionSums(uint numVoxels, const long long* sumX, const long long* sumY, const long long* sumZ, int* high, int* low){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        const long long* sums[3] = {sumX, sumY, sumZ};
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            long long value = sums[dim][index];
            high[(size_t)dim*numVoxels + index] = (int)(value >> TENSION_LOW_BITS);
            low[(size_t)dim*numVoxels + index] = (int)(value & ((1ll << TENSION_LOW_BITS) - 1));
        }
    }
}

//each of this partition's own faces' velocity, changed by the force on it over its mass: scale is -dt (sigma/rho) / dx^2 times the particles a voxel
//holds at rest, over the fixed point's unit. A wall's faces stay at rest, and a face with no mass has no particle near enough to have pushed it
__global__ void pushFacesByTension(uint numOwnVoxels, uint numVoxels, const char* solids, const int* high, const int* low, const float* massX, const float* massY, const float* massZ,
                                   float scale, float* ux, float* uy, float* uz){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numOwnVoxels && !solids[index]){
        const float* mass[3] = {massX, massY, massZ};
        float* velocities[3] = {ux, uy, uz};
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            long long value = ((long long)high[(size_t)dim*numVoxels + index] << TENSION_LOW_BITS) + low[(size_t)dim*numVoxels + index];
            if(value != 0 && mass[dim][index] > 0.0f){
                velocities[dim][index] += scale*(float)value / mass[dim][index];
            }
        }
    }
}

//adds dt of the two fluids' surface tension to the faces (pressureSolve, in applySurfaceTension's place). Every voxel's values come from its own
//neighbours as their owners hold them, the ghosts take their owners' after each step, and the particles' forces are summed in fixed point, so the
//result is the same however the nodes are split
void Particles::tensionOnMixture(){
    uint numVoxels = voxelIDsUsed.size();
    uint blocks = numVoxels / BLOCKSIZE + 1;
    size_t count = numVoxels > 0 ? numVoxels : 1;
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    const int passes = TENSION_BLUR;
    static const double perFace = 1.0 / surfaceEnergyPerFace(passes);
    float* fields[7];       //share, perRest, potential, a spare for the blur, then the faces' potentials: x, y and z
    for(float*& field : fields){
        gpuErrchk(cudaMallocAsync((void**)&field, sizeof(float)*count, stream));
    }
    float* share = fields[0];
    float* perRest = fields[1];
    float* potential = fields[2];
    float* spare = fields[3];
    float* ofFaces[3] = {fields[4], fields[5], fields[6]};
    long long* sums[3];
    int* high;
    int* low;
    for(long long*& sum : sums){
        gpuErrchk(cudaMallocAsync((void**)&sum, sizeof(long long)*count, stream));
        gpuErrchk(cudaMemsetAsync(sum, 0, sizeof(long long)*count, stream));
    }
    gpuErrchk(cudaMallocAsync((void**)&high, 3*sizeof(int)*count, stream));
    gpuErrchk(cudaMallocAsync((void**)&low, 3*sizeof(int)*count, stream));
    const char* near = obstacles.count() > 0 ? obstacleNear.devPtr() : nullptr;
    const uint* lowers[3] = {neighborNx.devPtr(), neighborNy.devPtr(), neighborNz.devPtr()};
    const uint* uppers[3] = {neighborPx.devPtr(), neighborPy.devPtr(), neighborPz.devPtr()};
    auto blur = [&](int axis, float*& field, float outside){   //one pass along an axis, in place as far as the caller sees
        if(numVoxels > 0){
            blurAlong<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), lowers[axis], uppers[axis], field, outside, spare);
        }
        std::swap(field, spare);
        context->fillGhosts(field, stream);
    };
    if(numVoxels > 0){
        shareOfVoxels<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(),
            neighborNz.devPtr(), neighborPz.devPtr(), voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), (const int*)liquidWeights[0].devPtr(),
            (const int*)liquidWeights[1].devPtr(), (const int*)liquidWeights[2].devPtr(), liquidWeightUnit, twoPhase.densityRatio, (float)restParticlesPerVoxel, share, perRest);
    }
    context->fillGhosts(share, stream);
    context->fillGhosts(perRest, stream);
    for(int pass = 0; pass < passes; ++pass){
        for(int axis = 0; axis < 3; ++axis){
            blur(axis, share, 0.0f);
        }
    }
    if(numVoxels > 0){
        potentialOfVoxels<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(),
            neighborNz.devPtr(), neighborPz.devPtr(), near, share, passes, wellSlope(0.0f, passes), wellSlope(1.0f, passes), (float)perFace, (float)wallWetting, potential);
    }
    context->fillGhosts(potential, stream);
    for(int pass = 0; pass < passes; ++pass){   //mu, the slope with the blurred c, back onto the c the particles made: the blur's passes again, last first
        for(int axis = 2; axis >= 0; --axis){
            blur(axis, potential, 0.0f);
        }
    }
    if(numVoxels > 0){
        potentialOfFaces<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborNy.devPtr(), neighborNz.devPtr(), potential, perRest,
            ofFaces[0], ofFaces[1], ofFaces[2]);
    }
    for(float* ofFace : ofFaces){
        context->fillGhosts(ofFace, stream);
    }
    tensionOfParticles(ofFaces, sums);  //each of the liquid's particles down mu's slope, onto the faces it moves with (particles.cu)
    if(numVoxels > 0){
        splitTensionSums<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, sums[0], sums[1], sums[2], high, low);
    }
    for(int dim = 0; dim < 3; ++dim){   //particles near this partition's edge reach into ghost voxels: their owners add those sums in
        context->reduceGhosts(high + (size_t)dim*numVoxels, stream);
        context->reduceGhosts(low + (size_t)dim*numVoxels, stream);
    }
    if(numOwnVoxels > 0){
        float scale = (float)(-dt*surfaceTension / (voxelSize*voxelSize)*restParticlesPerVoxel / TENSION_UNIT);
        pushFacesByTension<<<numOwnVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numOwnVoxels, numVoxels, solids.devPtr(), high, low, voxelWeightsX.devPtr(), voxelWeightsY.devPtr(),
            voxelWeightsZ.devPtr(), scale, voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr());
    }
    for(float* field : {share, perRest, potential, spare, ofFaces[0], ofFaces[1], ofFaces[2]}){
        gpuErrchk(cudaFreeAsync(field, stream));
    }
    for(long long* sum : sums){
        gpuErrchk(cudaFreeAsync(sum, stream));
    }
    gpuErrchk(cudaFreeAsync(high, stream));
    gpuErrchk(cudaFreeAsync(low, stream));
    gpuErrchk(cudaPeekAtLastError());
    for(CudaVec<float>* velocity : {&voxelsUx, &voxelsUy, &voxelsUz}){
        context->fillGhosts(velocity->devPtr(), stream);
    }
}

//per fluid: its particles' count, position and energy sums, fastest speed, and count by height. Each thread sums a share of the particles, then adds
//its sums in: the doubles come out in whatever order the threads land, which a test's numbers can live with
__global__ void sumPhases(uint numParticles, const double* px, const double* py, const double* pz, const float* vx, const float* vy, const float* vz, const uint* ids,
                          double floorY, double perBin, unsigned long long* counts, double* sums, unsigned int* fastestBits, unsigned int* heights){
    unsigned long long count[2] = {0, 0}, escaped[2] = {0, 0};
    double sum[2][4] = {};
    float fastest[2] = {0.0f, 0.0f};
    for(uint index = threadIdx.x + blockIdx.x*blockDim.x; index < numParticles; index += blockDim.x*gridDim.x){
        int phase = ids[index] >> 31;     //AIR_PARTICLE
        float speedSquared = vx[index]*vx[index] + vy[index]*vy[index] + vz[index]*vz[index];
        ++count[phase];
        escaped[phase] += ids[index] >> 30 & 1;     //ESCAPED_PARTICLE
        sum[phase][0] += px[index];
        sum[phase][1] += py[index];
        sum[phase][2] += pz[index];
        sum[phase][3] += 0.5*speedSquared;
        fastest[phase] = fmaxf(fastest[phase], sqrtf(speedSquared));
        int bin = min(max((int)((py[index] - floorY)*perBin), 0), PHASE_HEIGHT_BINS - 1);
        atomicAdd(heights + phase*PHASE_HEIGHT_BINS + bin, 1u);
    }
    for(int phase = 0; phase < 2; ++phase){
        if(count[phase] > 0){
            atomicAdd(counts + phase, count[phase]);
            atomicAdd(counts + 2 + phase, escaped[phase]);
            for(int which = 0; which < 4; ++which){
                atomicAdd(sums + 4*phase + which, sum[phase][which]);
            }
            atomicMax(fastestBits + phase, __float_as_uint(fastest[phase]));   //a non-negative float's bits order like the float
        }
    }
}

void Particles::phaseStatistics(PhaseStatistics& liquid, PhaseStatistics& air){
    if(size == 0){
        return;
    }
    struct Totals{
        unsigned long long counts[4];   //each fluid's particles, then how many of each are off the grid
        double sums[8];
        unsigned int fastestBits[2];
        unsigned int heights[2*PHASE_HEIGHT_BINS];
    };
    Totals* device;
    Totals host;
    gpuErrchk(cudaMallocAsync((void**)&device, sizeof(Totals), stream));
    gpuErrchk(cudaMemsetAsync(device, 0, sizeof(Totals), stream));
    double height = grid.sizeY*(double)grid.cellSize;
    sumPhases<<<256, 256, 0, stream>>>(size, px.devPtr(), py.devPtr(), pz.devPtr(), vx.devPtr(), vy.devPtr(), vz.devPtr(), particleIds.devPtr(),
        grid.negY, PHASE_HEIGHT_BINS / height, device->counts, device->sums, device->fastestBits, device->heights);
    gpuErrchk(cudaMemcpyAsync(&host, device, sizeof(Totals), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaFreeAsync(device, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    PhaseStatistics* phases[2] = {&liquid, &air};
    for(int phase = 0; phase < 2; ++phase){
        phases[phase]->count += host.counts[phase];
        phases[phase]->escaped += host.counts[2 + phase];
        for(int axis = 0; axis < 3; ++axis){
            phases[phase]->positionSum[axis] += host.sums[4*phase + axis];
        }
        phases[phase]->kineticEnergy += host.sums[4*phase + 3];
        float fastest;
        memcpy(&fastest, &host.fastestBits[phase], sizeof(fastest));
        phases[phase]->fastest = fmaxf(phases[phase]->fastest, fastest);
        for(int bin = 0; bin < PHASE_HEIGHT_BINS; ++bin){
            phases[phase]->heights[bin] += host.heights[phase*PHASE_HEIGHT_BINS + bin];
        }
    }
}
