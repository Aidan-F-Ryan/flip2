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
    if(freeSurfaceMode == FreeSurface::sharp || viscosity > 0.0 || surfaceTension > 0.0){
        std::cerr<<"Particles: two phases don't go with the sharp free surface, viscosity or surface tension yet\n";
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
