//Copyright 2023 Aberrant Behavior LLC

//Two phases (TwoPhase, particles.hu): air around the liquid, simulated with it. Experimental, after PF-FLIP (Braun, Bender and Thuerey 2025).
//
//The air is particles like the liquid's, each weighing 1/densityRatio of a liquid one. P2G (particles.cu) weighs every particle by its fluid, so a face's
//velocity is its momentum over its mass, the two fluids' together, and it keeps the plain weights too. From those, or from the liquid's level set, every
//face gets a lightness here: the liquid's density over the face's own, from 1 where it's all liquid to the density ratio where it's all air. Then:
//1. Each face of the pressure equations weighs its lightness times what it did (weighDensityFaces): the equations are the variable-density ones,
//   div((1/rho) grad p) = div u / dt, still symmetric and positive definite, so every solver takes them as they are: the multigrid's coarse grids
//   sum their equations from the unknowns' (multigridFunctions.cu), so they weigh what these faces do.
//2. The velocity update pushes each face by its lightness times what it did (lightenFaceUpdates, after cudaVelocityUpdate), so the velocities are the ones
//   the equations solved for.
//Nothing else changes: gravity accelerates both fluids alike and the pressure's answer to it differs with their densities, which is all buoyancy is.
//
//Where a face's lightness comes from (FaceDensity):
//- fractions: the plain weight over the mass, which is linear in the share of the weight on the face that's the liquid's.
//- phaseField: PF-FLIP's. The face's mass against what a voxel full of liquid at rest would put there, less a floor that keeps bunched-up air from
//  reading as liquid, square-rooted and clamped to 0..1; and a face between two voxels that are both liquid by it, or both air, is all one or the other.
//- levelSet: the share of the line between the two voxels' centres that's inside the liquid's level set (levelset.cu, built from the liquid's particles
//  alone and smoothed as the sharp free surface's is), the ghost fluid method's face density.
//- synthetic: from where the face is and nothing else, with every particle liquid. The fluid then flows through a pattern of densities fixed in space,
//  which means nothing physically, but hands the pressure solve the same kind of equations with no air particles needed: a test of the solve alone.
//
//Every voxel's values come from what its partition holds alike, so they're the same bit for bit however the nodes are split.

#include "particles.hu"
#include "liquidTile.hu"
#include "algorithms/voxelSolveFunctions.hu"   //NO_VOXEL, WALL_VOXEL
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>

static constexpr float PHASE_SHARPNESS = 1.0f;  //PF-FLIP's alpha: less and the phase field goes from air to liquid over less mass
static constexpr uint PLACE_THREADS = 128;      //per node block, where a kernel needs to know where each voxel is

//the liquid's density over a face's, when the share theta of the face is liquid and the rest air: 1 to ratio
__device__ inline float lightnessOf(float theta, float ratio){
    return 1.0f / (theta + (1.0f - theta)/ratio);
}

//fractions. Each stored voxel's three lower faces from P2G's sums: the plain weight over the mass is the liquid's density over the face's. A face no
//particle reached takes what its voxel's other two faces come to together, and with nothing on those either it's air
__global__ void lightnessFromFractions(uint numVoxels, const float* massX, const float* massY, const float* massZ, const int* plainX, const int* plainY, const int* plainZ,
                                       float plainUnit, float ratio, float* lightX, float* lightY, float* lightZ){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        const float* mass[3] = {massX, massY, massZ};
        const int* plain[3] = {plainX, plainY, plainZ};
        float* light[3] = {lightX, lightY, lightZ};
        float masses[3], plains[3];
        float allMass = 0.0f, allPlain = 0.0f;
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            masses[axis] = mass[axis][index];
            plains[axis] = plain[axis][index]*plainUnit;
            allMass += masses[axis];
            allPlain += plains[axis];
        }
        float unreached = allMass > 0.0f ? fminf(fmaxf(allPlain / allMass, 1.0f), ratio) : ratio;
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            light[axis][index] = masses[axis] > 0.0f ? fminf(fmaxf(plains[axis] / masses[axis], 1.0f), ratio) : unreached;
        }
    }
}

//phaseField, pass 1: each face's phase, 0 air to 1 liquid, from its mass alone (PF-FLIP's equation 7, with the liquid's density 1 and a voxel at rest
//holding rest particles): sqrt((mass/rest - floor)/sharpness), clamped, where the floor is log(ratio) times what air at rest puts there
__global__ void phaseOfFaces(uint numVoxels, const float* massX, const float* massY, const float* massZ, float rest, float ratio, float* phaseX, float* phaseY, float* phaseZ){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        const float* mass[3] = {massX, massY, massZ};
        float* phase[3] = {phaseX, phaseY, phaseZ};
        float floor = logf(ratio) / ratio;
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            float over = mass[axis][index] / rest - floor;
            phase[axis][index] = over > 0.0f ? fminf(sqrtf(over / PHASE_SHARPNESS), 1.0f) : 0.0f;
        }
    }
}

//pass 2: whether each stored voxel is liquid: the mean of its faces' phases is at least a half. An unknown has all six, its upper ones its upper
//neighbours'; anything else goes by the three it stores
__global__ void tagLiquidVoxels(uint numVoxels, const char* solveCodes, const uint* neighborPx, const uint* neighborPy, const uint* neighborPz,
                                const float* phaseX, const float* phaseY, const float* phaseZ, char* liquid){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        const uint* upper[3] = {neighborPx, neighborPy, neighborPz};
        const float* phase[3] = {phaseX, phaseY, phaseZ};
        float sum = 0.0f;
        int faces = 0;
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            sum += phase[axis][index];
            ++faces;
            if(solveCodes[index] && upper[axis][index] < WALL_VOXEL){
                sum += phase[axis][upper[axis][index]];
                ++faces;
            }
        }
        liquid[index] = sum >= 0.5f*faces;
    }
}

//pass 3: each face's lightness from its phase, in place; all liquid between two liquid voxels and all air between two of air
__global__ void lightnessFromPhases(uint numVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborNy, const uint* neighborNz, const char* liquid,
                                    float ratio, float* lightX, float* lightY, float* lightZ){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        const uint* lower[3] = {neighborNx, neighborNy, neighborNz};
        float* light[3] = {lightX, lightY, lightZ};
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            uint below = lower[axis][index];    //an unknown's lower neighbour, or for an air voxel above an unknown, that unknown
            bool belowLiquid = below < WALL_VOXEL ? liquid[below] : liquid[index];
            float phase = liquid[index] && belowLiquid ? 1.0f : !liquid[index] && !belowLiquid ? 0.0f : light[axis][index];
            light[axis][index] = lightnessOf(phase, ratio);
        }
    }
}

//levelSet. Each face's liquid share is how much of the line between the two voxels' centres the level set puts inside the liquid (negative inside): all
//or none where both ends agree, and otherwise where it crosses 0 between them
__global__ void lightnessFromLevelSet(uint numVoxels, const uint* neighborNx, const uint* neighborNy, const uint* neighborNz, const float* level, float ratio,
                                      float* lightX, float* lightY, float* lightZ){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        const uint* lower[3] = {neighborNx, neighborNy, neighborNz};
        float* light[3] = {lightX, lightY, lightZ};
        float here = level[index];
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            uint below = lower[axis][index];
            float there = below < WALL_VOXEL ? level[below] : here;
            float theta = here <= 0.0f && there <= 0.0f ? 1.0f : here > 0.0f && there > 0.0f ? 0.0f : fminf(here, there) / (fminf(here, there) - fmaxf(here, there));
            light[axis][index] = lightnessOf(theta, ratio);
        }
    }
}

//synthetic: whether a point in the world is air: above a height, inside a ball, or inside the nearest of a lattice of balls
__device__ inline bool syntheticAir(float3 point, TwoPhase settings){
    float3 from = make_float3(point.x - settings.syntheticCentre.x, point.y - settings.syntheticCentre.y, point.z - settings.syntheticCentre.z);
    if(settings.syntheticShape == 0){
        return from.y > 0.0f;
    }
    if(settings.syntheticShape == 2){
        float s = settings.syntheticSpacing;
        from = make_float3(from.x - s*rintf(from.x/s), from.y - s*rintf(from.y/s), from.z - s*rintf(from.z/s));
    }
    return from.x*from.x + from.y*from.y + from.z*from.z < settings.syntheticRadius*settings.syntheticRadius;
}

//synthetic, a block per node: each stored voxel's three lower faces by where their centres are
__global__ void lightnessFromPlace(VoxelPlaces places, TwoPhase settings, float* lightX, float* lightY, float* lightZ){
    uint first = blockIdx.x == 0 ? 0 : places.nodeVoxelEnds[blockIdx.x - 1];
    uint last = places.nodeVoxelEnds[blockIdx.x];
    uint cell = places.nodeCells[blockIdx.x];
    float* light[3] = {lightX, lightY, lightZ};
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        int3 voxel = places.voxelOf(cell, places.voxelSlots[index]);
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            float3 centre = places.point(voxel, axis == 0 ? 0.0f : 0.5f, axis == 1 ? 0.0f : 0.5f, axis == 2 ? 0.0f : 0.5f);
            light[axis][index] = syntheticAir(centre, settings) ? settings.densityRatio : 1.0f;
        }
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
    uint numVoxels = voxelIDsUsed.size();
    if(numVoxels == 0){
        return;
    }
    uint blocks = numVoxels / BLOCKSIZE + 1;
    float* light[3] = {faceLightness[0].devPtr(), faceLightness[1].devPtr(), faceLightness[2].devPtr()};
    float ratio = twoPhase.densityRatio;
    switch(twoPhase.faceDensity){
        case FaceDensity::fractions:
            lightnessFromFractions<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), (const int*)plainWeights[0].devPtr(),
                (const int*)plainWeights[1].devPtr(), (const int*)plainWeights[2].devPtr(), plainWeightUnit, ratio, light[0], light[1], light[2]);
            break;
        case FaceDensity::phaseField:{
            char* liquid;
            gpuErrchk(cudaMallocAsync((void**)&liquid, numVoxels, stream));
            phaseOfFaces<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), (float)restParticlesPerVoxel, ratio,
                light[0], light[1], light[2]);
            tagLiquidVoxels<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborPx.devPtr(), neighborPy.devPtr(), neighborPz.devPtr(), light[0], light[1], light[2], liquid);
            lightnessFromPhases<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborNy.devPtr(), neighborNz.devPtr(), liquid, ratio,
                light[0], light[1], light[2]);
            gpuErrchk(cudaFreeAsync(liquid, stream));
            break;
        }
        case FaceDensity::levelSet:
            lightnessFromLevelSet<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, neighborNx.devPtr(), neighborNy.devPtr(), neighborNz.devPtr(), surfaceLevel.devPtr(), ratio,
                light[0], light[1], light[2]);
            break;
        case FaceDensity::synthetic:
            if(numStoredNodes > 0){
                lightnessFromPlace<<<numStoredNodes, PLACE_THREADS, 0, stream>>>(voxelPlaces(), twoPhase, light[0], light[1], light[2]);
            }
            break;
    }
    for(float* lightness : light){  //the ghosts take their owners', which read their own neighbours
        context->fillGhosts(lightness, stream);
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
//- markBeyondBand counts the liquid's particles in every node cell of the domain. A cell with BAND_LIQUID_PARTICLES of them holds liquid (a few stray
//  drops get no air of their own: they fly through nothing, as they do with no air at all), and rings spread out from those cells a node at a time.
//  Air in a cell past the band's last ring is flagged as removed, and goes the way a sink's particles do.
//- fillAirBand (sources.cu, in emitParticles) makes air, at rest, on the seeding lattice's points in the band's voxels that hold no particle.
//With nothing past the band to hold the air up, gravity would pour it out of the band's foot: floatInAir takes the air's own weight off every face, so
//the pressure solved for is what's over the still air's, and that is 0 past the band at every height.
//Unoptimised, and for one partition: the counts are atomics on a dense array of node cells, and the rings a sweep of that array each

static constexpr uint BAND_LIQUID_PARTICLES = 16;   //two voxels' worth at 8 a voxel

__global__ void countLiquidInCells(uint numParticles, const uint* cells, const uint* ids, const char* removed, uint* counts){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numParticles && !(ids[index] & AIR_PARTICLE) && !removed[index]){
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

//after alignParticlesToGrid, so every particle's cell is where it is now, and markRemovedParticles, which sized the flags
void Particles::markBeyondBand(){
    if(!airBand()){
        return;
    }
    if(numRanks > 1){
        std::cerr<<"Particles: the air band is for one partition yet\n";
        exit(1);
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
        countLiquidInCells<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, gridCell.devPtr(), particleIds.devPtr(), removedFlags.devPtr(), counts);
    }
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
    if(size > 0){
        markAirBeyondBand<<<size / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(size, gridCell.devPtr(), particleIds.devPtr(), bandRings.devPtr(), (char)last,
            removedFlags.devPtr());
    }
    gpuErrchk(cudaFreeAsync(counts, stream));
    gpuErrchk(cudaFreeAsync(other, stream));
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
    unsigned long long count[2] = {0, 0};
    double sum[2][4] = {};
    float fastest[2] = {0.0f, 0.0f};
    for(uint index = threadIdx.x + blockIdx.x*blockDim.x; index < numParticles; index += blockDim.x*gridDim.x){
        int phase = ids[index] >> 31;     //AIR_PARTICLE
        float speedSquared = vx[index]*vx[index] + vy[index]*vy[index] + vz[index]*vz[index];
        ++count[phase];
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
        unsigned long long counts[2];
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
