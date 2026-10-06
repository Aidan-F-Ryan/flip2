//Copyright 2023 Aberrant Behavior LLC

//Two phases (TwoPhase, particles.hu): air around the liquid, simulated with it. Experimental, after PF-FLIP (Braun, Bender and Thuerey 2025).
//
//The air is particles like the liquid's, each weighing 1/densityRatio of a liquid one. P2G (particles.cu) weighs every particle by its fluid, so a face's
//velocity is its momentum over its mass, the two fluids' together, and it keeps the liquid's own weights too. From those, or from the liquid's level set,
//every face gets a lightness here: the liquid's density over the face's own, from 1 where it's all liquid to the density ratio where it's all air. Then:
//1. Each face of the pressure equations weighs its lightness times what it did (weighDensityFaces): the equations are the variable-density ones,
//   div((1/rho) grad p) = div u / dt, still symmetric and positive definite, so every solver takes them as they are: the multigrid's coarse grids
//   sum their equations from the unknowns' (multigridFunctions.cu), so they weigh what these faces do.
//2. The velocity update pushes each face by its lightness times what it did (lightenFaceUpdates, after cudaVelocityUpdate), so the velocities are the ones
//   the equations solved for.
//Nothing else changes: gravity accelerates both fluids alike and the pressure's answer to it differs with their densities, which is all buoyancy is.
//
//Where a face's lightness comes from (FaceDensity):
//- fractions: both fluids' weight on the face over its mass, which is linear in the share of that weight that's the liquid's.
//- phaseField: PF-FLIP's. The face's mass against what a voxel full of liquid at rest would put there, less a floor that keeps bunched-up air from
//  reading as liquid, square-rooted and clamped to 0..1; and a face between two voxels that are both liquid by it, or both air, is all one or the other.
//- levelSet: the share of the line between the two voxels' centres that's inside the liquid's level set (levelset.cu, built from the liquid's particles
//  alone and smoothed as the sharp free surface's is), the ghost fluid method's face density. Beside an obstacle the level set is read clear of it
//  (levelClearOfObstacles).
//- synthetic: from where the face is and nothing else, with every particle liquid. The fluid then flows through a pattern of densities fixed in space,
//  which means nothing physically, but hands the pressure solve the same kind of equations with no air particles needed: a test of the solve alone.
//Whichever it is, the particles settle the faces that only one fluid's reach (settleFacesOfOneFluid): all liquid, or all air.
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
static constexpr float CLEAR_OF_OBSTACLES = 1.5f;   //voxels out from an obstacle's surface where the level set is read for the voxels beside it
                                                    //(levelClearOfObstacles): what a voxel's centre takes in from the particles reaches that far

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

//a point moved out along the way out of every obstacle it's inside or within clearance of: out of an inside edge's two walls, or a corner's three, one
//after another. Within a tenth of the clearance is out: the point is to be past the surface's reach, not at a distance to the bit
__device__ inline float3 clearOfObstacles(const Obstacles& obstacles, float3 point, float clearance){
    for(int wall = 0; wall < 4; ++wall){
        float distance;
        float3 normal;
        if(nearestObstacle(obstacles, point, distance, normal) < 0 || distance >= 0.9f*clearance){
            break;
        }
        point = point + (clearance - distance)*normal;
    }
    return point;
}

//the stored voxel that is a voxel of the domain, wherever its node is; NO_VOXEL where nothing is stored. cellToNode is never cleared, so an entry only
//counts if nodeCells agrees (as loadNeighborNodes has it, particles.cu)
__device__ inline uint storedVoxel(const VoxelPlaces& places, int3 voxel, uint numUsedGridNodes, const uint* cellToNode, const uint* interiorVoxels){
    if(!places.inDomain(voxel)){
        return NO_VOXEL;
    }
    int width = places.interiorWidth;
    uint cell = voxel.x/width + places.grid.sizeX*(voxel.y/width + places.grid.sizeY*(voxel.z/width));
    uint node = cellToNode[cell];
    if(node >= numUsedGridNodes || places.nodeCells[node] != cell){
        return NO_VOXEL;
    }
    return interiorVoxels[(size_t)node*width*width*width + voxel.x % width + width*(voxel.y % width + width*(voxel.z % width))];
}

//a field of the stored voxels at a point of the world, blended from the eight voxels whose centres are around it; missing where one isn't stored
__device__ inline float fieldAt(const VoxelPlaces& places, float3 point, uint numUsedGridNodes, const uint* cellToNode, const uint* interiorVoxels, const float* field, float missing){
    float h = places.voxelSize();
    float at[3] = {(point.x - places.grid.negX)/h - 0.5f, (point.y - places.grid.negY)/h - 0.5f, (point.z - places.grid.negZ)/h - 0.5f};    //in voxels, from the first centre
    int base[3];
    float along[3];
    #pragma unroll
    for(int axis = 0; axis < 3; ++axis){
        base[axis] = (int)floorf(at[axis]);
        along[axis] = at[axis] - base[axis];
    }
    float sum = 0.0f;
    #pragma unroll
    for(int corner = 0; corner < 8; ++corner){
        int dx = corner & 1, dy = corner >> 1 & 1, dz = corner >> 2;
        float weight = (dx ? along[0] : 1.0f - along[0])*(dy ? along[1] : 1.0f - along[1])*(dz ? along[2] : 1.0f - along[2]);
        if(weight != 0.0f){     //a voxel with no say isn't looked up
            uint voxel = storedVoxel(places, make_int3(base[0] + dx, base[1] + dy, base[2] + dz), numUsedGridNodes, cellToNode, interiorVoxels);
            sum += weight*(voxel != NO_VOXEL ? field[voxel] : missing);
        }
    }
    return sum;
}

//The liquid's level set at the voxels beside an obstacle, read clear of the obstacle instead: per stored voxel, its value there, or for a voxel a
//surface passes near (obstacleNear), what its values out from the obstacle make it. What a voxel's centre takes in from the particles reaches a voxel
//and a half, and beside an obstacle part of that is inside it, where there's nothing to take in: the level set puts a half space of liquid there
//(levelset.cu), which is right deep in the liquid and wrong where its surface meets the obstacle. The surface read 0.1 to 0.25 of a voxel low beside
//a pillar standing in still water, a head the solve then moves the water to answer, and at centres inside the obstacle along the inside edges of a
//container it read as outside altogether, so the faces there weighed as air with water on them, and a glass of still water climbed its own corners
//to the ceiling.
//So the voxel takes the level set from CLEAR_OF_OBSTACLES out from every obstacle's surface, along the way out from its centre, carried back to it
//along the line through its value there and a voxel further out. Flat water then stays flat up to a wall whichever way the wall leans; a film on a
//floor, or the air between a ceiling and water under it, is still there for the voxels against them; and where the surface meets the obstacle square
//on, as still water meets an upright wall, both values are the same height's, so it crosses 0 beside the obstacle exactly where it does clear of it,
//however much the level set's values are squeezed towards 0 near an obstacle (they are, for three voxels out). And its own value stands where that
//reads deeper in the liquid: the level set errs towards outside beside an obstacle, so where it says liquid, liquid is there, and water a voxel wide
//up the corner of a glass, which nothing a voxel and a half out sees, would otherwise weigh as air and keep climbing.
//A block per node storing voxels
__global__ void levelClearOfObstacles(VoxelPlaces places, Obstacles obstacles, const char* near, uint numUsedGridNodes, const uint* cellToNode, const uint* interiorVoxels,
                                      const float* level, float* clear){
    uint first = blockIdx.x == 0 ? 0 : places.nodeVoxelEnds[blockIdx.x - 1];
    uint last = places.nodeVoxelEnds[blockIdx.x];
    uint cell = places.nodeCells[blockIdx.x];
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        float value = level[index];
        if(near[index] & NEAR_SURFACE){
            float h = places.voxelSize();
            float3 centre = places.point(places.voxelOf(cell, places.voxelSlots[index]), 0.5f, 0.5f, 0.5f);
            float3 out = clearOfObstacles(obstacles, centre, CLEAR_OF_OBSTACLES*h);
            float3 way = out - centre;
            float far = sqrtf(dot(way, way));
            if(far > 0.0f){
                float there = fieldAt(places, out, numUsedGridNodes, cellToNode, interiorVoxels, level, LIQUID_BAND);
                float further = fieldAt(places, out + (h/far)*way, numUsedGridNodes, cellToNode, interiorVoxels, level, LIQUID_BAND);
                value = fminf(value, there + (there - further)*far/h);
            }
        }
        clear[index] = value;
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

//Whichever way the faces' lightness was found, the particles settle the faces that only one fluid's reach: a face with none of the air's weight on it
//is all liquid, and one with none of the liquid's is all air. P2G's sums say which exactly: a liquid particle adds the same number to a face's mass
//and to the liquid's weight there, so the two are equal to the bit until an air particle adds to the mass. Away from obstacles every source says as much
//itself, both fluids' particles reaching a voxel and a half past the surface between them. Beside an obstacle a face's mass, the phase field's measure,
//falls with how much of its reach the obstacle takes, and reads as part air where there's only water: sealed in a full box, the water set off along
//the walls at over 1 m/s (0.1 with this). The level set has its own answer there (levelClearOfObstacles), and the particles' word still stands
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
//side. A voxel obstacles close has no room for either fluid, and no share (closed not nullptr): what isn't liquid there isn't air.
//And none at all in a voxel whose faces the pressure solve has as air, every one: the grid carries no liquid there, however much of it is there. The
//level set's faces are all air around liquid too thin or too small for it to show, a sheet or a blob under two voxels or so across, and liquid on such
//faces weighs what air does and goes where the air's pressure sends it, while by its density it's liquid still: a mist that renders as water. In a dam
//break at 25 mm voxels, 2 s in, 1,900 particles hung in the air like that, gaining 0.06 m/s downwards a frame where the droplets beside them gained
//free fall's 0.41. (By how much of the faces is liquid, and not whether any is, the top of still water turned to droplets wherever it lay within half
//a voxel under a layer of voxels' centres, and air left every bubble's skin: the level set's smoothing draws that a third of a voxel inside the air)
__global__ void findLiquidShare(uint numVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy,
                                const uint* neighborNz, const uint* neighborPz, const char* near, const char* closed,
                                const float* massX, const float* massY, const float* massZ, const int* liquidX, const int* liquidY, const int* liquidZ, float liquidUnit,
                                const float* lightX, const float* lightY, const float* lightZ, float ratio, float rest, float* share){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        const uint* lower[3] = {neighborNx, neighborNy, neighborNz};
        const uint* upper[3] = {neighborPx, neighborPy, neighborPz};
        const float* mass[3] = {massX, massY, massZ};
        const int* liquid[3] = {liquidX, liquidY, liquidZ};
        const float* light[3] = {lightX, lightY, lightZ};
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
        float allAir = fminf(lightnessOf(0.0f, ratio), ratio);    //what a face with no liquid to it has, from any of the sources or from settleFacesOfOneFluid
        float ofLiquid = 0.0f, ofBoth = 0.0f;
        float room = 0.0f;      //what the faces can weigh at rest, in faces away from the walls
        bool carried = !(1.0f < allAir);    //whether any face has liquid to it; with the two fluids as dense as each other the faces can't say
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
                    carried = carried || light[axis][ends[end]] < allAir;
                    room += inside[axis][2*end]*inside[(axis + 1) % 3][1]*inside[(axis + 2) % 3][1];
                }
            }
        }
        bool beside = near != nullptr && (near[index] & NEAR_SURFACE);
        share[index] = closed != nullptr && closed[index] ? nanf("") : !carried ? 0.0f : beside ? (ofBoth > 0.0f ? ofLiquid / ofBoth : 0.0f) : ofLiquid / (room*rest);
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
    switch(twoPhase.faceDensity){
        case FaceDensity::fractions:
            lightnessFromFractions<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), (const int*)liquidWeights[0].devPtr(),
                (const int*)liquidWeights[1].devPtr(), (const int*)liquidWeights[2].devPtr(), liquidWeightUnit, ratio, light[0], light[1], light[2]);
            break;
        case FaceDensity::phaseField:{
            char* liquid;
            gpuErrchk(cudaMallocAsync((void**)&liquid, numVoxels + 1, stream));
            phaseOfFaces<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), (float)restParticlesPerVoxel, ratio,
                light[0], light[1], light[2]);
            tagLiquidVoxels<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborPx.devPtr(), neighborPy.devPtr(), neighborPz.devPtr(), light[0], light[1], light[2], liquid);
            lightnessFromPhases<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborNy.devPtr(), neighborNz.devPtr(), liquid, ratio,
                light[0], light[1], light[2]);
            gpuErrchk(cudaFreeAsync(liquid, stream));
            break;
        }
        case FaceDensity::levelSet:{
            const float* level = surfaceLevel.devPtr();
            float* clear = nullptr;
            if(beside){
                gpuErrchk(cudaMallocAsync((void**)&clear, sizeof(float)*numVoxels, stream));
                levelClearOfObstacles<<<numStoredNodes, PLACE_THREADS, 0, stream>>>(voxelPlaces(), obstacles.state(), obstacleNear.devPtr(), numUsedGridNodes, cellToNode.devPtr(),
                    nodeInteriorVoxels.devPtr(), level, clear);
                level = clear;
            }
            lightnessFromLevelSet<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, neighborNx.devPtr(), neighborNy.devPtr(), neighborNz.devPtr(), level, ratio, light[0], light[1], light[2]);
            if(clear != nullptr){
                gpuErrchk(cudaFreeAsync(clear, stream));
            }
            break;
        }
        case FaceDensity::synthetic:
            if(numStoredNodes > 0){
                lightnessFromPlace<<<numStoredNodes, PLACE_THREADS, 0, stream>>>(voxelPlaces(), twoPhase, light[0], light[1], light[2]);
            }
            break;
    }
    if(twoPhase.particles()){
        settleFacesOfOneFluid<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), (const int*)liquidWeights[0].devPtr(),
            (const int*)liquidWeights[1].devPtr(), (const int*)liquidWeights[2].devPtr(), liquidWeightUnit, ratio, light[0], light[1], light[2]);
    }
    for(float* lightness : light){  //the ghosts take their owners', which read their own neighbours
        context->fillGhosts(lightness, stream);
    }
    if(twoPhase.escaping()){    //from the faces as the ghosts' exchange left them: a voxel's upper faces are its neighbours'
        findLiquidShare<<<blocks, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(),
            neighborNz.devPtr(), neighborPz.devPtr(), beside ? obstacleNear.devPtr() : nullptr, beside ? obstacleSolids.devPtr() : nullptr, voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), (const int*)liquidWeights[0].devPtr(),
            (const int*)liquidWeights[1].devPtr(), (const int*)liquidWeights[2].devPtr(), liquidWeightUnit, light[0], light[1], light[2], ratio, (float)restParticlesPerVoxel,
            liquidShare.devPtr());
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
