//Copyright 2023 Aberrant Behavior LLC

//Whitewater (whitewater.hu). Each substep, as soon as its length is known and before the pressure solve, the surface is read for where it's breaking
//up and how many particles that makes (Particles::findWhitewater, findWhitewaterSources). Then, once the grid has its new velocities and the fluid's
//particles have moved by them (Particles::stepWhitewater),
//  - the particles there are fly through that grid for the substep, each as what it is (flyWhitewater for those in this partition's nodes, a block per
//    node as the fluid's own kernels have it; finishWhitewater for the rest, which have no grid around them, and then for what ends any of them);
//  - the new ones are made (emitWhitewater);
//  - all of them are put in the order of their node cells again, which drops the ones that are gone;
//  - and with more than one partition, those that have left this one's planes go to the partitions they're in now (exchangeWhitewater).
//Nothing here writes anything of the fluid's.
//
//Where the surface breaks. Three things make whitewater, and the velocities the particles bring to the grid each substep, before the pressure has
//sorted them out, show all three at the surface:
//- liquid closing on liquid, or on a wall: a jet reaching a pool, a wave coming down on the water ahead of it, a flood meeting a pillar. There the
//  velocities converge, which no liquid can: -div u is how fast the two are closing. The pressure stops it within the substep, so the field after
//  the solve shows nothing of it. The air between them is folded in, and what can't get out of the way is thrown off: bubbles and spray both
//- the surface folding. With n its normal and S the strain rate, the surface's own area changes at div u - n S n: where that's negative, it's being
//  drawn together and under (what Ma, Shi and Kirby 2011 take for the air entrained under a breaking wave). And its tilt changes at the slope along
//  it of the velocity across it, grad (u.n) less its part along n: a wave's face steepening until it goes over, or the ring round a falling jet,
//  where the pool's surface is pulled down beside liquid that's already going down, for as long as the jet falls. Both make bubbles
//- the surface stretching, where its area is growing: a sheet climbing a wall, a crown's rim, thinning until it tears: spray
//A shear under the surface, a river's, does none of these, however strong; nor does a tank turning as a whole, at any speed a tank turns at; nor
//anything a liquid at rest does.
//
//How fast it has to be. A piece of liquid of size d comes away from the rest, or a pocket of air that size is folded in, when the liquid either side
//of it is moving apart or together fast enough to pay for it: the kinetic energy of that difference, w, against the surface a ball of that size
//has, and the work of lifting it its own height. For a ball, rho w^2 d = 12 sigma + 2 rho g d^2: a Weber number of 12, and gravity's share. That
//asks least of w at d0 = sqrt(6 sigma / (rho g)), 6.6 mm for water, where it's 0.51 m/s (a jet has to reach about that to carry air into a pool);
//the constants are a ball's, an order of magnitude's and not a measurement's. A voxel finer than d0 can only shed what's smaller than itself, which
//takes more. The difference itself is the one across the voxel: the rate (the closing's plus the folding's, or plus the stretching's) times its
//width, which is what the grid shows of a front it can't hold any sharper. Against a wall that's the front's whole jump; between two bodies of
//liquid, whose velocities P2G blends over a voxel or two, somewhat less.
//
//How much. Past that, the surface makes whitewater at a speed in proportion to how far past it is, a volume of AIR_FOLDED (w - w0) of bubbles and
//SPRAY_THROWN (w - w0) of spray per unit of its area and time. The grid shows a front a voxel thick, and closing for as long as it takes to cross
//one, however thin the layer of air that's really caught under it: so the air's volume is taken in the proportion of what's folded in, d0 across
//(or a voxel, if that's finer), to the voxel. Without that, liquid landing on liquid took a voxel's depth of air down with it over all the area
//it landed on: a dam break's water was a quarter air by volume two seconds in, and looked it. The constant is an order of magnitude's: a 3 m/s
//jet plunging into a pool takes down some percent of its own flow in air (Bin 1993 reviews what such jets are measured to take). The second is
//the look's: set so that a dam break against a pillar throws the fan of spray an artist expects of it, several times what the liquid really
//sheds, since each particle is drawn far larger than the droplets it stands for. WhitewaterSettings scales both. A voxel's share of the surface is
//the liquid's share's slope across it (the slope adds up to one across a surface, whichever way it faces), so no voxel has to be called a
//surface voxel.
//
//What sizes. Each particle stands for the same volume, so its size is drawn by volume: the size a litre of spray is mostly in, not the size most
//droplets are. What breaks off at the breakup scale d0 shatters, the finer the further past breaking it is: into pieces d0 over the square root of
//its Weber number's ratio to 12, d = sqrt(12 sigma d0 / rho) / w, the scaling of a splash's rim: 2.4 mm at 1 m/s for water and under a millimetre at
//3, spread log-normally about that. Air folded in is torn down to what the turbulence can't break further (Hinze 1955),
//d = 0.725 (sigma/rho)^(3/5) epsilon^(-2/5) with epsilon = w^3 / d0, a few tenths of a millimetre; and by number there are r^(-10/3) of the bubbles
//larger than that and r^(-3/2) of the smaller (Deane and Stokes 2002), so by volume most of the air is in bubbles a millimetre or more across, up to
//the pocket's own size, which are up and gone in a second or two, and a few percent is in the fine ones that hang as a haze.

#include "particles.hu"
#include "gridSampling.hu"
#include "liquidTile.hu"
#include "transport.hu"
#include "algorithms/radixSortKernels.hu"
#include "algorithms/particleToGridFunctions.hu"
#include "algorithms/voxelSolveFunctions.hu"
#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <algorithm>
#include <type_traits>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

static constexpr float SURFACE_SHARE = 0.5f;    //the liquid's share at its surface: a bubble that rises to less is foam, and a droplet that falls into more has landed
static constexpr float SUNK_SHARE = 0.75f;      //foam drawn under to this much liquid is a bubble again
static constexpr float BARE_SHARE = 0.05f;      //and foam left with this little under it is spray
static constexpr float FACE_LIQUID = 0.1f;      //a face holding less of its rest weight of liquid than this says nothing of the liquid's velocity
static constexpr float BREAKUP_WEBER = 12.0f;   //rho w^2 d / sigma for a ball of diameter d whose kinetic energy at w is its surface's
static constexpr float AIR_FOLDED = 1.0f;       //the volume of bubbles a breaking surface makes, per unit area and time, over how far past breaking its speed is
static constexpr float SPRAY_THROWN = 8.0f;     //and of spray
static constexpr float HINZE = 0.725f;          //Hinze's constant, for the largest bubble turbulence leaves whole
static constexpr float SIZE_SPREAD = 0.4f;      //the log-normal's width: a standard deviation of 0.4 in the logarithm, a factor of 1.5
static constexpr float SMAGORINSKY = 0.17f;     //Smagorinsky's constant: the eddies a voxel can't hold mix what's in it at (0.17 dx)^2 times the strain rate
static constexpr uint MOST_PER_VOXEL = 4096;    //bubbles a voxel makes in a substep at most, and droplets: only so that no rate, however wrong, overflows a count
static constexpr uint WHITEWATER_THREADS = 256; //per node block in flyWhitewater, a particle each

//what the kernels read of the grid
struct WhitewaterGrid{
    uint numUsedGridNodes;
    uint numOwnNodes;
    const uint* nodeCells;
    const uint* cellToNode;
    const uint* interiorVoxels;
    const float* velocities[3];
    const float* before[3];     //the grid's velocities before the forces and the pressure, as FLIP has them
    const float* weights[3];    //P2G's weight on each face, every fluid's
    const float* share;         //the liquid's share of each stored voxel
    Grid grid;
    int interiorWidth;
    float voxelSize;
    const char* letGo;          //the voxels boundaries let go of in this substep's solve (boundaries.cu), or nullptr with none that can
};

//the particles a kernel moves
struct WhitewaterParticles{
    double* x;
    double* y;
    double* z;
    float* u;
    float* v;
    float* w;
    const uint* cells;
    float* lives;
    float* radii;
    const float* births;
    char* kinds;
    const unsigned long long* ids;
};

//what a droplet's or a bubble's flight takes of the two fluids
struct WhitewaterFluids{
    float3 gravity;
    float pull;             //gravity's length
    float tension;          //the liquid's surface tension over its density
    float airDensity;       //over the liquid's
    float airViscosity;
    float liquidViscosity;
    bool phases;            //whether the grid carries the air too: its velocity is then what a droplet flies through
    float scale;            //metres: the size of what breaks off the surface (Particles::whitewaterScale)
    float breaks;           //m/s: the difference in speed it takes to break it
    float dropletScale;     //droplets' radii against what the model gives
};

//the radius of a droplet shed by liquid breaking up at a speed, by a normal number: the splash's rim's size at that speed (see the top), spread
//log-normally, and no larger than what breaks off whole
__device__ inline float dropletRadius(float tension, float scale, float speed, float normal, float dropletScale){
    return fminf(0.5f*sqrtf(BREAKUP_WEBER*tension*scale) / speed*expf(SIZE_SPREAD*normal), 0.5f*scale)*dropletScale;
}

//the flight (EscapedPath, gridSampling.hu) of a droplet of this radius in air, or of a bubble of it in the liquid. A bubble flattens as it rises, the
//more the larger it is against the surface tension that rounds it, and its drag levels where Tomiyama et al. (1998) have it, by its Eotvos number
//(as voxelVelsToParticles has an escaped bubble's)
__device__ inline EscapedFlight whitewaterFlight(const WhitewaterFluids& fluids, bool droplet, float radius){
    EscapedFlight flight;
    flight.gravity = fluids.gravity;
    flight.radius = radius;
    if(droplet){
        flight.viscosity = fluids.airViscosity;
        flight.density = 1.0f / fluids.airDensity;
        flight.bluff = 0.44f;
    }
    else{
        float eotvos = fluids.pull*(1.0f - fluids.airDensity)*4.0f*radius*radius / fluids.tension;
        flight.viscosity = fluids.liquidViscosity;
        flight.density = fluids.airDensity;
        flight.bluff = 8.0f/3.0f*eotvos / (eotvos + 4.0f);
    }
    return flight;
}

//a number from 0 up to 1, the which-th of key's
__device__ inline float whitewaterChance(unsigned long long key, unsigned long long which){
    return (mixBits64(key + which) >> 40)*(1.0f / 16777216.0f);
}

//a voxel's own key for a substep, from the substep's: by where the voxel is in the domain, so the same however the nodes are stored or split
__device__ inline unsigned long long whitewaterVoxelKey(unsigned long long key, int3 voxel, int3 domainVoxels){
    return mixBits64(key ^ ((unsigned long long)voxel.x + (unsigned long long)domainVoxels.x*((unsigned long long)voxel.y + (unsigned long long)domainVoxels.y*voxel.z)));
}

//How fast the eddies smaller than a voxel mix what the liquid carries at a point of a tile, over the voxel's size: Smagorinsky's (0.17 dx)^2 |S|, with
//|S| = sqrt(2 S:S) the size of the strain rate S, the symmetric part of the velocity's gradient, taken from half a voxel either side along each axis.
//A side with no velocity takes the middle's (around), and adds nothing
__device__ inline float mixingAt(const float* tile, int width, float3 point, int3 origin, int3 domainVoxels, float3 around){
    float gradient[3][3];   //component a's change along b, per voxel
    #pragma unroll
    for(int b = 0; b < 3; ++b){
        float3 half = make_float3(b == 0 ? 0.5f : 0.0f, b == 1 ? 0.5f : 0.0f, b == 2 ? 0.5f : 0.0f);
        float3 ahead = sampleVelocity(tile, width, point + half, origin, domainVoxels, around, WallStick());
        float3 behind = sampleVelocity(tile, width, point - half, origin, domainVoxels, around, WallStick());
        gradient[0][b] = ahead.x - behind.x;
        gradient[1][b] = ahead.y - behind.y;
        gradient[2][b] = ahead.z - behind.z;
    }
    float strain = 0.0f;    //2 S:S
    #pragma unroll
    for(int a = 0; a < 3; ++a){
        strain += 2.0f*gradient[a][a]*gradient[a][a];
        #pragma unroll
        for(int b = a + 1; b < 3; ++b){
            strain += (gradient[a][b] + gradient[b][a])*(gradient[a][b] + gradient[b][a]);
        }
    }
    return SMAGORINSKY*SMAGORINSKY*sqrtf(strain);
}

//which way the liquid's share of the voxels grows at a point of a tile: the slope of the eight voxel centres around it, per voxel. Only its direction
//is used, to tell a droplet falling into the liquid from one leaving it, so where a centre holds nothing it counts as air
__device__ inline float3 shareSlope(const float* shares, int width, float3 point){
    float coordinates[3] = {point.x - 0.5f, point.y - 0.5f, point.z - 0.5f};
    int base[3];
    float fraction[3];
    #pragma unroll
    for(int axis = 0; axis < 3; ++axis){
        base[axis] = min(max((int)floorf(coordinates[axis]), 0), width - 2);
        fraction[axis] = fminf(fmaxf(coordinates[axis] - base[axis], 0.0f), 1.0f);
    }
    float corners[8];
    #pragma unroll
    for(int corner = 0; corner < 8; ++corner){
        float share = shares[base[0] + (corner & 1) + width*(base[1] + (corner >> 1 & 1) + width*(base[2] + (corner >> 2)))];
        corners[corner] = isnan(share) ? 0.0f : share;
    }
    float slope[3];
    #pragma unroll
    for(int axis = 0; axis < 3; ++axis){
        int a = (axis + 1) % 3, b = (axis + 2) % 3;
        float sum = 0.0f;
        #pragma unroll
        for(int corner = 0; corner < 4; ++corner){
            int da = corner & 1, db = corner >> 1;
            float weight = (da ? fraction[a] : 1.0f - fraction[a])*(db ? fraction[b] : 1.0f - fraction[b]);
            int low = (da << a) | (db << b);
            sum += weight*(corners[low | 1 << axis] - corners[low]);
        }
        slope[axis] = sum;
    }
    return make_float3(slope[0], slope[1], slope[2]);
}

//The whitewater in this partition's own nodes flies through the substep: a block per node, which finds its particles among the sorted cells and, if
//it has any, gathers a tile three nodes wide (the grid's new velocities, the liquid's share of each voxel, the velocities before: as flyEscaped's is)
//and gives each thread a particle.
//First what it is now, by the liquid's share where it is: a droplet in the liquid has landed, and is foam where it did, unless it's still on its way
//out of it (moving, against the liquid, the way the liquid thins: it was born at the surface, with the surface's liquid around it); a bubble at the
//surface is foam; foam drawn well under is a bubble, and foam the liquid has left is spray: a droplet, of the size liquid breaking up at its speed
//sheds, and no longer the bubble's it was (a fine bubble's radius makes a droplet that hangs in the air for seconds, where the liquid left it).
//Then its flight: a droplet's through the air (at rest, unless the grid carries the air) or a bubble's through the liquid, by the grid's velocity
//where the substep finds it and what that changed by over the substep; or, for foam at the surface, the liquid's own path through the grid, as the
//fluid's particles take it. Foam under the surface, short of deep enough to be a bubble again, takes a bubble's flight: it's as light as it was,
//and comes back up through any flow a bubble would.
//Its life is how long it lasts once it has reached the surface, and from the first time it's foam it runs, whatever it is after: a bubble that
//has come up bursts, and being drawn a centimetre under meanwhile doesn't save it. (Run only while it was foam, it stood still for most of
//every particle's time: the surface's own motion takes foam under the share that makes it a bubble again within a sixth of a second, and it's a
//sixth of a second more before it's back. Two thirds of a dam break's bubbles reached the surface within a second, and nine in ten of them
//were still there two seconds on.)
//A bubble is also carried about by the eddies smaller than a voxel, which the grid's velocity has nothing of: without them, the bubbles of one splash
//stay a sheet a particle thick for as long as they last, drawn out into a line round whatever vortex took them down. Those eddies mix what the liquid
//carries (mixingAt), so each substep the bubble takes a step drawn for that much mixing over dt, by numbers that are its own and the substep's (key)
__global__ void flyWhitewater(WhitewaterGrid g, WhitewaterFluids fluids, WhitewaterParticles p, uint count, float dt, unsigned long long key){
    extern __shared__ float tile[];     //the new velocities' three planes, the liquid's share, then the three before
    __shared__ uint nodes[27];
    __shared__ uint range[2];
    uint cell = g.nodeCells[blockIdx.x];
    if(threadIdx.x < 2){    //its particles: from the first whose cell is this one to the first whose cell is past it
        uint key = cell + threadIdx.x;
        uint low = 0, high = count;
        while(low < high){
            uint middle = low + (high - low) / 2;
            if(p.cells[middle] < key){
                low = middle + 1;
            }
            else{
                high = middle;
            }
        }
        range[threadIdx.x] = low;
    }
    loadTileNodes(nodes, cell, g.numUsedGridNodes, g.nodeCells, g.cellToNode, g.grid);
    __syncthreads();
    if(range[0] == range[1]){
        return;
    }
    Tile place(cell, g.grid, g.interiorWidth, g.interiorWidth);
    int slots = place.slots();
    for(int slot = threadIdx.x; slot < slots; slot += blockDim.x){
        int3 t = place.at(slot);
        bool inside = place.inDomain(t);
        uint voxel = inside ? place.voxel(t, nodes, g.numUsedGridNodes, g.interiorVoxels) : NO_VOXEL;
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){   //as loadTile has them: past the walls at rest, and none where nothing is stored or no particle reached
            bool reached = voxel != NO_VOXEL && g.weights[dim][voxel] > 0.0f;
            tile[dim*slots + slot] = !inside ? 0.0f : reached ? g.velocities[dim][voxel] : nanf("");
            tile[(4 + dim)*slots + slot] = !inside ? 0.0f : reached ? g.before[dim][voxel] : nanf("");
        }
        tile[3*slots + slot] = voxel != NO_VOXEL ? g.share[voxel] : nanf("");
    }
    __syncthreads();
    if(g.letGo != nullptr){     //where a wall has let go of a voxel its own face there carries the liquid's velocity on, as the fluid's particles have it (loadTile)
        const int strides[3] = {1, place.width, place.width*place.width};
        const int size[3] = {place.domainVoxels.x, place.domainVoxels.y, place.domainVoxels.z};
        for(int slot = threadIdx.x; slot < slots; slot += blockDim.x){
            int3 t = place.at(slot);
            uint voxel = place.inDomain(t) ? place.voxel(t, nodes, g.numUsedGridNodes, g.interiorVoxels) : NO_VOXEL;
            if(voxel == NO_VOXEL || !g.letGo[voxel]){
                continue;
            }
            int3 global = place.global(t);
            const int at[3] = {global.x, global.y, global.z};
            #pragma unroll
            for(int dim = 0; dim < 3; ++dim){
                int along = slot / strides[dim] % place.width;
                #pragma unroll
                for(int plane = 0; plane < 2; ++plane){     //the new velocities, and the ones before
                    float* faces = tile + (4*plane + dim)*slots;
                    if(at[dim] == 0 && along + 1 < place.width){
                        faces[slot] = faces[slot + strides[dim]];
                    }
                    if(at[dim] == size[dim] - 1 && along + 1 < place.width){
                        faces[slot + strides[dim]] = faces[slot];
                    }
                }
            }
        }
        __syncthreads();
    }
    const Grid& grid = g.grid;
    float step = dt / g.voxelSize;
    float3 rest = make_float3(0.0f, 0.0f, 0.0f);
    for(uint index = range[0] + threadIdx.x; index < range[1]; index += blockDim.x){
        float3 point = make_float3((p.x[index] - grid.negX)/g.voxelSize - place.origin.x, (p.y[index] - grid.negY)/g.voxelSize - place.origin.y,
                                   (p.z[index] - grid.negZ)/g.voxelSize - place.origin.z);
        float3 own = make_float3(p.u[index], p.v[index], p.w[index]);
        float liquid = shareAround(tile + 3*slots, place.width, point, place.origin, place.domainVoxels);
        liquid = isnan(liquid) ? 0.0f : liquid;
        float3 around = sampleVelocity(tile, place.width, point, place.origin, place.domainVoxels, rest, WallStick());
        unsigned long long mine = mixBits64(key ^ p.ids[index]);    //its own numbers for this substep
        char kind = p.kinds[index];
        if(kind == WHITEWATER_SPRAY){
            if(liquid > SURFACE_SHARE && dot(own - around, shareSlope(tile + 3*slots, place.width, point)) >= 0.0f){
                kind = WHITEWATER_FOAM;
            }
        }
        else if(kind == WHITEWATER_BUBBLE){
            if(liquid <= SURFACE_SHARE){
                kind = WHITEWATER_FOAM;
            }
        }
        else if(liquid > SUNK_SHARE){
            kind = WHITEWATER_BUBBLE;
        }
        else if(liquid < BARE_SHARE){
            kind = WHITEWATER_SPRAY;
            float3 through = fluids.phases ? own - around : own;    //against the air, which is at rest unless the grid carries it
            float normal = sqrtf(-2.0f*logf(fmaxf(whitewaterChance(mine, 6), 1.0e-7f)))*cosf(6.2831853f*whitewaterChance(mine, 7));     //Box and Muller's
            p.radii[index] = fmaxf(dropletRadius(fluids.tension, fluids.scale, fmaxf(sqrtf(dot(through, through)), fluids.breaks), normal, fluids.dropletScale), 1.0e-5f);
        }
        float life = p.lives[index];    //over 0 until it's first at the surface; from then, less than 0 by what's left of it, and 0 once that's run out
        if(kind == WHITEWATER_FOAM && life > 0.0f){
            life = -life;
        }
        if(life < 0.0f){
            life = fminf(life + dt, 0.0f);
        }
        p.lives[index] = life;
        double moved[3];
        float arrives[3];
        if(kind == WHITEWATER_FOAM && !(liquid > SURFACE_SHARE)){
            float3 velocity = velocityThrough(tile, place.width, point, place.origin, place.domainVoxels, own, step, true, WallStick());
            moved[0] = (double)dt*velocity.x;
            moved[1] = (double)dt*velocity.y;
            moved[2] = (double)dt*velocity.z;
            arrives[0] = velocity.x;
            arrives[1] = velocity.y;
            arrives[2] = velocity.z;
        }
        else{
            bool droplet = kind == WHITEWATER_SPRAY;
            float3 acceleration = rest;
            if(droplet && !fluids.phases){
                around = rest;      //the grid's velocity is the liquid's, and the air around a droplet is at rest
            }
            else{
                float3 was = sampleVelocity(tile + 4*slots, place.width, point, place.origin, place.domainVoxels, rest, WallStick());
                acceleration = make_float3((around.x - was.x)/dt, (around.y - was.y)/dt, (around.z - was.z)/dt);
            }
            EscapedPath path(own, around, acceleration, whitewaterFlight(fluids, droplet, p.radii[index]), dt);
            path.at(dt, moved, arrives);
            if(!droplet){
                //A step of variance 2 D dt along each axis, D the mixing: sqrt(2 D dt) times a normal number. Where D changes from place to place it's
                //the D where the step ends that sizes it (found from where the step by this place's D would end): steps sized by where they start
                //gather what they carry wherever the mixing is weak, which mixing doesn't
                float normal[3];
                #pragma unroll
                for(int axis = 0; axis < 3; ++axis){    //Box and Muller's
                    normal[axis] = sqrtf(-2.0f*logf(fmaxf(whitewaterChance(mine, 2*axis), 1.0e-7f)))*cosf(6.2831853f*whitewaterChance(mine, 2*axis + 1));
                }
                float reach = sqrtf(2.0f*mixingAt(tile, place.width, point, place.origin, place.domainVoxels, around)*step);    //in voxels: D dt / dx^2 is the mixing times dt / dx
                float3 ends = make_float3(point.x + reach*normal[0], point.y + reach*normal[1], point.z + reach*normal[2]);
                reach = sqrtf(2.0f*mixingAt(tile, place.width, ends, place.origin, place.domainVoxels, around)*step);
                #pragma unroll
                for(int axis = 0; axis < 3; ++axis){
                    moved[axis] += (double)(reach*g.voxelSize*normal[axis]);
                }
            }
        }
        p.x[index] += moved[0];
        p.y[index] += moved[1];
        p.z[index] += moved[2];
        p.u[index] = arrives[0];
        p.v[index] = arrives[1];
        p.w[index] = arrives[2];
        p.kinds[index] = kind;
    }
}

//where whitewater ends, and what the walls do to it
struct WhitewaterEnds{
    double low[3];          //the domain
    double high[3];
    float voxelSize;
    float now;              //seconds from the start
    float maxAge;
};

//Every particle, once flyWhitewater has moved the ones in this partition's nodes. One in a cell with no node of this partition's has no grid around it
//and so no liquid: it's a droplet, whatever it was, and flies through air at rest. Then, for all of them, what ends one: its life at the surface run out;
//any past its age; any in a sink, or at an open face; a droplet that has reached a wall or an obstacle, which it wets. A bubble or foam that has
//crossed a wall or gone into an obstacle is put back just inside or outside it, as the fluid's particles are, without the velocity that took it there
__global__ void finishWhitewater(WhitewaterGrid g, WhitewaterFluids fluids, WhitewaterParticles p, uint count, float dt, WhitewaterEnds ends, Sources sources, Obstacles obstacles){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index >= count){
        return;
    }
    char kind = p.kinds[index];
    if(kind == WHITEWATER_GONE){
        return;
    }
    uint cell = p.cells[index];
    uint node = g.cellToNode[cell];
    if(!(node < g.numOwnNodes && g.nodeCells[node] == cell)){
        kind = WHITEWATER_SPRAY;
        float3 rest = make_float3(0.0f, 0.0f, 0.0f);
        EscapedPath path(make_float3(p.u[index], p.v[index], p.w[index]), rest, rest, whitewaterFlight(fluids, true, p.radii[index]), dt);
        double moved[3];
        float arrives[3];
        path.at(dt, moved, arrives);
        p.x[index] += moved[0];
        p.y[index] += moved[1];
        p.z[index] += moved[2];
        p.u[index] = arrives[0];
        p.v[index] = arrives[1];
        p.w[index] = arrives[2];
    }
    double* position[3] = {p.x + index, p.y + index, p.z + index};
    float* velocity[3] = {p.u + index, p.v + index, p.w + index};
    bool gone = p.lives[index] == 0.0f || !(ends.now - p.births[index] < ends.maxAge);
    #pragma unroll
    for(int axis = 0; axis < 3; ++axis){
        double at = *position[axis];
        bool below = at < ends.low[axis], above = !(at < ends.high[axis]);    //a position that isn't a number is past everything
        gone = gone || (sources.openFaces >> 2*axis & 1 && at < ends.low[axis] + ends.voxelSize) || (sources.openFaces >> (2*axis + 1) & 1 && at >= ends.high[axis] - ends.voxelSize);
        if(below || above){
            gone = gone || kind == WHITEWATER_SPRAY || !(at == at);
            *position[axis] = below ? ends.low[axis] + 1.0e-3*ends.voxelSize : ends.high[axis] - 1.0e-3*ends.voxelSize;
            *velocity[axis] = 0.0f;
        }
    }
    for(int sink = 0; sink < sources.numSinks && !gone; ++sink){
        gone = sources.sinks[sink].contains(*position[0], *position[1], *position[2]);
    }
    for(int which = 0; which < obstacles.count && !gone; ++which){
        const ObstacleState& obstacle = obstacles.items[which];
        float3 x = make_float3((float)*position[0], (float)*position[1], (float)*position[2]);
        float3 normal;
        float distance = obstacleDistance(obstacle, x, normal);
        if(distance < 0.0f){
            if(kind == WHITEWATER_SPRAY){
                gone = true;
                break;
            }
            float push = 0.05f*ends.voxelSize - distance;
            *position[0] += push*normal.x;
            *position[1] += push*normal.y;
            *position[2] += push*normal.z;
            float3 relative = make_float3(*velocity[0], *velocity[1], *velocity[2]) - obstacleVelocity(obstacle, x + push*normal);
            float inward = dot(relative, normal);
            if(inward < 0.0f){
                *velocity[0] -= inward*normal.x;
                *velocity[1] -= inward*normal.y;
                *velocity[2] -= inward*normal.z;
            }
        }
    }
    p.kinds[index] = gone ? WHITEWATER_GONE : kind;
}

//The liquid's share of each stored voxel with no air simulated (two phases have findLiquidShare's, twophase.cu, which this follows): the liquid's
//weight on the voxel's six faces over what they'd hold at rest, where the walls leave room for it; never more than all of it. From CORRECTION_DEPTH
//voxels inside the footprint a voxel is all liquid whatever it weighs: beside an obstacle, which takes part of each face's reach, the weights alone
//would have the liquid thin all along it. None (NaN) in a voxel obstacles close
__global__ void findLiquidShareAlone(uint numVoxels, const char* solveCodes, const uint* neighborNx, const uint* neighborPx, const uint* neighborNy, const uint* neighborPy,
                                     const uint* neighborNz, const uint* neighborPz, const char* closed, const char* depth, const float* weightX, const float* weightY,
                                     const float* weightZ, float rest, float* share){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < numVoxels){
        const uint* lower[3] = {neighborNx, neighborNy, neighborNz};
        const uint* upper[3] = {neighborPx, neighborPy, neighborPz};
        const float* weight[3] = {weightX, weightY, weightZ};
        bool unknown = solveCodes[index];
        float inside[3][3];     //along each axis, how much of the reach of the voxel's lower face, of its centre and of its upper face is inside the walls
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            int low = 2, high = 2;  //the voxels between this one and the wall below it, and above: 2 for any more, which no face's reach crosses
            if(unknown){    //an unknown's neighbours are its own to read, and so are an unknown neighbour's: anything else's upper ones were never written
                uint below = lower[axis][index], above = upper[axis][index];
                low = below == WALL_VOXEL ? 0 : below < WALL_VOXEL && solveCodes[below] && lower[axis][below] == WALL_VOXEL ? 1 : 2;
                high = above == WALL_VOXEL ? 0 : above < WALL_VOXEL && solveCodes[above] && upper[axis][above] == WALL_VOXEL ? 1 : 2;
            }
            inside[axis][0] = 1.0f - pastWall(2*low) - pastWall(2*high + 2);
            inside[axis][1] = 1.0f - pastWall(2*low + 1) - pastWall(2*high + 1);
            inside[axis][2] = 1.0f - pastWall(2*low + 2) - pastWall(2*high);
        }
        float liquid = 0.0f, room = 0.0f;
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            uint above = unknown ? upper[axis][index] : NO_VOXEL;
            uint ends[2] = {index, above < WALL_VOXEL ? above : NO_VOXEL};
            #pragma unroll
            for(int end = 0; end < 2; ++end){
                if(ends[end] != NO_VOXEL){
                    liquid += weight[axis][ends[end]];
                    room += inside[axis][2*end]*inside[(axis + 1) % 3][1]*inside[(axis + 2) % 3][1];
                }
            }
        }
        share[index] = closed != nullptr && closed[index] ? nanf("") : unknown && depth[index] >= CORRECTION_DEPTH ? 1.0f : fminf(liquid / (room*rest), 1.0f);
    }
}

//what findWhitewaterSources takes besides the grid
struct WhitewaterSources{
    const char* solveCodes;
    const char* near;           //obstacleNear, or nullptr with no obstacles
    const float* liquid[3];     //the liquid's weight on each face: P2G's sums, or with two phases the liquid's own, in fixed point
    float liquidUnit;           //that fixed point's unit, or 0 for floats
    float rest;                 //a face's weight at rest
    float across;               //metres: a rate of closing, folding or stretching times this is the difference in speed the surface is breaking under
    float breaks;               //m/s: the difference it takes to break the surface
    float perSpeed;             //particles a voxel makes per second, per m/s past that, per unit of its share's slope: bubbles, and droplets
    float sprayPerSpeed;
    unsigned long long key;     //this substep's
};

//Where the surface is breaking up, and how many bubbles and droplets each voxel makes this substep: a block per node of this partition's, a thread
//per voxel of its interior, over a tile a voxel wider all round. A node none of whose voxels holds both liquid and something else is done as soon as
//it knows. The velocities are the ones P2G made of the particles', before the forces and the pressure, and their gradient comes from the faces that
//carry the liquid's velocity: those holding enough liquid, a wall's own (at rest), and the ones obstacles close (the obstacle's): so liquid running
//into a wall or a pillar closes on it. A difference with no such face either side isn't taken.
//Each count is its rate times dt, the fraction left over made a whole one that often, by a number that's the voxel's and the substep's alone: the
//same however the nodes are stored or split
__global__ void findWhitewaterSources(WhitewaterGrid g, WhitewaterSources s, float dt, uint* counts, uint* folded, float* folds, float* tears){
    extern __shared__ float tile[];     //three planes of face velocities, which of them carry the liquid's velocity (bits), and the liquid's share
    __shared__ uint nodes[27];
    __shared__ uint mixed;
    uint cell = g.nodeCells[blockIdx.x];
    loadTileNodes(nodes, cell, g.numUsedGridNodes, g.nodeCells, g.cellToNode, g.grid);
    if(threadIdx.x == 0){
        mixed = 0;
    }
    __syncthreads();
    Tile place(cell, g.grid, g.interiorWidth, 1);
    int slots = place.slots();
    float* shares = tile + 4*slots;
    bool some = false;
    for(int slot = threadIdx.x; slot < slots; slot += blockDim.x){  //the share first: -1 where there's a wall or an obstacle, which the slope reads as more of the same
        int3 t = place.at(slot);
        float share = -1.0f;
        if(place.inDomain(t)){
            uint voxel = place.voxel(t, nodes, g.numUsedGridNodes, g.interiorVoxels);
            float stored = voxel != NO_VOXEL ? g.share[voxel] : 0.0f;
            share = isnan(stored) ? -1.0f : fminf(fmaxf(stored, 0.0f), 1.0f);
            some = some || (place.interior(t) && share > 0.0f && share < 1.0f);
        }
        shares[slot] = share;
    }
    if(some){
        mixed = 1;      //every thread that writes it writes the same
    }
    __syncthreads();
    if(!mixed){
        return;
    }
    for(int slot = threadIdx.x; slot < slots; slot += blockDim.x){
        int3 t = place.at(slot);
        int3 global = place.global(t);
        int at[3] = {global.x, global.y, global.z};
        int size[3] = {place.domainVoxels.x, place.domainVoxels.y, place.domainVoxels.z};
        bool inside = place.inDomain(t);
        uint voxel = inside ? place.voxel(t, nodes, g.numUsedGridNodes, g.interiorVoxels) : NO_VOXEL;
        int carries = 0;
        #pragma unroll
        for(int dim = 0; dim < 3; ++dim){
            int a = (dim + 1) % 3, b = (dim + 2) % 3;
            bool wall = (at[dim] == 0 || at[dim] == size[dim]) && at[a] >= 0 && at[a] < size[a] && at[b] >= 0 && at[b] < size[b];   //a wall's own face
            float velocity = 0.0f;
            bool known = wall;
            if(!wall && voxel != NO_VOXEL){
                float liquid = s.liquidUnit > 0.0f ? ((const int*)s.liquid[dim])[voxel]*s.liquidUnit : s.liquid[dim][voxel];
                known = liquid >= FACE_LIQUID*s.rest || (s.near != nullptr && (s.near[voxel] & CLOSED_FACE << 2*dim));
                velocity = g.before[dim][voxel];
            }
            tile[dim*slots + slot] = velocity;
            carries |= known << dim;
        }
        tile[3*slots + slot] = (float)carries;
    }
    __syncthreads();
    const int strides[3] = {1, place.width, place.width*place.width};
    int interior = g.interiorWidth*g.interiorWidth*g.interiorWidth;
    for(int own = threadIdx.x; own < interior; own += blockDim.x){
        int3 t = make_int3(1 + own % g.interiorWidth, 1 + own / g.interiorWidth % g.interiorWidth, 1 + own / (g.interiorWidth*g.interiorWidth));
        int slot = place.slot(t);
        float share = shares[slot];
        uint voxel = place.voxel(t, nodes, g.numUsedGridNodes, g.interiorVoxels);
        if(voxel == NO_VOXEL || !s.solveCodes[voxel] || share < 0.0f){
            continue;
        }
        float inwards[3];       //the share's slope, per voxel: across a surface its length adds up to 1, and it points into the liquid
        float slope = 0.0f;
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            float below = shares[slot - strides[axis]], above = shares[slot + strides[axis]];
            inwards[axis] = 0.5f*((above < 0.0f ? share : above) - (below < 0.0f ? share : below));
            slope += inwards[axis]*inwards[axis];
        }
        slope = sqrtf(slope);
        if(slope == 0.0f){
            continue;
        }
        auto carries = [&](int dim, int where){
            return ((int)tile[3*slots + where] >> dim & 1) != 0;
        };
        float gradient[3][3];   //component a's change along b, per voxel
        #pragma unroll
        for(int a = 0; a < 3; ++a){
            const float* face = tile + a*slots;
            #pragma unroll
            for(int b = 0; b < 3; ++b){
                if(a == b){     //between the voxel's two faces
                    gradient[a][a] = carries(a, slot) && carries(a, slot + strides[a]) ? face[slot + strides[a]] - face[slot] : 0.0f;
                    continue;
                }
                float sum = 0.0f;   //across each of its two faces, between the faces either side along b, or to one of them
                int taken = 0;
                #pragma unroll
                for(int end = 0; end < 2; ++end){
                    int middle = slot + end*strides[a];
                    int low = middle - strides[b], high = middle + strides[b];
                    bool lowCarries = carries(a, low), middleCarries = carries(a, middle), highCarries = carries(a, high);
                    if(lowCarries && highCarries){
                        sum += 0.5f*(face[high] - face[low]);
                        ++taken;
                    }
                    else if(middleCarries && (lowCarries || highCarries)){
                        sum += highCarries ? face[high] - face[middle] : face[middle] - face[low];
                        ++taken;
                    }
                }
                gradient[a][b] = taken > 0 ? sum / taken : 0.0f;
            }
        }
        float across[3];        //how the velocity across the surface, u.n, changes along each axis; then its part along n, which is n S n; and the divergence
        float along = 0.0f, tilt = 0.0f, divergence = 0.0f;
        #pragma unroll
        for(int b = 0; b < 3; ++b){
            across[b] = (inwards[0]*gradient[0][b] + inwards[1]*gradient[1][b] + inwards[2]*gradient[2][b]) / slope;
            along += across[b]*inwards[b] / slope;
            tilt += across[b]*across[b];
            divergence += gradient[b][b];
        }
        tilt = sqrtf(fmaxf(tilt - along*along, 0.0f));  //what's left is along the surface: how fast it's tilting
        float surface = divergence - along;    //how fast the surface's own area is growing
        float closing = fmaxf(-divergence, 0.0f);
        float fold = (closing + fmaxf(-surface, 0.0f) + tilt) / g.voxelSize*s.across;      //the differences in speed folding air in, and throwing liquid off
        float tear = (closing + fmaxf(surface, 0.0f)) / g.voxelSize*s.across;
        if(!(fold > s.breaks) && !(tear > s.breaks)){
            continue;
        }
        unsigned long long key = whitewaterVoxelKey(s.key, place.global(t), place.domainVoxels);
        uint bubbles = (uint)fminf(s.perSpeed*slope*dt*fmaxf(fold - s.breaks, 0.0f) + whitewaterChance(key, 0), (float)MOST_PER_VOXEL);
        uint droplets = (uint)fminf(s.sprayPerSpeed*slope*dt*fmaxf(tear - s.breaks, 0.0f) + whitewaterChance(key, 1), (float)MOST_PER_VOXEL);
        counts[voxel] = bubbles + droplets;
        folded[voxel] = bubbles;
        folds[voxel] = fold;
        tears[voxel] = tear;
    }
}

//what emitWhitewater takes besides the grid and the particles' arrays
struct WhitewaterBirths{
    const uint* counts;         //how many each voxel makes, and where its first goes, from the first new one
    const unsigned long long* starts;
    const uint* folded;         //how many of those are bubbles: its first ones
    const float* folds;         //the differences in speed that are breaking its surface up: folding air in, and throwing liquid off
    const float* tears;
    uint first;                 //how many there are already, after which the new ones go
    uint room;                  //how many new ones there's room for
    const uint* neighbors[6];
    float tension;
    float scale;                //metres: the size of what breaks off
    float dropletScale;
    float bubbleScale;
    float foamLife;
    float now;
    unsigned long long key;
    double corner[3];           //the domain's
};

//The new particles: a block per node of this partition's, its threads sharing out the node's voxels, each voxel's particles going where the running
//sum of the counts puts them, its bubbles first. Each is somewhere in its voxel, with the liquid's velocity there (each component between the
//voxel's two faces of it); a droplet is thrown from that by up to the difference that's throwing it, any way that's outwards. Its radius is what
//that difference leaves whole (see the top); its life as foam is drawn so that foam bursts at a steady rate. Everything about it comes from numbers
//that are its voxel's, its place among the voxel's, and the substep's; its id it gets once the sort has placed it (nameWhitewater)
__global__ void emitWhitewater(VoxelPlaces places, WhitewaterGrid g, WhitewaterBirths b, double* px, double* py, double* pz, float* pu, float* pv, float* pw,
                               unsigned long long* ids, float* births, float* lives, float* radii, char* kinds){
    uint node = blockIdx.x;
    uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
    uint last = places.nodeVoxelEnds[node];
    uint cell = places.nodeCells[node];
    int3 domain = make_int3(places.grid.sizeX*places.interiorWidth, places.grid.sizeY*places.interiorWidth, places.grid.sizeZ*places.interiorWidth);
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        uint count = b.counts[index];
        if(count == 0 || b.starts[index] >= b.room){
            continue;
        }
        uint start = (uint)b.starts[index];
        int3 voxel = places.voxelOf(cell, places.voxelSlots[index]);
        float share = g.share[index];
        float low[3], high[3], outwards[3];
        float steepest = 0.0f;
        #pragma unroll
        for(int axis = 0; axis < 3; ++axis){
            uint below = b.neighbors[2*axis][index], above = b.neighbors[2*axis + 1][index];
            low[axis] = g.velocities[axis][index];
            high[axis] = above == WALL_VOXEL ? 0.0f : above == NO_VOXEL ? low[axis] : g.velocities[axis][above];
            float under = below == WALL_VOXEL ? share : below == NO_VOXEL ? 0.0f : g.share[below];
            float over = above == WALL_VOXEL ? share : above == NO_VOXEL ? 0.0f : g.share[above];
            outwards[axis] = (isnan(under) ? share : under) - (isnan(over) ? share : over);
            steepest += outwards[axis]*outwards[axis];
        }
        unsigned long long voxelKey = whitewaterVoxelKey(b.key, voxel, domain);
        for(uint which = 0; which < count && start + which < b.room; ++which){
            bool droplet = which >= b.folded[index];
            float speed = droplet ? b.tears[index] : b.folds[index];
            uint particle = b.first + start + which;
            unsigned long long key = mixBits64(voxelKey ^ ((unsigned long long)(which + 1) << 32));
            float in[3] = {whitewaterChance(key, 1), whitewaterChance(key, 2), whitewaterChance(key, 3)};
            float turn = 6.2831853f*whitewaterChance(key, 4), rise = 2.0f*whitewaterChance(key, 5) - 1.0f;    //a direction: any, as likely as any other
            float flat = sqrtf(fmaxf(1.0f - rise*rise, 0.0f));
            float thrown[3] = {flat*cosf(turn), flat*sinf(turn), rise};
            float through = droplet ? speed*whitewaterChance(key, 6) : 0.0f;
            if(thrown[0]*outwards[0] + thrown[1]*outwards[1] + thrown[2]*outwards[2] < 0.0f){   //a droplet leaves the way the liquid thins
                thrown[0] = -thrown[0];
                thrown[1] = -thrown[1];
                thrown[2] = -thrown[2];
            }
            px[particle] = b.corner[0] + ((double)voxel.x + in[0])*g.voxelSize;
            py[particle] = b.corner[1] + ((double)voxel.y + in[1])*g.voxelSize;
            pz[particle] = b.corner[2] + ((double)voxel.z + in[2])*g.voxelSize;
            pu[particle] = low[0] + in[0]*(high[0] - low[0]) + through*thrown[0];
            pv[particle] = low[1] + in[1]*(high[1] - low[1]) + through*thrown[1];
            pw[particle] = low[2] + in[2]*(high[2] - low[2]) + through*thrown[2];
            float radius;
            if(droplet){
                float normal = sqrtf(-2.0f*logf(fmaxf(whitewaterChance(key, 7), 1.0e-7f)))*cosf(6.2831853f*whitewaterChance(key, 8));     //Box and Muller's
                radius = dropletRadius(b.tension, b.scale, speed, normal, b.dropletScale);
            }
            else{   //by volume: r^(3/2) of it per unit of radius below Hinze's size, r^(-1/3) above, up to the pocket's own
                float hinze = fminf(0.5f*HINZE*powf(b.tension, 0.6f)*powf(b.scale / (speed*speed*speed), 0.4f), 0.5f*b.scale);
                float above = 1.5f*(powf(0.5f*b.scale / hinze, 2.0f/3.0f) - 1.0f), below = 0.4f;     //the volume either side, over Hinze's size
                float pick = whitewaterChance(key, 7)*(above + below);
                radius = hinze*(pick < below ? powf(pick / below, 0.4f) : powf(1.0f + (pick - below) / 1.5f, 1.5f))*b.bubbleScale;
            }
            radii[particle] = fmaxf(radius, 1.0e-5f);
            lives[particle] = fmaxf(-b.foamLife*logf(fmaxf(1.0f - whitewaterChance(key, 9), 1.0e-7f)), 1.0e-6f);    //over 0: it hasn't been at the surface yet
            births[particle] = b.now;
            ids[particle] = 0;      //until the sort has placed it (nameWhitewater)
            kinds[particle] = droplet ? WHITEWATER_SPRAY : WHITEWATER_BUBBLE;
        }
    }
}

//With less room than the surface would fill (WhitewaterSettings::capacity), every voxel makes the same fraction of what it would have, bubbles and
//droplets alike, the part left over made a whole one that often by a number that's the voxel's and the substep's: what's made thins evenly, rather
//than the voxels that come last in the count making none
__global__ void thinWhitewater(VoxelPlaces places, float keep, unsigned long long key, uint* counts, uint* folded){
    uint node = blockIdx.x;
    uint first = node == 0 ? 0 : places.nodeVoxelEnds[node - 1];
    uint last = places.nodeVoxelEnds[node];
    uint cell = places.nodeCells[node];
    int3 domain = make_int3(places.grid.sizeX*places.interiorWidth, places.grid.sizeY*places.interiorWidth, places.grid.sizeZ*places.interiorWidth);
    for(uint index = first + threadIdx.x; index < last; index += blockDim.x){
        uint count = counts[index];
        if(count == 0){
            continue;
        }
        unsigned long long voxelKey = whitewaterVoxelKey(key, places.voxelOf(cell, places.voxelSlots[index]), domain);
        uint bubbles = (uint)(folded[index]*keep + whitewaterChance(voxelKey, 2));
        uint droplets = (uint)((count - folded[index])*keep + whitewaterChance(voxelKey, 3));
        counts[index] = bubbles + droplets;
        folded[index] = bubbles;
    }
}

//how many the voxels make between them: the last one's place among them all, and its own
__global__ void totalWhitewater(const uint* counts, const unsigned long long* starts, uint numVoxels, unsigned long long* total){
    *total = starts[numVoxels - 1] + counts[numVoxels - 1];
}

//a particle's node cell, as rootCell finds the fluid's. The host's too: it shares a checkpoint's particles out between partitions by the same sums
__host__ __device__ inline uint whitewaterCell(double px, double py, double pz, const Grid& grid){
    uint x = floor((px - grid.negX) / grid.cellSize);
    uint y = floor((py - grid.negY) / grid.cellSize);
    uint z = floor((pz - grid.negZ) / grid.cellSize);
    return (x < grid.sizeX ? x : grid.sizeX - 1) + (y < grid.sizeY ? y : grid.sizeY - 1)*grid.sizeX + (z < grid.sizeZ ? z : grid.sizeZ - 1)*grid.sizeX*grid.sizeY;
}

//The sort keys of count particles from first on: twice each one's node cell, and one more for a particle made this substep, which those from firstNew
//on are: so each of those sorts after the rest of its cell, wherever the rest came from (exchangeWhitewater), and is told from them afterwards
//(nameWhitewater). One that's gone, or was never made, takes the cell past the last, which the sort puts at the end
__global__ void binWhitewater(uint first, uint count, const double* px, const double* py, const double* pz, const char* kinds, Grid grid, uint firstNew, uint* keys){
    uint index = first + threadIdx.x + blockIdx.x*blockDim.x;
    if(index < first + count){
        uint cell = kinds[index] != WHITEWATER_GONE ? whitewaterCell(px[index], py[index], pz[index], grid) : grid.sizeX*grid.sizeY*grid.sizeZ;
        keys[index] = 2*cell + (index >= firstNew ? 1 : 0);
    }
}

//which of the sorted are particles made this substep: 1 for each, 0 for the rest
__global__ void markNewWhitewater(uint count, const uint* keys, uint goneKey, uint* made){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < count){
        made[index] = keys[index] < goneKey ? keys[index] & 1 : 0;
    }
}

//Each particle made this substep gets its id: first, plus how many made this substep the sort put before it (before, the running sum of
//markNewWhitewater's). The sort's order doesn't depend on how the domain is split, so neither do the ids. And every key becomes the cell it's twice
__global__ void nameNewWhitewater(uint count, uint* keys, const uint* before, uint goneKey, unsigned long long first, unsigned long long* ids){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < count){
        uint key = keys[index];
        if(key < goneKey && (key & 1)){
            ids[index] = first + before[index];
        }
        keys[index] = key >> 1;
    }
}

__global__ void countWhitewater(const uint* sortedKeys, uint count, uint key, unsigned long long* before){   //how many of the sorted keys come before key
    uint low = 0, high = count;
    while(low < high){
        uint middle = low + (high - low) / 2;
        if(sortedKeys[middle] < key){
            low = middle + 1;
        }
        else{
            high = middle;
        }
    }
    *before = low;
}

__global__ void fillWhitewaterIndices(uint count, uint* indices){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < count){
        indices[index] = index;
    }
}

void WhitewaterPool::start(){
    gpuErrchk(cudaMalloc((void**)&found, 2*sizeof(unsigned long long)));
    gpuErrchk(cudaMallocHost((void**)&seen, 2*sizeof(unsigned long long)));
    gpuErrchk(cudaEventCreateWithFlags(&counted, cudaEventDisableTiming));
    gpuErrchk(cudaEventCreateWithFlags(&wanted, cudaEventDisableTiming));
    seen[0] = seen[1] = 0;
}

//Room for needed particles, the first keep of which stay what they are. The arrays grow by half again at least, so a pool that's filling isn't copied
//every substep, though not past most unless they have to: whitewater costs the memory of what there is of it, not of what there might be
void WhitewaterPool::reserve(uint needed, uint keep, uint most, cudaStream_t stream){
    if(needed <= slots){
        return;
    }
    size_t grown = std::max<size_t>(needed, std::min<size_t>(std::max<size_t>((size_t)slots + slots / 2, 65536), most));
    auto grow = [&](auto*& array, size_t each, bool kept){
        void* fresh;
        gpuErrchk(cudaMallocAsync(&fresh, each*grown, stream));
        if(kept && keep > 0){
            gpuErrchk(cudaMemcpyAsync(fresh, array, each*keep, cudaMemcpyDeviceToDevice, stream));
        }
        if(array != nullptr){
            gpuErrchk(cudaFreeAsync(array, stream));
        }
        array = (std::remove_reference_t<decltype(array)>)fresh;
    };
    for(double*& axis : position){
        grow(axis, sizeof(double), true);
    }
    for(float*& axis : velocity){
        grow(axis, sizeof(float), true);
    }
    grow(cells, sizeof(uint), true);
    grow(ids, sizeof(unsigned long long), true);
    grow(births, sizeof(float), true);
    grow(lives, sizeof(float), true);
    grow(radii, sizeof(float), true);
    grow(kinds, sizeof(char), true);
    grow(otherCells, sizeof(uint), false);
    grow(order, sizeof(uint), false);
    grow(spare8, 8, false);
    grow(spare4, 4, false);
    grow(spare1, 1, false);
    cub::DoubleBuffer<uint> keys(cells, otherCells);
    cub::DoubleBuffer<uint> origins((uint*)spare4, order);
    size_t bytes = 0;
    gpuErrchk(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, origins, (int)grown, 0, 32, stream));    //the most the sort can ask for
    if(bytes > sortBytes){
        if(sortSpace != nullptr){
            gpuErrchk(cudaFreeAsync(sortSpace, stream));
        }
        gpuErrchk(cudaMallocAsync(&sortSpace, bytes, stream));
        sortBytes = bytes;
    }
    slots = (uint)grown;
}

//done with what the surface makes this substep
void WhitewaterPool::forget(cudaStream_t stream){
    if(sources > 0){
        for(void* array : {(void*)counts, (void*)starts, (void*)folded, (void*)folds, (void*)tears}){
            gpuErrchk(cudaFreeAsync(array, stream));
        }
        counts = nullptr;
        starts = nullptr;
        folded = nullptr;
        folds = nullptr;
        tears = nullptr;
        sources = 0;
    }
    wanting = false;
}

void WhitewaterPool::release(cudaStream_t stream){
    if(found == nullptr){   //never started
        return;
    }
    for(void* array : {(void*)position[0], (void*)position[1], (void*)position[2], (void*)velocity[0], (void*)velocity[1], (void*)velocity[2], (void*)cells, (void*)ids,
                       (void*)births, (void*)lives, (void*)radii, (void*)kinds, (void*)otherCells, (void*)order, spare8, spare4, spare1, sortSpace, (void*)counts,
                       (void*)starts, (void*)folded, (void*)folds, (void*)tears}){
        if(array != nullptr){
            cudaFreeAsync(array, stream);
        }
    }
    cudaFree(found);
    found = nullptr;
    cudaFreeHost(seen);
    cudaEventDestroy(counted);
    cudaEventDestroy(wanted);
    slots = 0;
}

void Particles::setWhitewater(const WhitewaterSettings& settings){
    whitewater = settings;
    if(whitewater.on && whitewaterPool.found == nullptr){
        whitewaterPool.start();
    }
}

//this substep's key, for everything drawn in it
static unsigned long long whitewaterKey(unsigned long long seed, unsigned long long substep){
    return mixBits64(seed ^ mixBits64(substep));
}

//What the surface makes this substep (findWhitewaterSources), found as soon as the substep's length is, before the pressure solve: it reads the grid as
//P2G left it, which is all there is yet, and all it needs. How many particles that comes to goes to pinned memory as it's found, and has landed by the
//time the solve is over, which waits for the GPU more than once itself: so stepWhitewater knows how much room they take without a wait of its own
void Particles::findWhitewater(){
    if(!whitewater.on){
        return;
    }
    WhitewaterPool& pool = whitewaterPool;
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    int interiorWidth = 2<<refinementLevel;
    bool beside = obstacles.count() > 0 && numStoredNodes > 0;
    uint numVoxels = voxelIDsUsed.size();
    if(!twoPhase.on && numVoxels > 0){  //two phases have the liquid's share already (findFaceDensities)
        findLiquidShareAlone<<<numVoxels / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numVoxels, solveCodes.devPtr(), neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(),
            neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr(), beside ? obstacleSolids.devPtr() : nullptr, footprintDepth.devPtr(), voxelWeightsX.devPtr(),
            voxelWeightsY.devPtr(), voxelWeightsZ.devPtr(), (float)restParticlesPerVoxel, liquidShare.devPtr());
        context->fillGhosts(liquidShare.devPtr(), stream);
    }
    pool.forget(stream);    //what a substep that never got to stepWhitewater left
    if(numOwnVoxels == 0 || numOwnNodes == 0){
        return;
    }
    pool.sources = numOwnVoxels;
    gpuErrchk(cudaMallocAsync((void**)&pool.counts, sizeof(uint)*numOwnVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&pool.starts, sizeof(unsigned long long)*numOwnVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&pool.folded, sizeof(uint)*numOwnVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&pool.folds, sizeof(float)*numOwnVoxels, stream));
    gpuErrchk(cudaMallocAsync((void**)&pool.tears, sizeof(float)*numOwnVoxels, stream));
    gpuErrchk(cudaMemsetAsync(pool.counts, 0, sizeof(uint)*numOwnVoxels, stream));
    WhitewaterGrid g = whitewaterGrid();
    //what breaks off, and the difference in speed across it that it takes (see the top)
    double pull = std::sqrt((double)forces.gravity.x*forces.gravity.x + (double)forces.gravity.y*forces.gravity.y + (double)forces.gravity.z*forces.gravity.z);
    double sigma = whitewater.surfaceTension;
    double scale = whitewaterScale();
    WhitewaterSources s;
    s.solveCodes = solveCodes.devPtr();
    s.near = beside ? obstacleNear.devPtr() : nullptr;
    for(int dim = 0; dim < 3; ++dim){
        s.liquid[dim] = twoPhase.on ? liquidWeights[dim].devPtr() : g.weights[dim];
    }
    s.liquidUnit = twoPhase.on ? liquidWeightUnit : 0.0f;
    s.rest = (float)restParticlesPerVoxel;
    s.across = (float)voxelSize;
    s.breaks = (float)std::sqrt(BREAKUP_WEBER*sigma / scale + 2.0*pull*scale);
    s.perSpeed = (float)(whitewater.amount*whitewater.bubbles*AIR_FOLDED*whitewater.perVoxel / voxelSize*(scale / voxelSize));
    s.sprayPerSpeed = (float)(whitewater.amount*whitewater.spray*SPRAY_THROWN*whitewater.perVoxel / voxelSize);
    s.key = whitewaterKey(whitewater.seed, substepIndex);
    int tileWidth = interiorWidth + 2;
    findWhitewaterSources<<<numOwnNodes, 64, 5*sizeof(float)*tileWidth*tileWidth*tileWidth, stream>>>(g, s, (float)dt, pool.counts, pool.folded, pool.folds, pool.tears);
    gpuErrchk(cudaPeekAtLastError());
    placeWhitewater();
    totalWhitewater<<<1, 1, 0, stream>>>(pool.counts, pool.starts, numOwnVoxels, pool.found + 1);
    gpuErrchk(cudaMemcpyAsync(pool.seen + 1, pool.found + 1, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaEventRecord(pool.wanted, stream));
    pool.wanting = true;
}

//where each voxel's new particles go among them all: the running sum of the counts, in 64 bits, which no rate overflows
void Particles::placeWhitewater(){
    WhitewaterPool& pool = whitewaterPool;
    void* scratch = nullptr;
    size_t scratchBytes = 0;
    gpuErrchk(cub::DeviceScan::ExclusiveScan(scratch, scratchBytes, pool.counts, pool.starts, ::cuda::std::plus<>{}, 0ull, (int)pool.sources, stream));
    gpuErrchk(cudaMallocAsync(&scratch, scratchBytes, stream));
    gpuErrchk(cub::DeviceScan::ExclusiveScan(scratch, scratchBytes, pool.counts, pool.starts, ::cuda::std::plus<>{}, 0ull, (int)pool.sources, stream));
    gpuErrchk(cudaFreeAsync(scratch, stream));
}

//the size of what breaks off the surface, metres: the size it's easiest to break at, or a voxel if that's smaller (see the top)
double Particles::whitewaterScale() const{
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    double pull = std::sqrt((double)forces.gravity.x*forces.gravity.x + (double)forces.gravity.y*forces.gravity.y + (double)forces.gravity.z*forces.gravity.z);
    double smallest = pull > 0.0 ? std::sqrt(0.5*BREAKUP_WEBER*whitewater.surfaceTension / pull) : voxelSize;
    return std::min(voxelSize, smallest);
}

WhitewaterGrid Particles::whitewaterGrid(){
    return {numUsedGridNodes, numOwnNodes, nodeCells.devPtr(), cellToNode.devPtr(), nodeInteriorVoxels.devPtr(),
            {voxelsUx.devPtr(), voxelsUy.devPtr(), voxelsUz.devPtr()}, {voxelsUxOld.devPtr(), voxelsUyOld.devPtr(), voxelsUzOld.devPtr()},
            {voxelWeightsX.devPtr(), voxelWeightsY.devPtr(), voxelWeightsZ.devPtr()}, liquidShare.devPtr(), grid, 2<<refinementLevel,
            (float)(grid.cellSize / (2<<refinementLevel)), lettingGo() ? letGo.devPtr() : nullptr};
}

void Particles::stepWhitewater(){
    if(!whitewater.on){
        return;
    }
    WhitewaterPool& pool = whitewaterPool;
    uint count = whitewaterCount();
    unsigned long long wanted = 0;  //what findWhitewater found the surface makes: landed, as the count has, unless a solver one day never waits for the GPU
    if(pool.wanting){
        gpuErrchk(cudaEventSynchronize(pool.wanted));
        wanted = pool.seen[1];
        pool.wanting = false;
    }
    //the new ones go after the ones there are, as many as there's room for: all of them, short of the most this partition is to hold
    uint room = whitewater.capacity > count ? whitewater.capacity - count : 0;
    uint made = (uint)std::min<unsigned long long>(wanted, room);
    pool.dropped += wanted - made;
    pool.reserve(count + made, count, whitewater.capacity, stream);
    double voxelSize = grid.cellSize / (2<<refinementLevel);
    int interiorWidth = 2<<refinementLevel;
    int apronCells = (int)std::floor(radius);
    WhitewaterGrid g = whitewaterGrid();
    double pull = std::sqrt((double)forces.gravity.x*forces.gravity.x + (double)forces.gravity.y*forces.gravity.y + (double)forces.gravity.z*forces.gravity.z);
    double breakup = whitewaterScale();
    WhitewaterFluids fluids = {forces.gravity, (float)pull, whitewater.surfaceTension, twoPhase.on ? 1.0f / twoPhase.densityRatio : whitewater.airDensity,
                               twoPhase.on ? twoPhase.airViscosity : whitewater.airViscosity, whitewater.liquidViscosity, twoPhase.on, (float)breakup,
                               (float)std::sqrt(BREAKUP_WEBER*whitewater.surfaceTension / breakup + 2.0*pull*breakup), whitewater.dropletScale};
    WhitewaterParticles p = {pool.position[0], pool.position[1], pool.position[2], pool.velocity[0], pool.velocity[1], pool.velocity[2], pool.cells, pool.lives, pool.radii,
                             pool.births, pool.kinds, pool.ids};
    unsigned long long key = whitewaterKey(whitewater.seed, substepIndex);
    if(count > 0){
        if(numOwnNodes > 0){
            int tileWidth = 3*interiorWidth;
            flyWhitewater<<<numOwnNodes, WHITEWATER_THREADS, 7*sizeof(float)*tileWidth*tileWidth*tileWidth, stream>>>(g, fluids, p, count, (float)dt, mixBits64(key ^ 0x6a09e667f3bcc908ull));
        }
        WhitewaterEnds ends;
        ends.low[0] = grid.negX;
        ends.low[1] = grid.negY;
        ends.low[2] = grid.negZ;
        ends.high[0] = grid.negX + (double)grid.sizeX*grid.cellSize;
        ends.high[1] = grid.negY + (double)grid.sizeY*grid.cellSize;
        ends.high[2] = grid.negZ + (double)grid.sizeZ*grid.cellSize;
        ends.voxelSize = (float)voxelSize;
        ends.now = (float)elapsedTime;
        ends.maxAge = whitewater.maxAge;
        finishWhitewater<<<count / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(g, fluids, p, count, (float)dt, ends, sources, obstacles.state());
        gpuErrchk(cudaPeekAtLastError());
    }
    if(made > 0){
        VoxelPlaces places = {nodeIndexUsedVoxels.devPtr(), nodeCells.devPtr(), voxelIDsUsed.devPtr(), grid, interiorWidth, apronCells};
        if(made < wanted){  //less room than the surface would fill: every voxel makes its share of what there's room for, and where that falls short of the room, nothing
            thinWhitewater<<<numOwnNodes, 64, 0, stream>>>(places, (float)((double)made / (double)wanted), key, pool.counts, pool.folded);
            placeWhitewater();
            gpuErrchk(cudaMemsetAsync(pool.kinds + count, WHITEWATER_GONE, made, stream));
        }
        WhitewaterBirths b;
        b.counts = pool.counts;
        b.starts = pool.starts;
        b.folded = pool.folded;
        b.folds = pool.folds;
        b.tears = pool.tears;
        b.first = count;
        b.room = made;
        const uint* neighbors[6] = {neighborNx.devPtr(), neighborPx.devPtr(), neighborNy.devPtr(), neighborPy.devPtr(), neighborNz.devPtr(), neighborPz.devPtr()};
        for(int face = 0; face < 6; ++face){
            b.neighbors[face] = neighbors[face];
        }
        b.tension = whitewater.surfaceTension;
        b.scale = (float)whitewaterScale();
        b.dropletScale = whitewater.dropletScale;
        b.bubbleScale = whitewater.bubbleScale;
        b.foamLife = whitewater.foamLife;
        b.now = (float)elapsedTime;
        b.key = key;
        b.corner[0] = grid.negX;
        b.corner[1] = grid.negY;
        b.corner[2] = grid.negZ;
        emitWhitewater<<<numOwnNodes, 64, 0, stream>>>(places, g, b, pool.position[0], pool.position[1], pool.position[2], pool.velocity[0], pool.velocity[1], pool.velocity[2],
                                                       pool.ids, pool.births, pool.lives, pool.radii, pool.kinds);
        gpuErrchk(cudaPeekAtLastError());
    }
    pool.forget(stream);
    //in the order of their node cells again, those that are gone last
    uint bound = count + made;
    uint numCells = grid.sizeX*grid.sizeY*grid.sizeZ;
    if(bound > 0){
        binWhitewater<<<bound / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(0, bound, pool.position[0], pool.position[1], pool.position[2], pool.kinds, grid, count, pool.cells);
        orderWhitewater(bound);
    }
    pool.sorted = bound;
    if(numRanks > 1){
        exchangeWhitewater(made);
        return;
    }
    nameWhitewater(bound, pool.nextId);
    pool.nextId += made;    //every id there was room for, whether or not the surface used it
    //and how many are left, counted on the device; the host reads that when it next needs it, by when it's long since landed
    pool.count = 0;
    if(bound > 0){
        countWhitewater<<<1, 1, 0, stream>>>(pool.cells, bound, numCells, pool.found);
        gpuErrchk(cudaMemcpyAsync(pool.seen, pool.found, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaEventRecord(pool.counted, stream));
        pool.counting = true;
    }
}

//the particles made this substep get their ids, from first on in the order the sort left them in, and the first count keys become cells again:
//after the substep's last sort
void Particles::nameWhitewater(uint count, unsigned long long first){
    if(count == 0){
        return;
    }
    WhitewaterPool& pool = whitewaterPool;
    uint goneKey = 2*grid.sizeX*grid.sizeY*grid.sizeZ;
    uint* before = pool.otherCells;     //the keys' other array, which the sort has left spare
    markNewWhitewater<<<count / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(count, pool.cells, goneKey, before);
    void* scratch = nullptr;
    size_t scratchBytes = 0;
    gpuErrchk(cub::DeviceScan::ExclusiveSum(scratch, scratchBytes, before, before, (int)count, stream));
    gpuErrchk(cudaMallocAsync(&scratch, scratchBytes, stream));
    gpuErrchk(cub::DeviceScan::ExclusiveSum(scratch, scratchBytes, before, before, (int)count, stream));
    gpuErrchk(cudaFreeAsync(scratch, stream));
    nameNewWhitewater<<<count / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(count, pool.cells, before, goneKey, first, pool.ids);
    gpuErrchk(cudaPeekAtLastError());
}

//puts the first bound particles in the order of their keys, which binWhitewater left in the pool's cells: every array follows
void Particles::orderWhitewater(uint bound){
    WhitewaterPool& pool = whitewaterPool;
    int bits = keyBits(2*grid.sizeX*grid.sizeY*grid.sizeZ + 1);
    fillWhitewaterIndices<<<bound / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(bound, (uint*)pool.spare4);
    cub::DoubleBuffer<uint> keys(pool.cells, pool.otherCells);      //the sort goes back and forth between each pair, and says which it ended in
    cub::DoubleBuffer<uint> origins((uint*)pool.spare4, pool.order);
    size_t bytes = pool.sortBytes;
    gpuErrchk(cub::DeviceRadixSort::SortPairs(pool.sortSpace, bytes, keys, origins, (int)bound, 0, bits, stream));    //stable: those of a cell stay in the order they were in
    if(keys.Current() != pool.cells){
        std::swap(pool.cells, pool.otherCells);
    }
    if(origins.Current() != pool.order){
        pool.spare4 = pool.order;
        pool.order = origins.Current();
    }
    auto follow = [&](auto*& array, void*& spare){     //into the spare array its size, which it then leaves behind for the next
        auto* gathered = (std::remove_reference_t<decltype(array)>)spare;
        reorderGridIndices<<<bound / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(bound, pool.order, array, gathered);
        spare = array;
        array = gathered;
    };
    for(double*& axis : pool.position){
        follow(axis, pool.spare8);
    }
    follow(pool.ids, pool.spare8);
    for(float*& axis : pool.velocity){
        follow(axis, pool.spare4);
    }
    follow(pool.births, pool.spare4);
    follow(pool.lives, pool.spare4);
    follow(pool.radii, pool.spare4);
    follow(pool.kinds, pool.spare1);
    gpuErrchk(cudaPeekAtLastError());
}

//Whitewater that has left this partition's planes goes to the partition whose planes it's in now, and theirs comes here. After the sort each other
//partition's are a run of the sorted keys, as the cells run along z slowest; any partition's, not only the next one's, since a droplet isn't held to
//the fluid's speed and a long substep can take one past a thin partition. They arrive in partition order, those from below before this one's own and
//those from above after, which is the order one partition would have had them all in, and the sort that follows keeps it, the particles made this
//substep after the others of their cell (binWhitewater), where one partition's own sort leaves them. So the partitions' whitewater, one after another,
//is what a single partition would hold, in its order, and the ids of the ones made this substep run on from one partition's to the next's.
//This waits for the GPU, to know how many are leaving: with more than one partition only
void Particles::exchangeWhitewater(uint made){
    WhitewaterPool& pool = whitewaterPool;
    uint bound = pool.sorted;
    uint cellsPerPlane = grid.sizeX*grid.sizeY;
    std::vector<unsigned long long> before(numRanks + 1, 0);    //how many of the sorted come before each partition's first plane, and before the last one's end: the ones alive
    if(bound > 0){
        unsigned long long* found;
        gpuErrchk(cudaMallocAsync((void**)&found, sizeof(unsigned long long)*(numRanks + 1), stream));
        for(int other = 0; other <= numRanks; ++other){
            countWhitewater<<<1, 1, 0, stream>>>(pool.cells, bound, 2*partitionPlanes[other]*cellsPerPlane, found + other);    //the keys are twice the cells
        }
        gpuErrchk(cudaMemcpyAsync(before.data(), found, sizeof(unsigned long long)*(numRanks + 1), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaFreeAsync(found, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
    }
    size_t each = numRanks + 1;     //what each partition tells the rest: how many it has for each of them, and how many it made room for this substep
    std::vector<uint> leaving(each, 0);
    for(int other = 0; other < numRanks; ++other){
        leaving[other] = other == rank ? 0 : (uint)(before[other + 1] - before[other]);
    }
    leaving[numRanks] = made;
    std::vector<uint> all(numRanks*each);   //every partition's, in partition order
    transport->allGatherHost(leaving.data(), all.data(), sizeof(uint)*each);
    uint kept = (uint)(before[rank + 1] - before[rank]);
    uint arriving = 0, below = 0, moving = 0;
    unsigned long long madeBelow = 0, madeByAll = 0;
    bool any = false;
    for(int other = 0; other < numRanks; ++other){
        uint from = other == rank ? 0 : all[other*each + rank];
        arriving += from;
        below += other < rank ? from : 0;
        moving += leaving[other];
        madeBelow += other < rank ? all[other*each + numRanks] : 0;
        madeByAll += all[other*each + numRanks];
        for(int to = 0; to < numRanks; ++to){
            any = any || all[other*each + to] > 0;
        }
    }
    uint count = kept + arriving;
    pool.counting = false;
    pool.count = count;
    pool.sorted = count;
    std::vector<TransportSend> sends;
    std::vector<TransportReceive> receives;
    std::vector<void*> old;
    if(arriving + moving > 0){
        pool.reserve(count, bound, whitewater.capacity, stream);
        auto rebuild = [&](auto*& array, bool travels){    //an array's new contents: from the partitions below, kept, from those above. The same order of arrays on every partition
            using T = std::remove_pointer_t<std::remove_reference_t<decltype(array)>>;
            T* fresh;
            gpuErrchk(cudaMallocAsync((void**)&fresh, sizeof(T)*pool.slots, stream));
            if(kept > 0){
                gpuErrchk(cudaMemcpyAsync(fresh + below, array + before[rank], sizeof(T)*kept, cudaMemcpyDeviceToDevice, stream));
            }
            uint at = 0;
            for(int other = 0; other < numRanks && travels; ++other){
                uint from = other == rank ? kept : all[other*each + rank];
                if(other != rank && leaving[other] > 0){
                    sends.push_back({other, array + before[other], sizeof(T)*leaving[other]});
                }
                if(other != rank && from > 0){
                    receives.push_back({other, fresh + at, sizeof(T)*from});
                }
                at += from;
            }
            old.push_back(array);
            array = fresh;
        };
        for(double*& axis : pool.position){
            rebuild(axis, true);
        }
        for(float*& axis : pool.velocity){
            rebuild(axis, true);
        }
        rebuild(pool.ids, true);
        rebuild(pool.births, true);
        rebuild(pool.lives, true);
        rebuild(pool.radii, true);
        rebuild(pool.kinds, true);
        rebuild(pool.cells, false);     //the kept ones' keys; the arrivals' are found from where they are, none of them made this substep
    }
    if(any){    //every partition takes part if any has something to send, as all of them know
        transport->exchange(sends, receives, stream);
    }
    if(arriving + moving > 0){
        for(void* array : old){     //after it, in stream order
            gpuErrchk(cudaFreeAsync(array, stream));
        }
        if(arriving > 0){
            uint above = arriving - below;
            if(below > 0){
                binWhitewater<<<below / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(0, below, pool.position[0], pool.position[1], pool.position[2], pool.kinds, grid, 0xFFFFFFFFu, pool.cells);
            }
            if(above > 0){
                binWhitewater<<<above / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(below + kept, above, pool.position[0], pool.position[1], pool.position[2], pool.kinds, grid, 0xFFFFFFFFu, pool.cells);
            }
            orderWhitewater(count);
        }
    }
    nameWhitewater(count, pool.nextId + madeBelow);
    pool.nextId += madeByAll;
    gpuErrchk(cudaPeekAtLastError());
}

//how many there are: what the last sort counted, which is in pinned memory by the time anything asks (the substep has waited for the GPU since, more
//than once: building the grid does). The wait here is for the callers that ask straight after a substep, at a frame's end
uint Particles::whitewaterCount(){
    if(!whitewater.on){
        return 0;
    }
    WhitewaterPool& pool = whitewaterPool;
    if(pool.counting){
        gpuErrchk(cudaEventSynchronize(pool.counted));
        pool.count = (uint)pool.seen[0];
        pool.counting = false;
    }
    return pool.count;
}

//The whitewater as the cache takes it (cacheWriter.hu), each attribute a plane of every particle's value. A frame's: float32 P and v, the id, then age,
//life, radius and kind as float32s. A checkpoint's: P as float64, the id, v, then birth, life, radius and kind: the 8-byte planes first, so they sit on
//8-byte boundaries
__global__ void packWhitewater(uint count, const double* px, const double* py, const double* pz, const float* pu, const float* pv, const float* pw, const unsigned long long* ids,
                               const float* births, const float* lives, const float* radii, const char* kinds, bool checkpoint, double now, char* planes){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < count){
        size_t n = count;
        const double* positions[3] = {px, py, pz};
        const float* velocities[3] = {pu, pv, pw};
        float* floats;
        if(checkpoint){
            #pragma unroll
            for(int axis = 0; axis < 3; ++axis){
                ((double*)planes)[axis*n + index] = positions[axis][index];
            }
            ((unsigned long long*)planes)[3*n + index] = ids[index];
            floats = (float*)(planes + 32*n);
            #pragma unroll
            for(int axis = 0; axis < 3; ++axis){
                floats[axis*n + index] = velocities[axis][index];
            }
            floats[3*n + index] = births[index];
            floats += 4*n;
        }
        else{
            floats = (float*)planes;
            #pragma unroll
            for(int axis = 0; axis < 3; ++axis){
                floats[axis*n + index] = (float)positions[axis][index];
                floats[(3 + axis)*n + index] = velocities[axis][index];
            }
            ((unsigned long long*)planes)[3*n + index] = ids[index];    //6n floats in
            floats += 8*n;
            floats[index] = fmaxf((float)(now - births[index]), 0.0f);  //its age: never negative, though a birth can round to just after now
            floats += n;
        }
        floats[index] = lives[index];
        floats[n + index] = radii[index];
        floats[2*n + index] = (float)kinds[index];
    }
}

void Particles::copyWhitewaterToHost(char* host, bool checkpoint, cudaEvent_t copied){
    uint count = whitewaterCount();
    if(count > 0){
        WhitewaterPool& pool = whitewaterPool;
        size_t bytes = (size_t)count*(checkpoint ? 60 : 48);
        char* planes;
        gpuErrchk(cudaMallocAsync((void**)&planes, bytes, stream));
        packWhitewater<<<count / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(count, pool.position[0], pool.position[1], pool.position[2], pool.velocity[0], pool.velocity[1], pool.velocity[2],
            pool.ids, pool.births, pool.lives, pool.radii, pool.kinds, checkpoint, elapsedTime, planes);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaMemcpyAsync(host, planes, bytes, cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaFreeAsync(planes, stream));
    }
    gpuErrchk(cudaEventRecord(copied, stream));
}

//A checkpoint's whitewater, every partition's of the bake that wrote it one after another: the order one partition holds them in. This partition takes
//the ones in its own planes, by the cell the device would find each in, and they're alive and in order as if no bake had stopped. nextId: the id
//the next one made gets
void Particles::setWhitewaterParticles(uint count, const double* positions, const unsigned long long* ids, const float* velocities, const float* births, const float* lives,
                                       const float* radii, const float* kinds, unsigned long long nextId){
    WhitewaterPool& pool = whitewaterPool;
    pool.nextId = nextId;
    uint cellsPerPlane = grid.sizeX*grid.sizeY;
    std::vector<uint> mine;
    for(uint particle = 0; particle < count; ++particle){
        uint plane = whitewaterCell(positions[particle], positions[(size_t)count + particle], positions[2*(size_t)count + particle], grid) / cellsPerPlane;
        if(plane >= boxLo && plane < boxHi){
            mine.push_back(particle);
        }
    }
    uint kept = (uint)mine.size();
    pool.counting = false;
    pool.count = kept;
    pool.sorted = kept;
    if(kept == 0){
        return;
    }
    pool.reserve(kept, 0, whitewater.capacity, stream);
    auto send = [&](auto* into, const auto* from, size_t offset){    //from pageable memory: there before the call returns
        using T = std::remove_pointer_t<decltype(into)>;
        std::vector<T> own(kept);
        for(uint particle = 0; particle < kept; ++particle){
            own[particle] = (T)from[offset + mine[particle]];
        }
        gpuErrchk(cudaMemcpyAsync(into, own.data(), sizeof(T)*kept, cudaMemcpyHostToDevice, stream));
    };
    for(int axis = 0; axis < 3; ++axis){
        send(pool.position[axis], positions, (size_t)axis*count);
        send(pool.velocity[axis], velocities, (size_t)axis*count);
    }
    send(pool.ids, ids, 0);
    send(pool.births, births, 0);
    send(pool.lives, lives, 0);
    send(pool.radii, radii, 0);
    send(pool.kinds, kinds, 0);
    binWhitewater<<<kept / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(0, kept, pool.position[0], pool.position[1], pool.position[2], pool.kinds, grid, 0xFFFFFFFFu, pool.cells);
    orderWhitewater(kept);      //already in order, so it moves nothing
    nameWhitewater(kept, 0);    //none of them new: their keys to cells
    gpuErrchk(cudaStreamSynchronize(stream));
}

//a 64-bit hash of a whitewater particle's exact state
__device__ inline unsigned long long whitewaterHash(double x, double y, double z, float u, float v, float w, unsigned long long id, float life, float radius, char kind){
    unsigned long long hash = mixBits64((unsigned long long)__double_as_longlong(x));
    hash = mixBits64(hash ^ (unsigned long long)__double_as_longlong(y));
    hash = mixBits64(hash ^ (unsigned long long)__double_as_longlong(z));
    hash = mixBits64(hash ^ ((unsigned long long)__float_as_uint(v) << 32 | __float_as_uint(u)));
    hash = mixBits64(hash ^ ((unsigned long long)__float_as_uint(life) << 32 | __float_as_uint(w)));
    hash = mixBits64(hash ^ id);
    return mixBits64(hash ^ ((unsigned long long)(unsigned char)kind << 32 | __float_as_uint(radius)));
}

//the particles' count by kind, the sums of their hashes' halves and the fastest: integers, added up the same in any order
__global__ void sumWhitewater(uint count, const double* px, const double* py, const double* pz, const float* pu, const float* pv, const float* pw, const unsigned long long* ids,
                              const float* lives, const float* radii, const char* kinds, unsigned long long* sums, unsigned int* fastest){
    unsigned long long mine[5] = {0, 0, 0, 0, 0};
    unsigned int speed = 0;
    for(uint index = threadIdx.x + blockIdx.x*blockDim.x; index < count; index += blockDim.x*gridDim.x){
        char kind = kinds[index];
        if(kind == WHITEWATER_GONE){
            continue;
        }
        unsigned long long hash = whitewaterHash(px[index], py[index], pz[index], pu[index], pv[index], pw[index], ids[index], lives[index], radii[index], kind);
        ++mine[(int)kind];
        mine[3] += hash >> 32;
        mine[4] += hash & 0xFFFFFFFFull;
        speed = max(speed, __float_as_uint(sqrtf(pu[index]*pu[index] + pv[index]*pv[index] + pw[index]*pw[index])));
    }
    #pragma unroll
    for(int sum = 0; sum < 5; ++sum){
        if(mine[sum] != 0){
            atomicAdd(sums + sum, mine[sum]);
        }
    }
    atomicMax(fastest, speed);
}

void Particles::whitewaterStatistics(WhitewaterStatistics& statistics){
    uint count = whitewaterCount();
    WhitewaterPool& pool = whitewaterPool;
    statistics.dropped += pool.dropped;
    if(count == 0){
        return;
    }
    unsigned long long* sums;
    unsigned long long found[6];
    gpuErrchk(cudaMallocAsync((void**)&sums, 6*sizeof(unsigned long long), stream));
    gpuErrchk(cudaMemsetAsync(sums, 0, 6*sizeof(unsigned long long), stream));
    sumWhitewater<<<std::min(count / BLOCKSIZE + 1, 1024u), BLOCKSIZE, 0, stream>>>(count, pool.position[0], pool.position[1], pool.position[2], pool.velocity[0], pool.velocity[1],
        pool.velocity[2], pool.ids, pool.lives, pool.radii, pool.kinds, sums, (unsigned int*)(sums + 5));
    gpuErrchk(cudaMemcpyAsync(found, sums, sizeof(found), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaFreeAsync(sums, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
    for(int kind = 0; kind < 3; ++kind){
        statistics.kinds[kind] += found[kind];
        statistics.count += found[kind];
    }
    statistics.hashHigh += found[3];
    statistics.hashLow += found[4];
    unsigned int bits = (unsigned int)found[5];
    float speed;
    memcpy(&speed, &bits, sizeof(speed));
    statistics.fastest = std::max(statistics.fastest, speed);
}
