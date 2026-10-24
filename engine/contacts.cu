//Copyright 2026 Taimen LLC

//Contacts between the dynamic bodies and what they touch (bodies.hu): the domain's walls, the obstacles, and each other.
//
//Each body carries points over its surface, found once from its distance field (setUpContacts). Every substep, after the pressure and the friction
//have moved it (bodies.cu), each point is held against every partner whose reach it's in: a wall's plane, an obstacle's distance field (moving or
//not), another body's. A point under a partner's surface, or near enough that it would reach it this substep at the speed they close at, is a
//candidate (within three quarters of a voxel: the points are a voxel apart), and each body keeps up to CONTACT_SLOTS of its candidates against each partner: the deepest, the farthest from it, the one making the
//largest triangle with those, and the one farthest from all three, so a box resting on a floor is held at its corners. Every choice is an atomic
//minimum of (how good, which point), so it's the same whatever order the points come in, and every partition, which moves the bodies alike, finds the
//same contacts.
//
//The contacts are solved by sequential impulses (Catto's, as Bullet and Box2D have them): each contact in turn gets the impulse along its normal that
//brings the speed at which the two separate there to its target, never pulling (the impulse so far stays at or over 0), and impulses along two
//tangents that stop them sliding, within Coulomb's cone (friction times the normal impulse). The target is how close the gap lets them come this
//substep, or for a contact already touching, its bounce: the speed it hit at, times the bodies' bounce, past a threshold under which nothing bounces
//(two substeps of falling), so a resting body doesn't skip. Contacts sharing no body are independent, so they're dealt into batches no two of a body's
//in one (greedy, in slot order), and a batch is solved in parallel, one batch after another, by one block. A body sunk into its partner is pushed
//out by a velocity of its own (split impulse): it moves the body that substep and isn't its velocity, so getting out of the floor never makes
//anything bounce.
//
//Each contact's impulse carries over to the next substep, which applies it before the pressure solve (warm starting): a body resting on the bottom
//under water is held up from the substep's start, so the water sees it at rest, rather than seeing it fall and the floor stop it after. Its contact,
//found again after the solve by its surface point, starts from that impulse and corrects it; one not found again is taken back
//
//Friction and bounce combine as Bullet combines them, as products of the body's own and its partner's: another body's own, or a wall's or a still or
//moved obstacle's from the scene, which are RBD Bullet Solver's for its ground plane and collisions unless given (bounce 0.5, friction 1), so a piece
//of Houdini's defaults (bounce 0.25) bounces off a floor as it does there, at 0.125

#include "bodies.hu"
#include "algorithms/parallelPrefixSumKernels.hu"
#include <cub/cub.cuh>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

static constexpr int CONTACT_THREADS = 256;
static constexpr int VELOCITY_ITERATIONS = 64;
static constexpr int POSITION_ITERATIONS = 16;
static constexpr float PUSH_OUT = 0.2f;         //of how far a body is sunk into its partner, taken out each substep
static constexpr float SLOP = 0.05f;            //voxels a body may be sunk without being pushed out, so a resting one isn't pushed out and back
static constexpr int MAX_BATCHES = 256;
static constexpr int MAX_SURFACE_POINTS = 1 << 18;  //a body's: its lattice is coarsened to stay under

__device__ inline float3 cross(float3 a, float3 b){
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__ inline float3 matrixTimes(const float m[9], float3 v){
    return make_float3(m[0]*v.x + m[1]*v.y + m[2]*v.z, m[3]*v.x + m[4]*v.y + m[5]*v.z, m[6]*v.x + m[7]*v.y + m[8]*v.z);
}

// ---- each body's surface points ----

//the lattice points h apart over a body's bounds within half a spacing of its surface, a flag each: at least one layer of them for every face
__global__ void flagSurfaceSamples(ObstacleState shape, float3 low, float h, int3 samples, uint* flags){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < (uint)(samples.x*samples.y*samples.z)){
        float3 p = make_float3(low.x + (index % samples.x + 0.5f)*h, low.y + (index / samples.x % samples.y + 0.5f)*h, low.z + (index / (samples.x*samples.y) + 0.5f)*h);
        float3 normal;
        flags[index] = fabsf(obstacleDistance(shape, p, normal)) <= 0.5f*h;
    }
}

//and each moved onto the surface along its distance's gradient, into points in lattice order (ends: the flags' running sum, each one's end)
__global__ void placeSurfaceSamples(ObstacleState shape, float3 low, float h, int3 samples, const uint* ends, int body, float3* points, int* bodies){
    uint index = threadIdx.x + blockIdx.x*blockDim.x;
    if(index < (uint)(samples.x*samples.y*samples.z)){
        uint end = ends[index], start = index == 0 ? 0 : ends[index - 1];
        if(end != start){
            float3 p = make_float3(low.x + (index % samples.x + 0.5f)*h, low.y + (index / samples.x % samples.y + 0.5f)*h, low.z + (index / (samples.x*samples.y) + 0.5f)*h);
            float3 normal;
            float distance = obstacleDistance(shape, p, normal);
            points[start] = p - distance*normal;
            bodies[start] = body;
        }
    }
}

void Bodies::setUpContacts(const ObstacleSet& obstacleSet, double voxelSize){
    std::vector<float3> allPoints;
    std::vector<int> allBodies;
    for(int body = 0; body < numBodies; ++body){
        ObstacleState shape = obstacleSet.hostState(obstacles[body]);    //in its own space: the world's taken for it
        for(int i = 0; i < 12; ++i){
            shape.worldToObject[i] = i % 5 == 0 ? 1.0f : 0.0f;
        }
        shape.frame = nullptr;
        const BodyMass& mass = masses[body];
        float h = (float)voxelSize;
        //from a voxel and a quarter under its bounds: started a whole voxel under, a box's faces fall halfway between two layers of the lattice, which
        //then both miss them by just half a spacing, and rounding decides whether the face gets any points
        float3 low = make_float3(mass.objectLow.x - 1.25f*h, mass.objectLow.y - 1.25f*h, mass.objectLow.z - 1.25f*h);
        float3 size = make_float3(mass.objectHigh.x - mass.objectLow.x + 2.5f*h, mass.objectHigh.y - mass.objectLow.y + 2.5f*h, mass.objectHigh.z - mass.objectLow.z + 2.5f*h);
        //about one a voxel of surface: a body of a lot of surface gets coarser points, at most MAX_SURFACE_POINTS of them
        double area = 2.0*((double)size.x*size.y + (double)size.y*size.z + (double)size.z*size.x);
        if(area / ((double)h*h) > MAX_SURFACE_POINTS){
            h = (float)std::sqrt(area / MAX_SURFACE_POINTS);
        }
        int3 samples = make_int3((int)std::ceil(size.x / h), (int)std::ceil(size.y / h), (int)std::ceil(size.z / h));
        uint count = (uint)samples.x*samples.y*samples.z;
        uint* ends;
        float3* points;
        int* owners;
        gpuErrchk(cudaMalloc((void**)&ends, sizeof(uint)*count));
        flagSurfaceSamples<<<count / CONTACT_THREADS + 1, CONTACT_THREADS, 0, stream>>>(shape, low, h, samples, ends);
        cudaParallelPrefixSum(count, ends, stream);
        uint found = 0;
        gpuErrchk(cudaMemcpyAsync(&found, ends + count - 1, sizeof(uint), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));   //once, at the start
        gpuErrchk(cudaMalloc((void**)&points, sizeof(float3)*std::max(found, 1u)));
        gpuErrchk(cudaMalloc((void**)&owners, sizeof(int)*std::max(found, 1u)));
        placeSurfaceSamples<<<count / CONTACT_THREADS + 1, CONTACT_THREADS, 0, stream>>>(shape, low, h, samples, ends, body, points, owners);
        std::vector<float3> hostPoints(found);
        std::vector<int> hostOwners(found);
        if(found > 0){
            gpuErrchk(cudaMemcpyAsync(hostPoints.data(), points, sizeof(float3)*found, cudaMemcpyDeviceToHost, stream));
            gpuErrchk(cudaMemcpyAsync(hostOwners.data(), owners, sizeof(int)*found, cudaMemcpyDeviceToHost, stream));
        }
        gpuErrchk(cudaStreamSynchronize(stream));
        cudaFree(ends);
        cudaFree(points);
        cudaFree(owners);
        allPoints.insert(allPoints.end(), hostPoints.begin(), hostPoints.end());
        allBodies.insert(allBodies.end(), hostOwners.begin(), hostOwners.end());
        std::cerr<<"Body "<<body<<": "<<found<<" surface points for its contacts, "<<h / voxelSize<<" voxels apart\n";
    }
    numSurfacePoints = (int)allPoints.size();
    partners = 6 + obstacleSet.count();
    gpuErrchk(cudaMalloc((void**)&surfacePoints, sizeof(float3)*std::max(numSurfacePoints, 1)));
    gpuErrchk(cudaMalloc((void**)&surfaceBodies, sizeof(int)*std::max(numSurfacePoints, 1)));
    if(numSurfacePoints > 0){
        gpuErrchk(cudaMemcpyAsync(surfacePoints, allPoints.data(), sizeof(float3)*numSurfacePoints, cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(surfaceBodies, allBodies.data(), sizeof(int)*numSurfacePoints, cudaMemcpyHostToDevice, stream));
    }
    //what the contacts take per body; the pairs and their slots are made as each substep finds them (makePairs)
    gpuErrchk(cudaMalloc((void**)&deviceObstacles, sizeof(int)*numBodies));
    gpuErrchk(cudaMemcpyAsync(deviceObstacles, obstacles.data(), sizeof(int)*numBodies, cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMalloc((void**)&queryBoxes, 2*sizeof(float3)*numBodies));
    gpuErrchk(cudaMalloc((void**)&pairEnds, sizeof(uint)*numBodies));
    gpuErrchk(cudaMalloc((void**)&pushSums, sizeof(ExactSum)*BODY_LANES*numBodies));
    gpuErrchk(cudaMemsetAsync(pushSums, 0, sizeof(ExactSum)*BODY_LANES*numBodies, stream));
    gpuErrchk(cudaMalloc((void**)&islands, sizeof(int)*numBodies));
    gpuErrchk(cudaMalloc((void**)&islandBatches, sizeof(int)*(MAX_BATCHES + 2)*(size_t)numBodies));
    gpuErrchk(cudaMalloc((void**)&bodyBatches, sizeof(unsigned long long)*(MAX_BATCHES / 64)*(size_t)numBodies));
    gpuErrchk(cudaMalloc((void**)&solveBodies, sizeof(SolveBody)*numBodies));
    gpuErrchk(cudaMallocHost((void**)&pairTotal, sizeof(uint)));
    *pairTotal = 0;
    gpuErrchk(cudaEventCreateWithFlags(&pairsCounted, cudaEventDisableTiming));
    gpuErrchk(cudaEventRecord(pairsCounted, stream));
    numPairs = 0;
    numLast = 0;
    gpuErrchk(cudaStreamSynchronize(stream));
}

// ---- finding them ----

//what the contact passes need of the world: the obstacles where they are (the bodies' frames among them), which body each obstacle is, the domain
struct ContactWorld{
    Obstacles obstacles;                    //each one's body, if it's dynamic, in its state
    const int* obstacleOf;                  //per body, its obstacle, on the GPU
    float3 low, high;   //the domain's walls
    float wallBounce[6], wallContactFriction[6];    //what a body meets at each (Bodies::setWallContact)
    float voxel;
    float dt;
    float stillSpeed;   //the speed apart under which a contact doesn't bounce
};

//a float as an unsigned integer that orders as the floats do, and back
__device__ inline uint orderable(float value){
    uint bits = __float_as_uint(value);
    return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}

__device__ inline float fromOrderable(uint bits){
    return __uint_as_float(bits & 0x80000000u ? bits & 0x7FFFFFFFu : ~bits);
}

//a body's surface point in the world, where its frame puts it: own space = W x + t, so x = W^T (own - t)
__device__ inline float3 surfacePoint(const ContactWorld& world, int body, float3 local){
    const float* m = toObject(world.obstacles.items[world.obstacleOf[body]]);
    float3 d = make_float3(local.x - m[3], local.y - m[7], local.z - m[11]);
    return make_float3(m[0]*d.x + m[4]*d.y + m[8]*d.z, m[1]*d.x + m[5]*d.y + m[9]*d.z, m[2]*d.x + m[6]*d.y + m[10]*d.z);
}

//how far a point is from a partner, negative inside it, the partner's normal there (out of it), and its velocity there; false if the point's past its
//reach. Partners 0 to 5 are the walls, -x, +x, -y, +y, -z, +z; then the obstacles
__device__ inline bool partnerAt(const ContactWorld& world, int partner, float3 p, float& distance, float3& normal, float3& velocity){
    if(partner < 6){
        int axis = partner/2;
        bool upper = partner % 2;
        distance = upper ? component(world.high, axis) - component(p, axis) : component(p, axis) - component(world.low, axis);
        float sign = upper ? -1.0f : 1.0f;
        normal = make_float3(axis == 0 ? sign : 0.0f, axis == 1 ? sign : 0.0f, axis == 2 ? sign : 0.0f);
        velocity = make_float3(0.0f, 0.0f, 0.0f);
        return true;
    }
    const ObstacleState& obstacle = world.obstacles.items[partner - 6];
    float3 low = obstacle.frame != nullptr ? obstacle.frame->reachLow : obstacle.reachLow;
    float3 high = obstacle.frame != nullptr ? obstacle.frame->reachHigh : obstacle.reachHigh;
    if(p.x < low.x || p.y < low.y || p.z < low.z || p.x > high.x || p.y > high.y || p.z > high.z){
        return false;
    }
    distance = obstacleDistance(obstacle, p, normal);
    velocity = obstacleVelocity(obstacle, p);
    return true;
}

//One pass of the choice: every surface point against every partner it's a candidate for, and the best for this pass's slot of each manifold kept,
//as an atomic minimum of (how good, the point), pass 0 the deepest, 1 the farthest from it, 2 the one making the largest triangle with the two, 3 the
//farthest from all three. A manifold whose earlier pass found nothing has nothing for this one
__global__ void pickContacts(int pass, ContactWorld world, const float3* points, const int* pointBodies, int numPoints, const uint* pairEnds, const int* pairPartners,
                             unsigned long long* picks){
    for(int index = threadIdx.x + blockIdx.x*blockDim.x; index < numPoints; index += blockDim.x*gridDim.x){
        int body = pointBodies[index];
        const ObstacleState& self = world.obstacles.items[world.obstacleOf[body]];
        float3 p = surfacePoint(world, body, points[index]);
        float3 moving = obstacleVelocity(self, p);
        for(uint pair = body == 0 ? 0u : pairEnds[body - 1]; pair < pairEnds[body]; ++pair){    //what it can touch this substep
            int partner = pairPartners[pair];
            float distance;
            float3 normal, velocity;
            if(!partnerAt(world, partner, p, distance, normal, velocity)){
                continue;
            }
            //touching, or able to reach it this substep: within three quarters of a voxel, as a point near an edge can sit half a voxel up the face
            //beside the one touching (its nearest when the points were made), and is moved to the edge if picked (nearestToPartner); and as far again
            //as the two could close in a substep at their speeds. At the speed they close at now isn't enough: a box falling on one that's falling too
            //closes on it at none until the floor stops the one below
            float closing = sqrtf(dot(moving, moving)) + sqrtf(dot(velocity, velocity));
            if(!(distance < 0.75f*world.voxel + world.dt*closing)){
                continue;
            }
            int first = pair*CONTACT_SLOTS;
            float score = distance;
            if(pass > 0){
                float3 picked[CONTACT_SLOTS - 1];
                bool ready = true;
                for(int k = 0; k < pass; ++k){
                    unsigned long long key = picks[first + k];
                    ready = ready && key != ~0ull && (int)(key & 0xFFFFFFFFull) != index;
                    if(ready){
                        picked[k] = surfacePoint(world, body, points[(int)(key & 0xFFFFFFFFull)]);
                    }
                }
                //spread over the points as near as the deepest, within half a voxel of it: not up the sides of a box falling flat
                if(!ready || distance > fromOrderable((uint)(picks[first] >> 32)) + 0.5f*world.voxel){
                    continue;
                }
                float3 d0 = p - picked[0];
                if(pass == 1){
                    score = -dot(d0, d0);
                }
                else if(pass == 2){
                    float3 area = cross(picked[1] - picked[0], d0);
                    score = -dot(area, area);
                }
                else{
                    float3 d1 = p - picked[1], d2 = p - picked[2];
                    score = -fminf(dot(d0, d0), fminf(dot(d1, d1), dot(d2, d2)));
                }
            }
            atomicMin(picks + first + pass, (unsigned long long)orderable(score) << 32 | (unsigned int)index);
        }
    }
}

//A picked surface point moved along the body's surface to where its partner is nearest (Macklin, Erleben, Muller, Chentanez, Jeschke and Corse, Local
//optimization for robust signed distance field collision, 2020): the surface points are a voxel apart, and a ball's nearest to the floor is up to half
//a voxel from its bottom, where the floor's push would turn it. Each step goes down the partner's distance along the surface, as far as the angle
//between the body's normal and the partner's over the surface's curvature there (measured from the body's normals half a voxel apart), then back onto
//the surface, three quarters of a voxel in all at most, what the points' spacing leaves: on a ball, one step to its bottom from the point nearest it;
//on a flat face, none, so a box keeps its corners however it's tipped
__device__ inline float3 nearestToPartner(const ContactWorld& world, int body, int partner, float3 x){
    const ObstacleState& self = world.obstacles.items[world.obstacleOf[body]];
    float3 own;
    x = x - obstacleDistance(self, x, own)*own;
    float budget = 0.75f*world.voxel;  //no farther than the points are apart: a flat face tipped a hair would otherwise send them all to its low edge
    for(int step = 0; step < 4 && budget > 0.0f; ++step){
        float distance;
        float3 normal, velocity;
        if(!partnerAt(world, partner, x, distance, normal, velocity)){
            break;
        }
        float3 slide = normal - dot(normal, own)*own;   //where the partner's distance grows, along the surface
        float sine = sqrtf(dot(slide, slide));
        if(sine < 1.0e-4f){
            break;
        }
        float3 away = (-1.0f / sine)*slide;
        float h = 0.5f*world.voxel;
        float3 there = x + h*away, aside;
        there = there - obstacleDistance(self, there, aside)*aside;
        obstacleDistance(self, there, aside);
        float3 bend = aside - own;
        float curvature = sqrtf(dot(bend, bend)) / h;
        float arc = asinf(fminf(sine, 1.0f));
        float length = fminf(curvature > 0.0f ? arc / curvature : budget, budget);
        budget -= length;
        x = x + length*away;
        x = x - obstacleDistance(self, x, own)*own;
    }
    return x;
}

//per slot, its contact from its pick: where, its normal and gap, the levers, the effective masses, its friction, the speed apart it has to reach and the
//push out it needs, at its pick moved to where the partner is nearest. One within a quarter voxel of the manifold's earlier picks, so moved, adds
//nothing, and the slot's left empty
__global__ void buildContacts(ContactWorld world, const float3* points, const int* pairBodies, const int* pairPartners, const BodyState* states, const BodyMass* masses,
                              const BodyCoupling* couplings, const unsigned long long* picks, int numSlots, BodyContact* contacts){
    int slot = threadIdx.x + blockIdx.x*blockDim.x;
    if(slot >= numSlots){
        return;
    }
    BodyContact& contact = contacts[slot];
    contact.point = -1;
    unsigned long long key = picks[slot];
    if(key == ~0ull){
        return;
    }
    int k = slot % CONTACT_SLOTS, manifold = slot / CONTACT_SLOTS;
    int body = pairBodies[manifold], partner = pairPartners[manifold];
    int point = (int)(key & 0xFFFFFFFFull);
    float3 p = nearestToPartner(world, body, partner, surfacePoint(world, body, points[point]));
    for(int earlier = 0; earlier < k; ++earlier){
        unsigned long long before = picks[manifold*CONTACT_SLOTS + earlier];
        float3 q = nearestToPartner(world, body, partner, surfacePoint(world, body, points[(int)(before & 0xFFFFFFFFull)])) - p;
        if(dot(q, q) < 0.0625f*world.voxel*world.voxel){
            return;
        }
    }
    float distance;
    float3 normal, velocity;
    partnerAt(world, partner, p, distance, normal, velocity);
    //Which way they push, and whether this is a contact at all. Where they touch, the gap between them is least, so the partner's normal is square to
    //the body's surface there (the refinement looks for just that): a point on a flat face of the body whose partner's normal runs more than 30
    //degrees along that face isn't where they touch, but a point near it (on a box overhanging another's edge, beside the edge, where the edge's
    //normal points sideways), and isn't a contact. Where both surfaces are smooth they agree; a distance field's normal is ambiguous at an edge, where
    //it turns from one face's to the next's, so the normal is that of whichever surface is flat there: a box's corner resting on a face takes the
    //face's, a face resting on another box's edge takes its own, reversed. How flat each is: how far its normal turns over a quarter voxel, along the
    //surface either way; a wall is flat
    {
        const ObstacleState& self = world.obstacles.items[world.obstacleOf[body]];
        float3 own;
        obstacleDistance(self, p, own);
        float3 axis = fabsf(own.x) < 0.57f ? make_float3(1.0f, 0.0f, 0.0f) : fabsf(own.y) < 0.57f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(0.0f, 0.0f, 1.0f);
        float3 u = cross(own, axis);
        u = (1.0f / sqrtf(dot(u, u)))*u;
        float3 w = cross(own, u);
        float h = 0.25f*world.voxel;
        float ownTurn = 0.0f, withTurn = 0.0f;
        for(int k = 0; k < 4; ++k){
            float3 there = p + (k & 1 ? -h : h)*(k < 2 ? u : w), turned;
            obstacleDistance(self, there, turned);
            ownTurn = fmaxf(ownTurn, 1.0f - dot(turned, own));
            if(partner >= 6){
                obstacleDistance(world.obstacles.items[partner - 6], there, turned);
                withTurn = fmaxf(withTurn, 1.0f - dot(turned, normal));
            }
        }
        float3 along = normal - dot(normal, own)*own;
        if(ownTurn < 0.05f && dot(along, along) > 0.25f){
            return;
        }
        if(ownTurn < withTurn){
            normal = -1.0f*own;
        }
    }
    int other = partner >= 6 ? world.obstacles.items[partner - 6].body : -1;
    contact.point = point;
    contact.other = other;
    contact.body = body;
    contact.partner = partner;
    contact.k = k;
    contact.at = p;
    contact.normal = normal;
    contact.partnerVelocity = velocity;
    {
        const float* m = toObject(world.obstacles.items[world.obstacleOf[body]]);
        contact.local = make_float3(m[0]*p.x + m[1]*p.y + m[2]*p.z + m[3], m[4]*p.x + m[5]*p.y + m[6]*p.z + m[7], m[8]*p.x + m[9]*p.y + m[10]*p.z + m[11]);
    }
    contact.dt = world.dt;
    contact.color = -1;
    //two tangents square to the normal and to each other
    float3 axis = fabsf(normal.x) < 0.57f ? make_float3(1.0f, 0.0f, 0.0f) : fabsf(normal.y) < 0.57f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(0.0f, 0.0f, 1.0f);
    float3 t1 = cross(normal, axis);
    t1 = (1.0f / sqrtf(dot(t1, t1)))*t1;
    contact.tangents[0] = t1;
    contact.tangents[1] = cross(normal, t1);
    int involved[2] = {body, other};
    float3 speed = make_float3(0.0f, 0.0f, 0.0f);   //how fast they separate there, body less partner
    for(int side = 0; side < 2; ++side){
        int which = involved[side];
        if(which < 0){
            contact.lever[side] = make_float3(0.0f, 0.0f, 0.0f);
            continue;
        }
        const BodyState& state = states[which];
        float3 centre = make_float3((float)state.position[0], (float)state.position[1], (float)state.position[2]);
        float3 lever = p - centre;
        contact.lever[side] = lever;
        float3 v = make_float3((float)state.velocity[0], (float)state.velocity[1], (float)state.velocity[2]);
        float3 w = make_float3((float)state.angularVelocity[0], (float)state.angularVelocity[1], (float)state.angularVelocity[2]);
        float3 there = v + cross(w, lever);
        speed = side == 0 ? speed + there : speed - there;
    }
    if(other < 0){
        speed = speed - velocity;
    }
    for(int direction = 0; direction < 3; ++direction){
        float3 u = direction == 0 ? normal : contact.tangents[direction - 1];
        float k = 0.0f;
        for(int side = 0; side < 2; ++side){
            int which = involved[side];
            if(which < 0){
                continue;
            }
            const BodyCoupling& coupling = couplings[which];
            float3 turn = cross(contact.lever[side], u);
            float inverse[9];
            for(int i = 0; i < 9; ++i){
                inverse[i] = (float)coupling.inverseInertia[i];
            }
            k += (float)coupling.inverseMass + dot(turn, matrixTimes(inverse, turn));
        }
        contact.mass[direction] = k > 0.0f ? 1.0f / k : 0.0f;
    }
    //each times its partner's: another body's own, a wall's or a still or moved obstacle's (the scene's, RBD Bullet Solver's ground plane's and
    //collisions' unless given)
    const ObstacleState* still = partner >= 6 ? &world.obstacles.items[partner - 6] : nullptr;
    contact.friction = masses[body].contactFriction*(other >= 0 ? masses[other].contactFriction : still != nullptr ? still->contactFriction : world.wallContactFriction[partner]);
    float bounce = masses[body].bounce*(other >= 0 ? masses[other].bounce : still != nullptr ? still->bounce : world.wallBounce[partner]);
    //the speed apart it has to reach: one closing the gap within this substep, fast enough, bounces; otherwise it may close the gap, and no more
    float apart = dot(speed, normal);
    bool hits = apart < -fmaxf(distance, 0.0f) / world.dt;
    contact.target = hits && apart < -world.stillSpeed && bounce > 0.0f ? -bounce*apart : distance > 0.0f ? -distance / world.dt : 0.0f;
    float sunk = -distance - SLOP*world.voxel;
    contact.bias = sunk > 0.0f ? PUSH_OUT*sunk / world.dt : 0.0f;
    contact.depth = distance;
    for(int i = 0; i < 3; ++i){
        contact.impulse[i] = 0.0f;
    }
    contact.pseudo = 0.0f;
    contact.applied = 0.0f;
}

__device__ inline float3 contactImpulse(const BodyContact& contact){
    return contact.impulse[0]*contact.normal + contact.impulse[1]*contact.tangents[0] + contact.impulse[2]*contact.tangents[1];
}

// ---- what the contacts give the bodies, added up exactly ----

//An impulse at a point into a body's sums: the impulse, and its moment (r x P, as a float, as the bodies were pushed by it one at a time), so what many
//contacts give a body comes out the same in whatever order they give it, and they can give it all at once
__device__ inline void addPush(ExactSum* sums, int body, float3 lever, float3 impulse){
    float turn[3] = {lever.y*impulse.z - lever.z*impulse.y, lever.z*impulse.x - lever.x*impulse.z, lever.x*impulse.y - lever.y*impulse.x};
    ExactSum* lanes = sums + BODY_LANES*body;
    addExactly(impulse.x, lanes[0]);
    addExactly(impulse.y, lanes[1]);
    addExactly(impulse.z, lanes[2]);
    for(int axis = 0; axis < 3; ++axis){
        addExactly(turn[axis], lanes[3 + axis]);
    }
}

//and each body takes what it was given: v += P / m, w += I^-1 (r x P), in the coupling's units (voxels of liquid); the sums emptied for the next
__global__ void applyPushes(int numBodies, BodyState* states, const BodyCoupling* couplings, ExactSum* sums){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body >= numBodies){
        return;
    }
    double given[BODY_LANES];
    for(int lane = 0; lane < BODY_LANES; ++lane){
        ExactSum& sum = sums[BODY_LANES*body + lane];
        given[lane] = exactValue(sum);
        for(int digit = 0; digit < EXACT_DIGITS; ++digit){
            sum.digits[digit] = 0;
        }
        sum.nonFinite = 0;
    }
    BodyState& state = states[body];
    const BodyCoupling& coupling = couplings[body];
    double add[3];
    for(int row = 0; row < 3; ++row){
        add[row] = coupling.inverseInertia[3*row]*given[3] + coupling.inverseInertia[3*row + 1]*given[4] + coupling.inverseInertia[3*row + 2]*given[5];
    }
    for(int axis = 0; axis < 3; ++axis){
        state.velocity[axis] += coupling.inverseMass*given[axis];
        state.angularVelocity[axis] += add[axis];
    }
}

// ---- the warm start ----

//At a substep's start, before the pressure solve: each body's velocities as it starts the substep
__global__ void startWarm(int numBodies, BodyState* states){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body < numBodies){
        BodyState& state = states[body];
        for(int axis = 0; axis < 3; ++axis){
            state.before[axis] = state.velocity[axis];
            state.before[3 + axis] = state.angularVelocity[axis];
        }
    }
}

//the last substep's contacts' impulses again, in proportion to this substep's length, so a body resting on something is held up from the start, each into
//its bodies' sums (applyPushes gives them)
__global__ void warmStartContacts(int numLast, ContactWorld world, const BodyState* states, BodyContact* last, ExactSum* sums){
    int slot = threadIdx.x + blockIdx.x*blockDim.x;
    if(slot >= numLast || last[slot].point < 0){
        return;
    }
    BodyContact& contact = last[slot];
    //its push along its normal alone: friction is solved afresh each substep, from what's sliding then. Carried over, the friction a contact solve leaves
    //in four contacts on a face, pulling against each other at the edge of their cones, kicks the body once the pushes shift
    contact.impulse[0] *= contact.dt > 0.0f ? world.dt / contact.dt : 0.0f;
    contact.impulse[1] = 0.0f;
    contact.impulse[2] = 0.0f;
    contact.dt = world.dt;
    contact.applied = 1.0f;
    int body = contact.body;
    float3 at = surfacePoint(world, body, contact.local);   //where it is now, the bodies having moved
    const BodyState& state = states[body];
    contact.lever[0] = at - make_float3((float)state.position[0], (float)state.position[1], (float)state.position[2]);
    if(contact.other >= 0){
        const BodyState& otherState = states[contact.other];
        contact.lever[1] = at - make_float3((float)otherState.position[0], (float)otherState.position[1], (float)otherState.position[2]);
    }
    float3 impulse = contactImpulse(contact);
    addPush(sums, body, contact.lever[0], impulse);
    if(contact.other >= 0){
        addPush(sums, contact.other, contact.lever[1], -1.0f*impulse);
    }
}

//then each body's angular momentum, and its velocity and angular momentum once the warm start has held it up (held)
__global__ void finishWarm(int numBodies, BodyState* states, const BodyMass* masses){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body >= numBodies){
        return;
    }
    BodyState& state = states[body];
    double r[9], inertia[9];
    rotationOf(state.orientation, r);
    turned(r, masses[body].inertia, inertia);
    times(inertia, state.angularVelocity, state.angularMomentum);
    for(int axis = 0; axis < 3; ++axis){
        state.held[axis] = state.velocity[axis];
        state.held[3 + axis] = state.angularMomentum[axis];
    }
}

// ---- the substep's pairs ----

//What a body can touch this substep (broadphase): the walls its box reaches, and the obstacles (other bodies among them) whose reach boxes its box meets,
//its box being its own reach box (3 voxels past its bounds) grown by how far it and they can move in the substep (margin), and the obstacles found
//through the index (ObstacleIndex), each once: in the cell where its box and theirs first meet. With visit set it lists them, from where pairEnds says
//its list starts (unsorted); otherwise it counts them
__device__ inline uint visitPairs(const ContactWorld& world, int body, float3 low, float3 high, bool visit, uint at, int* partners){
    uint found = 0;
    float bounds[2][3] = {{world.low.x, world.low.y, world.low.z}, {world.high.x, world.high.y, world.high.z}};
    float box[2][3] = {{low.x, low.y, low.z}, {high.x, high.y, high.z}};
    for(int wall = 0; wall < 6; ++wall){
        int axis = wall/2;
        bool reaches = wall % 2 ? box[1][axis] >= bounds[1][axis] : box[0][axis] <= bounds[0][axis];
        if(reaches){
            if(visit){
                partners[at + found] = wall;
            }
            ++found;
        }
    }
    const ObstacleIndex& index = world.obstacles.index;
    int3 first, last;
    reachSpan(index, low, high, first, last);
    int own = world.obstacleOf[body];
    for(int z = first.z; z <= last.z; ++z){
        for(int y = first.y; y <= last.y; ++y){
            for(int x = first.x; x <= last.x; ++x){
                uint cell = (uint)x + (uint)index.cells.x*((uint)y + (uint)index.cells.y*(uint)z);
                for(uint entry = cell == 0 ? 0u : index.ends[cell - 1]; entry < index.ends[cell]; ++entry){
                    int obstacle = index.entries[entry];
                    if(obstacle == own){
                        continue;
                    }
                    float3 otherLow, otherHigh;
                    obstacleReach(world.obstacles.items[obstacle], otherLow, otherHigh);
                    if(otherLow.x > high.x || otherLow.y > high.y || otherLow.z > high.z || otherHigh.x < low.x || otherHigh.y < low.y || otherHigh.z < low.z){
                        continue;   //their boxes don't meet
                    }
                    int3 theirFirst, theirLast;
                    reachSpan(index, otherLow, otherHigh, theirFirst, theirLast);
                    if(x != max(first.x, theirFirst.x) || y != max(first.y, theirFirst.y) || z != max(first.z, theirFirst.z)){
                        continue;   //found in another cell
                    }
                    if(visit){
                        partners[at + found] = 6 + obstacle;
                    }
                    ++found;
                }
            }
        }
    }
    return found;
}

//per body, its box for the substep, and how many pairs it's in
__global__ void countPairs(int numBodies, ContactWorld world, float margin, float3* boxes, uint* counts){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body >= numBodies){
        return;
    }
    float3 low, high;
    obstacleReach(world.obstacles.items[world.obstacleOf[body]], low, high);
    low = make_float3(low.x - margin, low.y - margin, low.z - margin);
    high = make_float3(high.x + margin, high.y + margin, high.z + margin);
    boxes[2*body] = low;
    boxes[2*body + 1] = high;
    counts[body] = visitPairs(world, body, low, high, false, 0, nullptr);
}

//and its pairs, from where ends says, its partners in order (walls 0 to 5 first, then the obstacles as they're numbered), so slot order is the same
//whatever order the index gave them in
__global__ void fillPairs(int numBodies, ContactWorld world, const float3* boxes, const uint* ends, int* pairBodies, int* pairPartners){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body >= numBodies){
        return;
    }
    uint first = body == 0 ? 0u : ends[body - 1];
    uint count = visitPairs(world, body, boxes[2*body], boxes[2*body + 1], true, first, pairPartners);
    for(uint i = first; i < first + count; ++i){
        pairBodies[i] = body;
        int partner = pairPartners[i];
        uint j = i;
        for(; j > first && pairPartners[j - 1] > partner; --j){
            pairPartners[j] = pairPartners[j - 1];
        }
        pairPartners[j] = partner;
    }
}

//the pair of a body and a partner, by its place among the body's (sorted), or -1 if it has none this substep
__device__ inline int findPair(const uint* ends, const int* pairPartners, int body, int partner){
    int lowest = body == 0 ? 0 : (int)ends[body - 1], highest = (int)ends[body] - 1;
    while(lowest <= highest){
        int middle = (lowest + highest) / 2;
        if(pairPartners[middle] == partner){
            return middle;
        }
        if(pairPartners[middle] < partner){
            lowest = middle + 1;
        }
        else{
            highest = middle - 1;
        }
    }
    return -1;
}

//the last substep's contacts into this one's slots: the same body, partner and place in the manifold. One whose pair this substep hasn't (nothing it can
//touch any more) gives back what the warm start gave, as the carry would have taken it back
__global__ void remapContacts(int numLast, const BodyContact* last, const uint* ends, const int* pairPartners, BodyContact* remapped, ExactSum* sums){
    int slot = threadIdx.x + blockIdx.x*blockDim.x;
    if(slot >= numLast || last[slot].point < 0){
        return;
    }
    const BodyContact& contact = last[slot];
    int pair = findPair(ends, pairPartners, contact.body, contact.partner);
    if(pair >= 0){
        remapped[pair*CONTACT_SLOTS + contact.k] = contact;
    }
    else if(contact.applied != 0.0f){
        float3 given = contactImpulse(contact);
        addPush(sums, contact.body, contact.lever[0], -1.0f*given);
        if(contact.other >= 0){
            addPush(sums, contact.other, contact.lever[1], given);
        }
    }
}

__global__ void emptySlots(int numSlots, BodyContact* contacts){
    int slot = threadIdx.x + blockIdx.x*blockDim.x;
    if(slot < numSlots){
        contacts[slot].point = -1;
    }
}

// ---- a step's contacts carried, and solved ----

//The last step's contacts carried into this one's, a slot a thread: what the substep's start gave each (warmStartContacts) is taken back where it was
//given, and one found again (same slot, same surface point) starts from that push along its own normal, given where it is now, both into the bodies'
//sums (applyPushes)
__global__ void carryContacts(int numSlots, float dt, const BodyContact* last, BodyContact* contacts, ExactSum* sums){
    int slot = threadIdx.x + blockIdx.x*blockDim.x;
    if(slot >= numSlots || last[slot].point < 0){
        return;
    }
    const BodyContact& before = last[slot];
    int first = slot / CONTACT_SLOTS*CONTACT_SLOTS;
    int found = -1;
    for(int k = 0; k < CONTACT_SLOTS; ++k){
        if(contacts[first + k].point == before.point){
            found = first + k;
        }
    }
    float3 given = contactImpulse(before);
    int body = before.body;
    if(before.applied != 0.0f){
        addPush(sums, body, before.lever[0], -1.0f*given);
        if(before.other >= 0){
            addPush(sums, before.other, before.lever[1], given);
        }
    }
    if(found >= 0){
        BodyContact& now = contacts[found];
        now.impulse[0] = fmaxf(dot(given, now.normal), 0.0f)*(before.dt > 0.0f ? dt / before.dt : 0.0f);
        now.impulse[1] = 0.0f;
        now.impulse[2] = 0.0f;
        float3 carried = contactImpulse(now);
        addPush(sums, body, now.lever[0], carried);
        if(now.other >= 0){
            addPush(sums, now.other, now.lever[1], -1.0f*carried);
        }
    }
}

//Islands: the bodies their contacts join, each named by its lowest body (union-find, hooking the higher root under the lower, so it's the same whatever
//order the contacts come in). Contacts with walls and obstacles that aren't bodies join nothing
__global__ void startIslands(int numBodies, int* islands){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body < numBodies){
        islands[body] = body;
    }
}

__device__ inline int islandRoot(volatile int* islands, int body){
    while(islands[body] != body){
        body = islands[body];
    }
    return body;
}

__global__ void joinIslands(int numSlots, const BodyContact* contacts, int* islands){
    int slot = threadIdx.x + blockIdx.x*blockDim.x;
    if(slot >= numSlots || contacts[slot].point < 0 || contacts[slot].other < 0){
        return;
    }
    int a = contacts[slot].body, b = contacts[slot].other;
    for(;;){
        a = islandRoot(islands, a);
        b = islandRoot(islands, b);
        if(a == b){
            return;
        }
        if(a > b){
            int swap = a;
            a = b;
            b = swap;
        }
        int was = atomicCAS(islands + b, b, a);
        if(was == b){
            return;
        }
        b = was;
    }
}

//and each body's island, every one pointing straight at its lowest; each slot's key, its island and then itself, for sorting the slots by (empty ones last)
__global__ void finishIslands(int numBodies, int* islands){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body < numBodies){
        islands[body] = islandRoot(islands, body);
    }
}

__global__ void keyIslands(int numSlots, const BodyContact* contacts, const int* islands, unsigned long long* keys, int* slots){
    int slot = threadIdx.x + blockIdx.x*blockDim.x;
    if(slot < numSlots){
        keys[slot] = contacts[slot].point < 0 ? ~0ull : (unsigned long long)islands[contacts[slot].body] << 32 | (unsigned int)slot;
        slots[slot] = slot;
    }
}

//Each island's contacts into batches, a thread an island (its lowest body's), in slot order: each into the first batch none of its bodies is in yet, so
//no two contacts of a batch share a body and a batch can be solved all at once. Then its contacts in the solve's order, batch after batch, from where its
//first is among the sorted keys, and where each batch starts. A contact past the last batch but one goes in the last, which is solved one at a time
__global__ void dealIslands(int numBodies, int numSlots, const unsigned long long* keys, const int* sorted, BodyContact* contacts, unsigned long long* bodyBatches,
                            int* order, int* islandBatches){
    int island = threadIdx.x + blockIdx.x*blockDim.x;
    if(island >= numBodies){
        return;
    }
    int* batches = islandBatches + (size_t)island*(MAX_BATCHES + 2);
    batches[0] = 0;
    unsigned long long below = (unsigned long long)island << 32;
    int lowest = 0, highest = numSlots;     //the first key at or past the island's
    while(lowest < highest){
        int middle = (lowest + highest) / 2;
        if(keys[middle] < below){
            lowest = middle + 1;
        }
        else{
            highest = middle;
        }
    }
    int start = lowest, end = start;
    while(end < numSlots && keys[end] >> 32 == (unsigned long long)island && keys[end] != ~0ull){
        ++end;
    }
    if(end == start){
        return;     //no island of its own: it's in another's, or touches nothing
    }
    const int words = MAX_BATCHES / 64;
    for(int i = start; i < end; ++i){
        const BodyContact& contact = contacts[sorted[i]];
        for(int word = 0; word < words; ++word){
            bodyBatches[(size_t)contact.body*words + word] = 0;
            if(contact.other >= 0){
                bodyBatches[(size_t)contact.other*words + word] = 0;
            }
        }
    }
    int counts[MAX_BATCHES] = {};
    int used = 0;
    for(int i = start; i < end; ++i){
        BodyContact& contact = contacts[sorted[i]];
        unsigned long long* mine = bodyBatches + (size_t)contact.body*words;
        unsigned long long* theirs = contact.other >= 0 ? bodyBatches + (size_t)contact.other*words : nullptr;
        int batch = 0;
        while(batch < MAX_BATCHES - 1 && (mine[batch / 64] >> batch % 64 & 1ull || (theirs != nullptr && theirs[batch / 64] >> batch % 64 & 1ull))){
            ++batch;
        }
        mine[batch / 64] |= 1ull << batch % 64;
        if(theirs != nullptr){
            theirs[batch / 64] |= 1ull << batch % 64;
        }
        contact.color = batch;
        ++counts[batch];
        used = max(used, batch + 1);
    }
    batches[0] = used;
    int at = start;
    for(int batch = 0; batch < used; ++batch){
        batches[1 + batch] = at;
        at += counts[batch];
        counts[batch] = batches[1 + batch];
    }
    batches[1 + used] = end;
    for(int i = start; i < end; ++i){
        int slot = sorted[i];
        order[counts[contacts[slot].color]++] = slot;
    }
}

//every body's velocities as floats for the solve, and its inverse mass and inertia
__global__ void startSolve(int numBodies, const BodyState* states, const BodyCoupling* couplings, SolveBody* bodies){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body >= numBodies){
        return;
    }
    const BodyState& state = states[body];
    SolveBody& solving = bodies[body];
    solving.v = make_float3((float)state.velocity[0], (float)state.velocity[1], (float)state.velocity[2]);
    solving.w = make_float3((float)state.angularVelocity[0], (float)state.angularVelocity[1], (float)state.angularVelocity[2]);
    solving.pv = make_float3(0.0f, 0.0f, 0.0f);
    solving.pw = make_float3(0.0f, 0.0f, 0.0f);
    solving.inverseMass = (float)couplings[body].inverseMass;
    for(int i = 0; i < 9; ++i){
        solving.inverseInertia[i] = (float)couplings[body].inverseInertia[i];
    }
}

//The solve, a block an island (its lowest body's): each batch's contacts in parallel, batch after batch, forwards then backwards in turn (symmetric
//Gauss-Seidel), VELOCITY_ITERATIONS times; then the push out of what they're sunk into, on velocities of its own, POSITION_ITERATIONS times. An island
//touches only its own bodies, so the islands are solved all at once
__global__ void solveIslands(int numBodies, const int* order, const int* islandBatches, BodyContact* contacts, SolveBody* bodies){
    int island = blockIdx.x;
    if(island >= numBodies){
        return;
    }
    const int* batches = islandBatches + (size_t)island*(MAX_BATCHES + 2);
    int count = batches[0];
    if(count == 0){
        return;
    }
    auto apply = [&](bool pushing, int body, float3 lever, float3 impulse){
        SolveBody& solving = bodies[body];
        float3& linear = pushing ? solving.pv : solving.v;
        float3& angular = pushing ? solving.pw : solving.w;
        linear = linear + solving.inverseMass*impulse;
        angular = angular + matrixTimes(solving.inverseInertia, cross(lever, impulse));
    };
    auto solve = [&](bool pushing, int slot){
        BodyContact& contact = contacts[slot];
        int body = contact.body, other = contact.other;
        auto apart = [&](){     //how fast they separate at the contact, body less partner
            const SolveBody& mine = bodies[body];
            float3 speed = (pushing ? mine.pv : mine.v) + cross(pushing ? mine.pw : mine.w, contact.lever[0]);
            if(other >= 0){
                const SolveBody& theirs = bodies[other];
                return speed - ((pushing ? theirs.pv : theirs.v) + cross(pushing ? theirs.pw : theirs.w, contact.lever[1]));
            }
            return pushing ? speed : speed - contact.partnerVelocity;
        };
        if(pushing){
            float more = (contact.bias - dot(apart(), contact.normal))*contact.mass[0];
            float total = fmaxf(contact.pseudo + more, 0.0f);
            float3 impulse = (total - contact.pseudo)*contact.normal;
            contact.pseudo = total;
            apply(true, body, contact.lever[0], impulse);
            if(other >= 0){
                apply(true, other, contact.lever[1], -1.0f*impulse);
            }
            return;
        }
        float more = (contact.target - dot(apart(), contact.normal))*contact.mass[0];
        float total = fmaxf(contact.impulse[0] + more, 0.0f);
        float3 impulse = (total - contact.impulse[0])*contact.normal;
        contact.impulse[0] = total;
        apply(false, body, contact.lever[0], impulse);
        if(other >= 0){
            apply(false, other, contact.lever[1], -1.0f*impulse);
        }
        float3 speed = apart();
        float along[2] = {contact.impulse[1] - dot(speed, contact.tangents[0])*contact.mass[1], contact.impulse[2] - dot(speed, contact.tangents[1])*contact.mass[2]};
        float slide = sqrtf(along[0]*along[0] + along[1]*along[1]), most = contact.friction*contact.impulse[0];
        if(slide > most){
            along[0] *= most / slide;
            along[1] *= most / slide;
        }
        impulse = (along[0] - contact.impulse[1])*contact.tangents[0] + (along[1] - contact.impulse[2])*contact.tangents[1];
        contact.impulse[1] = along[0];
        contact.impulse[2] = along[1];
        apply(false, body, contact.lever[0], impulse);
        if(other >= 0){
            apply(false, other, contact.lever[1], -1.0f*impulse);
        }
    };
    for(int pass = 0; pass < VELOCITY_ITERATIONS + POSITION_ITERATIONS; ++pass){
        bool pushing = pass >= VELOCITY_ITERATIONS;
        for(int turn = 0; turn < count; ++turn){
            int batch = pass % 2 ? count - 1 - turn : turn;     //forwards, then backwards: no contact is always first
            int first = batches[1 + batch], last = batches[2 + batch];
            if(batch == MAX_BATCHES - 1){   //the last takes whatever didn't fit before it, so its contacts can share bodies: one at a time
                if(threadIdx.x == 0){
                    for(int i = first; i < last; ++i){
                        solve(pushing, order[i]);
                    }
                }
            }
            else{
                for(int i = first + threadIdx.x; i < last; i += blockDim.x){
                    solve(pushing, order[i]);
                }
            }
            __syncthreads();
        }
    }
}

//then the bodies take their new velocities, and their angular momentum and frames follow
__global__ void finishSolve(int numBodies, BodyState* states, const BodyMass* masses, const SolveBody* bodies, BodyFrame* frames){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body >= numBodies){
        return;
    }
    BodyState& state = states[body];
    const SolveBody& solving = bodies[body];
    float3 lv[2] = {solving.v, solving.pv}, av[2] = {solving.w, solving.pw};
    for(int axis = 0; axis < 3; ++axis){
        state.velocity[axis] = component(lv[0], axis);
        state.angularVelocity[axis] = component(av[0], axis);
        state.pseudo[axis] = component(lv[1], axis);
        state.pseudo[3 + axis] = component(av[1], axis);
    }
    double r[9], inertia[9];
    rotationOf(state.orientation, r);
    turned(r, masses[body].inertia, inertia);
    times(inertia, state.angularVelocity, state.angularMomentum);
    writeVelocity(state, frames[body]);
}

static ContactWorld contactWorld(const Obstacles& obstacleState, const int* obstacleOf, float3 low, float3 high, const float wallBounce[6], const float wallContactFriction[6],
                                 double voxel, double dt, float stillSpeed){
    ContactWorld world;
    world.obstacles = obstacleState;
    world.obstacleOf = obstacleOf;
    world.low = low;
    world.high = high;
    for(int wall = 0; wall < 6; ++wall){
        world.wallBounce[wall] = wallBounce[wall];
        world.wallContactFriction[wall] = wallContactFriction[wall];
    }
    world.voxel = (float)voxel;
    world.dt = (float)dt;
    world.stillSpeed = stillSpeed;
    return world;
}

//After the substep's solve and friction: what gravity, the pressure and the friction gave each body since the warm start, made a steady push over the
//substep (dt), which its steps take a piece of at a time, so that a pile's contacts hold up a step's weight at each; it starts them from where the
//warm start left it. The angular part is a torque, onto the angular momentum: the body turns between the steps
__global__ void spreadBodies(int numBodies, BodyState* states, const BodyMass* masses, double dt){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body >= numBodies){
        return;
    }
    BodyState& state = states[body];
    for(int axis = 0; axis < 3; ++axis){
        state.perSecond[axis] = (state.velocity[axis] - state.held[axis]) / dt;
        state.perSecond[3 + axis] = (state.angularMomentum[axis] - state.held[3 + axis]) / dt;
        state.velocity[axis] = state.held[axis];
        state.angularMomentum[axis] = state.held[3 + axis];
    }
    double r[9], inverse[9];
    rotationOf(state.orientation, r);
    turned(r, masses[body].inverseInertia, inverse);
    times(inverse, state.angularMomentum, state.angularVelocity);
}

//a step's piece of it, dt's worth; the angular velocity what the angular momentum makes of the inertia where the body's turned to now
__global__ void driveBodies(int numBodies, BodyState* states, const BodyMass* masses, double dt){
    int body = threadIdx.x + blockIdx.x*blockDim.x;
    if(body >= numBodies){
        return;
    }
    BodyState& state = states[body];
    for(int axis = 0; axis < 3; ++axis){
        state.velocity[axis] += dt*state.perSecond[axis];
        state.angularMomentum[axis] += dt*state.perSecond[3 + axis];
    }
    double r[9], inverse[9];
    rotationOf(state.orientation, r);
    turned(r, masses[body].inverseInertia, inverse);
    times(inverse, state.angularMomentum, state.angularVelocity);
}

void Bodies::reserve(size_t& capacity, size_t needed, std::initializer_list<std::pair<void**, size_t>> arrays){
    if(needed <= capacity){
        return;
    }
    size_t room = std::max(needed, capacity + capacity/2);  //some over, so a pile settling in doesn't make them again every substep
    for(const std::pair<void**, size_t>& array : arrays){
        if(*array.first != nullptr){
            gpuErrchk(cudaFreeAsync(*array.first, stream));
        }
        gpuErrchk(cudaMallocAsync(array.first, array.second*room, stream));
    }
    capacity = room;
}

void Bodies::makePairs(const Obstacles& obstacleState, float3 low, float3 high){
    gpuErrchk(cudaEventSynchronize(pairsCounted));      //done by now, as the pressure solve has waited on the stream since: it waits only if not
    numPairs = (int)*pairTotal;
    reserve(pairCapacity, (size_t)numPairs, {{(void**)&pairBodies, sizeof(int)}, {(void**)&pairPartners, sizeof(int)}});
    size_t numSlots = (size_t)numPairs*CONTACT_SLOTS;
    if(numSlots > slotCapacity){    //more room, the last contacts kept in it
        BodyContact* kept = lastContacts;
        lastContacts = nullptr;
        reserve(slotCapacity, numSlots, {{(void**)&picks, sizeof(unsigned long long)}, {(void**)&contacts, sizeof(BodyContact)}, {(void**)&lastContacts, sizeof(BodyContact)},
                                         {(void**)&remapped, sizeof(BodyContact)}, {(void**)&islandKeys, sizeof(unsigned long long)},
                                         {(void**)&sortedKeys, sizeof(unsigned long long)}, {(void**)&slotsIn, sizeof(int)}, {(void**)&slotsSorted, sizeof(int)},
                                         {(void**)&contactOrder, sizeof(int)}});
        if(kept != nullptr){
            if(numLast > 0){
                gpuErrchk(cudaMemcpyAsync(lastContacts, kept, sizeof(BodyContact)*numLast, cudaMemcpyDeviceToDevice, stream));
            }
            gpuErrchk(cudaFreeAsync(kept, stream));
        }
        size_t bytes = 0;
        gpuErrchk(cub::DeviceRadixSort::SortPairs(nullptr, bytes, islandKeys, sortedKeys, slotsIn, slotsSorted, (int)slotCapacity, 0, 64, stream));
        if(bytes > sortScratchBytes){
            if(sortScratch != nullptr){
                gpuErrchk(cudaFreeAsync(sortScratch, stream));
            }
            gpuErrchk(cudaMallocAsync(&sortScratch, bytes, stream));
            sortScratchBytes = bytes;
        }
    }
    ContactWorld world = contactWorld(obstacleState, deviceObstacles, low, high, wallBounce, wallContactFriction, voxel, 0.0, 0.0f);
    if(numPairs > 0){
        fillPairs<<<bodyBlocks(), 32, 0, stream>>>(numBodies, world, queryBoxes, pairEnds, pairBodies, pairPartners);
        emptySlots<<<(uint)(numSlots / 256 + 1), 256, 0, stream>>>((int)numSlots, remapped);
    }
    if(numLast > 0){
        remapContacts<<<numLast / 256 + 1, 256, 0, stream>>>(numLast, lastContacts, pairEnds, pairPartners, remapped, pushSums);
        applyPushes<<<bodyBlocks(), 32, 0, stream>>>(numBodies, states, couplings, pushSums);
    }
    gpuErrchk(cudaPeekAtLastError());
    std::swap(lastContacts, remapped);
    numLast = (int)numSlots;
}

void Bodies::step(double dt, double frameDt, const Obstacles& obstacleState, float3 low, float3 high, float3 gravity){
    if(numBodies == 0){
        return;
    }
    makePairs(obstacleState, low, high);
    //how many steps: the collision substeps' rate, and enough that no body's surface moves more than half a voxel in one, by how fast the fastest
    //moved at the substep's start and what it gained per second over the last; every partition reads the same, so they agree
    double perStep = frameDt / collisionSubsteps;
    double reach = ((double)fastest() + (double)acceleration()*dt)*dt;
    int steps = std::max({(int)std::ceil(dt / perStep - 1e-9), (int)std::ceil(reach / (0.5*voxel) - 1e-9), 1});
    double small = dt / steps;
    spreadBodies<<<bodyBlocks(), 32, 0, stream>>>(numBodies, states, deviceMasses, dt);
    for(int stepIndex = 0; stepIndex < steps; ++stepIndex){
        driveBodies<<<bodyBlocks(), 32, 0, stream>>>(numBodies, states, deviceMasses, small);
        touch(small, obstacleState, low, high, gravity);
        moveOnce(small, dt);
        writeFrames();      //where they are for the next step's contacts
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpyAsync(seen, states, sizeof(BodyState)*numBodies, cudaMemcpyDeviceToHost, stream));     //read after the next substep's waits
}

void Bodies::warmStart(double dt, const Obstacles& obstacleState, float3 gravity, float obstacleSpeed, float3 low, float3 high){
    if(numBodies == 0){
        return;
    }
    float g = sqrtf(gravity.x*gravity.x + gravity.y*gravity.y + gravity.z*gravity.z);
    ContactWorld world = contactWorld(obstacleState, deviceObstacles, low, high, wallBounce, wallContactFriction, voxel, dt, 2.0f*g*(float)dt);
    startWarm<<<bodyBlocks(), 32, 0, stream>>>(numBodies, states);
    if(numLast > 0){
        warmStartContacts<<<numLast / 256 + 1, 256, 0, stream>>>(numLast, world, states, lastContacts, pushSums);
    }
    applyPushes<<<bodyBlocks(), 32, 0, stream>>>(numBodies, states, couplings, pushSums);
    finishWarm<<<bodyBlocks(), 32, 0, stream>>>(numBodies, states, deviceMasses);
    //the substep's pairs counted, for step to make once the host has waited on the stream anyway (the pressure solve's): whatever a body can touch within
    //twice as far as the fastest point of anything moved at the substep's start and gained per second over the last could take it, and a voxel more
    double fastestPoint = std::max((double)fastest(), (double)obstacleSpeed);
    float margin = (float)(2.0*(fastestPoint*dt + (double)acceleration()*dt*dt) + voxel);
    countPairs<<<bodyBlocks(), 32, 0, stream>>>(numBodies, world, margin, queryBoxes, pairEnds);
    cudaParallelPrefixSum((uint)numBodies, pairEnds, stream);
    gpuErrchk(cudaMemcpyAsync(pairTotal, pairEnds + numBodies - 1, sizeof(uint), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaEventRecord(pairsCounted, stream));
    gpuErrchk(cudaPeekAtLastError());
}

void Bodies::touch(double dt, const Obstacles& obstacleState, float3 low, float3 high, float3 gravity){
    if(numSurfacePoints == 0){
        return;
    }
    float g = sqrtf(gravity.x*gravity.x + gravity.y*gravity.y + gravity.z*gravity.z);
    ContactWorld world = contactWorld(obstacleState, deviceObstacles, low, high, wallBounce, wallContactFriction, voxel, dt, 2.0f*g*(float)dt);
    int slots = numPairs*CONTACT_SLOTS;
    uint slotBlocks = (uint)(slots / 256 + 1);
    if(slots > 0){
        gpuErrchk(cudaMemsetAsync(picks, 0xFF, sizeof(unsigned long long)*slots, stream));
        uint blocks = std::min((uint)numSurfacePoints / CONTACT_THREADS + 1, 1024u);
        for(int pass = 0; pass < CONTACT_SLOTS; ++pass){
            pickContacts<<<blocks, CONTACT_THREADS, 0, stream>>>(pass, world, surfacePoints, surfaceBodies, numSurfacePoints, pairEnds, pairPartners, picks);
        }
        buildContacts<<<slotBlocks, 256, 0, stream>>>(world, surfacePoints, pairBodies, pairPartners, states, deviceMasses, couplings, picks, slots, contacts);
        carryContacts<<<slotBlocks, 256, 0, stream>>>(slots, (float)dt, lastContacts, contacts, pushSums);
        applyPushes<<<bodyBlocks(), 32, 0, stream>>>(numBodies, states, couplings, pushSums);
        startIslands<<<bodyBlocks(), 32, 0, stream>>>(numBodies, islands);
        joinIslands<<<slotBlocks, 256, 0, stream>>>(slots, contacts, islands);
        finishIslands<<<bodyBlocks(), 32, 0, stream>>>(numBodies, islands);
        keyIslands<<<slotBlocks, 256, 0, stream>>>(slots, contacts, islands, islandKeys, slotsIn);
        size_t bytes = sortScratchBytes;
        gpuErrchk(cub::DeviceRadixSort::SortPairs(sortScratch, bytes, islandKeys, sortedKeys, slotsIn, slotsSorted, slots, 0, 64, stream));
        dealIslands<<<bodyBlocks(), 32, 0, stream>>>(numBodies, slots, sortedKeys, slotsSorted, contacts, bodyBatches, contactOrder, islandBatches);
    }
    startSolve<<<bodyBlocks(), 32, 0, stream>>>(numBodies, states, couplings, solveBodies);
    if(slots > 0){
        solveIslands<<<numBodies, CONTACT_THREADS, 0, stream>>>(numBodies, contactOrder, islandBatches, contacts, solveBodies);
    }
    finishSolve<<<bodyBlocks(), 32, 0, stream>>>(numBodies, states, deviceMasses, solveBodies, frames);
    gpuErrchk(cudaPeekAtLastError());
    std::swap(contacts, lastContacts);  //this step's are what the next starts from
}
