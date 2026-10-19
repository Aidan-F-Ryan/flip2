//Copyright 2023 Aberrant Behavior LLC

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
//Friction and bounce combine as Bullet combines them, as products: a wall's and an obstacle's are 1 here, so a body's own hold against them

#include "bodies.hu"
#include "algorithms/parallelPrefixSumKernels.hu"
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
        ObstacleState shape = obstacleSet.state().items[obstacles[body]];    //in its own space: the world's taken for it
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
    int slots = numBodies*CONTACT_PARTNERS*CONTACT_SLOTS;
    gpuErrchk(cudaMalloc((void**)&surfacePoints, sizeof(float3)*std::max(numSurfacePoints, 1)));
    gpuErrchk(cudaMalloc((void**)&surfaceBodies, sizeof(int)*std::max(numSurfacePoints, 1)));
    gpuErrchk(cudaMalloc((void**)&picks, sizeof(unsigned long long)*slots));
    gpuErrchk(cudaMalloc((void**)&contacts, sizeof(BodyContact)*slots));
    gpuErrchk(cudaMalloc((void**)&lastContacts, sizeof(BodyContact)*slots));
    gpuErrchk(cudaMalloc((void**)&contactOrder, sizeof(int)*slots));
    gpuErrchk(cudaMalloc((void**)&batchStarts, sizeof(int)*(MAX_BATCHES + 1)));
    if(numSurfacePoints > 0){
        gpuErrchk(cudaMemcpyAsync(surfacePoints, allPoints.data(), sizeof(float3)*numSurfacePoints, cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaMemcpyAsync(surfaceBodies, allBodies.data(), sizeof(int)*numSurfacePoints, cudaMemcpyHostToDevice, stream));
    }
    std::vector<BodyContact> none(slots);
    for(BodyContact& contact : none){
        contact = BodyContact{};
        contact.point = -1;
    }
    gpuErrchk(cudaMemcpyAsync(contacts, none.data(), sizeof(BodyContact)*slots, cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(lastContacts, none.data(), sizeof(BodyContact)*slots, cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
}

// ---- finding them ----

//what the contact passes need of the world: the obstacles where they are (the bodies' frames among them), which body each obstacle is, the domain
struct ContactWorld{
    Obstacles obstacles;
    signed char bodyOf[MAX_OBSTACLES];      //per obstacle, its body, or -1
    signed char obstacleOf[MAX_OBSTACLES];  //per body, its obstacle
    float3 low, high;   //the domain's walls
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
__global__ void pickContacts(int pass, ContactWorld world, const float3* points, const int* pointBodies, int numPoints, unsigned long long* picks){
    for(int index = threadIdx.x + blockIdx.x*blockDim.x; index < numPoints; index += blockDim.x*gridDim.x){
        int body = pointBodies[index];
        const ObstacleState& self = world.obstacles.items[world.obstacleOf[body]];
        float3 p = surfacePoint(world, body, points[index]);
        float3 moving = obstacleVelocity(self, p);
        for(int partner = 0; partner < 6 + world.obstacles.count; ++partner){
            if(partner >= 6 && partner - 6 == world.obstacleOf[body]){
                continue;
            }
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
            int first = (body*CONTACT_PARTNERS + partner)*CONTACT_SLOTS;
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
__global__ void buildContacts(ContactWorld world, const float3* points, const BodyState* states, const BodyMass* masses, const BodyCoupling* couplings,
                              const unsigned long long* picks, int numSlots, BodyContact* contacts){
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
    int body = manifold / CONTACT_PARTNERS, partner = manifold % CONTACT_PARTNERS;
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
    int other = partner >= 6 ? world.bodyOf[partner - 6] : -1;
    contact.point = point;
    contact.other = other;
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
    contact.friction = masses[body].contactFriction*(other >= 0 ? masses[other].contactFriction : 1.0f);
    float bounce = masses[body].bounce*(other >= 0 ? masses[other].bounce : 1.0f);
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

//an impulse at a point onto a body's velocity and turning: v += P / m, w += I^-1 (r x P), in the coupling's units (voxels of liquid)
__device__ inline void push(BodyState& state, const BodyCoupling& coupling, float3 lever, float3 impulse){
    double turn[3] = {lever.y*impulse.z - lever.z*impulse.y, lever.z*impulse.x - lever.x*impulse.z, lever.x*impulse.y - lever.y*impulse.x};
    double add[3];
    for(int row = 0; row < 3; ++row){
        add[row] = coupling.inverseInertia[3*row]*turn[0] + coupling.inverseInertia[3*row + 1]*turn[1] + coupling.inverseInertia[3*row + 2]*turn[2];
    }
    state.velocity[0] += coupling.inverseMass*impulse.x;
    state.velocity[1] += coupling.inverseMass*impulse.y;
    state.velocity[2] += coupling.inverseMass*impulse.z;
    for(int axis = 0; axis < 3; ++axis){
        state.angularVelocity[axis] += add[axis];
    }
}

__device__ inline float3 contactImpulse(const BodyContact& contact){
    return contact.impulse[0]*contact.normal + contact.impulse[1]*contact.tangents[0] + contact.impulse[2]*contact.tangents[1];
}

//The last substep's contacts carried into this one's, one thread, in slot order: what the substep's start gave each (warmStartBodies) is taken back
//where it was given, and one found again (same slot, same surface point) starts from that push along its own normal, given where it is now. Then the batches: each contact, in slot order, in the first batch none of its bodies is in yet
__global__ void carryContacts(int numBodies, int numSlots, float dt, BodyState* states, const BodyCoupling* couplings, const BodyContact* last, BodyContact* contacts,
                              int* order, int* batchStarts){
    for(int slot = 0; slot < numSlots; ++slot){
        const BodyContact& before = last[slot];
        if(before.point < 0){
            continue;
        }
        int first = slot / CONTACT_SLOTS*CONTACT_SLOTS;
        int found = -1;
        for(int k = 0; k < CONTACT_SLOTS; ++k){
            if(contacts[first + k].point == before.point){
                found = first + k;
            }
        }
        //what the substep's start gave, if it did, taken back where it was given; and what carries over, for this step's length, given where the
        //contact is now
        float3 given = contactImpulse(before);
        int body = slot / (CONTACT_PARTNERS*CONTACT_SLOTS);
        if(before.applied != 0.0f){
            push(states[body], couplings[body], before.lever[0], -1.0f*given);
            if(before.other >= 0){
                push(states[before.other], couplings[before.other], before.lever[1], given);
            }
        }
        if(found >= 0){
            BodyContact& now = contacts[found];
            now.impulse[0] = fmaxf(dot(given, now.normal), 0.0f)*(before.dt > 0.0f ? dt / before.dt : 0.0f);
            now.impulse[1] = 0.0f;
            now.impulse[2] = 0.0f;
            float3 carried = contactImpulse(now);
            push(states[body], couplings[body], now.lever[0], carried);
            if(now.other >= 0){
                push(states[now.other], couplings[now.other], now.lever[1], -1.0f*carried);
            }
        }
    }
    unsigned long long used[MAX_OBSTACLES][MAX_BATCHES / 64] = {};
    int counts[MAX_BATCHES] = {};
    int batches = 0;
    for(int slot = 0; slot < numSlots; ++slot){
        BodyContact& contact = contacts[slot];
        if(contact.point < 0){
            continue;
        }
        int body = slot / (CONTACT_PARTNERS*CONTACT_SLOTS);
        int batch = 0;
        while(batch < MAX_BATCHES - 1 && (used[body][batch / 64] >> batch % 64 & 1ull || (contact.other >= 0 && used[contact.other][batch / 64] >> batch % 64 & 1ull))){
            ++batch;
        }
        used[body][batch / 64] |= 1ull << batch % 64;
        if(contact.other >= 0){
            used[contact.other][batch / 64] |= 1ull << batch % 64;
        }
        contact.color = batch;
        ++counts[batch];
        batches = max(batches, batch + 1);
    }
    batchStarts[0] = 0;
    for(int batch = 0; batch < batches; ++batch){
        batchStarts[batch + 1] = batchStarts[batch] + counts[batch];
        counts[batch] = batchStarts[batch];
    }
    for(int batch = batches; batch < MAX_BATCHES; ++batch){
        batchStarts[batch + 1] = batchStarts[batches];
    }
    for(int slot = 0; slot < numSlots; ++slot){
        if(contacts[slot].point >= 0){
            order[counts[contacts[slot].color]++] = slot;
        }
    }
    (void)numBodies;
}

//The solve, one block: the bodies' velocities in shared memory, each batch's contacts in parallel, batch after batch, forwards then backwards in turn
//(symmetric Gauss-Seidel), VELOCITY_ITERATIONS times; then
//the push out of what they're sunk into, on velocities of its own, POSITION_ITERATIONS times. Then the bodies take their new velocities, and their
//angular momentum and frames follow
__global__ void solveContacts(int numBodies, BodyState* states, const BodyMass* masses, const BodyCoupling* couplings, BodyFrame* frames, BodyContact* contacts,
                              const int* order, const int* batchStarts){
    __shared__ float3 v[MAX_OBSTACLES], w[MAX_OBSTACLES], pv[MAX_OBSTACLES], pw[MAX_OBSTACLES];
    __shared__ float inverseMass[MAX_OBSTACLES], inverseInertia[MAX_OBSTACLES][9];
    for(int body = threadIdx.x; body < numBodies; body += blockDim.x){
        const BodyState& state = states[body];
        v[body] = make_float3((float)state.velocity[0], (float)state.velocity[1], (float)state.velocity[2]);
        w[body] = make_float3((float)state.angularVelocity[0], (float)state.angularVelocity[1], (float)state.angularVelocity[2]);
        pv[body] = make_float3(0.0f, 0.0f, 0.0f);
        pw[body] = make_float3(0.0f, 0.0f, 0.0f);
        inverseMass[body] = (float)couplings[body].inverseMass;
        for(int i = 0; i < 9; ++i){
            inverseInertia[body][i] = (float)couplings[body].inverseInertia[i];
        }
    }
    __syncthreads();
    int batches = 0;
    while(batches < MAX_BATCHES && batchStarts[batches + 1] > batchStarts[batches]){
        ++batches;
    }
    auto apply = [&](float3* linear, float3* angular, int body, float3 lever, float3 impulse){
        linear[body] = linear[body] + inverseMass[body]*impulse;
        angular[body] = angular[body] + matrixTimes(inverseInertia[body], cross(lever, impulse));
    };
    for(int pass = 0; pass < VELOCITY_ITERATIONS + POSITION_ITERATIONS; ++pass){
        bool pushing = pass >= VELOCITY_ITERATIONS;
        float3* linear = pushing ? pv : v;
        float3* angular = pushing ? pw : w;
        for(int turn = 0; turn < batches; ++turn){
            int batch = pass % 2 ? batches - 1 - turn : turn;     //forwards, then backwards: no contact is always first
            for(int i = batchStarts[batch] + threadIdx.x; i < batchStarts[batch + 1]; i += blockDim.x){
                BodyContact& contact = contacts[order[i]];
                int body = order[i] / (CONTACT_PARTNERS*CONTACT_SLOTS), other = contact.other;
                auto apart = [&](){     //how fast they separate at the contact, body less partner
                    float3 speed = linear[body] + cross(angular[body], contact.lever[0]);
                    return other >= 0 ? speed - (linear[other] + cross(angular[other], contact.lever[1])) : pushing ? speed : speed - contact.partnerVelocity;
                };
                if(pushing){
                    float more = (contact.bias - dot(apart(), contact.normal))*contact.mass[0];
                    float total = fmaxf(contact.pseudo + more, 0.0f);
                    float3 impulse = (total - contact.pseudo)*contact.normal;
                    contact.pseudo = total;
                    apply(linear, angular, body, contact.lever[0], impulse);
                    if(other >= 0){
                        apply(linear, angular, other, contact.lever[1], -1.0f*impulse);
                    }
                    continue;
                }
                float more = (contact.target - dot(apart(), contact.normal))*contact.mass[0];
                float total = fmaxf(contact.impulse[0] + more, 0.0f);
                float3 impulse = (total - contact.impulse[0])*contact.normal;
                contact.impulse[0] = total;
                apply(linear, angular, body, contact.lever[0], impulse);
                if(other >= 0){
                    apply(linear, angular, other, contact.lever[1], -1.0f*impulse);
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
                apply(linear, angular, body, contact.lever[0], impulse);
                if(other >= 0){
                    apply(linear, angular, other, contact.lever[1], -1.0f*impulse);
                }
            }
            __syncthreads();
        }
    }
    for(int body = threadIdx.x; body < numBodies; body += blockDim.x){
        BodyState& state = states[body];
        float3 lv[2] = {v[body], pv[body]}, av[2] = {w[body], pw[body]};
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
}

//At a substep's start, before the pressure solve: the last substep's contacts' impulses again, in proportion to this substep's length, so a body
//resting on something is held up from the start. One thread, in slot order. Each body's velocities then are what it starts the substep with
__global__ void warmStartBodies(int numBodies, int numSlots, BodyState* states, const BodyMass* masses, const BodyCoupling* couplings, ContactWorld world,
                                BodyContact* last){
    for(int body = 0; body < numBodies; ++body){
        BodyState& state = states[body];
        for(int axis = 0; axis < 3; ++axis){
            state.before[axis] = state.velocity[axis];
            state.before[3 + axis] = state.angularVelocity[axis];
        }
    }
    for(int slot = 0; slot < numSlots; ++slot){
        BodyContact& contact = last[slot];
        if(contact.point < 0){
            continue;
        }
        //its push along its normal alone: friction is solved afresh each substep, from what's sliding then. Carried over, the friction a contact
        //solve leaves in four contacts on a face, pulling against each other at the edge of their cones, kicks the body once the pushes shift
        contact.impulse[0] *= contact.dt > 0.0f ? world.dt / contact.dt : 0.0f;
        contact.impulse[1] = 0.0f;
        contact.impulse[2] = 0.0f;
        contact.dt = world.dt;
        contact.applied = 1.0f;
        int body = slot / (CONTACT_PARTNERS*CONTACT_SLOTS);
        float3 at = surfacePoint(world, body, contact.local);   //where it is now, the bodies having moved
        const BodyState& state = states[body];
        contact.lever[0] = at - make_float3((float)state.position[0], (float)state.position[1], (float)state.position[2]);
        if(contact.other >= 0){
            const BodyState& otherState = states[contact.other];
            contact.lever[1] = at - make_float3((float)otherState.position[0], (float)otherState.position[1], (float)otherState.position[2]);
        }
        float3 impulse = contactImpulse(contact);
        push(states[body], couplings[body], contact.lever[0], impulse);
        if(contact.other >= 0){
            push(states[contact.other], couplings[contact.other], contact.lever[1], -1.0f*impulse);
        }
    }
    for(int body = 0; body < numBodies; ++body){
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
}

static ContactWorld contactWorld(const Obstacles& obstacleState, const std::vector<int>& obstacles, float3 low, float3 high, double voxel, double dt, float stillSpeed){
    ContactWorld world;
    world.obstacles = obstacleState;
    for(int i = 0; i < MAX_OBSTACLES; ++i){
        world.bodyOf[i] = -1;
        world.obstacleOf[i] = -1;
    }
    for(size_t body = 0; body < obstacles.size(); ++body){
        world.bodyOf[obstacles[body]] = (signed char)body;
        world.obstacleOf[body] = (signed char)obstacles[body];
    }
    world.low = low;
    world.high = high;
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

void Bodies::step(double dt, double frameDt, const Obstacles& obstacleState, float3 low, float3 high, float3 gravity){
    if(numBodies == 0){
        return;
    }
    //how many steps: the collision substeps' rate, and enough that no body's surface moves more than half a voxel in one, by how fast the fastest
    //moved at the substep's start and what it gained per second over the last; every partition reads the same, so they agree
    double perStep = frameDt / collisionSubsteps;
    double reach = ((double)fastest() + (double)acceleration()*dt)*dt;
    int steps = std::max({(int)std::ceil(dt / perStep - 1e-9), (int)std::ceil(reach / (0.5*voxel) - 1e-9), 1});
    double small = dt / steps;
    spreadBodies<<<1, 32, 0, stream>>>(numBodies, states, deviceMasses, dt);
    for(int stepIndex = 0; stepIndex < steps; ++stepIndex){
        driveBodies<<<1, 32, 0, stream>>>(numBodies, states, deviceMasses, small);
        touch(small, obstacleState, low, high, gravity);
        moveOnce(small, dt);
        writeFrames();      //where they are for the next step's contacts
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemcpyAsync(seen, states, sizeof(BodyState)*numBodies, cudaMemcpyDeviceToHost, stream));     //read after the next substep's waits
}

void Bodies::warmStart(double dt, const Obstacles& obstacleState, float3 gravity){
    if(numBodies == 0){
        return;
    }
    float g = sqrtf(gravity.x*gravity.x + gravity.y*gravity.y + gravity.z*gravity.z);
    ContactWorld world = contactWorld(obstacleState, obstacles, make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f), voxel, dt, 2.0f*g*(float)dt);
    warmStartBodies<<<1, 1, 0, stream>>>(numBodies, numBodies*CONTACT_PARTNERS*CONTACT_SLOTS, states, deviceMasses, couplings, world, lastContacts);
    gpuErrchk(cudaPeekAtLastError());
}

void Bodies::touch(double dt, const Obstacles& obstacleState, float3 low, float3 high, float3 gravity){
    if(numSurfacePoints == 0){
        return;
    }
    float g = sqrtf(gravity.x*gravity.x + gravity.y*gravity.y + gravity.z*gravity.z);
    ContactWorld world = contactWorld(obstacleState, obstacles, low, high, voxel, dt, 2.0f*g*(float)dt);
    int slots = numBodies*CONTACT_PARTNERS*CONTACT_SLOTS;
    gpuErrchk(cudaMemsetAsync(picks, 0xFF, sizeof(unsigned long long)*slots, stream));
    uint blocks = std::min((uint)numSurfacePoints / CONTACT_THREADS + 1, 1024u);
    for(int pass = 0; pass < CONTACT_SLOTS; ++pass){
        pickContacts<<<blocks, CONTACT_THREADS, 0, stream>>>(pass, world, surfacePoints, surfaceBodies, numSurfacePoints, picks);
    }
    buildContacts<<<slots / CONTACT_THREADS + 1, CONTACT_THREADS, 0, stream>>>(world, surfacePoints, states, deviceMasses, couplings, picks, slots, contacts);
    carryContacts<<<1, 1, 0, stream>>>(numBodies, slots, (float)dt, states, couplings, lastContacts, contacts, contactOrder, batchStarts);
    solveContacts<<<1, CONTACT_THREADS, 0, stream>>>(numBodies, states, deviceMasses, couplings, frames, contacts, contactOrder, batchStarts);
    gpuErrchk(cudaPeekAtLastError());
    std::swap(contacts, lastContacts);  //this substep's are what the next starts from
}
