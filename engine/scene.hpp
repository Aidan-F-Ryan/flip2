//Copyright 2023 Aberrant Behavior LLC

#ifndef SCENE_HPP
#define SCENE_HPP

//A simulation described as a JSON file (schema "flip2.scene/1"): the domain, the solver's settings, the fluid it starts with, the forces on it and what
//to write out. Plain C++, so the DCC exporters, the command line and tests all read the same thing. Every length is in metres, every time in seconds, and
//+y is up unless the gravity says otherwise. An example, with every key and its default:
//
//  {
//    "schema": "flip2.scene/1",
//    "fps": 24, "frames": 120,
//    "domain": {"min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5], "voxelSize": 0.0078125,     //rounded up to whole nodes of 4^3 voxels
//               "open": []},                                               //faces that delete the fluid reaching them: "-x", "+x", "-y", "+y", "-z", "+z"
//    "solver": {"flipRatio": 0.95, "cfl": 4, "densityCorrectionTime": 0.1, "pressureSolver": "multigrid", "advection": "rk3", "dotProducts": "exact",
//               "transfer": "flip",                                        //or "apic": particles carry their velocity's gradient; flipRatio 0 for pure APIC
//               "viscousCfl": 6,                                           //with viscosity: voxels it may spread across in a substep; 0 for no limit
//               "freeSurface": "footprint"},                               //or "sharp": the pressure is 0 at the level set's surface (ghost fluid),
//                                                                           //which costs more a substep and, with surface tension, 0.7 the timestep
//    "gravity": [0, -9.8, 0],
//    "liquid": {"density": 1000, "viscosity": 0, "surfaceTension": 0,       //kg/m^3; Pa s, dynamic (water 0.001, honey 2 to 10); N/m (water 0.073). With
//               "contactAngle": 60},                                        //viscosity, it sticks to walls and obstacles; with surface tension, the
//                                                                           //timestep keeps to the capillary limit, sqrt(density voxelSize^3 / 2 pi it),
//                                                                           //and its surface meets the domain's walls at contactAngle degrees: under 90
//                                                                           //it wets them, over 90 it beads up on them
//    "particlesPerVoxel": 8, "seed": 1,                                     //1, 8 or 27: seeded on a jittered 1^3, 2^3 or 3^3 lattice per voxel
//    "fluids": [{"shape": "box", "min": [...], "max": [...], "velocity": [0, 0, 0]},
//               {"shape": "sphere", "center": [...], "radius": 0.1, "velocity": [0, 0, 0]},
//               {"mesh": "blob.obj", "transform": [...], "velocity": [0, 0, 0]}],         //fluids, emitters and sinks can be meshes, placed as obstacles are
//    "emitters": [{"shape": "box", "min": [...], "max": [...], "velocity": [0, 0, 2]}],   //inflow: keeps its shape full of fluid moving at velocity
//    "sinks": [{"shape": "sphere", "center": [...], "radius": 0.1}],                       //outflow: deletes the fluid inside it
//    "obstacles": [{"mesh": "rock.obj", "friction": 0, "thickness": 0,                   //or "vertices": "v.npy", "triangles": "t.npy"; or a box or sphere;
//                                                                                         //or "vdb": "rock.vdb", "grid": "surface", "velocityGrid": "vel"
//                   "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1],                    //its own space to the world: row-major, acting on column vectors
//                   "keyframes": [{"time": 0, "transform": [...]}, ...],                 //instead of a transform, it moves: linearly, rotations by slerp
//                   "deforming": [{"time": 0, "mesh": "f1.obj"}, {"time": 0.04, "vertices": "f2.npy"}, ...]}],   //or it deforms: see SceneObstacle
//    "forces": [{"type": "point", "position": [...], "strength": 9.8, "radius": 0, "falloff": 1},       //strength in m/s^2, towards it (negative: away)
//               {"type": "vortex", "position": [...], "axis": [0, 1, 0], "strength": 5, "radius": 0, "falloff": 1},
//               {"type": "turbulence", "strength": 2, "scale": 0.1, "speed": 0.5, "seed": 0},
//               {"type": "wind", "velocity": [2, 0, 0], "drag": 1, "depth": 2},                //drag per second, on the voxels within depth of the surface
//               {"type": "volume", "vdb": "push.vdb", "grid": "", "mode": "force",             //a VDB of vectors (its first, or the one grid names), fixed in
//                "strength": 1, "drag": 1}],                                                    //the world, acting where it has active voxels and not
//                                                                                               //elsewhere: accelerations in m/s^2 times strength, or with
//                                                                                               //mode "velocity", velocities times strength, which the
//                                                                                               //fluid there takes up at drag per second
//    "partitions": 1, "devices": 0,
//    "output": {"dir": ".", "cache": true, "compression": "zstd",           //the cache DCCs read (cacheWriter.hu), in dir; "zstd", "lz4" or "none"
//               "attributes": ["id", "age"],                               //what its frames carry besides P and v: each particle's id, which it keeps
//                                                                          //and no other ever has, and its age in seconds. 12 bytes a particle
//                                                                          //between them, about 3 once compressed
//               "checkpoints": {"every": 10, "keep": 2},                   //in the cache, for flip2 resume: every so many frames (0: only on cancel and at
//                                                                          //the end), keeping the newest few
//               "positions": false, "diagnostics": ""}                     //N.bin, float32 x, y, z per particle; diagnostics: a file name in dir, or ""
//  }

#include <array>
#include <memory>
#include <string>
#include <vector>

//A distance field in the engine's own sparse layout (SolidSDF, obstacles.hu), sampled at the domain's voxel size: what a VDB level set is resampled to
//(volumes.hpp). Bricks of 8^3 samples cover its bounds, x fastest; only those near its surface hold samples, and the rest are all inside or all outside.
//A force's volume of vectors is kept the same way, at its own spacing: table says which bricks hold samples, velocities holds the vectors, and pool, in
//place of distances, how much of each sample is the volume's own: 1 among its active voxels, 0 past them, where its vector is 0 too
struct SceneField{
    double origin[3] = {0.0, 0.0, 0.0};     //sample (0, 0, 0)
    double spacing = 0.0;                   //between samples
    int bricks[3] = {0, 0, 0};
    float band = 0.0f;                      //distances are clamped to +-band
    std::vector<int> table;                 //per brick: its place in pool, in bricks, or -1 all outside, or -2 all inside
    std::vector<float> pool;                //per brick there: 8^3 signed distances, negative inside, x fastest
    std::vector<float> velocities;          //with a velocity grid: per brick in pool, 8^3 x velocities, then y, then z; otherwise empty
    float fastest = 0.0f;                   //the largest speed among them
    double low[3] = {0.0, 0.0, 0.0};        //the bounds of its surface
    double high[3] = {0.0, 0.0, 0.0};
};

//Something solid the fluid flows around: a triangle mesh, a box or a sphere, in its own space, placed in the world by a transform or moved by keyframed
//transforms. Each transform's scale is baked into the shape at the first key, so it moves rigidly from there on, and the fluid sees its true distances.
//Or a deforming mesh: samples of its vertices in the world over time, with the same triangles throughout, each an OBJ (whose triangles have to be the
//mesh's) or an (n, 3) .npy of vertices. Between samples its vertices move linearly, and before the first and after the last it holds still
struct SceneObstacle{
    enum Kind{MESH, BOX, SPHERE, FIELD};    //FIELD: a VDB level set, placed rigidly, with no scale
    Kind kind = BOX;
    std::vector<float> vertices;    //mesh: x, y, z per vertex
    std::vector<int> triangles;     //mesh: 3 vertex indices per triangle
    double min[3] = {0.0, 0.0, 0.0};        //box
    double max[3] = {0.0, 0.0, 0.0};
    double centre[3] = {0.0, 0.0, 0.0};     //sphere
    double radius = 0.0;
    std::vector<double> keyTimes;                       //empty: it stays where transforms[0] puts it
    std::vector<std::array<double, 16>> transforms;     //own space to world, row-major, acting on column vectors (translation in elements 3, 7, 11)
    double friction = 0.0;      //0: the fluid slips past freely; 1: the fluid touching it moves with it
    double thickness = 0.0;     //mesh: 0 for a closed surface; for an open one, like a ground plane, the shell's thickness around it
    std::vector<double> sampleTimes;                    //deforming: when each sample is; empty: it doesn't deform
    std::shared_ptr<const std::vector<float>> samples;  //deforming: each sample's vertices in turn, x, y, z each, in the world (shared by the scene's copies)
    std::shared_ptr<const SceneField> field;            //FIELD: its distances, and velocities if it came with a velocity grid (then it isn't placed)
};

//a fluid, emitter or sink: a box or sphere in the world, or a closed mesh placed and moved as an obstacle is ("mesh" or "vertices" and "triangles", with
//a "transform", "keyframes" or "deforming" samples), which only the GPU tests: it seeds a mesh fluid's particles at the start, where no box or sphere
//fluid, earlier mesh fluid or obstacle is
struct SceneShape{
    enum Kind{BOX, SPHERE, MESH};
    Kind kind = BOX;
    double min[3] = {0.0, 0.0, 0.0};        //a box's corners
    double max[3] = {0.0, 0.0, 0.0};
    double centre[3] = {0.0, 0.0, 0.0};     //a sphere's
    double radius = 0.0;
    double velocity[3] = {0.0, 0.0, 0.0};   //what its fluid starts with
    bool air = false;                       //a fluid's "phase": "air": it's the scene's air (Scene::air), not its liquid. Air comes after every liquid fluid
    bool carve = false;                     //an air fluid's "carve": true: it takes its space out of the liquid (a bubble); otherwise liquid has what both hold
    SceneObstacle mesh;                     //a mesh's, and where it is (friction and thickness unused)

    bool contains(const double point[3]) const;     //never, for a mesh
    void bounds(double low[3], double high[3]) const;
};

struct SceneForce{
    enum Kind{POINT, VORTEX, TURBULENCE, WIND, VOLUME};
    Kind kind = POINT;
    double position[3] = {0.0, 0.0, 0.0};   //point: where it pulls to; vortex: a point on its axis
    double axis[3] = {0.0, 1.0, 0.0};       //vortex: its axis, which needn't be unit length
    double velocity[3] = {0.0, 0.0, 0.0};   //wind: the air's
    double strength = 0.0;                  //point, vortex, turbulence: m/s^2 at full strength
    double radius = 0.0;                    //point, vortex: where it fades to nothing; 0: no fade, it reaches everywhere
    double falloff = 1.0;                   //point, vortex: the fade, (1 - r/radius)^falloff
    double scale = 0.1;                     //turbulence: the size of its eddies
    double speed = 0.5;                     //turbulence: how fast its pattern drifts, in eddies per second
    unsigned int seed = 0;                  //turbulence
    double drag = 1.0;                      //wind: how quickly the surface takes up its velocity, per second; volume, in velocity mode: the fluid inside it
    double depth = 2.0;                     //wind: how many voxels below the surface it reaches
    std::shared_ptr<const SceneField> field;    //volume: its vectors (volumes.hpp's loadVectorField); strength scales them
    bool velocities = false;                //volume: whether they're velocities to take up, rather than accelerations
};

struct Scene{
    std::string path;           //where it was read from: relative output paths are relative to the working directory, not to it
    double fps = 24.0;
    int frames = 120;
    double domainMin[3] = {-0.5, -0.5, -0.5};
    unsigned int nodes[3] = {32, 32, 32};   //the domain in nodes of 4^3 voxels
    double nodeSize = 1.0 / 32.0;           //metres per node
    double flipRatio = 0.95;
    double cfl = 4.0;
    double densityCorrectionTime = 0.1;
    std::string pressureSolver = "multigrid";
    std::string advection = "rk3";
    std::string dotProducts = "exact";
    std::string transfer = "flip";
    double viscousCfl = 6.0;        //with viscosity: voxels it may spread across in a substep; 0 for no limit
    std::string freeSurface = "footprint";  //or "sharp": where the pressure solve puts the liquid's surface (FreeSurface, particles.hu)
    double gravity[3] = {0.0, -9.8, 0.0};
    double density = 1000.0;        //the liquid's, kg/m^3: what turns its viscosity and surface tension into accelerations
    double viscosity = 0.0;         //dynamic, Pa s
    double surfaceTension = 0.0;    //N/m
    double contactAngle = 60.0;     //degrees, through the liquid, where its surface meets the domain's walls
    //"air": {...}: a second, lighter fluid simulated with the liquid (TwoPhase, particles.hu; experimental). Fluids with "phase": "air" are seeded as it
    bool air = false;
    double airDensity = 1.2;        //kg/m^3: with the liquid's, the density ratio
    std::string airFaceDensity = "fractions";   //how a face's density is found: "fractions", "phaseField", "levelSet", or "synthetic" (no air particles)
    double airFlipRatio = -1.0;     //the air's share of FLIP; negative: the liquid's
    int airBand = 0;                //voxels of air kept around the liquid; 0 keeps all of it
    std::string airSyntheticShape = "plane";    //synthetic: air above "centre"'s height ("plane"), in a "ball" there, or in "balls" on a lattice from there
    double airSyntheticCentre[3] = {0.0, 0.0, 0.0};
    double airSyntheticRadius = 0.0;
    double airSyntheticSpacing = 0.0;
    int particlesPerVoxel = 8;
    unsigned long long seed = 1;
    std::vector<SceneShape> fluids;
    std::vector<SceneShape> emitters;
    std::vector<SceneShape> sinks;
    unsigned int openFaces = 0;     //bits: -x, +x, -y, +y, -z, +z
    std::vector<SceneObstacle> obstacles;
    std::vector<SceneForce> forces;
    int partitions = 1;
    int devices = 0;            //0: every GPU there is
    std::string outputDirectory = ".";
    bool writeCache = true;     //frames/NNNN/ and cache.json in the output directory (cacheWriter.hu)
    std::string compression = "zstd";   //of the cache's shards: "zstd" or "lz4" (blosc), or "none"
    bool writeIds = true;       //whether the cache's frames carry each particle's "id"
    bool writeAges = true;      //and its "age"
    int checkpointEvery = 10;   //frames between the cache's checkpoints: 0 for none but on cancel and at the end
    int keepCheckpoints = 2;    //the newest kept
    bool writePositions = false;    //N.bin, float32 x, y, z per particle, gathered into one file per frame: for quick looks
    std::string diagnostics;    //a file name in the output directory, or empty for none

    double voxelSize() const{
        return nodeSize / 4.0;
    }
};

//reads and checks a scene file, and the meshes it names (relative to its own directory); throws std::runtime_error naming the file and what's wrong
//with it. Keys it doesn't know are reported on stderr, in case they're typos, and otherwise ignored
Scene loadScene(const std::string& path);

//a triangle mesh from a Wavefront OBJ file (its v and f lines; polygons are split into fans) or from two NumPy .npy arrays, vertices (N x 3 floats) and
//triangles (M x 3 integers); throws std::runtime_error saying what's wrong
void loadObj(const std::string& path, std::vector<float>& vertices, std::vector<int>& triangles);
void loadNpyMesh(const std::string& verticesPath, const std::string& trianglesPath, std::vector<float>& vertices, std::vector<int>& triangles);

//the fluid the scene starts with: particlesPerVoxel per voxel, on a lattice of 1, 2 or 3 per side, each jittered within its lattice cell by a hash of
//the seed and its place in the domain, so the same scene always seeds the same particles. Where shapes overlap, the earlier one's velocity wins
//Fluids are seeded in the order the scene lists them, so with air, which comes last, the liquid's particles are the first *liquid of them
void seedParticles(const Scene& scene, std::vector<double>& x, std::vector<double>& y, std::vector<double>& z, std::vector<float>& u, std::vector<float>& v, std::vector<float>& w,
                   size_t* liquid = nullptr);

#endif
