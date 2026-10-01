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
//    "solver": {"flipRatio": 0.95, "cfl": 4, "densityCorrectionTime": 0.1, "pressureSolver": "multigrid", "advection": "rk3", "dotProducts": "exact"},
//    "gravity": [0, -9.8, 0],
//    "particlesPerVoxel": 8, "seed": 1,                                     //1, 8 or 27: seeded on a jittered 1^3, 2^3 or 3^3 lattice per voxel
//    "fluids": [{"shape": "box", "min": [...], "max": [...], "velocity": [0, 0, 0]},
//               {"shape": "sphere", "center": [...], "radius": 0.1, "velocity": [0, 0, 0]}],
//    "emitters": [{"shape": "box", "min": [...], "max": [...], "velocity": [0, 0, 2]}],   //inflow: keeps its shape full of fluid moving at velocity
//    "sinks": [{"shape": "sphere", "center": [...], "radius": 0.1}],                       //outflow: deletes the fluid inside it
//    "forces": [{"type": "point", "position": [...], "strength": 9.8, "radius": 0, "falloff": 1},       //strength in m/s^2, towards it (negative: away)
//               {"type": "vortex", "position": [...], "axis": [0, 1, 0], "strength": 5, "radius": 0, "falloff": 1},
//               {"type": "turbulence", "strength": 2, "scale": 0.1, "speed": 0.5, "seed": 0},
//               {"type": "wind", "velocity": [2, 0, 0], "drag": 1, "depth": 2}],               //drag per second, on the voxels within depth of the surface
//    "partitions": 1, "devices": 0,
//    "output": {"dir": ".", "positions": true, "diagnostics": ""}             //diagnostics: a file name in dir, or "" for none
//  }

#include <string>
#include <vector>

struct SceneShape{
    enum Kind{BOX, SPHERE};
    Kind kind = BOX;
    double min[3] = {0.0, 0.0, 0.0};        //a box's corners
    double max[3] = {0.0, 0.0, 0.0};
    double centre[3] = {0.0, 0.0, 0.0};     //a sphere's
    double radius = 0.0;
    double velocity[3] = {0.0, 0.0, 0.0};   //what its fluid starts with

    bool contains(const double point[3]) const;
    void bounds(double low[3], double high[3]) const;
};

struct SceneForce{
    enum Kind{POINT, VORTEX, TURBULENCE, WIND};
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
    double drag = 1.0;                      //wind: how quickly the surface takes up its velocity, per second
    double depth = 2.0;                     //wind: how many voxels below the surface it reaches
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
    double gravity[3] = {0.0, -9.8, 0.0};
    int particlesPerVoxel = 8;
    unsigned long long seed = 1;
    std::vector<SceneShape> fluids;
    std::vector<SceneShape> emitters;
    std::vector<SceneShape> sinks;
    unsigned int openFaces = 0;     //bits: -x, +x, -y, +y, -z, +z
    std::vector<SceneForce> forces;
    int partitions = 1;
    int devices = 0;            //0: every GPU there is
    std::string outputDirectory = ".";
    bool writePositions = true;
    std::string diagnostics;    //a file name in the output directory, or empty for none

    double voxelSize() const{
        return nodeSize / 4.0;
    }
};

//reads and checks a scene file; throws std::runtime_error naming the file and what's wrong with it. Keys it doesn't know are reported on stderr, in
//case they're typos, and otherwise ignored
Scene loadScene(const std::string& path);

//the fluid the scene starts with: particlesPerVoxel per voxel, on a lattice of 1, 2 or 3 per side, each jittered within its lattice cell by a hash of
//the seed and its place in the domain, so the same scene always seeds the same particles. Where shapes overlap, the earlier one's velocity wins
void seedParticles(const Scene& scene, std::vector<double>& x, std::vector<double>& y, std::vector<double>& z, std::vector<float>& u, std::vector<float>& v, std::vector<float>& w);

#endif
