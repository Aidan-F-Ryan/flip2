//Copyright 2023 Aberrant Behavior LLC

//flip2's command line:
//
//  flip2 bake scene.json [--frames N] [--out DIR]
//
//runs the scene (see scene.hpp) and writes its frames into the output directory: N.bin, float32 x, y, z per particle, and the diagnostics file if the
//scene asks for one. Standard output carries only events, one JSON object per line, for a DCC or a farm to follow:
//
//  {"event":"start","particles":1240000,"frames":120,"nodes":[32,32,32],"voxelSize":0.0078125,"ranks":1}
//  {"event":"frame","frame":1,"seconds":0.041}
//  {"event":"done","frames":120,"seconds":5.2}
//  {"event":"error","message":"scene.json: domain: needs a positive \"voxelSize\""}
//
//Everything else the engine says goes to standard error. As with main, FLIP2_PARTITIONS splits the domain between partitions in this process, and
//FLIP2_WORLD_SIZE, FLIP2_RANK, FLIP2_RENDEZVOUS, FLIP2_DEVICE and FLIP2_TRANSPORT make this process one rank of several (tools/launch-local.sh)

#include "testing.h"
#include "scene.hpp"
#include "tcpTransport.hu"
#ifdef FLIP2_WITH_NCCL
#include "ncclTransport.hu"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

static bool printsEvents = true;    //only rank 0 does, across processes

static void event(const std::string& json){
    if(printsEvents){
        std::fputs((json + "\n").c_str(), stdout);
        std::fflush(stdout);
    }
}

static std::string quoted(const std::string& text){     //a JSON string
    std::string out = "\"";
    for(char c : text){
        if(c == '"' || c == '\\'){
            out += '\\';
            out += c;
        }
        else if((unsigned char)c < 0x20){
            char escaped[8];
            std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
            out += escaped;
        }
        else{
            out += c;
        }
    }
    return out + "\"";
}

static int failure(const std::string& message){
    event("{\"event\":\"error\",\"message\":" + quoted(message) + "}");
    std::cerr<<message<<"\n";
    return 1;
}

static int usage(){
    std::cerr<<"usage: flip2 bake scene.json [--frames N] [--out DIR]\n";
    return 2;
}

static FluidShape toFluidShape(const SceneShape& shape){
    FluidShape out;
    out.kind = shape.kind == SceneShape::SPHERE ? FluidShape::SPHERE : FluidShape::BOX;
    shape.bounds(out.low, out.high);
    for(int axis = 0; axis < 3; ++axis){
        out.centre[axis] = shape.centre[axis];
        out.velocity[axis] = (float)shape.velocity[axis];
    }
    out.radius = shape.radius;
    return out;
}

static ForceField toForceField(const SceneForce& force){
    ForceField field;
    field.kind = force.kind == SceneForce::POINT ? FORCE_POINT : force.kind == SceneForce::VORTEX ? FORCE_VORTEX : force.kind == SceneForce::TURBULENCE ? FORCE_TURBULENCE : FORCE_WIND;
    field.position = make_float3((float)force.position[0], (float)force.position[1], (float)force.position[2]);
    double length = std::sqrt(force.axis[0]*force.axis[0] + force.axis[1]*force.axis[1] + force.axis[2]*force.axis[2]);
    field.axis = length > 0.0 ? make_float3((float)(force.axis[0]/length), (float)(force.axis[1]/length), (float)(force.axis[2]/length)) : make_float3(0.0f, 1.0f, 0.0f);
    field.velocity = make_float3((float)force.velocity[0], (float)force.velocity[1], (float)force.velocity[2]);
    field.strength = (float)force.strength;
    field.radius = (float)force.radius;
    field.falloff = (float)force.falloff;
    field.scale = (float)force.scale;
    field.speed = (float)force.speed;
    field.seed = force.seed;
    field.drag = (float)force.drag;
    field.depth = (float)force.depth;
    return field;
}

int main(int argc, char** argv){
    std::cout.rdbuf(std::cerr.rdbuf());     //the engine's own messages go to standard error, so standard output carries only the events
    if(argc < 3 || std::string(argv[1]) != "bake"){
        return usage();
    }
    std::string scenePath = argv[2];
    int frames = -1;
    std::string outputDirectory;
    for(int arg = 3; arg < argc; ++arg){
        std::string option = argv[arg];
        if(option == "--frames" && arg + 1 < argc){
            frames = std::atoi(argv[++arg]);
        }
        else if(option == "--out" && arg + 1 < argc){
            outputDirectory = argv[++arg];
        }
        else{
            return usage();
        }
    }
    Scene scene;
    try{
        scene = loadScene(scenePath);
    }
    catch(const std::exception& error){
        return failure(error.what());
    }
    if(frames >= 0){
        scene.frames = frames;
    }
    if(!outputDirectory.empty()){
        scene.outputDirectory = outputDirectory;
    }
    if(const char* partitions = std::getenv("FLIP2_PARTITIONS")){
        scene.partitions = std::max(1, std::atoi(partitions));
    }

    std::vector<double> x, y, z;
    std::vector<float> u, v, w;
    seedParticles(scene, x, y, z, u, v, w);
    if(x.empty() && scene.emitters.empty()){
        return failure(scenePath + ": it has no fluid: its fluids are empty or outside the domain, and it has no emitters");
    }
    std::error_code made;
    std::filesystem::create_directories(scene.outputDirectory, made);
    if(made){
        return failure(scene.outputDirectory + ": can't make the output directory: " + made.message());
    }

    //the ranks, as main sets them up: this process is one rank of FLIP2_WORLD_SIZE, or every partition is here
    int worldSize = std::getenv("FLIP2_WORLD_SIZE") ? std::atoi(std::getenv("FLIP2_WORLD_SIZE")) : 1;
    std::string transport = std::getenv("FLIP2_TRANSPORT") ? std::getenv("FLIP2_TRANSPORT") : "";
    std::unique_ptr<ParticleSystemTester> simulation;
    int ranks = scene.partitions;
    if(worldSize > 1 || !transport.empty()){
        const char* rank = std::getenv("FLIP2_RANK");
        const char* rendezvous = std::getenv("FLIP2_RENDEZVOUS");
        if(rank == nullptr || rendezvous == nullptr){
            return failure("running as a rank needs FLIP2_RANK and FLIP2_RENDEZVOUS too");
        }
        printsEvents = std::atoi(rank) == 0;
        int device = std::getenv("FLIP2_DEVICE") ? std::atoi(std::getenv("FLIP2_DEVICE")) : 0;
        std::unique_ptr<Transport> link;
        if(transport.empty() || transport == "tcp"){
            link = std::make_unique<TcpTransport>(std::atoi(rank), worldSize, rendezvous);
        }
        else if(transport == "nccl"){
#ifdef FLIP2_WITH_NCCL
            link = std::make_unique<NcclTransport>(std::atoi(rank), worldSize, rendezvous);
#else
            return failure("FLIP2_TRANSPORT=nccl, but this build has no NCCL: build with NCCL_HOME set to where it's installed");
#endif
        }
        else{
            return failure("FLIP2_TRANSPORT is tcp or nccl, not " + transport);
        }
        ranks = worldSize;
        simulation = std::make_unique<ParticleSystemTester>((uint)x.size(), std::move(link), device);
    }
    else{
        simulation = std::make_unique<ParticleSystemTester>((uint)x.size(), scene.partitions, scene.devices);
    }

    simulation->setDomain(scene.domainMin[0], scene.domainMin[1], scene.domainMin[2], scene.nodes[0], scene.nodes[1], scene.nodes[2], scene.nodeSize);
    simulation->setParticles(x, y, z, u, v, w, scene.particlesPerVoxel);
    simulation->setFlipRatio(scene.flipRatio);
    simulation->setDensityCorrectionTime(scene.densityCorrectionTime);
    simulation->setCfl(scene.cfl);
    simulation->setPressureSolver(scene.pressureSolver == "sor" ? PressureSolver::sor : scene.pressureSolver == "cg" ? PressureSolver::cg :
                                  scene.pressureSolver == "jacobi" ? PressureSolver::jacobi : PressureSolver::multigrid);
    simulation->setRungeKutta3(scene.advection == "rk3");
    simulation->setDotProductSums(scene.dotProducts == "blocks" ? DotProductSums::perBlock : DotProductSums::exact);
    simulation->setGravity(make_float3((float)scene.gravity[0], (float)scene.gravity[1], (float)scene.gravity[2]));
    std::vector<ForceField> fields;
    for(const SceneForce& force : scene.forces){
        fields.push_back(toForceField(force));
    }
    simulation->setForceFields(fields);
    Sources sources;
    sources.latticePerSide = scene.particlesPerVoxel == 27 ? 3 : scene.particlesPerVoxel == 8 ? 2 : 1;
    sources.seed = scene.seed;
    sources.openFaces = scene.openFaces;
    for(const SceneShape& emitter : scene.emitters){
        sources.emitters[sources.numEmitters++] = toFluidShape(emitter);
    }
    for(const SceneShape& sink : scene.sinks){
        sources.sinks[sources.numSinks++] = toFluidShape(sink);
    }
    simulation->setSources(sources);
    simulation->setObstacles(scene.obstacles);      //after setDomain: they're voxelized at its voxel size

    std::string directory = scene.outputDirectory + "/";
    std::string diagnostics = scene.diagnostics.empty() ? "" : directory + scene.diagnostics;
    auto start = std::chrono::steady_clock::now();
    simulation->initialize();
    char line[512];
    std::snprintf(line, sizeof(line), "{\"event\":\"start\",\"particles\":%zu,\"frames\":%d,\"nodes\":[%u,%u,%u],\"voxelSize\":%.9g,\"ranks\":%d}",
                  x.size(), scene.frames, scene.nodes[0], scene.nodes[1], scene.nodes[2], scene.voxelSize(), ranks);
    event(line);
    if(!diagnostics.empty()){
        simulation->writeDiagnostics(diagnostics, 0);
    }
    if(scene.writePositions){
        simulation->writePositionsToFile(directory + "0.bin");
    }
    for(int frame = 1; frame <= scene.frames; ++frame){
        auto frameStart = std::chrono::steady_clock::now();
        simulation->solveFrame(scene.fps);
        if(!diagnostics.empty()){
            simulation->writeDiagnostics(diagnostics, frame);
        }
        if(scene.writePositions){
            simulation->writePositionsToFile(directory + std::to_string(frame) + ".bin");
        }
        std::snprintf(line, sizeof(line), "{\"event\":\"frame\",\"frame\":%d,\"seconds\":%.4f,\"particles\":%zu}", frame,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - frameStart).count(), simulation->particlesHere());
        event(line);
    }
    simulation.reset();     //finishes writing the frames
    std::snprintf(line, sizeof(line), "{\"event\":\"done\",\"frames\":%d,\"seconds\":%.3f}", scene.frames, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    event(line);
    return 0;
}
