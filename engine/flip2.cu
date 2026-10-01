//Copyright 2023 Aberrant Behavior LLC

//flip2's command line:
//
//  flip2 bake scene.json [--frames N] [--out DIR] [--overwrite]
//  flip2 verify DIR
//
//bake runs the scene (see scene.hpp) and writes into the output directory its cache (cacheWriter.hu: cache.json and frames/NNNN/, which a DCC can read
//while the bake runs), and if the scene asks, N.bin (float32 x, y, z per particle) and the diagnostics file. It won't bake over a cache already there
//unless told to with --overwrite, which deletes it first. verify checks every committed frame of the cache in DIR: each shard there, whole, with the size
//and XXH64 its commit record gives. Standard output carries only events, one JSON object per line, for a DCC or a farm to follow:
//
//  {"event":"start","particles":1240000,"frames":120,"nodes":[32,32,32],"voxelSize":0.0078125,"ranks":1}
//  {"event":"frame","frame":1,"seconds":0.041}                 a frame simulated
//  {"event":"committed","frame":1}                             and its cache frame on the disk, whole; from the cache's thread, so it can come later
//  {"event":"done","frames":120,"seconds":5.2}                 every frame committed
//  {"event":"verified","frames":121,"unfinished":0,"shards":121,"particles":150040000,"bytes":1712345678,"problems":0}
//  {"event":"error","message":"scene.json: domain: needs a positive \"voxelSize\""}
//
//Everything else the engine says goes to standard error. As with main, FLIP2_PARTITIONS splits the domain between partitions in this process, and
//FLIP2_WORLD_SIZE, FLIP2_RANK, FLIP2_RENDEZVOUS, FLIP2_DEVICE and FLIP2_TRANSPORT make this process one rank of several (tools/launch-local.sh)

#include "testing.h"
#include "scene.hpp"
#include "json.hpp"
#include "xxhash64.hpp"
#include "tcpTransport.hu"
#ifdef FLIP2_WITH_NCCL
#include "ncclTransport.hu"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

static bool printsEvents = true;    //only rank 0 does, across processes
static std::mutex eventLock;        //the cache's thread says when frames are committed

static void event(const std::string& json){
    if(printsEvents){
        std::lock_guard<std::mutex> lock(eventLock);
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
    std::cerr<<"usage: flip2 bake scene.json [--frames N] [--out DIR] [--overwrite]\n"
               "       flip2 verify DIR\n";
    return 2;
}

static bool readFile(const std::string& path, std::string& contents){
    std::ifstream file(path, std::ios::binary);
    if(!file){
        return false;
    }
    std::stringstream text;
    text<<file.rdbuf();
    contents = text.str();
    return true;
}

static std::string hex(uint64_t value){
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", (unsigned long long)value);
    return text;
}

//flip2 verify: every committed frame's shards, each the size its record gives, hashing to its XXH64, with a shard's header and the right particle count
static int verify(const std::string& directory){
    std::string text;
    if(!readFile(directory + "/cache.json", text)){
        return failure(directory + ": no cache.json, so no cache here");
    }
    int committed = -1;
    try{
        Json cache = JsonReader(text, directory + "/cache.json").document();
        if(const Json* last = cache.find("committed")){
            committed = (int)last->number;
        }
    }
    catch(const std::exception& error){
        return failure(error.what());
    }
    std::vector<std::filesystem::path> frames;
    std::error_code listed;
    for(const auto& entry : std::filesystem::directory_iterator(directory + "/frames", listed)){
        if(entry.is_directory()){
            frames.push_back(entry.path());
        }
    }
    std::sort(frames.begin(), frames.end());
    size_t done = 0, unfinished = 0, shards = 0, problems = 0;
    unsigned long long particles = 0, bytes = 0;
    auto problem = [&](const std::string& what){
        if(problems++ < 20){
            event("{\"event\":\"error\",\"message\":" + quoted(what) + "}");
        }
        std::cerr<<what<<"\n";
    };
    std::vector<char> chunk(4 << 20);
    for(const auto& frame : frames){
        std::string record;
        if(!readFile((frame / "commit.json").string(), record)){
            ++unfinished;
            continue;
        }
        ++done;
        try{
            Json commit = JsonReader(record, (frame / "commit.json").string()).document();
            const Json* list = commit.find("shards");
            if(list == nullptr){
                problem((frame / "commit.json").string() + ": no shards");
                continue;
            }
            for(const Json& shard : list->items){
                const Json* file = shard.find("file");
                const Json* size = shard.find("bytes");
                const Json* hash = shard.find("xxh64");
                const Json* count = shard.find("particles");
                if(file == nullptr || size == nullptr || hash == nullptr || count == nullptr){
                    problem((frame / "commit.json").string() + ": a shard without its file, bytes, particles and xxh64");
                    continue;
                }
                ++shards;
                std::string path = (frame / file->text).string();
                std::ifstream in(path, std::ios::binary);
                if(!in){
                    problem(path + ": missing");
                    continue;
                }
                Xxh64 digest;
                unsigned long long length = 0;
                char header[24] = {};
                while(in){
                    in.read(chunk.data(), chunk.size());
                    size_t got = (size_t)in.gcount();
                    if(length < sizeof(header)){
                        std::memcpy(header + length, chunk.data(), std::min(got, (size_t)(sizeof(header) - length)));
                    }
                    digest.update(chunk.data(), got);
                    length += got;
                }
                unsigned long long headerCount;
                std::memcpy(&headerCount, header + 16, 8);
                if(length != (unsigned long long)size->number){
                    problem(path + ": " + std::to_string(length) + " bytes, but its commit record says " + std::to_string((unsigned long long)size->number));
                }
                else if(hex(digest.digest()) != hash->text){
                    problem(path + ": its XXH64 is " + hex(digest.digest()) + ", but its commit record says " + hash->text);
                }
                else if(std::memcmp(header, "FLIP2SHD", 8) != 0 || headerCount != (unsigned long long)count->number){
                    problem(path + ": not a shard of " + std::to_string((unsigned long long)count->number) + " particles");
                }
                particles += (unsigned long long)count->number;
                bytes += length;
            }
        }
        catch(const std::exception& error){
            problem(error.what());
        }
    }
    for(int frame = 0; frame <= committed; ++frame){     //cache.json's committed frame and every one before it
        char name[16];
        std::snprintf(name, sizeof(name), "%04d", frame);
        if(!std::filesystem::exists(directory + "/frames/" + name + "/commit.json")){
            problem(directory + "/frames/" + name + ": cache.json says it's committed, but it has no commit record");
        }
    }
    char line[512];
    std::snprintf(line, sizeof(line), "{\"event\":\"verified\",\"frames\":%zu,\"unfinished\":%zu,\"shards\":%zu,\"particles\":%llu,\"bytes\":%llu,\"problems\":%zu}",
                  done, unfinished, shards, particles, bytes, problems);
    event(line);
    return problems > 0 ? 1 : 0;
}

//a mesh adds its description to meshes, which the shape names by its place there
static FluidShape toFluidShape(const SceneShape& shape, std::vector<SceneObstacle>& meshes){
    FluidShape out;
    out.kind = shape.kind == SceneShape::SPHERE ? FluidShape::SPHERE : shape.kind == SceneShape::MESH ? FluidShape::MESH : FluidShape::BOX;
    shape.bounds(out.low, out.high);
    for(int axis = 0; axis < 3; ++axis){
        out.centre[axis] = shape.centre[axis];
        out.velocity[axis] = (float)shape.velocity[axis];
    }
    out.radius = shape.radius;
    if(shape.kind == SceneShape::MESH){
        out.meshIndex = (int)meshes.size();
        meshes.push_back(shape.mesh);
    }
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
    if(argc == 3 && std::string(argv[1]) == "verify"){
        return verify(argv[2]);
    }
    if(argc < 3 || std::string(argv[1]) != "bake"){
        return usage();
    }
    std::string scenePath = argv[2];
    int frames = -1;
    std::string outputDirectory;
    bool overwrite = false;
    for(int arg = 3; arg < argc; ++arg){
        std::string option = argv[arg];
        if(option == "--frames" && arg + 1 < argc){
            frames = std::atoi(argv[++arg]);
        }
        else if(option == "--out" && arg + 1 < argc){
            outputDirectory = argv[++arg];
        }
        else if(option == "--overwrite"){
            overwrite = true;
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
    bool meshFluids = std::any_of(scene.fluids.begin(), scene.fluids.end(), [](const SceneShape& fluid){ return fluid.kind == SceneShape::MESH; });
    if(x.empty() && scene.emitters.empty() && !meshFluids){
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
    bool alone = worldSize <= 1 && transport.empty();
    bool rankZero = alone || (std::getenv("FLIP2_RANK") && std::atoi(std::getenv("FLIP2_RANK")) == 0);
    std::string sceneText;
    if(scene.writeCache && !readFile(scenePath, sceneText)){
        return failure(scenePath + ": can't read it again to hash it for the cache");
    }
    //a cache already here is only replaced when asked; rank 0 deletes it before it joins the others, who wait for it to before they write anything
    if(scene.writeCache && rankZero && std::filesystem::exists(scene.outputDirectory + "/cache.json")){
        if(!overwrite){
            return failure(scene.outputDirectory + " holds a cache already: bake with --overwrite to replace it, or give another --out");
        }
        std::error_code removed;
        std::filesystem::remove_all(scene.outputDirectory + "/frames", removed);
        if(!removed){
            std::filesystem::remove(scene.outputDirectory + "/cache.json", removed);
        }
        if(removed){
            return failure(scene.outputDirectory + ": can't delete the cache there: " + removed.message());
        }
    }
    std::unique_ptr<ParticleSystemTester> simulation;
    int ranks = scene.partitions;
    if(!alone){
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
    simulation->setApic(scene.transfer == "apic");
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
    std::vector<SceneObstacle> sourceMeshes;
    for(const SceneShape& emitter : scene.emitters){
        sources.emitters[sources.numEmitters++] = toFluidShape(emitter, sourceMeshes);
    }
    for(const SceneShape& sink : scene.sinks){
        sources.sinks[sources.numSinks++] = toFluidShape(sink, sourceMeshes);
    }
    simulation->setSources(sources);
    std::vector<FluidShape> fluids;
    for(const SceneShape& fluid : scene.fluids){
        fluids.push_back(toFluidShape(fluid, sourceMeshes));
    }
    if(!sourceMeshes.empty()){
        simulation->setSourceMeshes(sourceMeshes, fluids);      //after setDomain and setSources
    }
    simulation->setObstacles(scene.obstacles);      //after setDomain: they're voxelized at its voxel size
    if(scene.writeCache){
        CacheDescription description;
        std::error_code resolved;
        std::filesystem::path absolute = std::filesystem::absolute(scenePath, resolved);
        description.scenePath = resolved ? scenePath : absolute.lexically_normal().string();
        Xxh64 hash;
        hash.update(sceneText.data(), sceneText.size());
        description.sceneHash = hex(hash.digest());
        description.fps = scene.fps;
        description.frames = scene.frames;
        description.compression = scene.compression;
        simulation->startCache(scene.outputDirectory, description, [](int frame){ event("{\"event\":\"committed\",\"frame\":" + std::to_string(frame) + "}"); });
    }

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
    if(scene.writeCache){
        simulation->writeCacheFrame(0);
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
        if(scene.writeCache){
            simulation->writeCacheFrame(frame);
            std::string error = simulation->cacheError();
            if(!error.empty()){
                return failure(error);
            }
        }
        std::snprintf(line, sizeof(line), "{\"event\":\"frame\",\"frame\":%d,\"seconds\":%.4f,\"particles\":%zu}", frame,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - frameStart).count(), simulation->particlesHere());
        event(line);
    }
    std::string error = simulation->finishCache();  //every frame committed
    if(!error.empty()){
        return failure(error);
    }
    simulation.reset();     //finishes writing the frames
    std::snprintf(line, sizeof(line), "{\"event\":\"done\",\"frames\":%d,\"seconds\":%.3f}", scene.frames, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    event(line);
    return 0;
}
