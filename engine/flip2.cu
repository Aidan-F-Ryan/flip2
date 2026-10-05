//Copyright 2023 Aberrant Behavior LLC

//flip2's command line:
//
//  flip2 bake scene.json [--frames N] [--out DIR] [--overwrite]
//  flip2 resume DIR [--frames N] [--force]
//  flip2 verify DIR
//  flip2 export DIR [--frames A-B] [--out DIR] [--overwrite] [--follow] [--format bgeo.sc|bgeo]
//  flip2 mesh DIR [--frames A-B] [--out DIR] [--overwrite] [--follow] [--format bgeo.sc|bgeo] [--separation S] [--voxel-scale S] [--influence-scale S]
//             [--radius-scale S] [--smoothing N] [--fields] [--field-voxel-scale S] [--no-surface]
//  flip2 info
//
//bake runs the scene (see scene.hpp) and writes into the output directory its cache (cacheWriter.hu: cache.json, frames/NNNN/, which a DCC can read
//while the bake runs, and checkpoints/NNNN/), and if the scene asks, N.bin (float32 x, y, z per particle) and the diagnostics file. It won't bake over a
//cache already there unless told to with --overwrite, which deletes it first, with what was exported and meshed from it (DIR/export). The first SIGINT
//or SIGTERM lets the frame being simulated finish, checkpoints it and exits with status 3; a second exits at once. A pressure solve that doesn't converge
//ends the bake there with an error (status 1), the frames before it committed, for resume to take up from the newest checkpoint. resume carries the bake in DIR on from its newest checkpoint, exactly as if it had never
//stopped, on as many ranks as it's run with (so on more or fewer GPUs too): the frames after the checkpoint move to frames.discarded/ and are worked out
//again. It reads the scene the cache names, which has to be unchanged unless --force, and goes on to the frame the bake was going to; --frames takes it
//further. verify checks every
//committed frame of the cache in DIR: each shard there, whole, with the size and XXH64 its commit record gives. export writes the committed frames as files
//DCCs load natively (see exportCache), and mesh the liquid's surface in each (meshCache). info says what this build is and runs a
//kernel on every GPU it finds, to tell whether it runs there (exiting 1 if there's none it does). Standard output carries only events, one JSON object per
//line, for a DCC or a farm to follow:
//
//  {"event":"start","particles":1240000,"frames":120,"nodes":[32,32,32],"voxelSize":0.0078125,"ranks":1}     with "resumedFrom":40 when resuming
//  {"event":"frame","frame":1,"seconds":0.041}                 a frame simulated
//  {"event":"committed","frame":1}                             and its cache frame on the disk, whole; from the cache's thread, so it can come later
//  {"event":"checkpoint","frame":10}                           a checkpoint on the disk, whole
//  {"event":"cancelled","frame":57,"seconds":3.1}              stopped by a signal after frame 57, which is committed and checkpointed
//  {"event":"done","frames":120,"seconds":5.2}                 every frame committed
//  {"event":"verified","frames":121,"unfinished":0,"shards":121,"particles":150040000,"bytes":1712345678,"problems":0}
//  {"event":"exported","frame":42,"particles":1240000,"file":"/shots/a/export/houdini/particles.0042.bgeo.sc"}
//  {"event":"exportDone","frames":121,"exported":121,"seconds":9.8}
//  {"event":"meshed","frame":42,"particles":1240000,"points":310000,"faces":309000,"bandCells":52000,"rounds":2,"seconds":0.05,"file":"/shots/a/export/houdini/surface.0042.bgeo.sc"}
//      with --fields, and ...,"surfaceVoxels":90000,"velocityVoxels":180000,"fieldSeconds":0.01,"writeSeconds":0.05,"fields":"/shots/a/export/houdini/fields.0042.vdb"}
//  {"event":"meshDone","frames":121,"meshed":121,"seconds":12.1}
//  {"event":"info","build":"7dc95a4","architectures":"86","cudaRuntime":13040,"driver":13040,"gpus":[{"index":0,"name":"...","sm":86,"memoryMB":24135,"runs":true}],...}
//  {"event":"error","message":"scene.json: domain: needs a positive \"voxelSize\""}
//
//Everything else the engine says goes to standard error. As with main, FLIP2_PARTITIONS splits the domain between partitions in this process, and
//FLIP2_WORLD_SIZE, FLIP2_RANK, FLIP2_RENDEZVOUS, FLIP2_DEVICE and FLIP2_TRANSPORT make this process one rank of several (tools/launch-local.sh)

#include "testing.h"
#include "scene.hpp"
#include "json.hpp"
#include "xxhash64.hpp"
#include "bgeo.hpp"
#include "surface.hu"
#include "volumes.hpp"
#include "tcpTransport.hu"
#ifdef FLIP2_WITH_NCCL
#include "ncclTransport.hu"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

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
               "       flip2 resume DIR [--frames N] [--force]\n"
               "       flip2 verify DIR\n"
               "       flip2 export DIR [--frames A-B] [--out DIR] [--overwrite] [--follow] [--format bgeo.sc|bgeo]\n"
               "       flip2 mesh DIR [--frames A-B] [--out DIR] [--overwrite] [--follow] [--format bgeo.sc|bgeo] [--separation S] [--voxel-scale S]\n"
               "                  [--influence-scale S] [--radius-scale S] [--smoothing N] [--fields] [--field-voxel-scale S] [--no-surface]\n"
               "       flip2 info\n";
    return 2;
}

#ifndef FLIP2_ARCHITECTURES
#define FLIP2_ARCHITECTURES "unknown"
#endif

__global__ void probe(int* landed){
    *landed = 1;
}

//flip2 info: the build, and every GPU there is, with whether this build's kernels run on it: tried by launching one, as no table of architectures is
//as sure. 0 if they run on at least one
static int info(){
    int runtime = 0, driver = 0, devices = 0;
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    cudaError_t counted = cudaGetDeviceCount(&devices);
    std::string gpus;
    int running = 0;
    for(int device = 0; counted == cudaSuccess && device < devices; ++device){
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device);
        cudaSetDevice(device);
        int* landed = nullptr;
        int found = 0;
        cudaError_t tried = cudaMalloc((void**)&landed, sizeof(int));
        if(tried == cudaSuccess){
            probe<<<1, 1>>>(landed);
            tried = cudaGetLastError();
        }
        if(tried == cudaSuccess){
            tried = cudaMemcpy(&found, landed, sizeof(int), cudaMemcpyDeviceToHost);
        }
        cudaFree(landed);
        bool runs = tried == cudaSuccess && found == 1;
        running += runs;
        char entry[512];
        std::snprintf(entry, sizeof(entry), "%s{\"index\":%d,\"name\":%s,\"sm\":%d,\"memoryMB\":%zu,\"runs\":%s%s%s}", device ? "," : "", device, quoted(properties.name).c_str(),
                      properties.major*10 + properties.minor, properties.totalGlobalMem >> 20, runs ? "true" : "false", runs ? "" : ",\"error\":",
                      runs ? "" : quoted(cudaGetErrorString(tried)).c_str());
        gpus += entry;
    }
    bool openvdb = false;
#ifdef FLIP2_WITH_OPENVDB
    openvdb = true;
#endif
    bool nccl = false;
#ifdef FLIP2_WITH_NCCL
    nccl = true;
#endif
    std::string line = "{\"event\":\"info\",\"build\":" + quoted(FLIP2_BUILD_ID) + ",\"architectures\":" + quoted(FLIP2_ARCHITECTURES) + ",\"cudaRuntime\":" +
                       std::to_string(runtime) + ",\"driver\":" + std::to_string(driver) + ",\"gpus\":[" + gpus + "],\"compression\":" + (cacheCompresses() ? "true" : "false") +
                       ",\"openvdb\":" + (openvdb ? "true" : "false") + ",\"nccl\":" + (nccl ? "true" : "false") +
                       (counted == cudaSuccess ? std::string() : ",\"error\":" + quoted(cudaGetErrorString(counted))) + "}";
    event(line);
    return running > 0 ? 0 : 1;
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

//whether a cache record (cache.json, commit.json, ckpt.json) has a version newer than this flip2 reads, which it then mustn't (docs/cache-format.md)
static bool newerThanKnown(const Json& record){
    const Json* version = record.find("version");
    return version != nullptr && version->number > 1;
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
        if(newerThanKnown(cache)){
            return failure(directory + "/cache.json: version " + std::to_string((int)cache.find("version")->number) + ", newer than this flip2 reads (1)");
        }
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
            if(newerThanKnown(commit)){
                problem((frame / "commit.json").string() + ": a version newer than this flip2 reads");
                continue;
            }
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

static std::string frameName(int frame){    //as the cache's folders and the exports name it
    char name[16];
    std::snprintf(name, sizeof(name), "%04d", frame);
    return name;
}

//makes files of each of the cache's committed frames, for export and mesh: for every frame from first to last (or the bake's last, if last is -1) one of
//whose files, targets(frame), is missing or older than the frame's commit record (a resume worked it out again), or every one with overwrite, reads its
//shards, each checked against its commit record's XXH64, and has make(frame, shards, targets, why) write them (saying so as an event) or say why not. follow
//waits for frames still to come, and for the cache itself if the bake hasn't started it, so a DCC can show a bake's frames as it commits them; it stops
//when they've all been made. Ends with the event finished, counting the files written as written
static int eachCommittedFrame(const std::string& directory, int first, int last, bool overwrite, bool follow, const char* finished, const char* written,
                              const std::function<std::vector<std::string>(int)>& targets,
                              const std::function<bool(int, const std::vector<ShardData>&, const std::vector<std::string>&, std::string&)>& make){
    auto start = std::chrono::steady_clock::now();
    size_t made = 0, problems = 0;
    std::vector<bool> done;     //per frame in range: made, or found already made
    for(;;){
        std::string text;
        if(!readFile(directory + "/cache.json", text)){
            if(follow){     //the bake hasn't started writing its cache yet
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
                continue;
            }
            return failure(directory + ": no cache.json, so no cache here");
        }
        int bakeFrames = -1;
        try{
            Json cache = JsonReader(text, directory + "/cache.json").document();
            if(newerThanKnown(cache)){
                return failure(directory + "/cache.json: a version newer than this flip2 reads");
            }
            if(const Json* frames = cache.find("frames")){
                bakeFrames = (int)frames->number;
            }
        }
        catch(const std::exception& error){
            return failure(error.what());
        }
        int end = last >= 0 ? last : bakeFrames;
        if(end < first){
            return failure(directory + ": no frames from " + std::to_string(first) + (last >= 0 ? " to " + std::to_string(last) : ""));
        }
        done.resize(end - first + 1, false);
        size_t waiting = 0;
        for(int frame = first; frame <= end; ++frame){
            if(done[frame - first]){
                continue;
            }
            std::string name = frameName(frame);
            std::string record = directory + "/frames/" + name + "/commit.json";
            std::vector<std::string> files = targets(frame);
            std::error_code missing;
            auto committedAt = std::filesystem::last_write_time(record, missing);
            if(missing){
                ++waiting;      //not committed yet
                continue;
            }
            bool current = !overwrite;
            for(const std::string& file : files){
                std::error_code absent;
                auto madeAt = std::filesystem::last_write_time(file, absent);
                current = current && !absent && madeAt >= committedAt;
            }
            if(current){
                done[frame - first] = true;
                continue;
            }
            std::string why;
            std::vector<ShardData> shards;
            try{
                std::string contents;
                if(!readFile(record, contents)){
                    why = record + ": unreadable";
                }
                Json commit = why.empty() ? JsonReader(contents, record).document() : Json();
                const Json* list = commit.find("shards");
                if(why.empty() && (newerThanKnown(commit) || list == nullptr)){
                    why = record + (list == nullptr ? ": no shards" : ": a version newer than this flip2 reads");
                }
                for(size_t index = 0; why.empty() && list != nullptr && index < list->items.size(); ++index){
                    const Json* shardFile = list->items[index].find("file");
                    const Json* hash = list->items[index].find("xxh64");
                    shards.emplace_back();
                    if(shardFile == nullptr || hash == nullptr || !readShard(directory + "/frames/" + name + "/" + shardFile->text, hash->text, shards.back(), why)){
                        why = why.empty() ? record + ": a shard without its file and xxh64" : why;
                    }
                }
            }
            catch(const std::exception& error){
                why = error.what();
            }
            for(const std::string& file : files){     //there again, if a bake starting over has cleared its exports away since
                std::error_code ignored;
                std::filesystem::create_directories(std::filesystem::path(file).parent_path(), ignored);
            }
            if(why.empty() && make(frame, shards, files, why)){
                ++made;
                done[frame - first] = true;
            }
            else{
                ++problems;
                done[frame - first] = true;     //not tried again
                event("{\"event\":\"error\",\"message\":" + quoted("frame " + std::to_string(frame) + ": " + why) + "}");
                std::cerr<<"frame "<<frame<<": "<<why<<"\n";
            }
        }
        if(!follow || waiting == 0){
            char line[256];
            std::snprintf(line, sizeof(line), "{\"event\":\"%s\",\"frames\":%zu,\"%s\":%zu,\"unfinished\":%zu,\"problems\":%zu,\"seconds\":%.3f}", finished,
                          done.size(), written, made, waiting, problems, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
            event(line);
            return problems > 0 ? 1 : 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
}

static bool makeDirectory(const std::string& path){
    std::error_code made;
    std::filesystem::create_directories(path, made);
    if(made){
        failure(path + ": " + made.message());
        return false;
    }
    return true;
}

//flip2 export: the cache's committed frames, as files DCCs load natively. For now Houdini's: OUT/particles.NNNN.bgeo.sc (OUT is DIR/export/houdini
//unless --out says; .bgeo, uncompressed, with format "bgeo"), each a frame's every shard in one point cloud, in rank order (the order one partition would
//have held them), with P and v, and id and age if the cache's frames carry them (bgeo.cu). Frame NNNN is the cache's: frame 0 is the start, at time
//0. Frames are picked, redone and followed as eachCommittedFrame says
static int exportCache(const std::string& directory, int first, int last, std::string out, bool overwrite, bool follow, const std::string& format){
    if(out.empty()){
        out = directory + "/export/houdini";
    }
    if(!makeDirectory(out)){
        return 1;
    }
    return eachCommittedFrame(directory, first, last, overwrite, follow, "exportDone", "exported", [&](int frame){
        return std::vector<std::string>{out + "/particles." + frameName(frame) + "." + format};
    }, [&](int frame, const std::vector<ShardData>& shards, const std::vector<std::string>& targets, std::string& why){
        const std::string& target = targets[0];
        std::vector<const ShardData*> pieces;
        unsigned long long particles = 0;
        for(const ShardData& shard : shards){
            pieces.push_back(&shard);
            particles += shard.particles;
        }
        if(!writeParticlesBgeo(target, pieces, std::string("flip2 ") + FLIP2_BUILD_ID, why)){
            return false;
        }
        event("{\"event\":\"exported\",\"frame\":" + std::to_string(frame) + ",\"particles\":" + std::to_string(particles) + ",\"file\":" + ::quoted(target) + "}");
        return true;
    });
}

//flip2 mesh: the liquid's surface in each of the cache's committed frames, worked out on the GPU (surface.cu), as OUT/surface.NNNN.bgeo.sc (OUT and
//format as for export): closed quads facing outwards, their points carrying the liquid's velocity v for motion blur. With fields, also OUT/fields.NNNN.vdb, the
//liquid's fields as Houdini's FLIP outputs them for its whitewater: "surface", a level set, and "vel", its velocity (surface.hu, volumes.hpp); with
//surface false, only them. The particle separation is the bake's (half its voxel size) unless settings give one. Frames are picked, redone and
//followed as eachCommittedFrame says
static int meshCache(const std::string& directory, int first, int last, std::string out, bool overwrite, bool follow, const std::string& format,
                     SurfaceSettings settings, bool surface, bool fields, const FieldSettings& fieldSettings){
    SurfaceSettings checked = settings;
    if(checked.separation <= 0.0f){
        checked.separation = 1.0f;      //the bake's, which can only be checked once there's a cache
    }
    std::string problem = checked.problem();
    if(problem.empty() && fields){
        problem = fieldSettings.problem();
    }
    if(!problem.empty()){
        return failure("mesh: " + problem);
    }
    if(!surface && !fields){
        return failure("mesh: neither the surface nor the fields asked for");
    }
    if(out.empty()){
        out = directory + "/export/houdini";
    }
    if(!makeDirectory(out)){
        return 1;
    }
    if(const char* device = std::getenv("FLIP2_DEVICE")){
        cudaSetDevice(std::atoi(device));
    }
    std::unique_ptr<SurfaceMesher> mesher;      //made at the first frame, on the GPU: so not at all, if there's nothing to mesh
    std::vector<float> positions, velocities;
    SurfaceMesh mesh;
    FluidFields fluidFields;
    return eachCommittedFrame(directory, first, last, overwrite, follow, "meshDone", "meshed", [&](int frame){
        std::vector<std::string> files;
        if(surface){
            files.push_back(out + "/surface." + frameName(frame) + "." + format);
        }
        if(fields){
            files.push_back(out + "/fields." + frameName(frame) + ".vdb");
        }
        return files;
    }, [&](int frame, const std::vector<ShardData>& shards, const std::vector<std::string>& targets, std::string& why){
        if(settings.separation <= 0.0f){
            std::string text;
            const Json* voxelSize = nullptr;
            Json cache;
            try{
                if(readFile(directory + "/cache.json", text)){
                    cache = JsonReader(text, directory + "/cache.json").document();
                    voxelSize = cache.find("voxelSize");
                }
            }
            catch(const std::exception& error){
                why = error.what();
                return false;
            }
            if(voxelSize == nullptr || !(voxelSize->number > 0.0)){
                why = directory + "/cache.json: no voxelSize, so no particle separation to mesh with";
                return false;
            }
            settings.separation = (float)(voxelSize->number/2.0);
        }
        unsigned long long particles = 0;
        for(const ShardData& shard : shards){
            const ShardData::Attribute* p = shard.find("P");
            const ShardData::Attribute* v = shard.find("v");
            for(const ShardData::Attribute* attribute : {p, v}){
                if(attribute == nullptr || attribute->type != 1 || attribute->components != 3 || attribute->planes.size() != 3*sizeof(float)*shard.particles){
                    why = "a shard without float32 P and v";
                    return false;
                }
            }
            particles += shard.particles;
        }
        positions.resize(3*particles);
        velocities.resize(3*particles);
        float low[3] = {INFINITY, INFINITY, INFINITY}, high[3] = {-INFINITY, -INFINITY, -INFINITY};
        unsigned long long offset = 0;
        for(const ShardData& shard : shards){   //every shard's planes into one x, y and z plane each
            const float* p = (const float*)shard.find("P")->planes.data();
            const float* v = (const float*)shard.find("v")->planes.data();
            for(int axis = 0; axis < 3; ++axis){
                std::copy(p + axis*shard.particles, p + (axis + 1)*shard.particles, positions.begin() + axis*particles + offset);
                std::copy(v + axis*shard.particles, v + (axis + 1)*shard.particles, velocities.begin() + axis*particles + offset);
                for(uint64_t particle = 0; particle < shard.particles; ++particle){
                    low[axis] = std::min(low[axis], p[axis*shard.particles + particle]);
                    high[axis] = std::max(high[axis], p[axis*shard.particles + particle]);
                }
            }
            offset += shard.particles;
        }
        try{
            if(!mesher){
                mesher.reset(new SurfaceMesher);
            }
            mesher->mesh(positions.data(), velocities.data(), particles, low, high, settings, mesh);
            if(fields){
                mesher->fields(fieldSettings, fluidFields);
            }
        }
        catch(const std::exception& error){
            why = error.what();
            return false;
        }
        char numbers[200];
        std::snprintf(numbers, sizeof(numbers), ",\"particles\":%llu,\"points\":%zu,\"faces\":%zu,\"bandCells\":%zu,\"rounds\":%d,\"seconds\":%.4f",
                      particles, mesh.points.size()/3, mesh.quads.size()/4, mesher->bandCells, mesher->rounds, mesher->seconds);
        std::string said = "{\"event\":\"meshed\",\"frame\":" + std::to_string(frame) + numbers;
        if(surface){
            if(!writeSurfaceBgeo(targets[0], mesh, std::string("flip2 ") + FLIP2_BUILD_ID, why)){
                return false;
            }
            said += ",\"file\":" + ::quoted(targets[0]);
        }
        if(fields){
            auto writing = std::chrono::steady_clock::now();
            if(!writeFluidFields(targets.back(), fluidFields, why)){
                return false;
            }
            std::snprintf(numbers, sizeof(numbers), ",\"surfaceVoxels\":%zu,\"velocityVoxels\":%zu,\"fieldSeconds\":%.4f,\"writeSeconds\":%.4f,\"fields\":",
                          fluidFields.distances.size(), fluidFields.velocities.size()/3, mesher->fieldSeconds,
                          std::chrono::duration<double>(std::chrono::steady_clock::now() - writing).count());
            said += numbers + ::quoted(targets.back());
        }
        event(said + "}");
        return true;
    });
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
    field.kind = force.kind == SceneForce::POINT ? FORCE_POINT : force.kind == SceneForce::VORTEX ? FORCE_VORTEX : force.kind == SceneForce::TURBULENCE ? FORCE_TURBULENCE :
                 force.kind == SceneForce::VOLUME ? FORCE_VOLUME : FORCE_WIND;
    field.mode = force.velocities ? VOLUME_VELOCITY : VOLUME_FORCE;
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

//SIGINT and SIGTERM: the first lets the frame being simulated finish, checkpoints it and stops (a farm pre-empting the job, or a DCC's cancel); a second
//stops at once, leaving the cache as its last commits and checkpoint left it
static volatile std::sig_atomic_t cancelRequested = 0;

static void onCancel(int signal){
    if(cancelRequested){
        std::signal(signal, SIG_DFL);
        std::raise(signal);
        return;
    }
    cancelRequested = 1;
}

//a checkpoint's record (cacheWriter.hu)
struct Checkpoint{
    int frame = -1;
    double time = 0.0;
    unsigned long long substep = 0;
    bool apic = false;
    unsigned long long particles = 0;
    bool hasNextId = false;     //one from before particles had ids has none
    unsigned long long nextId = 0;
    std::string sceneHash;
    std::vector<uint> planes;   //its ranks' first node planes, and the last one's end
    struct State{
        std::string file;
        int rank;
        std::string xxh64;
    };
    std::vector<State> states;  //in rank order
};

//the newest checkpoint in directory whose record is down; frame -1 if there's none
static bool newestCheckpoint(const std::string& directory, Checkpoint& out, std::string& why){
    std::vector<int> frames;
    std::error_code listed;
    for(const auto& entry : std::filesystem::directory_iterator(directory + "/checkpoints", listed)){
        std::string name = entry.path().filename().string();
        if(entry.is_directory() && !name.empty() && std::all_of(name.begin(), name.end(), ::isdigit) && std::filesystem::exists(entry.path() / "ckpt.json")){
            frames.push_back(std::atoi(name.c_str()));
        }
    }
    out = Checkpoint();
    if(frames.empty()){
        return true;
    }
    char name[16];
    std::snprintf(name, sizeof(name), "%04d", *std::max_element(frames.begin(), frames.end()));
    std::string path = directory + "/checkpoints/" + name + "/ckpt.json";
    std::string text;
    if(!readFile(path, text)){
        why = path + ": can't read it";
        return false;
    }
    try{
        Json record = JsonReader(text, path).document();
        if(newerThanKnown(record)){
            throw std::runtime_error(path + ": a version newer than this flip2 reads");
        }
        auto need = [&](const char* key){
            const Json* value = record.find(key);
            if(value == nullptr){
                throw std::runtime_error(path + ": no \"" + key + "\"");
            }
            return value;
        };
        out.frame = (int)need("frame")->number;
        out.time = need("time")->number;
        out.substep = (unsigned long long)need("substep")->number;
        out.apic = need("apic")->boolean;
        out.particles = (unsigned long long)need("particles")->number;
        if(const Json* nextId = record.find("nextId")){
            out.hasNextId = true;
            out.nextId = (unsigned long long)nextId->number;
        }
        out.sceneHash = need("sceneXxh64")->text;
        for(const Json& plane : need("partitionPlanes")->items){
            out.planes.push_back((uint)plane.number);
        }
        for(const Json& state : need("states")->items){
            const Json* file = state.find("file");
            const Json* rank = state.find("rank");
            const Json* hash = state.find("xxh64");
            if(file == nullptr || rank == nullptr || hash == nullptr){
                throw std::runtime_error(path + ": a state without its file, rank and xxh64");
            }
            out.states.push_back({std::string(name) + "/" + file->text, (int)rank->number, hash->text});
        }
        if(out.planes.size() != out.states.size() + 1){
            throw std::runtime_error(path + ": its ranks' planes and states don't match");
        }
    }
    catch(const std::exception& error){
        why = error.what();
        return false;
    }
    return true;
}

//the checkpoint's particles in the ranks whose planes overlap node planes [low, high), in rank order: the order one partition would hold them in. Each
//state shard is checked against its record's XXH64 as it's read. ids and births come back empty from a checkpoint made before particles had them
static bool loadCheckpoint(const std::string& directory, const Checkpoint& checkpoint, uint low, uint high, std::vector<double>& x, std::vector<double>& y,
                           std::vector<double>& z, std::vector<float>& u, std::vector<float>& v, std::vector<float>& w,
                           std::vector<std::vector<float>>& gradients, std::vector<uint>& ids, std::vector<float>& births, std::string& why){
    gradients.assign(9, {});
    bool identified = true;
    for(const Checkpoint::State& state : checkpoint.states){
        if(checkpoint.planes[state.rank] >= high || checkpoint.planes[state.rank + 1] <= low){
            continue;
        }
        ShardData shard;
        if(!readShard(directory + "/checkpoints/" + state.file, state.xxh64, shard, why)){
            return false;
        }
        const ShardData::Attribute* positions = shard.find("P");
        const ShardData::Attribute* velocities = shard.find("v");
        const ShardData::Attribute* gradient = shard.find("c");
        if(positions == nullptr || velocities == nullptr || positions->type != 2 || velocities->type != 1 || (checkpoint.apic && (gradient == nullptr || gradient->type != 1))){
            why = directory + "/checkpoints/" + state.file + ": not a checkpoint's state";
            return false;
        }
        size_t count = shard.particles;
        const double* p = (const double*)positions->planes.data();
        const float* velocity = (const float*)velocities->planes.data();
        x.insert(x.end(), p, p + count);
        y.insert(y.end(), p + count, p + 2*count);
        z.insert(z.end(), p + 2*count, p + 3*count);
        u.insert(u.end(), velocity, velocity + count);
        v.insert(v.end(), velocity + count, velocity + 2*count);
        w.insert(w.end(), velocity + 2*count, velocity + 3*count);
        if(checkpoint.apic){
            const float* c = (const float*)gradient->planes.data();
            for(int k = 0; k < 9; ++k){
                gradients[k].insert(gradients[k].end(), c + k*count, c + (k + 1)*count);
            }
        }
        const ShardData::Attribute* id = shard.find("id");
        const ShardData::Attribute* birth = shard.find("birth");
        if(id != nullptr && birth != nullptr && id->type == 3 && birth->type == 1 && id->components == 1 && birth->components == 1){
            const uint64_t* i = (const uint64_t*)id->planes.data();     //the engine keeps their low 32 bits
            const float* b = (const float*)birth->planes.data();
            for(size_t particle = 0; particle < count; ++particle){
                ids.push_back((uint)i[particle]);
            }
            births.insert(births.end(), b, b + count);
        }
        else{
            identified = false;
        }
    }
    if(!identified){
        ids.clear();
        births.clear();
    }
    return true;
}

//a resumed bake works out everything after its checkpoint again: the frames after it go to frames.discarded/ (replacing any there), and a checkpoint
//newer than it, which can only be one a crash left unfinished, is deleted
static bool discardAfter(const std::string& directory, int frame, std::string& why){
    std::error_code failed;
    std::filesystem::create_directories(directory + "/frames.discarded", failed);
    for(const char* kind : {"frames", "checkpoints"}){
        std::vector<std::filesystem::path> later;
        std::error_code listed;
        for(const auto& entry : std::filesystem::directory_iterator(directory + "/" + kind, listed)){
            std::string name = entry.path().filename().string();
            if(entry.is_directory() && !name.empty() && std::all_of(name.begin(), name.end(), ::isdigit) && std::atoi(name.c_str()) > frame){
                later.push_back(entry.path());
            }
        }
        for(const auto& path : later){
            if(std::string(kind) == "frames"){
                std::filesystem::path into = directory + "/frames.discarded/" + path.filename().string();
                std::filesystem::remove_all(into, failed);
                if(!failed){
                    std::filesystem::rename(path, into, failed);
                }
            }
            else{
                std::filesystem::remove_all(path, failed);
            }
            if(failed){
                why = path.string() + ": can't put it aside: " + failed.message();
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char** argv){
    std::cout.rdbuf(std::cerr.rdbuf());     //the engine's own messages go to standard error, so standard output carries only the events
    if(argc == 3 && std::string(argv[1]) == "verify"){
        return verify(argv[2]);
    }
    if(argc == 2 && std::string(argv[1]) == "info"){
        return info();
    }
    if(argc >= 3 && (std::string(argv[1]) == "export" || std::string(argv[1]) == "mesh")){
        bool meshing = std::string(argv[1]) == "mesh";
        int first = 0, last = -1;
        std::string out;
        bool overwrite = false, follow = false;
        SurfaceSettings surface;
        FieldSettings fieldSettings;
        bool meshSurface = true, fields = false;
        std::string format = "bgeo.sc";
        for(int arg = 3; arg < argc; ++arg){
            std::string option = argv[arg];
            if(option == "--format" && arg + 1 < argc){
                format = argv[++arg];
                if(format != "bgeo" && format != "bgeo.sc"){
                    return usage();
                }
            }
            else if(meshing && option == "--fields"){
                fields = true;
            }
            else if(meshing && option == "--no-surface"){
                meshSurface = false;
            }
            else if(meshing && arg + 1 < argc && (option == "--separation" || option == "--voxel-scale" || option == "--influence-scale" || option == "--radius-scale" ||
                                                  option == "--smoothing" || option == "--field-voxel-scale")){
                char* end = nullptr;
                std::string given = argv[++arg];
                double value = std::strtod(given.c_str(), &end);
                if(end == given.c_str() || *end != '\0'){
                    return usage();
                }
                if(option == "--separation"){
                    surface.separation = (float)value;
                    if(!(surface.separation > 0.0f)){
                        return failure("mesh: the particle separation has to be positive");
                    }
                }
                else if(option == "--voxel-scale"){
                    surface.voxelScale = (float)value;
                }
                else if(option == "--influence-scale"){
                    surface.influenceScale = (float)value;
                }
                else if(option == "--radius-scale"){
                    surface.radiusScale = (float)value;
                }
                else if(option == "--field-voxel-scale"){
                    fieldSettings.voxelScale = (float)value;
                }
                else{
                    surface.smoothing = (int)value;
                    if(surface.smoothing != value){
                        return usage();
                    }
                }
            }
            else if(option == "--frames" && arg + 1 < argc){
                std::string range = argv[++arg];
                size_t dash = range.find('-', 1);
                first = std::atoi(range.substr(0, dash).c_str());
                last = dash == std::string::npos ? first : std::atoi(range.substr(dash + 1).c_str());
            }
            else if(option == "--out" && arg + 1 < argc){
                out = argv[++arg];
            }
            else if(option == "--overwrite"){
                overwrite = true;
            }
            else if(option == "--follow"){
                follow = true;
            }
            else{
                return usage();
            }
        }
        return meshing ? meshCache(argv[2], first, last, out, overwrite, follow, format, surface, meshSurface, fields, fieldSettings)
                       : exportCache(argv[2], first, last, out, overwrite, follow, format);
    }
    if(argc < 3 || (std::string(argv[1]) != "bake" && std::string(argv[1]) != "resume")){
        return usage();
    }
    bool resuming = std::string(argv[1]) == "resume";
    std::string scenePath = resuming ? "" : argv[2];
    std::string outputDirectory = resuming ? argv[2] : "";
    int frames = -1;
    bool overwrite = false;
    bool force = false;
    for(int arg = 3; arg < argc; ++arg){
        std::string option = argv[arg];
        if(option == "--frames" && arg + 1 < argc){
            frames = std::atoi(argv[++arg]);
        }
        else if(!resuming && option == "--out" && arg + 1 < argc){
            outputDirectory = argv[++arg];
        }
        else if(!resuming && option == "--overwrite"){
            overwrite = true;
        }
        else if(resuming && option == "--force"){
            force = true;
        }
        else{
            return usage();
        }
    }
    Checkpoint checkpoint;
    std::string cachedSceneHash;
    int cachedFrames = -1;      //the frames the bake was started with
    if(resuming){   //the scene the cache says made it, and where to carry on from
        std::string text;
        if(!readFile(outputDirectory + "/cache.json", text)){
            return failure(outputDirectory + ": no cache.json, so no bake to resume here");
        }
        try{
            Json cache = JsonReader(text, outputDirectory + "/cache.json").document();
            if(newerThanKnown(cache)){
                return failure(outputDirectory + "/cache.json: a version newer than this flip2 reads");
            }
            const Json* scene = cache.find("scene");
            const Json* path = scene ? scene->find("path") : nullptr;
            const Json* hash = scene ? scene->find("xxh64") : nullptr;
            if(path == nullptr || hash == nullptr){
                return failure(outputDirectory + "/cache.json: it doesn't say which scene made it");
            }
            scenePath = path->text;
            cachedSceneHash = hash->text;
            if(const Json* bakeFrames = cache.find("frames")){
                cachedFrames = (int)bakeFrames->number;
            }
        }
        catch(const std::exception& error){
            return failure(error.what());
        }
        std::string why;
        if(!newestCheckpoint(outputDirectory, checkpoint, why)){
            return failure(why);
        }
        if(checkpoint.frame < 0){
            return failure(outputDirectory + ": no checkpoint to resume from");
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
    else if(resuming && cachedFrames >= 0){     //the same bake, to where it was going
        scene.frames = cachedFrames;
    }
    if(!outputDirectory.empty()){
        scene.outputDirectory = outputDirectory;
    }
    if(resuming){
        scene.writeCache = true;    //into the cache it resumes
    }
    if(const char* partitions = std::getenv("FLIP2_PARTITIONS")){
        scene.partitions = std::max(1, std::atoi(partitions));
    }
    std::string sceneText;
    if(scene.writeCache && !readFile(scenePath, sceneText)){
        return failure(scenePath + ": can't read it again to hash it for the cache");
    }
    Xxh64 sceneHash;
    sceneHash.update(sceneText.data(), sceneText.size());
    if(resuming){
        if(!force && (hex(sceneHash.digest()) != cachedSceneHash || checkpoint.sceneHash != cachedSceneHash)){
            return failure(scenePath + " has changed since the bake in " + outputDirectory + " started (its XXH64 is " + hex(sceneHash.digest()) + ", the cache's " +
                           cachedSceneHash + "), so resuming wouldn't carry on the same bake: resume with --force to carry on with the scene as it is now");
        }
        if(scene.frames <= checkpoint.frame){
            return failure(outputDirectory + ": its newest checkpoint is at frame " + std::to_string(checkpoint.frame) + ", and the scene goes up to frame " +
                           std::to_string(scene.frames) + ": resume with --frames to go further");
        }
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
    int myRank = alone ? 0 : (std::getenv("FLIP2_RANK") ? std::atoi(std::getenv("FLIP2_RANK")) : -1);
    bool rankZero = myRank == 0;
    //rank 0 clears the way before it joins the others, who wait for it to before they write anything: a cache already here is only replaced when asked,
    //and a resumed one puts aside what came after its checkpoint
    if(scene.writeCache && rankZero && resuming){
        std::string why;
        if(!discardAfter(scene.outputDirectory, checkpoint.frame, why)){
            return failure(why);
        }
    }
    else if(scene.writeCache && rankZero && std::filesystem::exists(scene.outputDirectory + "/cache.json")){
        if(!overwrite){
            return failure(scene.outputDirectory + " holds a cache already: bake with --overwrite to replace it, give another --out, or flip2 resume it");
        }
        std::error_code removed;
        for(const char* part : {"frames", "checkpoints", "frames.discarded", "export"}){    //and what was exported and meshed from it
            if(!removed){
                std::filesystem::remove_all(scene.outputDirectory + "/" + part, removed);
            }
        }
        if(!removed){
            std::filesystem::remove(scene.outputDirectory + "/cache.json", removed);
        }
        if(removed){
            return failure(scene.outputDirectory + ": can't delete the cache there: " + removed.message());
        }
    }

    std::vector<double> x, y, z;

    size_t liquidSeeds = 0;     //with air: how many of the seeded particles are the liquid's, which come first (seedParticles)
    std::vector<float> u, v, w;
    std::vector<std::vector<float>> gradients;
    std::vector<uint> ids;
    std::vector<float> births;
    if(resuming){   //every rank's particles here, or across processes, those of the checkpoint's ranks whose planes overlap this rank's
        uint low = 0, high = scene.nodes[2];
        if(!alone){
            std::vector<uint> planes = Simulation::planesFor(scene.nodes[2], worldSize);
            if(planes.empty() || myRank < 0 || myRank >= worldSize){
                return failure("rank " + std::to_string(myRank) + " of " + std::to_string(worldSize) + " can't hold its share of " + std::to_string(scene.nodes[2]) + " node planes");
            }
            low = planes[myRank];
            high = planes[myRank + 1];
        }
        std::string why;
        if(!loadCheckpoint(scene.outputDirectory, checkpoint, low, high, x, y, z, u, v, w, gradients, ids, births, why)){
            return failure(why);
        }
        if((ids.empty() && !x.empty()) || !checkpoint.hasNextId){   //made before particles had ids: they're numbered as they come, from here on
            std::cerr<<scene.outputDirectory<<": its checkpoint has no particle ids, so they start again from this frame"<<(alone ? "" : ", and ranks in separate "
                       "processes will hand out the same ones")<<"\n";
            ids.clear();
            births.clear();
        }
    }
    else{
        seedParticles(scene, x, y, z, u, v, w, &liquidSeeds);
        bool meshFluids = std::any_of(scene.fluids.begin(), scene.fluids.end(), [](const SceneShape& fluid){ return fluid.kind == SceneShape::MESH; });
        if(x.empty() && scene.emitters.empty() && !meshFluids){
            return failure(scenePath + ": it has no fluid: its fluids are empty or outside the domain, and it has no emitters");
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
    if(resuming && scene.transfer == "apic" && checkpoint.apic){    //otherwise, as at the start, they have none
        simulation->setAffine(gradients);
    }
    if(resuming && !ids.empty()){
        simulation->setIdentities(ids, births, checkpoint.nextId);
    }
    simulation->setDensityCorrectionTime(scene.densityCorrectionTime);
    simulation->setCfl(scene.cfl);
    simulation->setPressureSolver(scene.pressureSolver == "sor" ? PressureSolver::sor : scene.pressureSolver == "cg" ? PressureSolver::cg :
                                  scene.pressureSolver == "jacobi" ? PressureSolver::jacobi : PressureSolver::multigrid);
    simulation->setRungeKutta3(scene.advection == "rk3");
    simulation->setDotProductSums(scene.dotProducts == "blocks" ? DotProductSums::perBlock : DotProductSums::exact);
    simulation->setGravity(make_float3((float)scene.gravity[0], (float)scene.gravity[1], (float)scene.gravity[2]));
    simulation->setViscosity(scene.viscosity / scene.density);
    simulation->setSurfaceTension(scene.surfaceTension / scene.density);
    simulation->setContactAngle(scene.contactAngle);
    simulation->setViscousCfl(scene.viscousCfl);
    simulation->setFreeSurface(scene.freeSurface == "sharp" ? FreeSurface::sharp : FreeSurface::footprint);
    if(scene.air){      //a second fluid (TwoPhase, particles.hu): the particles seeded after the liquid's, and any made later, are air
        TwoPhase twoPhase;
        twoPhase.on = true;
        twoPhase.densityRatio = (float)(scene.density / scene.airDensity);
        twoPhase.faceDensity = scene.airFaceDensity == "phaseField" ? FaceDensity::phaseField : scene.airFaceDensity == "levelSet" ? FaceDensity::levelSet :
                               scene.airFaceDensity == "synthetic" ? FaceDensity::synthetic : FaceDensity::fractions;
        twoPhase.airFlipRatio = (float)(scene.airFlipRatio < 0.0 ? scene.flipRatio : scene.airFlipRatio);
        twoPhase.band = scene.airBand;
        twoPhase.escapes = scene.airEscaped && twoPhase.densityRatio > 1.0f;   //with no difference in density there's no telling the liquid's weight on a face
        twoPhase.dropletRadius = (float)scene.airDropletRadius;
        twoPhase.airViscosity = (float)scene.airViscosity;
        twoPhase.syntheticShape = scene.airSyntheticShape == "ball" ? 1 : scene.airSyntheticShape == "balls" ? 2 : 0;
        twoPhase.syntheticCentre = make_float3((float)scene.airSyntheticCentre[0], (float)scene.airSyntheticCentre[1], (float)scene.airSyntheticCentre[2]);
        twoPhase.syntheticRadius = (float)scene.airSyntheticRadius;
        twoPhase.syntheticSpacing = (float)scene.airSyntheticSpacing;
        if(twoPhase.particles() && !resuming){  //the seeds after the liquid's are air; a checkpoint's ids say which are which themselves
            simulation->markAirParticles(liquidSeeds);
        }
        simulation->setTwoPhase(twoPhase);
        if(printsEvents){
            std::cerr<<"Two phases: density ratio "<<twoPhase.densityRatio<<", face densities by "<<scene.airFaceDensity<<"\n";
        }
    }
    std::vector<ForceField> fields;
    std::vector<std::shared_ptr<const SceneField>> volumes;     //each volume force's vectors, which every partition puts on its own GPU
    for(const SceneForce& force : scene.forces){
        fields.push_back(toForceField(force));
        volumes.push_back(force.field);
        if(force.kind == SceneForce::VOLUME && force.field && printsEvents){
            const SceneField& f = *force.field;
            std::cerr<<"Force volume: "<<f.velocities.size()/(3*512)<<" bricks of "<<f.spacing<<" m samples, from ("<<f.low[0]<<", "<<f.low[1]<<", "<<f.low[2]<<") to ("
                     <<f.high[0]<<", "<<f.high[1]<<", "<<f.high[2]<<"), its longest vector "<<f.fastest<<(force.velocities ? " m/s" : " m/s^2")<<" ("
                     <<std::round((f.velocities.size() + f.pool.size())*sizeof(float)/104857.6)/10.0<<" MB)\n";
        }
    }
    simulation->setForceFields(fields, volumes);
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
        description.sceneHash = hex(sceneHash.digest());
        description.fps = scene.fps;
        description.frames = scene.frames;
        description.compression = scene.compression;
        description.ids = scene.writeIds;
        description.ages = scene.writeAges;
        description.keepCheckpoints = scene.keepCheckpoints;
        simulation->startCache(scene.outputDirectory, description, resuming ? checkpoint.frame : -1, [](const char* what, int frame){
            event("{\"event\":\"" + std::string(what) + "\",\"frame\":" + std::to_string(frame) + "}");
        });
    }

    std::string directory = scene.outputDirectory + "/";
    std::string diagnostics = scene.diagnostics.empty() ? "" : directory + scene.diagnostics;
    auto start = std::chrono::steady_clock::now();
    char line[512];
    int firstFrame = 1;
    if(resuming){
        if(!diagnostics.empty()){
            simulation->continueDiagnostics(diagnostics, checkpoint.frame);
        }
        simulation->resume(checkpoint.time, checkpoint.substep);
        std::snprintf(line, sizeof(line), "{\"event\":\"start\",\"particles\":%llu,\"frames\":%d,\"nodes\":[%u,%u,%u],\"voxelSize\":%.9g,\"ranks\":%d,\"resumedFrom\":%d}",
                      checkpoint.particles, scene.frames, scene.nodes[0], scene.nodes[1], scene.nodes[2], scene.voxelSize(), ranks, checkpoint.frame);
        event(line);
        char name[16];
        std::snprintf(name, sizeof(name), "%04d", checkpoint.frame);
        if(!std::filesystem::exists(directory + "frames/" + name + "/commit.json")){    //a checkpoint is only committed after its frame, but just in case
            simulation->writeCacheFrame(checkpoint.frame);
        }
        firstFrame = checkpoint.frame + 1;
    }
    else{
        simulation->initialize();
        std::snprintf(line, sizeof(line), "{\"event\":\"start\",\"particles\":%zu,\"frames\":%d,\"nodes\":[%u,%u,%u],\"voxelSize\":%.9g,\"ranks\":%d}",
                      x.size(), scene.frames, scene.nodes[0], scene.nodes[1], scene.nodes[2], scene.voxelSize(), ranks);
        event(line);
        if(!diagnostics.empty()){
            simulation->writeDiagnostics(diagnostics, 0);
            if(scene.air && scene.airFaceDensity != "synthetic"){
                simulation->writePhaseDiagnostics(directory + "phases.jsonl", 0);
            }
        }
        if(scene.writePositions){
            simulation->writePositionsToFile(directory + "0.bin");
        }
        if(scene.writeCache){
            simulation->writeCacheFrame(0);
        }
    }
    std::signal(SIGINT, onCancel);
    std::signal(SIGTERM, onCancel);
    int checkpointed = resuming ? checkpoint.frame : -1;    //the newest checkpoint's frame, which resume would carry on from
    for(int frame = firstFrame; frame <= scene.frames; ++frame){
        auto frameStart = std::chrono::steady_clock::now();
        simulation->solveFrame(scene.fps);
        std::string unsolved = simulation->solveError();
        if(!unsolved.empty()){  //a pressure solve that didn't converge: no pressure, so no frame, and no going on. Every rank finds the same
            std::string message = "frame " + std::to_string(frame) + ": " + unsolved;
            if(scene.writeCache){
                std::string error = simulation->finishCache();  //the frames before this one committed
                if(!error.empty()){
                    return failure(error);
                }
                message += ". The cache holds the frames before it; ";
                message += checkpointed >= 0 ? "flip2 resume carries on from its checkpoint at frame " + std::to_string(checkpointed) +
                                               " (with --force once the scene is changed, say to another solver.pressureSolver)"
                                             : "it has no checkpoint to resume from yet";
            }
            return failure(message);
        }
        if(!diagnostics.empty()){
            simulation->writeDiagnostics(diagnostics, frame);
            if(scene.air && scene.airFaceDensity != "synthetic"){
                simulation->writePhaseDiagnostics(directory + "phases.jsonl", frame);
            }
        }
        if(scene.writePositions){
            simulation->writePositionsToFile(directory + std::to_string(frame) + ".bin");
        }
        bool stopping = simulation->anyRank(cancelRequested != 0);     //every rank stops after the same frame
        bool last = frame == scene.frames;
        if(scene.writeCache){
            simulation->writeCacheFrame(frame);
            if(stopping || last || (scene.checkpointEvery > 0 && frame % scene.checkpointEvery == 0)){
                simulation->writeCheckpoint(frame);
                checkpointed = frame;
            }
            std::string error = simulation->cacheError();
            if(!error.empty()){
                return failure(error);
            }
        }
        std::snprintf(line, sizeof(line), "{\"event\":\"frame\",\"frame\":%d,\"seconds\":%.4f,\"particles\":%zu}", frame,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - frameStart).count(), simulation->particlesHere());
        event(line);
        if(stopping && !last){
            std::string error = simulation->finishCache();  //the frame and its checkpoint committed
            if(!error.empty()){
                return failure(error);
            }
            simulation.reset();
            std::snprintf(line, sizeof(line), "{\"event\":\"cancelled\",\"frame\":%d,\"seconds\":%.3f}", frame,
                          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
            event(line);
            return 3;
        }
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
