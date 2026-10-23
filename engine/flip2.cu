//Copyright 2023 Aberrant Behavior LLC

//flip2's command line:
//
//  flip2 bake scene.json [--frames N] [--out DIR] [--overwrite]
//  flip2 resume DIR [--frames N] [--force]
//  flip2 verify DIR
//  flip2 export DIR [--frames A-B] [--out DIR] [--overwrite] [--follow] [--format bgeo.sc|bgeo]
//  flip2 mesh DIR [--frames A-B] [--out DIR] [--overwrite] [--follow] [--format bgeo.sc|bgeo] [--separation S] [--voxel-scale S] [--influence-scale S]
//             [--radius-scale S] [--smoothing N] [--fields] [--field-voxel-scale S] [--no-surface] [--workers N]
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
//  {"event":"frame","frame":1,"seconds":0.041,"particles":1240000}    a frame simulated; with whitewater, and ,"whitewater":310000}, how much of it there
//      is, and ,"whitewaterDropped":5000} in a frame whose surface would have made that many more had whitewater.maxParticles left room
//  {"event":"committed","frame":1}                             and its cache frame on the disk, whole; from the cache's thread, so it can come later
//  {"event":"checkpoint","frame":10}                           a checkpoint on the disk, whole
//  {"event":"cancelled","frame":57,"seconds":3.1}              stopped by a signal after frame 57, which is committed and checkpointed
//  {"event":"done","frames":120,"seconds":5.2}                 every frame committed
//  {"event":"verified","frames":121,"unfinished":0,"shards":121,"particles":150040000,"bytes":1712345678,"problems":0}
//  {"event":"exported","frame":42,"particles":1240000,"file":"/shots/a/export/houdini/particles.0042.bgeo.sc"}
//      with whitewater, and ...,"whitewater":310000,"whitewaterFile":"..."; with escaping air, and ...,"bubbles":21000,"bubblesFile":"..."; with dynamic
//      bodies, and ...,"bodies":3,"bodiesFile":".../bodies.0042.bgeo.sc"}
//  {"event":"exportDone","frames":121,"exported":121,"seconds":9.8}
//  {"event":"meshed","frame":42,"particles":1240000,"points":310000,"faces":309000,"bandCells":52000,"rounds":2,"seconds":0.05,"file":"/shots/a/export/houdini/surface.0042.bgeo.sc"}
//      with --fields, and ...,"surfaceVoxels":90000,"velocityVoxels":180000,"fieldSeconds":0.01,"writeSeconds":0.05,"fields":"/shots/a/export/houdini/fields.0042.vdb"}
//  {"event":"meshDone","frames":121,"meshed":121,"seconds":12.1}
//  {"event":"info","build":"7dc95a4","architectures":"86","cudaRuntime":13040,"driver":13040,"gpus":[{"index":0,"name":"...","uuid":"GPU-7c2ab39e-...","pci":"0000:0A:00.0",
//      "sm":86,"memoryMB":24135,"runs":true}],...}
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
#include <atomic>
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
               "                  [--influence-scale S] [--radius-scale S] [--smoothing N] [--fields] [--field-voxel-scale S] [--no-surface] [--workers N]\n"
               "                  [--solids overlap|cut]\n"
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
        //its UUID as nvidia-smi writes it, which CUDA_VISIBLE_DEVICES takes too and which names the same GPU however the devices are numbered, and
        //where it sits on the PCI bus
        const unsigned char* id = (const unsigned char*)properties.uuid.bytes;
        char uuid[48];
        std::snprintf(uuid, sizeof(uuid), "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", id[0], id[1], id[2], id[3], id[4], id[5], id[6],
                      id[7], id[8], id[9], id[10], id[11], id[12], id[13], id[14], id[15]);
        char bus[32] = "";
        cudaDeviceGetPCIBusId(bus, sizeof(bus), device);
        char entry[768];
        std::snprintf(entry, sizeof(entry), "%s{\"index\":%d,\"name\":%s,\"uuid\":\"%s\",\"pci\":%s,\"sm\":%d,\"memoryMB\":%zu,\"runs\":%s%s%s}", device ? "," : "", device,
                      quoted(properties.name).c_str(), uuid, quoted(bus).c_str(), properties.major*10 + properties.minor, properties.totalGlobalMem >> 20,
                      runs ? "true" : "false", runs ? "" : ",\"error\":", runs ? "" : quoted(cudaGetErrorString(tried)).c_str());
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
            std::vector<const Json*> lists = {list};    //the particles' shards, and the whitewater's, the bodies' and the bubbles' from a bake with any
            for(const char* key : {"whitewater", "bodies", "bubbles"}){
                const Json* more = commit.find(key);
                if(more != nullptr && more->kind == Json::OBJECT && more->find("shards") != nullptr){
                    lists.push_back(more->find("shards"));
                }
            }
            for(const Json* each : lists) for(const Json& shard : each->items){
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
                particles += each == list ? (unsigned long long)count->number : 0;
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

//the memory this machine has free for a process to take, in bytes, as Linux counts it (MemAvailable); 0 where it can't be read
static unsigned long long availableMemory(){
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    unsigned long long kilobytes = 0;
    while(meminfo >> key){
        if(key == "MemAvailable:" && meminfo >> kilobytes){
            return 1024ULL*kilobytes;
        }
        meminfo.ignore(1 << 10, '\n');
    }
    return 0;
}

//the particles frame's commit record says it has, 0 if it can't be read
static unsigned long long committedParticles(const std::string& directory, int frame){
    std::string text, record = directory + "/frames/" + frameName(frame) + "/commit.json";
    try{
        if(readFile(record, text)){
            const Json* count = JsonReader(text, record).document().find("particles");
            return count != nullptr ? (unsigned long long)count->number : 0;
        }
    }
    catch(const std::exception&){
    }
    return 0;
}

//makes files of each of the cache's committed frames, for export and mesh: for every frame from first to last (or the bake's last, if last is -1) one of
//whose files, targets(frame), is missing or older than the frame's commit record (a resume worked it out again), or every one with overwrite, reads its
//shards, each checked against its commit record's XXH64, and has make(frame, shards, whitewater, record, targets, why) write them (saying so as an event)
//or say why not, record being the frame's commit record. follow
//waits for frames still to come, and for the cache itself if the bake hasn't started it, so a DCC can show a bake's frames as it commits them; it stops
//when they've all been made. Ends with the event finished, counting the files written as written. With workers over 1 it works on up to that many frames
//at once, each read and made on a thread of its own, which make must allow: fewer when the frames' memory, bytesPerParticle for each particle of the first
//of them, would come to more than half of what the machine has free (asked each time it looks for frames)
static int eachCommittedFrame(const std::string& directory, int first, int last, bool overwrite, bool follow, const char* finished, const char* written,
                              int workers, unsigned long long bytesPerParticle, const std::function<std::vector<std::string>(int)>& targets,
                              const std::function<bool(int, const std::vector<ShardData>&, const std::vector<ShardData>&, const Json&, const std::vector<std::string>&,
                                                       std::string&)>& make){
    auto start = std::chrono::steady_clock::now();
    size_t made = 0, problems = 0;
    std::vector<bool> done;     //per frame in range: made, or found already made
    std::mutex tally;           //made, problems and done, with frames made at once
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
        std::vector<int> todo;      //committed frames to make
        for(int frame = first; frame <= end; ++frame){
            if(done[frame - first]){
                continue;
            }
            std::string record = directory + "/frames/" + frameName(frame) + "/commit.json";
            std::error_code missing;
            auto committedAt = std::filesystem::last_write_time(record, missing);
            if(missing){
                ++waiting;      //not committed yet
                continue;
            }
            bool current = !overwrite;
            for(const std::string& file : targets(frame)){
                std::error_code absent;
                auto madeAt = std::filesystem::last_write_time(file, absent);
                current = current && !absent && madeAt >= committedAt;
            }
            if(current){
                done[frame - first] = true;
                continue;
            }
            todo.push_back(frame);
        }
        unsigned long long running = std::min<unsigned long long>(std::max(workers, 1), todo.size());
        if(running > 1 && bytesPerParticle > 0){
            unsigned long long free = availableMemory(), each = bytesPerParticle*committedParticles(directory, todo[0]);
            if(free > 0 && each > 0){
                running = std::max(1ULL, std::min(running, free/2/each));
            }
        }
        auto makeFrame = [&](int frame){
            std::string name = frameName(frame);
            std::string record = directory + "/frames/" + name + "/commit.json";
            std::vector<std::string> files = targets(frame);
            std::string why;
            std::vector<ShardData> shards, whitewater;   //the frame's particles, and its whitewater if the bake made any
            Json commit;
            try{
                std::string contents;
                if(!readFile(record, contents)){
                    why = record + ": unreadable";
                }
                commit = why.empty() ? JsonReader(contents, record).document() : Json();
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
                const Json* more = why.empty() ? commit.find("whitewater") : nullptr;
                const Json* moreList = more != nullptr ? more->find("shards") : nullptr;
                for(size_t index = 0; why.empty() && moreList != nullptr && index < moreList->items.size(); ++index){
                    const Json* shardFile = moreList->items[index].find("file");
                    const Json* hash = moreList->items[index].find("xxh64");
                    whitewater.emplace_back();
                    if(shardFile == nullptr || hash == nullptr || !readShard(directory + "/frames/" + name + "/" + shardFile->text, hash->text, whitewater.back(), why)){
                        why = why.empty() ? record + ": a whitewater shard without its file and xxh64" : why;
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
            bool fine = false;
            try{
                fine = why.empty() && make(frame, shards, whitewater, commit, files, why);
            }
            catch(const std::exception& error){     //on a thread of its own, nothing above it would catch it
                why = error.what();
            }
            std::lock_guard<std::mutex> lock(tally);
            done[frame - first] = true;     //made, or not tried again
            if(fine){
                ++made;
            }
            else{
                ++problems;
                event("{\"event\":\"error\",\"message\":" + quoted("frame " + std::to_string(frame) + ": " + why) + "}");
                std::cerr<<"frame "<<frame<<": "<<why<<"\n";
            }
        };
        if(running <= 1){
            for(int frame : todo){
                makeFrame(frame);
            }
        }
        else{
            std::atomic<size_t> next{0};
            std::vector<std::thread> pool;
            for(unsigned long long worker = 0; worker < running; ++worker){
                pool.emplace_back([&](){
                    for(size_t index = next++; index < todo.size(); index = next++){
                        makeFrame(todo[index]);
                    }
                });
            }
            for(std::thread& thread : pool){
                thread.join();
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

//a frame's dynamic bodies, as Houdini's RBD Simulation Points have them: where each is and how it moves from the frame's bodies shard (its commit record's
//"bodies"), its name and pivot from cache.json, and orient x y z w where the shard has w x y z. Says why not if anything's missing or malformed
static bool bodiesOf(const std::string& directory, int frame, const Json& listing, std::vector<BgeoBody>& out, std::string& why){
    const Json* shards = listing.kind == Json::OBJECT ? listing.find("shards") : nullptr;
    const Json* file = shards != nullptr && shards->items.size() == 1 ? shards->items[0].find("file") : nullptr;
    const Json* hash = shards != nullptr && shards->items.size() == 1 ? shards->items[0].find("xxh64") : nullptr;
    if(file == nullptr || hash == nullptr){
        why = "\"bodies\" has no shard with its file and xxh64";
        return false;
    }
    char name[16];
    std::snprintf(name, sizeof(name), "%04d", frame);
    ShardData shard;
    if(!readShard(directory + "/frames/" + name + "/" + file->text, hash->text, shard, why)){
        return false;
    }
    std::string text;
    Json cache;
    try{
        if(!readFile(directory + "/cache.json", text)){
            why = directory + "/cache.json: can't read it";
            return false;
        }
        cache = JsonReader(text, directory + "/cache.json").document();
    }
    catch(const std::exception& error){
        why = error.what();
        return false;
    }
    const Json* described = cache.find("bodies");
    if(described == nullptr || described->kind != Json::ARRAY || described->items.size() != shard.particles){
        why = directory + "/cache.json: its \"bodies\" don't name the frame's " + std::to_string(shard.particles);
        return false;
    }
    const ShardData::Attribute* columns[4] = {shard.find("P"), shard.find("orient"), shard.find("v"), shard.find("w")};
    const uint32_t components[4] = {3, 4, 3, 3};
    for(int column = 0; column < 4; ++column){
        if(columns[column] == nullptr || columns[column]->type != 2 || columns[column]->components != components[column]){
            why = file->text + ": no float64 P, orient, v and w";
            return false;
        }
    }
    size_t count = (size_t)shard.particles;
    auto value = [&](int column, uint32_t component, size_t body){
        return (float)((const double*)columns[column]->planes.data())[component*count + body];
    };
    for(size_t body = 0; body < count; ++body){
        BgeoBody point;
        const Json& about = described->items[body];
        const Json* named = about.find("name");
        const Json* pivot = about.find("pivot");
        point.name = named != nullptr && named->kind == Json::STRING ? named->text : "";
        for(int axis = 0; axis < 3; ++axis){
            point.position[axis] = value(0, axis, body);
            point.velocity[axis] = value(2, axis, body);
            point.angularVelocity[axis] = value(3, axis, body);
            point.pivot[axis] = pivot != nullptr && pivot->kind == Json::ARRAY && pivot->items.size() == 3 ? (float)pivot->items[axis].number : 0.0f;
            point.orientation[axis] = value(1, axis + 1, body);
        }
        point.orientation[3] = value(1, 0, body);
        out.push_back(point);
    }
    return true;
}

//flip2 export: the cache's committed frames, as files DCCs load natively. For now Houdini's: OUT/particles.NNNN.bgeo.sc (OUT is DIR/export/houdini
//unless --out says; .bgeo, uncompressed, with format "bgeo"), each a frame's every shard in one point cloud, in rank order (the order one partition would
//have held them), with P and v, and id and age if the cache's frames carry them (bgeo.cu). Frame NNNN is the cache's: frame 0 is the start, at time
//0. A bake with whitewater has each frame's beside it, OUT/whitewater.NNNN.bgeo.sc, a point cloud too, with life, radius, kind, and the pscale its
//particles are meant to be drawn at: half the spacing of what a voxel's volume of whitewater is drawn as. A bake whose air escapes has each frame's
//bubbles, OUT/bubbles.NNNN.bgeo.sc, as the whitewater's bubbles are (writeBubblesBgeo): pscale half the spacing of its particles. A bake with dynamic bodies has
//OUT/bodies.NNNN.bgeo.sc too, a point per body as RBD Bullet Solver's Simulation Points have them, which Transform Pieces moves the pieces by
//(writeBodiesBgeo). Frames are picked, redone and followed as eachCommittedFrame says
static int exportCache(const std::string& directory, int first, int last, std::string out, bool overwrite, bool follow, const std::string& format){
    if(out.empty()){
        out = directory + "/export/houdini";
    }
    if(!makeDirectory(out)){
        return 1;
    }
    return eachCommittedFrame(directory, first, last, overwrite, follow, "exportDone", "exported", 1, 0, [&](int frame){
        return std::vector<std::string>{out + "/particles." + frameName(frame) + "." + format};
    }, [&](int frame, const std::vector<ShardData>& shards, const std::vector<ShardData>& whitewater, const Json& record, const std::vector<std::string>& targets,
           std::string& why){
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
        std::string also;
        //the radius to draw whitewater or bubbles at (key: which): half the spacing of what cache.json says a voxel's volume of them is drawn as
        auto drawnAt = [&](const char* key, float& pscale){
            double voxelSize = 0.0, perVoxel = 0.0;
            std::string text;
            if(readFile(directory + "/cache.json", text)){
                try{
                    Json cache = JsonReader(text, directory + "/cache.json").document();
                    const Json* size = cache.find("voxelSize");
                    const Json* described = cache.find(key);
                    const Json* count = described != nullptr ? described->find("perVoxel") : nullptr;
                    voxelSize = size != nullptr ? size->number : 0.0;
                    perVoxel = count != nullptr ? count->number : 0.0;
                }
                catch(const std::exception&){
                }
            }
            if(!(voxelSize > 0.0) || !(perVoxel > 0.0)){
                why = directory + "/cache.json: no voxelSize and " + key + ".perVoxel to size the " + key + " by";
                return false;
            }
            pscale = (float)(0.5*voxelSize / std::cbrt(perVoxel));
            return true;
        };
        if(!whitewater.empty()){
            float pscale;
            if(!drawnAt("whitewater", pscale)){
                return false;
            }
            std::vector<const ShardData*> more;
            unsigned long long points = 0;
            for(const ShardData& shard : whitewater){
                more.push_back(&shard);
                points += shard.particles;
            }
            std::string file = out + "/whitewater." + frameName(frame) + "." + format;
            if(!writeWhitewaterBgeo(file, more, pscale, std::string("flip2 ") + FLIP2_BUILD_ID, why)){
                return false;
            }
            also = ",\"whitewater\":" + std::to_string(points) + ",\"whitewaterFile\":" + ::quoted(file);
        }
        if(const Json* bubbles = record.find("bubbles")){   //a bake whose air escapes: its bubbles, which its shards hold as the commit record lists them
            const Json* list = bubbles->find("shards");
            std::vector<ShardData> read(list != nullptr ? list->items.size() : 0);
            std::vector<const ShardData*> more;
            unsigned long long points = 0;
            for(size_t index = 0; index < read.size(); ++index){
                const Json* shardFile = list->items[index].find("file");
                const Json* hash = list->items[index].find("xxh64");
                if(shardFile == nullptr || hash == nullptr || !readShard(directory + "/frames/" + frameName(frame) + "/" + shardFile->text, hash->text, read[index], why)){
                    why = why.empty() ? frameName(frame) + ": a bubbles shard without its file and xxh64" : why;
                    return false;
                }
                more.push_back(&read[index]);
                points += read[index].particles;
            }
            float pscale;
            if(!drawnAt("bubbles", pscale)){
                return false;
            }
            std::string file = out + "/bubbles." + frameName(frame) + "." + format;
            if(!writeBubblesBgeo(file, more, pscale, std::string("flip2 ") + FLIP2_BUILD_ID, why)){
                return false;
            }
            also += ",\"bubbles\":" + std::to_string(points) + ",\"bubblesFile\":" + ::quoted(file);
        }
        if(const Json* bodies = record.find("bodies")){     //a bake with dynamic bodies: each one where it is, as Houdini's RBD Simulation Points
            std::vector<BgeoBody> points;
            if(!bodiesOf(directory, frame, *bodies, points, why)){
                why = frameName(frame) + ": " + why;
                return false;
            }
            std::string file = out + "/bodies." + frameName(frame) + "." + format;
            if(!writeBodiesBgeo(file, points, std::string("flip2 ") + FLIP2_BUILD_ID, why)){
                return false;
            }
            also += ",\"bodies\":" + std::to_string(points.size()) + ",\"bodiesFile\":" + ::quoted(file);
        }
        event("{\"event\":\"exported\",\"frame\":" + std::to_string(frame) + ",\"particles\":" + std::to_string(particles) + ",\"file\":" + ::quoted(target) + also + "}");
        return true;
    });
}

//a dynamic body's frame where a frame's record has it (bodiesOf), as writeBodyFrames makes it in a bake: the world to its own space, (R^T, pivot - R^T x),
//its reach (the box round where its own bounds are, 3 voxels wider every way) and its velocity at a world point, v + w x (point - x)
static BodyFrame bodyFrameOf(const BgeoBody& body, float3 objectLow, float3 objectHigh, double voxelSize){
    double x = body.orientation[0], y = body.orientation[1], z = body.orientation[2], w = body.orientation[3];
    double r[9] = {1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
                   2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
                   2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)};    //row-major
    BodyFrame frame;
    for(int row = 0; row < 3; ++row){
        double moved = 0.0;
        for(int column = 0; column < 3; ++column){
            frame.worldToObject[4*row + column] = (float)r[3*column + row];
            moved += r[3*column + row]*body.position[column];
        }
        frame.worldToObject[4*row + 3] = (float)(body.pivot[row] - moved);
    }
    double spin[9] = {0.0, -body.angularVelocity[2], body.angularVelocity[1],
                      body.angularVelocity[2], 0.0, -body.angularVelocity[0],
                      -body.angularVelocity[1], body.angularVelocity[0], 0.0};
    for(int row = 0; row < 3; ++row){
        double moved = 0.0;
        for(int column = 0; column < 3; ++column){
            frame.velocity[4*row + column] = (float)spin[3*row + column];
            moved += spin[3*row + column]*body.position[column];
        }
        frame.velocity[4*row + 3] = (float)(body.velocity[row] - moved);
    }
    double low[3] = {INFINITY, INFINITY, INFINITY}, high[3] = {-INFINITY, -INFINITY, -INFINITY};
    for(int corner = 0; corner < 8; ++corner){
        double local[3] = {(corner & 1 ? objectHigh.x : objectLow.x) - body.pivot[0], (corner & 2 ? objectHigh.y : objectLow.y) - body.pivot[1],
                           (corner & 4 ? objectHigh.z : objectLow.z) - body.pivot[2]};
        for(int row = 0; row < 3; ++row){
            double world = r[3*row]*local[0] + r[3*row + 1]*local[1] + r[3*row + 2]*local[2] + body.position[row];
            low[row] = std::min(low[row], world);
            high[row] = std::max(high[row], world);
        }
    }
    double reach = 3.0*voxelSize;
    frame.reachLow = make_float3((float)(low[0] - reach), (float)(low[1] - reach), (float)(low[2] - reach));
    frame.reachHigh = make_float3((float)(high[0] + reach), (float)(high[1] + reach), (float)(high[2] + reach));
    return frame;
}

//flip2 mesh: the liquid's surface in each of the cache's committed frames, worked out on the GPU (surface.cu), as OUT/surface.NNNN.bgeo.sc (OUT and
//format as for export): closed quads facing outwards, their points carrying the liquid's velocity v for motion blur, flat against the domain's walls
//where the liquid reaches them (MeshWalls), and carried on into the scene's obstacles where it touches them, or with overlap false stopped at their
//surfaces (MeshSolids): the obstacles as the bake had them, built from the scene the cache names, at the frame's time, dynamic bodies where the frame's
//records put them. With fields, also OUT/fields.NNNN.vdb, the
//liquid's fields as Houdini's FLIP outputs them for its whitewater: "surface", a level set, and "vel", its velocity (surface.hu, volumes.hpp); with
//surface false, only them. The particle separation is the bake's (half its voxel size) unless settings give one. Frames are picked, redone and
//followed as eachCommittedFrame says, up to workers at once: each frame's reading, gathering and writing on its thread, and its meshing on the GPU
//in turns, on one mesher, since that's the least of a frame's time and one mesher's memory is what the GPU can spare beside a bake
static const unsigned long long MESH_BYTES_PER_PARTICLE = 80;   //what a frame being meshed holds in memory: its shards (P, v, id and age, 36 bytes a
                                                                //particle), its positions and velocities gathered (24) and its mesh and file (under 20)
static int meshCache(const std::string& directory, int first, int last, std::string out, bool overwrite, bool follow, const std::string& format,
                     SurfaceSettings settings, bool surface, bool fields, const FieldSettings& fieldSettings, int workers, bool overlap){
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
    int device = -1;    //the GPU FLIP2_DEVICE names, which every thread that meshes has to choose for itself
    if(const char* chosen = std::getenv("FLIP2_DEVICE")){
        device = std::atoi(chosen);
        cudaSetDevice(device);
    }
    std::unique_ptr<SurfaceMesher> mesher;      //made at the first frame, on the GPU: so not at all, if there's nothing to mesh
    std::mutex turn;                            //the mesher's, and the separation's and the walls' while they're still to be read from the cache
    MeshWalls walls;                            //the domain's walls (cache.json's domain, less its open faces), which the surface meets flat
    bool cacheRead = false;
    std::unique_ptr<ObstacleSet> obstacles;     //the scene's obstacles, built at the first frame, and moved to each frame's time
    cudaStream_t obstacleStream = nullptr;
    BodyFrame* bodyFrames = nullptr;            //the dynamic bodies' frames, on the GPU, which the obstacles read
    std::vector<int> bodyObstacles;             //each body's obstacle among those built (cache.json's "bodies" are the scene's dynamic obstacles in order)
    double fps = 24.0, voxelSize = 0.0;
    return eachCommittedFrame(directory, first, last, overwrite, follow, "meshDone", "meshed", workers, MESH_BYTES_PER_PARTICLE, [&](int frame){
        std::vector<std::string> files;
        if(surface){
            files.push_back(out + "/surface." + frameName(frame) + "." + format);
        }
        if(fields){
            files.push_back(out + "/fields." + frameName(frame) + ".vdb");
        }
        return files;
    }, [&](int frame, const std::vector<ShardData>& shards, const std::vector<ShardData>&, const Json& commit, const std::vector<std::string>& targets, std::string& why){
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
        std::vector<float> positions(3*particles), velocities(3*particles);
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
        SurfaceMesh mesh;
        FluidFields fluidFields;
        size_t bandCells = 0;
        int rounds = 0;
        double seconds = 0.0, fieldSeconds = 0.0;
        {
            std::lock_guard<std::mutex> lock(turn);
            if(!cacheRead){
                std::string text;
                Json cache;
                try{
                    if(readFile(directory + "/cache.json", text)){
                        cache = JsonReader(text, directory + "/cache.json").document();
                    }
                }
                catch(const std::exception& error){
                    why = error.what();
                    return false;
                }
                const Json* voxel = cache.find("voxelSize");
                if(voxel == nullptr || !(voxel->number > 0.0)){
                    why = directory + "/cache.json: no voxelSize, so no particle separation to mesh with, and no obstacles";
                    return false;
                }
                voxelSize = voxel->number;
                if(settings.separation <= 0.0f){
                    settings.separation = (float)(voxelSize/2.0);
                }
                if(const Json* rate = cache.find("fps")){
                    fps = rate->number > 0.0 ? rate->number : fps;
                }
                const Json* nodes = cache.find("nodes");
                const Json* nodeSize = cache.find("nodeSize");
                const Json* domainMin = cache.find("domainMin");
                if(nodes != nullptr && nodeSize != nullptr && domainMin != nullptr && nodes->items.size() == 3 && domainMin->items.size() == 3){
                    walls.closed = 63u;     //a cache from before "open" was written: walls all round
                    for(int axis = 0; axis < 3; ++axis){
                        walls.low[axis] = (float)domainMin->items[axis].number;
                        walls.high[axis] = (float)(domainMin->items[axis].number + nodes->items[axis].number*nodeSize->number);
                    }
                    if(const Json* open = cache.find("open")){
                        const char* faces[6] = {"-x", "+x", "-y", "+y", "-z", "+z"};
                        for(const Json& face : open->items){
                            for(int which = 0; which < 6; ++which){
                                if(face.kind == Json::STRING && face.text == faces[which]){
                                    walls.closed &= ~(1u << which);
                                }
                            }
                        }
                    }
                    //the scene's obstacles, as the bake built them; a scene that can't be read now meshes with none, and says so
                    const Json* scene = cache.find("scene");
                    const Json* path = scene != nullptr ? scene->find("path") : nullptr;
                    Scene described;
                    bool loaded = false;
                    try{
                        if(path != nullptr && path->kind == Json::STRING){
                            described = loadScene(path->text);
                            loaded = true;
                        }
                    }
                    catch(const std::exception& error){
                        std::cerr<<"mesh: the scene's obstacles aren't met: "<<error.what()<<"\n";
                    }
                    if(loaded && !described.obstacles.empty()){
                        if(device >= 0){
                            cudaSetDevice(device);
                        }
                        try{
                            gpuErrchk(cudaStreamCreate(&obstacleStream));
                            obstacles.reset(new ObstacleSet);
                            Grid grid((float)walls.low[0], (float)walls.low[1], (float)walls.low[2], (uint)nodes->items[0].number, (uint)nodes->items[1].number,
                                      (uint)nodes->items[2].number, (float)nodeSize->number);
                            obstacles->build(described.obstacles, voxelSize, grid, obstacleStream);
                            for(size_t index = 0; index < described.obstacles.size(); ++index){
                                if(described.obstacles[index].dynamic){
                                    bodyObstacles.push_back(obstacles->builtFrom(index));
                                }
                            }
                            if(!bodyObstacles.empty()){
                                gpuErrchk(cudaMalloc((void**)&bodyFrames, sizeof(BodyFrame)*bodyObstacles.size()));
                                for(size_t body = 0; body < bodyObstacles.size(); ++body){
                                    if(bodyObstacles[body] >= 0){
                                        obstacles->setFrame(bodyObstacles[body], bodyFrames + body, (int)body);
                                    }
                                }
                            }
                        }
                        catch(const std::exception& error){
                            why = std::string("building the scene's obstacles: ") + error.what();
                            return false;
                        }
                    }
                }
                cacheRead = true;
            }
            if(obstacles){      //where they are at the frame's time: dynamic bodies where its records have them
                if(device >= 0){
                    cudaSetDevice(device);
                }
                if(!bodyObstacles.empty()){
                    std::vector<BgeoBody> bodies;
                    const Json* listing = commit.kind == Json::OBJECT ? commit.find("bodies") : nullptr;
                    if(listing == nullptr || !bodiesOf(directory, frame, *listing, bodies, why) || bodies.size() != bodyObstacles.size()){
                        why = "the frame's bodies: " + (why.empty() ? std::string("not as many as cache.json names") : why);
                        return false;
                    }
                    std::vector<BodyFrame> frames(bodies.size());
                    for(size_t body = 0; body < bodies.size(); ++body){
                        float3 low = make_float3(0.0f, 0.0f, 0.0f), high = low;
                        if(bodyObstacles[body] >= 0){
                            obstacles->objectBounds(bodyObstacles[body], low, high);
                        }
                        frames[body] = bodyFrameOf(bodies[body], low, high, voxelSize);
                    }
                    gpuErrchk(cudaMemcpyAsync(bodyFrames, frames.data(), sizeof(BodyFrame)*frames.size(), cudaMemcpyHostToDevice, obstacleStream));
                }
                obstacles->update(frame / fps);
                obstacles->index();
                gpuErrchk(cudaStreamSynchronize(obstacleStream));
            }
            try{
                if(device >= 0){
                    cudaSetDevice(device);
                }
                if(!mesher){
                    mesher.reset(new SurfaceMesher);
                }
                MeshSolids solids;
                solids.obstacles = obstacles ? &obstacles->state() : nullptr;
                solids.overlap = overlap;
                mesher->mesh(positions.data(), velocities.data(), particles, low, high, settings, mesh, walls, solids);
                if(fields){
                    mesher->fields(fieldSettings, fluidFields);     //of the particles it has just meshed
                }
            }
            catch(const std::exception& error){
                why = error.what();
                return false;
            }
            bandCells = mesher->bandCells;
            rounds = mesher->rounds;
            seconds = mesher->seconds;
            fieldSeconds = mesher->fieldSeconds;
        }
        char numbers[200];
        std::snprintf(numbers, sizeof(numbers), ",\"particles\":%llu,\"points\":%zu,\"faces\":%zu,\"bandCells\":%zu,\"rounds\":%d,\"seconds\":%.4f",
                      particles, mesh.points.size()/3, mesh.quads.size()/4, bandCells, rounds, seconds);
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
                          fluidFields.distances.size(), fluidFields.velocities.size()/3, fieldSeconds,
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
    double acceleration = 0.0;  //the most its last substep accelerated a face at; one from before that was kept has 0, and the next substep goes by the forces alone
    unsigned long long nextId = 0;
    std::string sceneHash;
    std::vector<uint> planes;   //its ranks' first node planes, and the last one's end
    struct State{
        std::string file;
        int rank;
        std::string xxh64;
    };
    std::vector<State> states;  //in rank order
    std::vector<State> whitewater;  //its ranks' whitewater, from a bake with any
    unsigned long long whitewaterNextId = 0;    //and the id the next of its particles gets
    std::vector<State> bodies;      //the dynamic bodies' states (Bodies::restore), from a bake with any: one shard, none from one made before checkpoints kept them
    std::vector<State> contacts;    //and the contacts their next substep starts from
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
        if(const Json* acceleration = record.find("acceleration")){
            out.acceleration = acceleration->number;
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
        const Json* whitewater = record.find("whitewater");
        const Json* shards = whitewater != nullptr ? whitewater->find("shards") : nullptr;
        if(const Json* nextId = whitewater != nullptr ? whitewater->find("nextId") : nullptr){
            out.whitewaterNextId = (unsigned long long)nextId->number;
        }
        for(size_t index = 0; shards != nullptr && index < shards->items.size(); ++index){
            const Json* file = shards->items[index].find("file");
            const Json* rank = shards->items[index].find("rank");
            const Json* hash = shards->items[index].find("xxh64");
            if(file == nullptr || rank == nullptr || hash == nullptr){
                throw std::runtime_error(path + ": a whitewater shard without its file, rank and xxh64");
            }
            out.whitewater.push_back({std::string(name) + "/" + file->text, (int)rank->number, hash->text});
        }
        for(const char* key : {"bodies", "contacts"}){  //a bake with dynamic bodies: their states and contacts, a shard each
            const Json* part = record.find(key);
            const Json* listed = part != nullptr && part->kind == Json::OBJECT ? part->find("shards") : nullptr;
            for(size_t index = 0; listed != nullptr && index < listed->items.size(); ++index){
                const Json* file = listed->items[index].find("file");
                const Json* hash = listed->items[index].find("xxh64");
                if(file == nullptr || hash == nullptr){
                    throw std::runtime_error(path + ": a " + key + " shard without its file and xxh64");
                }
                (std::string(key) == "bodies" ? out.bodies : out.contacts).push_back({std::string(name) + "/" + file->text, 0, hash->text});
            }
        }
    }
    catch(const std::exception& error){
        why = error.what();
        return false;
    }
    return true;
}

//the checkpoint's particles in the ranks whose planes overlap node planes [low, high), in rank order: the order one partition would hold them in. Each
//state shard is checked against its record's XXH64 as it's read. ids and births come back empty from a checkpoint made before particles had them, and
//breakup (how far the churning has broken each air particle's air: escaping air) from one without it
static bool loadCheckpoint(const std::string& directory, const Checkpoint& checkpoint, uint low, uint high, std::vector<double>& x, std::vector<double>& y,
                           std::vector<double>& z, std::vector<float>& u, std::vector<float>& v, std::vector<float>& w,
                           std::vector<std::vector<float>>& gradients, std::vector<uint>& ids, std::vector<float>& births, std::vector<float>& breakup, std::string& why){
    gradients.assign(9, {});
    bool identified = true, sized = true;
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
        const ShardData::Attribute* broken = shard.find("breakup");
        if(broken != nullptr && broken->type == 1 && broken->components == 1){
            const float* b = (const float*)broken->planes.data();
            breakup.insert(breakup.end(), b, b + count);
        }
        else{
            sized = false;
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
    if(!sized){
        breakup.clear();
    }
    return true;
}

//the checkpoint's whitewater, every rank's one after another, each attribute a plane per component as Particles::setWhitewaterParticles takes them
struct CheckpointWhitewater{
    std::vector<double> positions;      //x, then y, then z
    std::vector<unsigned long long> ids;
    std::vector<float> velocities, births, lives, radii, kinds;
    size_t count = 0;
};

static bool loadCheckpointWhitewater(const std::string& directory, const Checkpoint& checkpoint, uint low, uint high, CheckpointWhitewater& out, std::string& why){
    std::vector<ShardData> shards;  //those of the checkpoint's ranks whose planes overlap node planes [low, high), as loadCheckpoint takes the particles'
    for(const Checkpoint::State& state : checkpoint.whitewater){
        if(checkpoint.planes[state.rank] >= high || checkpoint.planes[state.rank + 1] <= low){
            continue;
        }
        shards.emplace_back();
        if(!readShard(directory + "/checkpoints/" + state.file, state.xxh64, shards.back(), why)){
            return false;
        }
        const ShardData& shard = shards.back();
        auto holds = [&](const char* name, uint32_t type, uint32_t components){
            const ShardData::Attribute* attribute = shard.find(name);
            return attribute != nullptr && attribute->type == type && attribute->components == components;
        };
        if(!holds("P", 2, 3) || !holds("id", 3, 1) || !holds("v", 1, 3) || !holds("birth", 1, 1) || !holds("life", 1, 1) || !holds("radius", 1, 1) || !holds("kind", 1, 1)){
            why = directory + "/checkpoints/" + state.file + ": not a checkpoint's whitewater";
            return false;
        }
        out.count += shard.particles;
    }
    //planes of every rank's values together: component by component, rank after rank
    auto gather = [&](const char* name, int components, auto& into){
        using T = typename std::remove_reference_t<decltype(into)>::value_type;
        for(int component = 0; component < components; ++component){
            for(const ShardData& shard : shards){
                const T* plane = (const T*)shard.find(name)->planes.data() + (size_t)component*shard.particles;
                into.insert(into.end(), plane, plane + shard.particles);
            }
        }
    };
    gather("P", 3, out.positions);
    gather("id", 1, out.ids);
    gather("v", 3, out.velocities);
    gather("birth", 1, out.births);
    gather("life", 1, out.lives);
    gather("radius", 1, out.radii);
    gather("kind", 1, out.kinds);
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
        bool overlap = true;    //the surface carried on into the obstacles (--solids overlap), or stopped at their surfaces (cut)
        std::string format = "bgeo.sc";
        int workers = std::max(1, std::min(8, (int)std::thread::hardware_concurrency()/2));    //frames meshed at once: the GPU's part is small, the rest isn't
        for(int arg = 3; arg < argc; ++arg){
            std::string option = argv[arg];
            if(meshing && option == "--workers" && arg + 1 < argc){
                char* end = nullptr;
                std::string given = argv[++arg];
                long count = std::strtol(given.c_str(), &end, 10);
                if(end == given.c_str() || *end != '\0' || count < 1 || count > 256){
                    return usage();
                }
                workers = (int)count;
            }
            else if(option == "--format" && arg + 1 < argc){
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
            else if(meshing && option == "--solids" && arg + 1 < argc){
                std::string how = argv[++arg];
                if(how != "overlap" && how != "cut"){
                    return usage();
                }
                overlap = how == "overlap";
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
        return meshing ? meshCache(argv[2], first, last, out, overwrite, follow, format, surface, meshSurface, fields, fieldSettings, workers, overlap)
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
        bool dynamic = std::any_of(scene.obstacles.begin(), scene.obstacles.end(), [](const SceneObstacle& obstacle){ return obstacle.dynamic; });
        if(dynamic && (checkpoint.bodies.size() != 1 || checkpoint.contacts.size() != 1)){
            return failure(outputDirectory + ": its newest checkpoint, at frame " + std::to_string(checkpoint.frame) + ", is from a flip2 that didn't keep the dynamic "
                           "bodies in its checkpoints as shards, so the bake can't carry on from it: bake it again");
        }
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
    std::vector<float> breakup; //how far the churning has broken each air particle's air, with escaping air
    uint low = 0, high = scene.nodes[2];    //the node planes this process holds: all of them, or across processes, its rank's
    if(resuming){   //every rank's particles here, or across processes, those of the checkpoint's ranks whose planes overlap this rank's
        if(!alone){
            std::vector<uint> planes = Simulation::planesFor(scene.nodes[2], worldSize);
            if(planes.empty() || myRank < 0 || myRank >= worldSize){
                return failure("rank " + std::to_string(myRank) + " of " + std::to_string(worldSize) + " can't hold its share of " + std::to_string(scene.nodes[2]) + " node planes");
            }
            low = planes[myRank];
            high = planes[myRank + 1];
        }
        std::string why;
        if(!loadCheckpoint(scene.outputDirectory, checkpoint, low, high, x, y, z, u, v, w, gradients, ids, births, breakup, why)){
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
    for(int face = 0; face < 6; ++face){
        simulation->setWallContactAngle(face, scene.wallContactAngle[face]);
        simulation->setWallFriction(face, scene.wallFriction[face]);
    }
    simulation->setViscousCfl(scene.viscousCfl);
    simulation->setFreeSurface(scene.freeSurface == "levelSet" ? FreeSurface::sharp : FreeSurface::footprint);
    if(scene.air){      //a second fluid (TwoPhase, particles.hu): the particles seeded after the liquid's, and any made later, are air
        TwoPhase twoPhase;
        twoPhase.on = true;
        twoPhase.densityRatio = (float)(scene.density / scene.airDensity);
        twoPhase.soundSpeed = (float)scene.airSoundSpeed;
        twoPhase.airFlipRatio = (float)(scene.airFlipRatio < 0.0 ? scene.flipRatio : scene.airFlipRatio);
        twoPhase.band = scene.airBand;
        twoPhase.escapes = scene.airEscaped && twoPhase.densityRatio > 1.0f;   //with no difference in density there's no telling the liquid's weight on a face
        twoPhase.dropletRadius = (float)scene.airDropletRadius;
        twoPhase.airViscosity = (float)scene.airViscosity;
        twoPhase.bubbleRadius = (float)scene.airBubbleRadius;
        twoPhase.liquidViscosity = (float)(scene.airLiquidViscosity / scene.density);
        twoPhase.surfaceTension = (float)(scene.airSurfaceTension / scene.density);
        if(!resuming){  //the seeds after the liquid's are air; a checkpoint's ids say which are which themselves
            simulation->markAirParticles(liquidSeeds);
        }
        simulation->setTwoPhase(twoPhase);
        if(resuming && twoPhase.escaping()){    //the air's breakup as the checkpoint has it: one made before has its flying bubbles at the size every
            if(breakup.size() != x.size()){     //bubble flew at then, and the rest unbroken
                float then = twoPhase.bubbleRadius > 0.0f ? twoPhase.bubbleRadius
                                                          : (float)(scene.voxelSize()*std::cbrt(3.0 / (4.0*3.14159265358979323846*scene.particlesPerVoxel)));
                breakup.assign(x.size(), 0.0f);
                for(size_t particle = 0; particle < ids.size() && particle < breakup.size(); ++particle){
                    if((ids[particle] & (AIR_PARTICLE | ESCAPED_PARTICLE)) == (AIR_PARTICLE | ESCAPED_PARTICLE)){
                        breakup[particle] = then;
                    }
                }
            }
            simulation->setAirBreakup(breakup);
        }
        if(printsEvents){
            std::cerr<<"Two phases: density ratio "<<twoPhase.densityRatio<<"\n";
        }
    }
    if(scene.whitewater){   //spray, foam and bubbles (whitewater.hu), with the two fluids' properties as the scene has them, whether or not it simulates the air
        WhitewaterSettings whitewater;
        whitewater.on = true;
        whitewater.amount = (float)scene.whitewaterAmount;
        whitewater.spray = (float)scene.whitewaterSpray;
        whitewater.bubbles = (float)scene.whitewaterBubbles;
        whitewater.perVoxel = scene.whitewaterPerVoxel;
        whitewater.capacity = (uint)scene.whitewaterMaxParticles;
        whitewater.foamLife = (float)scene.whitewaterFoamLife;
        whitewater.maxAge = (float)scene.whitewaterMaxAge;
        whitewater.surfaceTension = (float)((scene.surfaceTension > 0.0 ? scene.surfaceTension : scene.airSurfaceTension) / scene.density);
        whitewater.airDensity = (float)(scene.airDensity / scene.density);
        whitewater.airViscosity = (float)scene.airViscosity;
        whitewater.liquidViscosity = (float)((scene.viscosity > 0.0 ? scene.viscosity : scene.airLiquidViscosity) / scene.density);
        whitewater.dropletScale = (float)scene.whitewaterDropletScale;
        whitewater.bubbleScale = (float)scene.whitewaterBubbleScale;
        whitewater.seed = scene.seed;
        simulation->setWhitewater(whitewater);
        if(resuming && !checkpoint.whitewater.empty()){     //as the checkpoint left it; one from a bake without any starts with none
            CheckpointWhitewater saved;
            std::string why;
            if(!loadCheckpointWhitewater(scene.outputDirectory, checkpoint, low, high, saved, why)){
                return failure(why);
            }
            simulation->setWhitewaterParticles((uint)saved.count, saved.positions.data(), saved.ids.data(), saved.velocities.data(), saved.births.data(), saved.lives.data(),
                                               saved.radii.data(), saved.kinds.data(), checkpoint.whitewaterNextId);
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
    SourceSettings sources;
    sources.latticePerSide = scene.particlesPerVoxel == 27 ? 3 : scene.particlesPerVoxel == 8 ? 2 : 1;
    sources.seed = scene.seed;
    sources.openFaces = scene.openFaces;
    std::vector<SceneObstacle> sourceMeshes;
    for(const SceneShape& emitter : scene.emitters){
        sources.emitters.push_back(toFluidShape(emitter, sourceMeshes));
    }
    for(const SceneShape& sink : scene.sinks){
        sources.sinks.push_back(toFluidShape(sink, sourceMeshes));
    }
    simulation->setSources(sources);
    std::vector<FluidShape> fluids;
    for(const SceneShape& fluid : scene.fluids){
        fluids.push_back(toFluidShape(fluid, sourceMeshes));
    }
    if(!sourceMeshes.empty()){
        simulation->setSourceMeshes(sourceMeshes, fluids);      //after setDomain and setSources
    }
    simulation->setObstacles(scene.obstacles, scene.density);      //after setDomain: they're voxelized at its voxel size, and the dynamic ones weighed against the liquid
    simulation->setCollisionSubsteps(scene.collisionSubsteps);
    if(resuming && checkpoint.bodies.size() == 1 && checkpoint.contacts.size() == 1){   //the dynamic bodies as the checkpoint left them, in place of where the scene starts them
        ShardData bodies, contacts;
        std::string why;
        if(!readShard(outputDirectory + "/checkpoints/" + checkpoint.bodies[0].file, checkpoint.bodies[0].xxh64, bodies, why) ||
           !readShard(outputDirectory + "/checkpoints/" + checkpoint.contacts[0].file, checkpoint.contacts[0].xxh64, contacts, why)){
            return failure(why);
        }
        why = simulation->restoreBodies(bodies, contacts);
        if(!why.empty()){
            return failure(outputDirectory + ": " + why);
        }
    }
    unsigned int wallsLetGo = 0;    //the walls that let go of the liquid, a bit each
    for(int face = 0; face < 6; ++face){
        wallsLetGo |= scene.wallHold[face] ? 0u : 1u << face;
    }
    simulation->setWallsLetGo(wallsLetGo);
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
        description.whitewaterPerVoxel = scene.whitewater ? scene.whitewaterPerVoxel : 0;
        description.openFaces = scene.openFaces;
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
        simulation->resume(checkpoint.time, checkpoint.substep, checkpoint.acceleration);
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
            if(scene.air){
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
    unsigned long long whitewaterDropped = 0;   //what whitewater.maxParticles had kept from being made by the frame before
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
            if(scene.air){
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
        std::string more;
        if(scene.whitewater){   //how much whitewater there is here, and how much more the surface would have made this frame with room for it
            unsigned long long whitewater, dropped;
            simulation->whitewaterHere(whitewater, dropped);
            more = ",\"whitewater\":" + std::to_string(whitewater) + (dropped > whitewaterDropped ? ",\"whitewaterDropped\":" + std::to_string(dropped - whitewaterDropped) : std::string());
            if(dropped > whitewaterDropped && whitewaterDropped == 0 && printsEvents){
                std::cerr<<"Whitewater: at whitewater.maxParticles ("<<(unsigned long long)scene.whitewaterMaxParticles<<"), so what the surface makes is thinned from frame "<<frame<<" on\n";
            }
            whitewaterDropped = dropped;
        }
        std::snprintf(line, sizeof(line), "{\"event\":\"frame\",\"frame\":%d,\"seconds\":%.4f,\"particles\":%zu%s}", frame,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - frameStart).count(), simulation->particlesHere(), more.c_str());
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
