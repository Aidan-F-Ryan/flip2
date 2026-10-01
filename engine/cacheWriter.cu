//Copyright 2023 Aberrant Behavior LLC

#include "cacheWriter.hu"
#include "json.hpp"
#include "xxhash64.hpp"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
#ifdef FLIP2_WITH_BLOSC
#include <blosc.h>
#endif
#ifndef FLIP2_BUILD_ID
#define FLIP2_BUILD_ID "unknown"
#endif

namespace{

constexpr int PLANES = 6;   //P x, y, z, then v x, y, z
constexpr uint32_t CODEC_RAW = 0;
constexpr uint32_t CODEC_BLOSC = 1;
constexpr uint32_t TYPE_FLOAT32 = 1;

struct ShardHeader{
    char magic[8];
    uint32_t version;
    uint32_t attributes;
    uint64_t particles;
    int32_t frame;
    uint32_t rank;
    uint32_t worldSize;
    uint32_t flags;
    float low[3];
    float high[3];
};
static_assert(sizeof(ShardHeader) == 64, "a shard's header is 64 bytes");

struct ShardAttribute{
    char name[16];
    uint32_t type;
    uint32_t components;
    uint32_t codec;
    uint32_t reserved;
    uint64_t offset;
    uint64_t storedBytes;
    uint64_t rawBytes;
    uint64_t reserved2;
};
static_assert(sizeof(ShardAttribute) == 64, "a shard's attribute entry is 64 bytes");

std::string frameName(int frame){
    char name[16];
    std::snprintf(name, sizeof(name), "%04d", frame);
    return name;
}

std::string shardName(int rank, const char* extension){
    char name[32];
    std::snprintf(name, sizeof(name), "particles.r%03d.%s", rank, extension);
    return name;
}

std::string hex(uint64_t value){
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", (unsigned long long)value);
    return text;
}

std::string number(double value){
    char text[32];
    std::snprintf(text, sizeof(text), "%.17g", value);
    return text;
}

std::string triple(const float values[3]){
    char text[64];
    std::snprintf(text, sizeof(text), "[%.9g,%.9g,%.9g]", values[0], values[1], values[2]);
    return text;
}

std::string quoted(const std::string& text){    //a JSON string
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

bool writeAll(int file, const void* data, size_t bytes){
    const char* at = (const char*)data;
    while(bytes > 0){
        ssize_t wrote = ::write(file, at, bytes);
        if(wrote < 0){
            if(errno == EINTR){
                continue;
            }
            return false;
        }
        at += wrote;
        bytes -= (size_t)wrote;
    }
    return true;
}

//flushes a directory's entries to the disk: a file just made or renamed in it survives a crash only once its directory has been
bool syncDirectory(const std::string& path){
    int directory = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
    if(directory < 0){
        return false;
    }
    bool synced = ::fsync(directory) == 0;
    ::close(directory);
    return synced;
}

//puts text in path whole or not at all: into a temporary file beside it, renamed over it. Durably, the file is flushed to the disk before the rename and
//the rename after it, so a crash leaves the old file or the new one; otherwise other processes see one or the other, but a crash can lose it
bool replaceFile(const std::string& path, const std::string& text, std::string& why, bool durably = true){
    std::string temporary = path + ".tmp";
    int file = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if(file < 0){
        why = temporary + ": " + std::strerror(errno);
        return false;
    }
    bool written = writeAll(file, text.data(), text.size()) && (!durably || ::fsync(file) == 0);
    int error = errno;
    ::close(file);
    if(!written){
        why = temporary + ": " + std::strerror(error);
        return false;
    }
    if(::rename(temporary.c_str(), path.c_str()) != 0){
        why = path + ": " + std::strerror(errno);
        return false;
    }
    if(durably && !syncDirectory(std::filesystem::path(path).parent_path().string())){
        why = path + ": flushing its directory: " + std::strerror(errno);
        return false;
    }
    return true;
}

}

CacheWriter::CacheWriter(const std::string& directory, const CacheDescription& description, const std::vector<int>& localRanks, std::function<void(int)> committed,
                         int numBuffers)
: directory(directory)
, description(description)
, localRanks(localRanks)
, commits(std::find(localRanks.begin(), localRanks.end(), 0) != localRanks.end())
, committed(std::move(committed))
{
#ifndef FLIP2_WITH_BLOSC
    if(this->description.compression != "none"){
        std::cerr<<"CacheWriter: this build has no blosc (install libblosc-dev and rebuild), so the cache's frames go uncompressed\n";
        this->description.compression = "none";
    }
#endif
    for(int i = 0; i < numBuffers; ++i){    //allocated as frames ask for them
        hostBuffers.push_back(nullptr);
        capacities.push_back(0);
        freeBuffers.push_back(i);
    }
    std::error_code made;
    std::filesystem::create_directories(directory + "/frames", made);
    if(made){
        fail(directory + "/frames: " + made.message());
    }
    else if(commits){
        describe();
    }
    thread = std::thread(&CacheWriter::run, this);
}

CacheWriter::~CacheWriter(){
    {
        std::lock_guard<std::mutex> lock(mutex);
        stopping = true;
    }
    changed.notify_all();
    thread.join();
    if(timing.frames > 0){
        double per = 1000.0/timing.frames;
        std::fprintf(stderr, "CacheWriter: %d frames; per frame, %.1f ms compressing, %.1f writing, %.1f flushing to the disk, %.1f committing; the simulation waited "
                     "for a buffer %d times, %.1f ms in all\n", timing.frames, per*timing.compress, per*timing.write, per*timing.flush, per*timing.commit, timing.waits,
                     1000.0*timing.waited);
    }
    for(float* hostBuffer : hostBuffers){
        if(hostBuffer != nullptr){
            cudaFreeHost(hostBuffer);
        }
    }
}

int CacheWriter::acquire(size_t floats){
    int buffer;
    {
        std::unique_lock<std::mutex> lock(mutex);
        if(freeBuffers.empty()){    //the disk is behind the simulation
            auto start = std::chrono::steady_clock::now();
            changed.wait(lock, [this]{ return !freeBuffers.empty(); });
            timing.waited += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            ++timing.waits;
        }
        buffer = freeBuffers.back();
        freeBuffers.pop_back();
    }
    floats = std::max<size_t>(floats, 1024);
    if(capacities[buffer] < floats){    //nobody else touches a buffer between acquire and submit
        if(hostBuffers[buffer] != nullptr){
            gpuErrchk(cudaFreeHost(hostBuffers[buffer]));
        }
        gpuErrchk(cudaHostAlloc((void**)&hostBuffers[buffer], sizeof(float)*floats, cudaHostAllocPortable));    //portable: pinned for every GPU's copies
        capacities[buffer] = floats;
    }
    return buffer;
}

void CacheWriter::submit(int buffer, int frame, double time, const std::vector<CacheShard>& shards, const std::vector<cudaEvent_t>& copies){
    {
        std::lock_guard<std::mutex> lock(mutex);
        jobs.push({buffer, frame, time, shards, copies});
    }
    changed.notify_all();
}

std::string CacheWriter::error(){
    std::lock_guard<std::mutex> lock(mutex);
    return failure;
}

void CacheWriter::fail(const std::string& why){
    std::lock_guard<std::mutex> lock(mutex);
    if(failure.empty()){
        failure = "writing the cache: " + why;
    }
}

void CacheWriter::flush(){
    std::unique_lock<std::mutex> lock(mutex);
    changed.wait(lock, [this]{ return jobs.empty() && !busy; });
}

void CacheWriter::run(){
    while(true){
        Job job;
        {
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock, [this]{ return stopping || !jobs.empty(); });
            if(jobs.empty()){
                return;     //stopping, and nothing left to write
            }
            job = jobs.front();
            jobs.pop();
            busy = true;
        }
        process(job);
        {
            std::lock_guard<std::mutex> lock(mutex);
            busy = false;
        }
        changed.notify_all();
    }
}

void CacheWriter::process(const Job& job){
    for(cudaEvent_t copy : job.copies){
        gpuErrchk(cudaEventSynchronize(copy));
    }
    bool fine = error().empty();    //after a failure nothing more is committed, so the frames committed stay the ones before it
    std::string frameDirectory = directory + "/frames/" + frameName(job.frame);
    if(fine){
        std::error_code made;
        std::filesystem::create_directories(frameDirectory, made);
        if(made){
            fail(frameDirectory + ": " + made.message());
            fine = false;
        }
    }
    std::vector<ShardRecord> records;
    for(const CacheShard& shard : job.shards){
        ShardRecord record;
        fine = fine && writeShard(frameDirectory, job.frame, shard, hostBuffers[job.buffer] + shard.offset, record);
        if(fine && !commits){   //rank 0, in another process, commits it from this record
            std::string why;
            std::string text = "{\"rank\":" + std::to_string(record.rank) + ",\"particles\":" + std::to_string(record.particles) + ",\"bytes\":" +
                               std::to_string(record.bytes) + ",\"xxh64\":\"" + hex(record.hash) + "\",\"low\":" + triple(record.low) + ",\"high\":" + triple(record.high) + "}\n";
            if(!replaceFile(frameDirectory + "/" + shardName(record.rank, "json"), text, why, false)){   //only for rank 0 to read: it needn't survive a crash
                fail(why);
                fine = false;
            }
        }
        records.push_back(record);
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        freeBuffers.push_back(job.buffer);
    }
    changed.notify_all();
    if(!fine || !commits){
        return;
    }
    for(int rank = 0; rank < description.worldSize; ++rank){    //the ranks in other processes, as their records turn up
        if(std::find(localRanks.begin(), localRanks.end(), rank) != localRanks.end()){
            continue;
        }
        std::string path = frameDirectory + "/" + shardName(rank, "json");
        ShardRecord record;
        auto start = std::chrono::steady_clock::now();
        int minutes = 0;
        while(!readRecord(path, rank, record)){
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            int waited = (int)(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / 60.0);
            if(waited > minutes){
                minutes = waited;
                std::cerr<<"CacheWriter: frame "<<job.frame<<" has waited "<<minutes<<" minute"<<(minutes > 1 ? "s" : "")<<" for rank "<<rank<<"'s shard\n";
            }
            if(minutes >= 30){
                fail("frame " + std::to_string(job.frame) + ": rank " + std::to_string(rank) + "'s shard never came");
                return;
            }
        }
        records.push_back(record);
    }
    std::sort(records.begin(), records.end(), [](const ShardRecord& a, const ShardRecord& b){ return a.rank < b.rank; });
    auto start = std::chrono::steady_clock::now();
    bool down = commit(frameDirectory, job.frame, job.time, records);
    timing.commit += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ++timing.frames;
    if(down && committed){
        committed(job.frame);
    }
    bool idle;
    {
        std::lock_guard<std::mutex> lock(mutex);
        idle = jobs.empty();
    }
    if(down && (idle || std::chrono::steady_clock::now() - described > std::chrono::seconds(1))){
        describe();
    }
}

bool CacheWriter::describe(){
    if(lastDescribed == lastCommitted){
        return true;
    }
    std::string why;
    if(!replaceFile(directory + "/cache.json", cacheJson(), why)){
        fail(why);
        return false;
    }
    lastDescribed = lastCommitted;
    described = std::chrono::steady_clock::now();
    return true;
}

//a shard: compressed (or not), hashed as it's written, and flushed to the disk before its record says it's there
bool CacheWriter::writeShard(const std::string& frameDirectory, int frame, const CacheShard& shard, const float* planes, ShardRecord& record){
    size_t count = shard.particles;
    record.rank = shard.rank;
    record.particles = count;
    for(int axis = 0; axis < 3; ++axis){
        float low = INFINITY, high = -INFINITY;
        const float* plane = planes + axis*count;
        for(size_t i = 0; i < count; ++i){
            low = std::min(low, plane[i]);
            high = std::max(high, plane[i]);
        }
        record.low[axis] = count > 0 ? low : 0.0f;
        record.high[axis] = count > 0 ? high : 0.0f;
    }
    size_t planeBytes = sizeof(float)*count;
    bool compress = description.compression != "none" && count > 0;
#ifdef FLIP2_WITH_BLOSC
    compress = compress && planeBytes <= (size_t)BLOSC_MAX_BUFFERSIZE;
#endif
    const char* stored[PLANES];
    uint64_t storedBytes[PLANES];
    for(int plane = 0; plane < PLANES; ++plane){
        stored[plane] = (const char*)(planes + plane*count);
        storedBytes[plane] = planeBytes;
    }
    auto start = std::chrono::steady_clock::now();
#ifdef FLIP2_WITH_BLOSC
    if(compress){
        size_t room = planeBytes + BLOSC_MAX_OVERHEAD;
        if(scratch.size() < PLANES*room){
            scratch.resize(PLANES*room);
        }
        //a thread per plane, each running blosc on one thread: with more, blosc places its blocks in the order they finish, and the same frame would
        //come out a different file each time
        bool zstd = description.compression == "zstd";     //zstd at level 1 with bit shuffle packs simulation floats best for the time; lz4 is faster
        std::atomic<bool> packed{true};
        std::vector<std::thread> workers;
        for(int plane = 0; plane < PLANES; ++plane){
            workers.emplace_back([&, plane]{
                char* into = scratch.data() + plane*room;
                int bytes = blosc_compress_ctx(zstd ? 1 : 5, zstd ? BLOSC_BITSHUFFLE : BLOSC_SHUFFLE, sizeof(float), planeBytes, planes + plane*count, into, room,
                                               zstd ? BLOSC_ZSTD_COMPNAME : BLOSC_LZ4_COMPNAME, 0, 1);
                if(bytes <= 0){
                    packed = false;
                    return;
                }
                stored[plane] = into;
                storedBytes[plane] = (uint64_t)bytes;
            });
        }
        for(std::thread& worker : workers){
            worker.join();
        }
        if(!packed){
            fail("frame " + std::to_string(frame) + ": blosc couldn't compress rank " + std::to_string(shard.rank) + "'s shard");
            return false;
        }
    }
#endif
    auto compressed = std::chrono::steady_clock::now();
    timing.compress += std::chrono::duration<double>(compressed - start).count();
    ShardHeader header = {};
    std::memcpy(header.magic, "FLIP2SHD", 8);
    header.version = 1;
    header.attributes = 2;
    header.particles = count;
    header.frame = frame;
    header.rank = (uint32_t)shard.rank;
    header.worldSize = (uint32_t)description.worldSize;
    for(int axis = 0; axis < 3; ++axis){
        header.low[axis] = record.low[axis];
        header.high[axis] = record.high[axis];
    }
    ShardAttribute attributes[2] = {};
    uint64_t offset = sizeof(ShardHeader) + sizeof(attributes);
    for(int attribute = 0; attribute < 2; ++attribute){
        ShardAttribute& entry = attributes[attribute];
        std::strncpy(entry.name, attribute == 0 ? "P" : "v", sizeof(entry.name));
        entry.type = TYPE_FLOAT32;
        entry.components = 3;
        entry.codec = compress ? CODEC_BLOSC : CODEC_RAW;
        entry.offset = offset;
        entry.storedBytes = 3*sizeof(uint64_t);
        for(int component = 0; component < 3; ++component){
            entry.storedBytes += storedBytes[3*attribute + component];
        }
        entry.rawBytes = 3*planeBytes;
        offset += entry.storedBytes;
    }
    std::string path = frameDirectory + "/" + shardName(shard.rank, "f2p");
    int file = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if(file < 0){
        fail(path + ": " + std::strerror(errno));
        return false;
    }
    Xxh64 hash;
    auto put = [&](const void* data, size_t bytes){
        hash.update(data, bytes);
        return writeAll(file, data, bytes);
    };
    bool written = put(&header, sizeof(header)) && put(attributes, sizeof(attributes));
    for(int attribute = 0; attribute < 2 && written; ++attribute){
        written = put(storedBytes + 3*attribute, 3*sizeof(uint64_t));
        for(int component = 0; component < 3 && written; ++component){
            written = put(stored[3*attribute + component], storedBytes[3*attribute + component]);
        }
    }
    auto wrote = std::chrono::steady_clock::now();
    timing.write += std::chrono::duration<double>(wrote - compressed).count();
    written = written && ::fsync(file) == 0;
    timing.flush += std::chrono::duration<double>(std::chrono::steady_clock::now() - wrote).count();
    int error = errno;
    ::close(file);
    if(!written){
        fail(path + ": " + std::strerror(error));
        return false;
    }
    record.bytes = offset;
    record.hash = hash.digest();
    return true;
}

//another process's record of its shard, if it's there yet: it's renamed into place whole, so one that's there reads
bool CacheWriter::readRecord(const std::string& path, int rank, ShardRecord& record){
    std::ifstream file(path, std::ios::binary);
    if(!file){
        return false;
    }
    std::stringstream text;
    text<<file.rdbuf();
    std::string contents = text.str();
    try{
        Json json = JsonReader(contents, path).document();
        const Json* particles = json.find("particles");
        const Json* bytes = json.find("bytes");
        const Json* hash = json.find("xxh64");
        const Json* low = json.find("low");
        const Json* high = json.find("high");
        if(particles == nullptr || bytes == nullptr || hash == nullptr || low == nullptr || high == nullptr || low->items.size() != 3 || high->items.size() != 3){
            return false;
        }
        record.rank = rank;
        record.particles = (uint64_t)particles->number;
        record.bytes = (uint64_t)bytes->number;
        record.hash = std::strtoull(hash->text.c_str(), nullptr, 16);
        for(int axis = 0; axis < 3; ++axis){
            record.low[axis] = (float)low->items[axis].number;
            record.high[axis] = (float)high->items[axis].number;
        }
        return true;
    }
    catch(const std::exception&){
        return false;
    }
}

//the frame's commit record, written last: the frame is in the cache once it's down
bool CacheWriter::commit(const std::string& frameDirectory, int frame, double time, const std::vector<ShardRecord>& records){
    uint64_t total = 0;
    for(const ShardRecord& record : records){
        total += record.particles;
    }
    std::string text = "{\"flip2\":\"commit\",\"version\":1,\"frame\":" + std::to_string(frame) + ",\"time\":" + number(time) + ",\"particles\":" + std::to_string(total) +
                       ",\"shards\":[";
    for(size_t i = 0; i < records.size(); ++i){
        const ShardRecord& record = records[i];
        text += std::string(i ? "," : "") + "\n  {\"file\":\"" + shardName(record.rank, "f2p") + "\",\"rank\":" + std::to_string(record.rank) + ",\"particles\":" +
                std::to_string(record.particles) + ",\"bytes\":" + std::to_string(record.bytes) + ",\"xxh64\":\"" + hex(record.hash) + "\",\"low\":" + triple(record.low) +
                ",\"high\":" + triple(record.high) + "}";
    }
    text += "]}\n";
    //the shards were flushed as they were written, and flushing the frame's directory after the rename makes their names last along with the record's;
    //then the frame's own name in frames/
    std::string why;
    if(!replaceFile(frameDirectory + "/commit.json", text, why)){
        fail(why);
        return false;
    }
    if(!syncDirectory(directory + "/frames")){
        fail(directory + "/frames: flushing it: " + std::strerror(errno));
        return false;
    }
    lastCommitted = std::max(lastCommitted, frame);
    return true;
}

std::string CacheWriter::cacheJson() const{
    const CacheDescription& d = description;
    std::string planes;
    for(size_t i = 0; i < d.partitionPlanes.size(); ++i){
        planes += (i ? "," : "") + std::to_string(d.partitionPlanes[i]);
    }
    return "{\"flip2\":\"cache\",\"version\":1,\n"
           " \"scene\":{\"path\":" + quoted(d.scenePath) + ",\"xxh64\":\"" + d.sceneHash + "\"},\n"
           " \"fps\":" + number(d.fps) + ",\"frames\":" + std::to_string(d.frames) + ",\"committed\":" + std::to_string(lastCommitted) + ",\n"
           " \"worldSize\":" + std::to_string(d.worldSize) + ",\"partitionPlanes\":[" + planes + "],\n"
           " \"nodes\":[" + std::to_string(d.nodes[0]) + "," + std::to_string(d.nodes[1]) + "," + std::to_string(d.nodes[2]) + "],\"nodeSize\":" + number(d.nodeSize) +
           ",\"voxelSize\":" + number(d.voxelSize) + ",\"domainMin\":[" + number(d.domainMin[0]) + "," + number(d.domainMin[1]) + "," + number(d.domainMin[2]) + "],\n"
           " \"gpu\":" + quoted(d.gpu) + ",\"sm\":" + std::to_string(d.sm) + ",\"cudaRuntime\":" + std::to_string(d.cudaRuntime) + ",\"build\":" + quoted(FLIP2_BUILD_ID) + ",\n"
           " \"shards\":{\"format\":\"f2p\",\"version\":1,\"compression\":\"" + (d.compression == "none" ? std::string("none") : "blosc-" + d.compression) + "\","
           "\"attributes\":[{\"name\":\"P\",\"type\":\"float32\",\"components\":3},{\"name\":\"v\",\"type\":\"float32\",\"components\":3}]}}\n";
}
