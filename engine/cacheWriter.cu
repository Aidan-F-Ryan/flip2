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
#include <iterator>
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

constexpr uint32_t CODEC_RAW = 0;
constexpr uint32_t CODEC_BLOSC = 1;
constexpr uint32_t TYPE_FLOAT32 = 1;
constexpr uint32_t TYPE_FLOAT64 = 2;
constexpr uint32_t TYPE_UINT64 = 3;

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

std::string shardName(const char* base, int rank, const char* extension){    //particles.r003.f2p, state.r003.json
    char name[48];
    std::snprintf(name, sizeof(name), "%s.r%03d.%s", base, rank, extension);
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

bool cacheCompresses(){
#ifdef FLIP2_WITH_BLOSC
    return true;
#else
    return false;
#endif
}

const ShardData::Attribute* ShardData::find(const std::string& name) const{
    for(const Attribute& attribute : attributes){
        if(attribute.name == name){
            return &attribute;
        }
    }
    return nullptr;
}

bool readShard(const std::string& path, const std::string& xxh64, ShardData& out, std::string& why){
    std::ifstream file(path, std::ios::binary);
    if(!file){
        why = path + ": missing";
        return false;
    }
    std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if(!xxh64.empty()){
        Xxh64 hash;
        hash.update(data.data(), data.size());
        if(hex(hash.digest()) != xxh64){
            why = path + ": its XXH64 is " + hex(hash.digest()) + ", not the " + xxh64 + " its record says: it's damaged";
            return false;
        }
    }
    ShardHeader header;
    if(data.size() < sizeof(header)){
        why = path + ": too short for a shard";
        return false;
    }
    std::memcpy(&header, data.data(), sizeof(header));
    if(std::memcmp(header.magic, "FLIP2SHD", 8) != 0 || header.version != 1 || data.size() < sizeof(header) + header.attributes*sizeof(ShardAttribute)){
        why = path + ": not a version 1 flip2 shard";
        return false;
    }
    out = ShardData();
    out.particles = header.particles;
    out.frame = header.frame;
    out.rank = (int)header.rank;
    out.worldSize = (int)header.worldSize;
    for(uint32_t index = 0; index < header.attributes; ++index){
        ShardAttribute entry;
        std::memcpy(&entry, data.data() + sizeof(header) + index*sizeof(entry), sizeof(entry));
        ShardData::Attribute attribute;
        attribute.name = std::string(entry.name, strnlen(entry.name, sizeof(entry.name)));
        attribute.type = entry.type;
        attribute.components = entry.components;
        if(entry.type != TYPE_FLOAT32 && entry.type != TYPE_FLOAT64 && entry.type != TYPE_UINT64){  //a type a later version added: skipped, as
            continue;                                                                               //docs/cache-format.md says readers do
        }
        size_t valueBytes = entry.type == TYPE_FLOAT32 ? 4 : 8;
        size_t planeBytes = valueBytes*header.particles;
        if(entry.offset + entry.storedBytes > data.size() || entry.rawBytes != planeBytes*entry.components){
            why = path + ": attribute " + attribute.name + " doesn't fit the file";
            return false;
        }
        attribute.planes.resize(entry.rawBytes);
        size_t at = entry.offset + 8*entry.components;
        for(uint32_t component = 0; component < entry.components; ++component){
            uint64_t stored;
            std::memcpy(&stored, data.data() + entry.offset + 8*component, 8);
            if(at + stored > data.size()){
                why = path + ": attribute " + attribute.name + " runs past the end of the file";
                return false;
            }
            char* into = attribute.planes.data() + component*planeBytes;
            if(entry.codec == CODEC_RAW){
                if(stored != planeBytes){
                    why = path + ": attribute " + attribute.name + " is the wrong size";
                    return false;
                }
                std::memcpy(into, data.data() + at, planeBytes);
            }
            else{
#ifdef FLIP2_WITH_BLOSC
                if(planeBytes > 0 && blosc_decompress_ctx(data.data() + at, into, planeBytes, 1) != (int)planeBytes){
                    why = path + ": attribute " + attribute.name + " doesn't decompress";
                    return false;
                }
#else
                why = path + ": its planes are compressed with blosc, which this build lacks (install libblosc-dev and rebuild)";
                return false;
#endif
            }
            at += stored;
        }
        out.attributes.push_back(std::move(attribute));
    }
    return true;
}

CacheWriter::CacheWriter(const std::string& directory, const CacheDescription& description, const std::vector<int>& localRanks, int committedBefore,
                         std::function<void(const char*, int)> done, int numBuffers)
: directory(directory)
, description(description)
, localRanks(localRanks)
, commits(std::find(localRanks.begin(), localRanks.end(), 0) != localRanks.end())
, done(std::move(done))
, lastCommitted(committedBefore)
{
#ifndef FLIP2_WITH_BLOSC
    if(this->description.compression != "none"){
        std::cerr<<"CacheWriter: this build has no blosc (install libblosc-dev and rebuild), so the cache's frames go uncompressed\n";
        this->description.compression = "none";
    }
#endif
    this->description.keepCheckpoints = std::max(1, this->description.keepCheckpoints);
    for(int i = 0; i < numBuffers; ++i){    //allocated as jobs ask for them
        hostBuffers.push_back(nullptr);
        capacities.push_back(0);
        freeBuffers.push_back(i);
    }
    std::error_code made;
    std::filesystem::create_directories(directory + "/frames", made);
    if(!made){
        std::filesystem::create_directories(directory + "/checkpoints", made);
    }
    if(made){
        fail(directory + ": making frames/ and checkpoints/ in it: " + made.message());
    }
    else if(commits){
        std::error_code listed;     //a resumed bake keeps the checkpoints it had
        for(const auto& entry : std::filesystem::directory_iterator(directory + "/checkpoints", listed)){
            std::string name = entry.path().filename().string();
            if(entry.is_directory() && std::filesystem::exists(entry.path() / "ckpt.json") && !name.empty() && std::all_of(name.begin(), name.end(), ::isdigit)){
                lastCheckpoint = std::max(lastCheckpoint, std::atoi(name.c_str()));
            }
        }
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
    if(timing.frames + timing.checkpoints > 0){
        double per = 1000.0/(timing.frames + timing.checkpoints);
        std::fprintf(stderr, "CacheWriter: %d frames and %d checkpoints; per job, %.1f ms compressing, %.1f writing, %.1f flushing to the disk, %.1f committing; the "
                     "simulation waited for a buffer %d times, %.1f ms in all\n", timing.frames, timing.checkpoints, per*timing.compress, per*timing.write, per*timing.flush,
                     per*timing.commit, timing.waits, 1000.0*timing.waited);
    }
    for(char* hostBuffer : hostBuffers){
        if(hostBuffer != nullptr){
            cudaFreeHost(hostBuffer);
        }
    }
}

int CacheWriter::acquire(size_t bytes){
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
    bytes = std::max<size_t>(bytes, 4096);
    if(capacities[buffer] < bytes){     //nobody else touches a buffer between acquire and submit
        if(hostBuffers[buffer] != nullptr){
            gpuErrchk(cudaFreeHost(hostBuffers[buffer]));
        }
        gpuErrchk(cudaHostAlloc((void**)&hostBuffers[buffer], bytes, cudaHostAllocPortable));     //portable: pinned for every GPU's copies
        capacities[buffer] = bytes;
    }
    return buffer;
}

void CacheWriter::submit(int buffer, int frame, double time, const std::vector<CacheShard>& shards, const std::vector<cudaEvent_t>& copies, const std::vector<CacheShard>& whitewater){
    Job job{false, buffer, frame, CheckpointState(), shards, copies, whitewater};
    job.state.time = time;
    {
        std::lock_guard<std::mutex> lock(mutex);
        jobs.push(job);
    }
    changed.notify_all();
}

void CacheWriter::submitCheckpoint(int buffer, int frame, const CheckpointState& state, const std::vector<CacheShard>& shards, const std::vector<cudaEvent_t>& copies,
                                   const std::vector<CacheShard>& whitewater){
    {
        std::lock_guard<std::mutex> lock(mutex);
        jobs.push({true, buffer, frame, state, shards, copies, whitewater});
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
    bool fine = error().empty();    //after a failure nothing more is committed, so what's committed stays what came before it
    const char* base = job.checkpoint ? "state" : "particles";
    std::string place = directory + (job.checkpoint ? "/checkpoints/" : "/frames/") + frameName(job.frame);
    std::vector<Layout> layout = {{"P", 3, job.checkpoint ? 8 : 4, false}, {"v", 3, 4, false}};    //in the order the partitions pack them
    if(job.checkpoint && job.state.apic){
        layout.push_back({"c", 9, 4, false});
    }
    if(job.checkpoint || description.ids){
        layout.push_back({"id", 1, 8, true});
    }
    if(job.checkpoint){
        layout.push_back({"birth", 1, 4, false});
    }
    else if(description.ages){
        layout.push_back({"age", 1, 4, false});
    }
    if(fine){
        std::error_code made;
        std::filesystem::create_directories(place, made);
        if(made){
            fail(place + ": " + made.message());
            fine = false;
        }
    }
    std::vector<ShardRecord> records;
    for(const CacheShard& shard : job.shards){
        ShardRecord record;
        fine = fine && writeShard(place + "/" + shardName(base, shard.rank, "f2p"), job.frame, shard, hostBuffers[job.buffer] + shard.offset, layout, record);
        record.acceleration = job.checkpoint ? job.state.acceleration : 0.0;
        if(fine && !commits){   //rank 0, in another process, commits it from this record
            std::string why;
            std::string text = "{\"rank\":" + std::to_string(record.rank) + ",\"particles\":" + std::to_string(record.particles) + ",\"bytes\":" +
                               std::to_string(record.bytes) + ",\"xxh64\":\"" + hex(record.hash) + "\",\"low\":" + triple(record.low) + ",\"high\":" + triple(record.high) +
                               (job.checkpoint ? ",\"acceleration\":" + number(record.acceleration) : std::string()) + "}\n";
            if(!replaceFile(place + "/" + shardName(base, record.rank, "json"), text, why, false)){   //only for rank 0 to read: it needn't survive a crash
                fail(why);
                fine = false;
            }
        }
        records.push_back(record);
    }
    //the whitewater's shards, beside the particles', and their records for rank 0 the same way
    std::vector<Layout> whitewaterLayout = job.checkpoint
        ? std::vector<Layout>{{"P", 3, 8, false}, {"id", 1, 8, true}, {"v", 3, 4, false}, {"birth", 1, 4, false}, {"life", 1, 4, false}, {"radius", 1, 4, false}, {"kind", 1, 4, false}}
        : std::vector<Layout>{{"P", 3, 4, false}, {"v", 3, 4, false}, {"id", 1, 8, true}, {"age", 1, 4, false}, {"life", 1, 4, false}, {"radius", 1, 4, false}, {"kind", 1, 4, false}};
    std::vector<ShardRecord> whitewaterRecords;
    for(const CacheShard& shard : job.whitewater){
        ShardRecord record;
        fine = fine && writeShard(place + "/" + shardName("whitewater", shard.rank, "f2p"), job.frame, shard, hostBuffers[job.buffer] + shard.offset, whitewaterLayout, record);
        if(fine && !commits){
            std::string why;
            std::string text = "{\"rank\":" + std::to_string(record.rank) + ",\"particles\":" + std::to_string(record.particles) + ",\"bytes\":" +
                               std::to_string(record.bytes) + ",\"xxh64\":\"" + hex(record.hash) + "\",\"low\":" + triple(record.low) + ",\"high\":" + triple(record.high) + "}\n";
            if(!replaceFile(place + "/" + shardName("whitewater", record.rank, "json"), text, why, false)){
                fail(why);
                fine = false;
            }
        }
        whitewaterRecords.push_back(record);
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        freeBuffers.push_back(job.buffer);
    }
    changed.notify_all();
    if(!fine || !commits){
        return;
    }
    for(int rank = 0; rank < description.worldSize; ++rank){    //the ranks in other processes, as their records turn up: the particles', and with whitewater its own
        if(std::find(localRanks.begin(), localRanks.end(), rank) != localRanks.end()){
            continue;
        }
        for(const char* kind : {base, "whitewater"}){
            if(kind != base && description.whitewaterPerVoxel <= 0){
                continue;
            }
            std::string path = place + "/" + shardName(kind, rank, "json");
            ShardRecord record;
            auto start = std::chrono::steady_clock::now();
            int minutes = 0;
            while(!readRecord(path, rank, record)){
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                int waited = (int)(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / 60.0);
                if(waited > minutes){
                    minutes = waited;
                    std::cerr<<"CacheWriter: "<<place<<" has waited "<<minutes<<" minute"<<(minutes > 1 ? "s" : "")<<" for rank "<<rank<<"'s "<<kind<<" shard\n";
                }
                if(minutes >= 30){
                    fail(place + ": rank " + std::to_string(rank) + "'s " + kind + " shard never came");
                    return;
                }
            }
            (kind == base ? records : whitewaterRecords).push_back(record);
        }
    }
    for(std::vector<ShardRecord>* list : {&records, &whitewaterRecords}){
        std::sort(list->begin(), list->end(), [](const ShardRecord& a, const ShardRecord& b){ return a.rank < b.rank; });
    }
    auto start = std::chrono::steady_clock::now();
    bool down = job.checkpoint ? commitCheckpoint(place, job, records, whitewaterRecords) : commit(place, job.frame, job.state.time, records, whitewaterRecords);
    timing.commit += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    ++(job.checkpoint ? timing.checkpoints : timing.frames);
    if(down && done){
        done(job.checkpoint ? "checkpoint" : "committed", job.frame);
    }
    if(down && job.checkpoint){
        pruneCheckpoints();
    }
    bool idle;
    {
        std::lock_guard<std::mutex> lock(mutex);
        idle = jobs.empty();
    }
    if(down && (idle || job.checkpoint || std::chrono::steady_clock::now() - described > std::chrono::seconds(1))){
        describe();
    }
}

bool CacheWriter::describe(){
    if(lastDescribed == lastCommitted && checkpointDescribed == lastCheckpoint){
        return true;
    }
    std::string why;
    if(!replaceFile(directory + "/cache.json", cacheJson(), why)){
        fail(why);
        return false;
    }
    lastDescribed = lastCommitted;
    checkpointDescribed = lastCheckpoint;
    described = std::chrono::steady_clock::now();
    return true;
}

//a shard: compressed (or not), hashed as it's written, and flushed to the disk before its record says it's there
bool CacheWriter::writeShard(const std::string& path, int frame, const CacheShard& shard, const char* data, const std::vector<Layout>& layout, ShardRecord& record){
    size_t count = shard.particles;
    record.rank = shard.rank;
    record.particles = count;
    struct Plane{
        const char* stored;
        uint64_t storedBytes;
        size_t rawBytes;
        int valueBytes;
        const char* raw;
    };
    std::vector<Plane> planes;
    size_t at = 0;
    for(const Layout& attribute : layout){
        for(int component = 0; component < attribute.components; ++component){
            size_t bytes = (size_t)attribute.bytes*count;
            planes.push_back({data + at, bytes, bytes, attribute.bytes, data + at});
            at += bytes;
        }
    }
    for(int axis = 0; axis < 3; ++axis){    //the bounds, from the positions: the first attribute
        double low = INFINITY, high = -INFINITY;
        for(size_t i = 0; i < count; ++i){
            double x = layout[0].bytes == 8 ? ((const double*)planes[axis].raw)[i] : ((const float*)planes[axis].raw)[i];
            low = std::min(low, x);
            high = std::max(high, x);
        }
        record.low[axis] = count > 0 ? (float)low : 0.0f;
        record.high[axis] = count > 0 ? (float)high : 0.0f;
    }
    bool compress = description.compression != "none" && count > 0;
#ifdef FLIP2_WITH_BLOSC
    for(const Plane& plane : planes){
        compress = compress && plane.rawBytes <= (size_t)BLOSC_MAX_BUFFERSIZE;
    }
#endif
    auto start = std::chrono::steady_clock::now();
#ifdef FLIP2_WITH_BLOSC
    if(compress){
        std::vector<size_t> into(planes.size());
        size_t room = 0;
        for(size_t plane = 0; plane < planes.size(); ++plane){
            into[plane] = room;
            room += planes[plane].rawBytes + BLOSC_MAX_OVERHEAD;
        }
        if(scratch.size() < room){
            scratch.resize(room);
        }
        //a thread per plane, each running blosc on one thread: with more, blosc places its blocks in the order they finish, and the same frame would
        //come out a different file each time
        bool zstd = description.compression == "zstd";     //zstd at level 1 with bit shuffle packs simulation floats best for the time; lz4 is faster
        std::atomic<bool> packed{true};
        std::vector<std::thread> workers;
        for(size_t plane = 0; plane < planes.size(); ++plane){
            workers.emplace_back([&, plane]{
                Plane& mine = planes[plane];
                char* out = scratch.data() + into[plane];
                int bytes = blosc_compress_ctx(zstd ? 1 : 5, zstd ? BLOSC_BITSHUFFLE : BLOSC_SHUFFLE, mine.valueBytes, mine.rawBytes, mine.raw, out,
                                               mine.rawBytes + BLOSC_MAX_OVERHEAD, zstd ? BLOSC_ZSTD_COMPNAME : BLOSC_LZ4_COMPNAME, 0, 1);
                if(bytes <= 0){
                    packed = false;
                    return;
                }
                mine.stored = out;
                mine.storedBytes = (uint64_t)bytes;
            });
        }
        for(std::thread& worker : workers){
            worker.join();
        }
        if(!packed){
            fail(path + ": blosc couldn't compress it");
            return false;
        }
    }
#endif
    auto compressed = std::chrono::steady_clock::now();
    timing.compress += std::chrono::duration<double>(compressed - start).count();
    ShardHeader header = {};
    std::memcpy(header.magic, "FLIP2SHD", 8);
    header.version = 1;
    header.attributes = (uint32_t)layout.size();
    header.particles = count;
    header.frame = frame;
    header.rank = (uint32_t)shard.rank;
    header.worldSize = (uint32_t)description.worldSize;
    for(int axis = 0; axis < 3; ++axis){
        header.low[axis] = record.low[axis];
        header.high[axis] = record.high[axis];
    }
    std::vector<ShardAttribute> entries(layout.size());
    uint64_t offset = sizeof(ShardHeader) + entries.size()*sizeof(ShardAttribute);
    size_t first = 0;   //the attribute's first plane
    for(size_t attribute = 0; attribute < layout.size(); ++attribute){
        ShardAttribute& entry = entries[attribute];
        std::memset(&entry, 0, sizeof(entry));
        std::strncpy(entry.name, layout[attribute].name.c_str(), sizeof(entry.name));
        entry.type = layout[attribute].integer ? TYPE_UINT64 : layout[attribute].bytes == 8 ? TYPE_FLOAT64 : TYPE_FLOAT32;
        entry.components = (uint32_t)layout[attribute].components;
        entry.codec = compress ? CODEC_BLOSC : CODEC_RAW;
        entry.offset = offset;
        entry.storedBytes = entry.components*sizeof(uint64_t);
        entry.rawBytes = 0;
        for(uint32_t component = 0; component < entry.components; ++component){
            entry.storedBytes += planes[first + component].storedBytes;
            entry.rawBytes += planes[first + component].rawBytes;
        }
        offset += entry.storedBytes;
        first += entry.components;
    }
    int file = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if(file < 0){
        fail(path + ": " + std::strerror(errno));
        return false;
    }
    Xxh64 hash;
    auto put = [&](const void* bytes, size_t length){
        hash.update(bytes, length);
        return writeAll(file, bytes, length);
    };
    bool written = put(&header, sizeof(header)) && put(entries.data(), entries.size()*sizeof(ShardAttribute));
    first = 0;
    for(size_t attribute = 0; attribute < layout.size() && written; ++attribute){
        for(int component = 0; component < layout[attribute].components && written; ++component){
            written = put(&planes[first + component].storedBytes, sizeof(uint64_t));
        }
        for(int component = 0; component < layout[attribute].components && written; ++component){
            written = put(planes[first + component].stored, planes[first + component].storedBytes);
        }
        first += layout[attribute].components;
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
        if(const Json* acceleration = json.find("acceleration")){
            record.acceleration = acceleration->number;
        }
        return true;
    }
    catch(const std::exception&){
        return false;
    }
}

//the shards' list in a commit record
std::string CacheWriter::shardsJson(const std::vector<ShardRecord>& records, const char* base) const{
    std::string text;
    for(size_t i = 0; i < records.size(); ++i){
        const ShardRecord& record = records[i];
        text += std::string(i ? "," : "") + "\n  {\"file\":\"" + shardName(base, record.rank, "f2p") + "\",\"rank\":" + std::to_string(record.rank) + ",\"particles\":" +
                std::to_string(record.particles) + ",\"bytes\":" + std::to_string(record.bytes) + ",\"xxh64\":\"" + hex(record.hash) + "\",\"low\":" + triple(record.low) +
                ",\"high\":" + triple(record.high) + "}";
    }
    return text;
}

//the frame's commit record, written last: the frame is in the cache once it's down
//the whitewater's part of a commit record or a checkpoint's: how many there are, more (a checkpoint's next id), and their shards. Nothing without whitewater
std::string CacheWriter::whitewaterJson(const std::vector<ShardRecord>& whitewater, const std::string& more) const{
    if(description.whitewaterPerVoxel <= 0){
        return "";
    }
    uint64_t total = 0;
    for(const ShardRecord& record : whitewater){
        total += record.particles;
    }
    return ",\n \"whitewater\":{\"particles\":" + std::to_string(total) + more + ",\"shards\":[" + shardsJson(whitewater, "whitewater") + "]}";
}

bool CacheWriter::commit(const std::string& frameDirectory, int frame, double time, const std::vector<ShardRecord>& records, const std::vector<ShardRecord>& whitewater){
    uint64_t total = 0;
    for(const ShardRecord& record : records){
        total += record.particles;
    }
    std::string text = "{\"flip2\":\"commit\",\"version\":1,\"frame\":" + std::to_string(frame) + ",\"time\":" + number(time) + ",\"particles\":" + std::to_string(total) +
                       ",\"shards\":[" + shardsJson(records, "particles") + "]" + whitewaterJson(whitewater) + "}\n";
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

//a checkpoint's record, written last as a frame's is: what flip2 resume needs besides the particles, which are in the state shards
bool CacheWriter::commitCheckpoint(const std::string& checkpointDirectory, const Job& job, const std::vector<ShardRecord>& records, const std::vector<ShardRecord>& whitewater){
    uint64_t total = 0;
    for(const ShardRecord& record : records){
        total += record.particles;
    }
    std::string planes;
    for(size_t i = 0; i < description.partitionPlanes.size(); ++i){
        planes += (i ? "," : "") + std::to_string(description.partitionPlanes[i]);
    }
    double acceleration = 0.0;  //every rank's largest, as the next substep takes it
    for(const ShardRecord& record : records){
        acceleration = std::max(acceleration, record.acceleration);
    }
    std::string text = "{\"flip2\":\"checkpoint\",\"version\":1,\"frame\":" + std::to_string(job.frame) + ",\"time\":" + number(job.state.time) + ",\"substep\":" +
                       std::to_string(job.state.substep) + ",\"apic\":" + (job.state.apic ? "true" : "false") + ",\"nextId\":" + std::to_string(job.state.nextId) + ",\"acceleration\":" + number(acceleration) +
                       ",\"particles\":" + std::to_string(total) +
                       ",\n \"worldSize\":" + std::to_string(description.worldSize) + ",\"partitionPlanes\":[" + planes + "],\"sceneXxh64\":\"" + description.sceneHash +
                       "\",\"build\":" + quoted(FLIP2_BUILD_ID) + ",\n \"states\":[" + shardsJson(records, "state") + "]" + whitewaterJson(whitewater, ",\"nextId\":" + std::to_string(job.state.whitewaterNextId)) + "}\n";
    std::string why;
    if(!replaceFile(checkpointDirectory + "/ckpt.json", text, why)){
        fail(why);
        return false;
    }
    if(!syncDirectory(directory + "/checkpoints")){
        fail(directory + "/checkpoints: flushing it: " + std::strerror(errno));
        return false;
    }
    lastCheckpoint = std::max(lastCheckpoint, job.frame);
    return true;
}

//deletes all but the newest keepCheckpoints checkpoints, and any unfinished one older than the newest (one a crash left); not newer ones, which other
//ranks may still be writing
void CacheWriter::pruneCheckpoints(){
    std::vector<std::pair<int, bool>> found;    //frame, whether it's committed
    std::error_code listed;
    for(const auto& entry : std::filesystem::directory_iterator(directory + "/checkpoints", listed)){
        std::string name = entry.path().filename().string();
        if(entry.is_directory() && !name.empty() && std::all_of(name.begin(), name.end(), ::isdigit)){
            found.push_back({std::atoi(name.c_str()), std::filesystem::exists(entry.path() / "ckpt.json")});
        }
    }
    std::sort(found.begin(), found.end(), [](const std::pair<int, bool>& a, const std::pair<int, bool>& b){ return a.first > b.first; });
    int kept = 0;
    for(const auto& [frame, committed] : found){
        bool keep = committed ? ++kept <= description.keepCheckpoints : frame > lastCheckpoint;
        if(!keep){
            std::error_code removed;
            std::filesystem::remove_all(directory + "/checkpoints/" + frameName(frame), removed);
        }
    }
}

std::string CacheWriter::cacheJson() const{
    const CacheDescription& d = description;
    std::string planes;
    for(size_t i = 0; i < d.partitionPlanes.size(); ++i){
        planes += (i ? "," : "") + std::to_string(d.partitionPlanes[i]);
    }
    return "{\"flip2\":\"cache\",\"version\":1,\n"
           " \"scene\":{\"path\":" + quoted(d.scenePath) + ",\"xxh64\":\"" + d.sceneHash + "\"},\n"
           " \"fps\":" + number(d.fps) + ",\"frames\":" + std::to_string(d.frames) + ",\"committed\":" + std::to_string(lastCommitted) + ",\"checkpoint\":" +
           std::to_string(lastCheckpoint) + ",\n"
           " \"worldSize\":" + std::to_string(d.worldSize) + ",\"partitionPlanes\":[" + planes + "],\n"
           " \"nodes\":[" + std::to_string(d.nodes[0]) + "," + std::to_string(d.nodes[1]) + "," + std::to_string(d.nodes[2]) + "],\"nodeSize\":" + number(d.nodeSize) +
           ",\"voxelSize\":" + number(d.voxelSize) + ",\"domainMin\":[" + number(d.domainMin[0]) + "," + number(d.domainMin[1]) + "," + number(d.domainMin[2]) + "],\n"
           " \"gpu\":" + quoted(d.gpu) + ",\"sm\":" + std::to_string(d.sm) + ",\"cudaRuntime\":" + std::to_string(d.cudaRuntime) + ",\"build\":" + quoted(FLIP2_BUILD_ID) + ",\n"
           " \"shards\":{\"format\":\"f2p\",\"version\":1,\"compression\":\"" + (d.compression == "none" ? std::string("none") : "blosc-" + d.compression) + "\","
           "\"attributes\":[{\"name\":\"P\",\"type\":\"float32\",\"components\":3},{\"name\":\"v\",\"type\":\"float32\",\"components\":3}" +
           (d.ids ? ",{\"name\":\"id\",\"type\":\"uint64\",\"components\":1}" : "") + (d.ages ? ",{\"name\":\"age\",\"type\":\"float32\",\"components\":1}" : "") + "]}" +
           (d.whitewaterPerVoxel > 0 ? ",\n \"whitewater\":{\"perVoxel\":" + std::to_string(d.whitewaterPerVoxel) + "}" : std::string()) + "}\n";
}
