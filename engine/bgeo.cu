//Copyright 2023 Aberrant Behavior LLC

//Houdini's .bgeo is its JSON geometry schema in its binary JSON encoding (Houdini 12 on). The reference for both is in every Houdini install:
//$HFS/houdini/public/binary_json/binary_json.py (the encoding) and $HFS/houdini/public/hgeo/hgeo.py (the schema). The file starts with a magic number,
//then each JSON value is led by a byte saying what follows. Strings are defined once as numbered tokens and referred to by number after that, as Houdini
//writes them; a uniform array is one type byte for all its values, which then follow packed. A point cloud is
//
//  ["fileversion", ..., "hasindex", false, "pointcount", n, "vertexcount", 0, "primitivecount", 0, "info", {...},
//   "topology", ["pointref", ["indices", []]],
//   "attributes", ["pointattributes", [[definition, values], ...]],
//   "primitives", []]
//
//with P first and the other attributes after it in alphabetical order, as Houdini has them: age (a float), id (an int32, marked as an integer that isn't
//to be blended, as Houdini marks its own ids) and v. Each attribute's values are a page of raw data: every point's tuple in order, x, y, z interleaved, in pages of 1024 points (Houdini's own page
//size), none of them flagged constant. That's what Houdini 22 writes for a point cloud, down to the keys' order. A mesh of quads adds its vertices, each
//naming its point, to the topology's indices, a quad's 4 after another's, and one run of polygons to the primitives:
//
//  "primitives", [[["type", "p_r"], ["s_v", 0, "n_p", quads, "r_v", [4, quads]]]]
//
//which is how Houdini 22 writes a run of polygons with 4 vertices each in binary: its first vertex, how many there are, and their vertex counts' runs.
//
//A path ending .sc is written compressed, as Houdini's .bgeo.sc (Blosc-compressed geometry): "scf1" and 8 bytes of 0; then the file as it would be
//uncompressed, in chunks of 1 MiB (the last one shorter), each a Blosc chunk; then an index of big-endian 64-bit numbers: where each chunk after the
//first starts, counted from the end of those 12 bytes, the chunk size, the last chunk's size and the index's own size so far; and last, "1fcs". That's
//how Houdini 22 writes one, its chunks LZ4 with a byte shuffle of 4, as these are: zstd makes them a few percent smaller, but Houdini twice as slow to read
#include "bgeo.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <unordered_map>
#ifdef FLIP2_WITH_BLOSC
#include <blosc.h>
#endif

namespace{

enum : unsigned char{       //binary_json.py's JID_ values
    ID_MAP_BEGIN = 0x7b, ID_MAP_END = 0x7d, ID_ARRAY_BEGIN = 0x5b, ID_ARRAY_END = 0x5d, ID_INT8 = 0x11, ID_INT16 = 0x12, ID_INT32 = 0x13, ID_INT64 = 0x14,
    ID_REAL32 = 0x19, ID_REAL64 = 0x1a, ID_FALSE = 0x30, ID_TRUE = 0x31, ID_TOKEN_DEFINE = 0x2b, ID_TOKEN_REFERENCE = 0x26, ID_UNIFORM_ARRAY = 0x40,
    ID_MAGIC = 0x7f
};
const uint32_t BINARY_MAGIC = 0x624a534e;   //"NSJb" in a little-endian file
const int PAGE_SIZE = 1024;

//where a file's bytes go: straight to it, or through Houdini's .sc compression
class Sink{
public:
    virtual ~Sink() = default;
    virtual void write(const void* data, size_t bytes) = 0;
    virtual bool finish() = 0;      //writes whatever it holds back; whether everything got written
};

class PlainSink : public Sink{
public:
    explicit PlainSink(std::FILE* file) : file(file), buffer(1 << 22){
        std::setvbuf(file, buffer.data(), _IOFBF, buffer.size());
    }
    void write(const void* data, size_t bytes) override{
        ok = ok && std::fwrite(data, 1, bytes, file) == bytes;
    }
    bool finish() override{
        return ok && std::fflush(file) == 0;
    }
private:
    std::FILE* file;
    std::vector<char> buffer;
    bool ok = true;
};

#ifdef FLIP2_WITH_BLOSC

//the .sc framing (above): the bytes held back until there's a batch of chunks, which are compressed at once, a thread each
class CompressingSink : public Sink{
public:
    static const size_t CHUNK = 1 << 20;
    static const size_t BATCH = 32;     //chunks compressed together

    explicit CompressingSink(std::FILE* file) : file(file){
        static const char magic[12] = {'s', 'c', 'f', '1', 0, 0, 0, 0, 0, 0, 0, 0};
        put(magic, sizeof(magic));
        pending.reserve(BATCH*CHUNK);
    }

    void write(const void* data, size_t bytes) override{
        const char* from = (const char*)data;
        while(bytes > 0){
            size_t taken = std::min(bytes, BATCH*CHUNK - pending.size());
            pending.insert(pending.end(), from, from + taken);
            from += taken;
            bytes -= taken;
            if(pending.size() == BATCH*CHUNK){
                compress();
            }
        }
    }

    bool finish() override{
        compress();
        for(uint64_t start : starts){
            putBigEndian(start);
        }
        putBigEndian(CHUNK);
        putBigEndian(lastChunk);
        putBigEndian(8*starts.size() + 16);
        put("1fcs", 4);
        return ok && std::fflush(file) == 0;
    }

private:
    //compresses and writes every byte held back, in chunks
    void compress(){
        size_t chunks = (pending.size() + CHUNK - 1) / CHUNK;
        std::vector<std::vector<char>> compressed(chunks);
        std::vector<int> sizes(chunks);
        #pragma omp parallel for schedule(dynamic, 1)
        for(long chunk = 0; chunk < (long)chunks; ++chunk){
            size_t bytes = std::min(CHUNK, pending.size() - chunk*CHUNK);
            compressed[chunk].resize(bytes + BLOSC_MAX_OVERHEAD);
            sizes[chunk] = blosc_compress_ctx(5, BLOSC_SHUFFLE, 4, bytes, pending.data() + chunk*CHUNK, compressed[chunk].data(), compressed[chunk].size(), "lz4",
                                              0, 1);
        }
        for(size_t chunk = 0; chunk < chunks; ++chunk){
            if(sizes[chunk] <= 0){
                ok = false;
                return;
            }
            if(chunkCount > 0){
                starts.push_back(written);
            }
            put(compressed[chunk].data(), sizes[chunk]);
            written += sizes[chunk];
            lastChunk = std::min(CHUNK, pending.size() - chunk*CHUNK);
            ++chunkCount;
        }
        pending.clear();
    }

    void put(const void* data, size_t bytes){
        ok = ok && std::fwrite(data, 1, bytes, file) == bytes;
    }

    void putBigEndian(uint64_t value){
        unsigned char bytes[8];
        for(int byte = 0; byte < 8; ++byte){
            bytes[byte] = (unsigned char)(value >> (56 - 8*byte));
        }
        put(bytes, 8);
    }

    std::FILE* file;
    std::vector<char> pending;
    std::vector<uint64_t> starts;   //each chunk's after the first, from the end of the magic number
    uint64_t written = 0;           //bytes of chunks written
    uint64_t lastChunk = 0;         //the last chunk's size, uncompressed
    size_t chunkCount = 0;
    bool ok = true;
};

#endif

//writes binary JSON, little-endian
class BinaryJson{
public:
    explicit BinaryJson(Sink& sink) : sink(sink){
        id(ID_MAGIC);
        raw(&BINARY_MAGIC, sizeof(BINARY_MAGIC));
    }

    void beginArray(){ id(ID_ARRAY_BEGIN); }
    void endArray(){ id(ID_ARRAY_END); }
    void beginMap(){ id(ID_MAP_BEGIN); }
    void endMap(){ id(ID_MAP_END); }
    void boolean(bool value){ id(value ? ID_TRUE : ID_FALSE); }

    void integer(long long value){     //in the fewest bytes that hold it, as Houdini writes them
        if(value >= -0x80 && value < 0x80){
            int8_t small = (int8_t)value;
            id(ID_INT8);
            raw(&small, 1);
        }
        else if(value >= -0x8000 && value < 0x8000){
            int16_t medium = (int16_t)value;
            id(ID_INT16);
            raw(&medium, 2);
        }
        else if(value >= -0x80000000LL && value < 0x80000000LL){
            int32_t large = (int32_t)value;
            id(ID_INT32);
            raw(&large, 4);
        }
        else{
            int64_t huge = value;
            id(ID_INT64);
            raw(&huge, 8);
        }
    }

    void real(double value){
        id(ID_REAL64);
        raw(&value, sizeof(value));
    }

    void string(const std::string& text){  //a token: defined the first time, referred to by number every time
        auto found = tokens.find(text);
        uint64_t number;
        if(found == tokens.end()){
            number = tokens.size();
            tokens.emplace(text, number);
            id(ID_TOKEN_DEFINE);
            length(number);
            length(text.size());
            raw(text.data(), text.size());
        }
        else{
            number = found->second;
        }
        id(ID_TOKEN_REFERENCE);
        length(number);
    }

    //a uniform array of count float32s, whose values the caller then writes with raw()
    void beginFloats(uint64_t count){
        id(ID_UNIFORM_ARRAY);
        id(ID_REAL32);
        length(count);
    }

    //a uniform array of count int32s, likewise
    void beginInts(uint64_t count){
        id(ID_UNIFORM_ARRAY);
        id(ID_INT32);
        length(count);
    }

    void raw(const void* data, size_t bytes){
        if(bytes > 0){
            sink.write(data, bytes);
        }
    }

private:
    void id(unsigned char value){
        raw(&value, 1);
    }

    void length(uint64_t value){    //one byte below 0xf1; otherwise a byte saying how many follow
        if(value < 0xf1){
            unsigned char small = (unsigned char)value;
            raw(&small, 1);
        }
        else if(value < 0xffff){
            uint16_t medium = (uint16_t)value;
            id(0xf2);
            raw(&medium, 2);
        }
        else if(value < 0xffffffffull){
            uint32_t large = (uint32_t)value;
            id(0xf4);
            raw(&large, 4);
        }
        else{
            int64_t huge = (int64_t)value;
            id(0xf8);
            raw(&huge, 8);
        }
    }

    Sink& sink;
    std::unordered_map<std::string, uint64_t> tokens;
};

//an attribute's three float32 planes from a shard, or nullptr if it hasn't them
const float* planesOf(const ShardData& shard, const char* name){
    const ShardData::Attribute* attribute = shard.find(name);
    if(attribute == nullptr || attribute->type != 1 || attribute->components != 3 || attribute->planes.size() != 3*sizeof(float)*shard.particles){
        return nullptr;
    }
    return (const float*)attribute->planes.data();
}

//an attribute's one plane of a type (1 float32, 3 uint64) from a shard, or nullptr if it hasn't it
const void* planeOf(const ShardData& shard, const char* name, uint32_t type){
    const ShardData::Attribute* attribute = shard.find(name);
    if(attribute == nullptr || attribute->type != type || attribute->components != 1 || attribute->planes.size() != (type == 1 ? 4 : 8)*shard.particles){
        return nullptr;
    }
    return attribute->planes.data();
}

//writes path through write(json), compressed if it ends .sc: to path.tmp, moved into place once it's whole, so nothing reading it ever sees half of one.
//Says why not if it can't
template<typename Write>
bool writeFile(const std::string& path, std::string& why, Write write){
    std::string temporary = path + ".tmp";
    std::FILE* file = std::fopen(temporary.c_str(), "wb");
    if(file == nullptr){
        why = "can't write " + temporary + ": " + std::strerror(errno);
        return false;
    }
    bool compressed = path.size() > 3 && path.compare(path.size() - 3, 3, ".sc") == 0;
    std::unique_ptr<Sink> sink;
    if(!compressed){
        sink.reset(new PlainSink(file));
    }
    else{
#ifdef FLIP2_WITH_BLOSC
        sink.reset(new CompressingSink(file));
#else
        std::fclose(file);
        std::remove(temporary.c_str());
        why = "this build of flip2 has no blosc, so it can't write .sc files";
        return false;
#endif
    }
    BinaryJson json(*sink);
    write(json);
    bool written = sink->finish();
    sink.reset();
    written = std::fclose(file) == 0 && written;
    if(!written){
        why = "couldn't write " + temporary;
        std::remove(temporary.c_str());
        return false;
    }
    std::error_code renamed;
    std::filesystem::rename(temporary, path, renamed);
    if(renamed){
        why = "couldn't move " + temporary + " into place: " + renamed.message();
        return false;
    }
    return true;
}

//the file's start, from its version through its info: the counts, what wrote it and, if there are points, their bounds
void writeHeader(BinaryJson& json, uint64_t points, uint64_t vertices, uint64_t primitives, const std::string& software, const float low[3], const float high[3]){
    json.string("fileversion");
    json.string("19.0.0");      //what reads it: the layout is Houdini 22's, unchanged since well before 19
    json.string("hasindex");
    json.boolean(false);
    json.string("pointcount");
    json.integer((long long)points);
    json.string("vertexcount");
    json.integer((long long)vertices);
    json.string("primitivecount");
    json.integer((long long)primitives);
    json.string("info");
    json.beginMap();
    json.string("software");
    json.string(software);
    if(points > 0){
        json.string("bounds");     //x, then y, then z: low and high
        json.beginArray();
        for(int axis = 0; axis < 3; ++axis){
            json.real(low[axis]);
            json.real(high[axis]);
        }
        json.endArray();
    }
    json.endMap();
}

//a point attribute of size values per point, float32s or with integers int32s, whose values, every point's in order, the caller then writes with raw()
//before endVector(). kind is what Houdini should take it for, which tells it how transforms and blends treat it: "point" (P), "vector" (v),
//"nonarithmetic_integer" (id), or nullptr for a plain number
void beginAttribute(BinaryJson& json, const char* name, uint64_t points, int size, bool integers, const char* kind){
    json.beginArray();
    json.beginArray();      //the definition
    json.string("scope");
    json.string("public");
    json.string("type");
    json.string("numeric");
    json.string("name");
    json.string(name);
    json.string("options");
    json.beginMap();
    if(kind != nullptr){
        json.string("type");
        json.beginMap();
        json.string("type");
        json.string("string");
        json.string("value");
        json.string(kind);
        json.endMap();
    }
    json.endMap();
    json.endArray();
    json.beginArray();      //the values
    json.string("size");
    json.integer(size);
    json.string("storage");
    json.string(integers ? "int32" : "fpreal32");
    json.string("defaults");
    json.beginArray();
    json.string("size");
    json.integer(1);
    json.string("storage");
    json.string(integers ? "int64" : "fpreal64");
    json.string("values");
    json.beginArray();
    if(integers){
        json.integer(0);
    }
    else{
        json.real(0.0);
    }
    json.endArray();
    json.endArray();
    json.string("values");
    json.beginArray();
    json.string("size");
    json.integer(size);
    json.string("storage");
    json.string(integers ? "int32" : "fpreal32");
    json.string("pagesize");
    json.integer(PAGE_SIZE);
    json.string("rawpagedata");
    if(integers){
        json.beginInts(size*points);
    }
    else{
        json.beginFloats(size*points);
    }
}

//a point attribute of 3 float32s per point: P is a position, anything else a direction
void beginVector(BinaryJson& json, const char* name, uint64_t points){
    beginAttribute(json, name, points, 3, false, std::strcmp(name, "P") == 0 ? "point" : "vector");
}

void endVector(BinaryJson& json){
    json.endArray();
    json.endArray();
    json.endArray();
}

}

bool writeParticlesBgeo(const std::string& path, const std::vector<const ShardData*>& shards, const std::string& software, std::string& why){
    uint64_t points = 0;
    float low[3] = {INFINITY, INFINITY, INFINITY}, high[3] = {-INFINITY, -INFINITY, -INFINITY};
    bool ids = true, ages = true;   //written if every shard has them
    for(const ShardData* shard : shards){
        const float* positions = planesOf(*shard, "P");
        if(positions == nullptr || planesOf(*shard, "v") == nullptr){
            why = "a shard without float32 P and v";
            return false;
        }
        ids = ids && planeOf(*shard, "id", 3) != nullptr;
        ages = ages && planeOf(*shard, "age", 1) != nullptr;
        for(int axis = 0; axis < 3; ++axis){
            for(uint64_t particle = 0; particle < shard->particles; ++particle){
                float value = positions[axis*shard->particles + particle];
                low[axis] = std::min(low[axis], value);
                high[axis] = std::max(high[axis], value);
            }
        }
        points += shard->particles;
    }
    return writeFile(path, why, [&](BinaryJson& json){
        json.beginArray();
        writeHeader(json, points, 0, 0, software, low, high);
        json.string("topology");
        json.beginArray();
        json.string("pointref");
        json.beginArray();
        json.string("indices");
        json.beginArray();
        json.endArray();
        json.endArray();
        json.endArray();
        json.string("attributes");
        json.beginArray();
        json.string("pointattributes");
        json.beginArray();
        std::vector<float> tuples;
        std::vector<int32_t> narrow;
        auto scalar = [&](const char* name, bool integers){     //a plane, shard after shard: floats as they are, ids as the int32s Houdini's are
            beginAttribute(json, name, points, 1, integers, integers ? "nonarithmetic_integer" : nullptr);
            for(const ShardData* shard : shards){
                const void* plane = planeOf(*shard, name, integers ? 3 : 1);
                if(integers){
                    narrow.resize(shard->particles);
                    for(uint64_t particle = 0; particle < shard->particles; ++particle){
                        narrow[particle] = (int32_t)((const uint64_t*)plane)[particle];
                    }
                    plane = narrow.data();
                }
                json.raw(plane, 4*shard->particles);
            }
            endVector(json);
        };
        for(const char* name : {"P", "v"}){
            if(std::strcmp(name, "v") == 0){    //between P and v, as Houdini orders them
                if(ages){
                    scalar("age", false);
                }
                if(ids){
                    scalar("id", true);
                }
            }
            beginVector(json, name, points);
            for(const ShardData* shard : shards){   //each shard's planes interleaved into tuples, a block at a time
                const float* planes = planesOf(*shard, name);
                const uint64_t block = 1 << 18;
                for(uint64_t first = 0; first < shard->particles; first += block){
                    uint64_t count = std::min(block, shard->particles - first);
                    tuples.resize(3*count);
                    for(uint64_t particle = 0; particle < count; ++particle){
                        for(int axis = 0; axis < 3; ++axis){
                            tuples[3*particle + axis] = planes[axis*shard->particles + first + particle];
                        }
                    }
                    json.raw(tuples.data(), sizeof(float)*tuples.size());
                }
            }
            endVector(json);
        }
        json.endArray();
        json.endArray();
        json.string("primitives");
        json.beginArray();
        json.endArray();
        json.endArray();
    });
}

bool writeSurfaceBgeo(const std::string& path, const SurfaceMesh& mesh, const std::string& software, std::string& why){
    uint64_t points = mesh.points.size()/3, quads = mesh.quads.size()/4;
    if(mesh.points.size() != 3*points || mesh.velocities.size() != mesh.points.size() || mesh.quads.size() != 4*quads || 4*quads > 0x7fffffffull){
        why = "a malformed mesh";
        return false;
    }
    float low[3] = {INFINITY, INFINITY, INFINITY}, high[3] = {-INFINITY, -INFINITY, -INFINITY};
    for(uint64_t point = 0; point < points; ++point){
        for(int axis = 0; axis < 3; ++axis){
            low[axis] = std::min(low[axis], mesh.points[3*point + axis]);
            high[axis] = std::max(high[axis], mesh.points[3*point + axis]);
        }
    }
    return writeFile(path, why, [&](BinaryJson& json){
        json.beginArray();
        writeHeader(json, points, 4*quads, quads, software, low, high);
        json.string("topology");
        json.beginArray();
        json.string("pointref");
        json.beginArray();
        json.string("indices");     //each quad's points turned around: Houdini has a polygon face the way its points run clockwise
        json.beginInts(4*quads);
        std::vector<int32_t> corners;
        const uint64_t block = 1 << 18;
        for(uint64_t first = 0; first < quads; first += block){
            uint64_t count = std::min(block, quads - first);
            corners.resize(4*count);
            for(uint64_t quad = 0; quad < count; ++quad){
                const int* from = &mesh.quads[4*(first + quad)];
                corners[4*quad] = from[0];
                corners[4*quad + 1] = from[3];
                corners[4*quad + 2] = from[2];
                corners[4*quad + 3] = from[1];
            }
            json.raw(corners.data(), sizeof(int32_t)*corners.size());
        }
        json.endArray();
        json.endArray();
        json.string("attributes");
        json.beginArray();
        json.string("pointattributes");
        json.beginArray();
        beginVector(json, "P", points);
        json.raw(mesh.points.data(), sizeof(float)*mesh.points.size());
        endVector(json);
        beginVector(json, "v", points);
        json.raw(mesh.velocities.data(), sizeof(float)*mesh.velocities.size());
        endVector(json);
        json.endArray();
        json.endArray();
        json.string("primitives");
        json.beginArray();
        if(quads > 0){
            json.beginArray();
            json.beginArray();
            json.string("type");
            json.string("p_r");
            json.endArray();
            json.beginArray();
            json.string("s_v");
            json.integer(0);
            json.string("n_p");
            json.integer((long long)quads);
            json.string("r_v");
            json.beginInts(2);
            int32_t run[2] = {4, (int32_t)quads};
            json.raw(run, sizeof(run));
            json.endArray();
            json.endArray();
        }
        json.endArray();
        json.endArray();
    });
}
