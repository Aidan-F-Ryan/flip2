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
//and each attribute's values are a page of raw data: every point's tuple in order, x, y, z interleaved, in pages of 1024 points (Houdini's own page
//size), none of them flagged constant. That's what Houdini 22 writes for a point cloud, down to the keys' order. A mesh of quads adds its vertices, each
//naming its point, to the topology's indices, a quad's 4 after another's, and one run of polygons to the primitives:
//
//  "primitives", [[["type", "p_r"], ["s_v", 0, "n_p", quads, "r_v", [4, quads]]]]
//
//which is how Houdini 22 writes a run of polygons with 4 vertices each in binary: its first vertex, how many there are, and their vertex counts' runs
#include "bgeo.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <unordered_map>

namespace{

enum : unsigned char{       //binary_json.py's JID_ values
    ID_MAP_BEGIN = 0x7b, ID_MAP_END = 0x7d, ID_ARRAY_BEGIN = 0x5b, ID_ARRAY_END = 0x5d, ID_INT8 = 0x11, ID_INT16 = 0x12, ID_INT32 = 0x13, ID_INT64 = 0x14,
    ID_REAL32 = 0x19, ID_REAL64 = 0x1a, ID_FALSE = 0x30, ID_TRUE = 0x31, ID_TOKEN_DEFINE = 0x2b, ID_TOKEN_REFERENCE = 0x26, ID_UNIFORM_ARRAY = 0x40,
    ID_MAGIC = 0x7f
};
const uint32_t BINARY_MAGIC = 0x624a534e;   //"NSJb" in a little-endian file
const int PAGE_SIZE = 1024;

//writes binary JSON to a file, little-endian
class BinaryJson{
public:
    explicit BinaryJson(std::FILE* file) : file(file){
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
        if(ok && bytes > 0){
            ok = std::fwrite(data, 1, bytes, file) == bytes;
        }
    }

    bool good() const{
        return ok;
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

    std::FILE* file;
    std::unordered_map<std::string, uint64_t> tokens;
    bool ok = true;
};

//an attribute's three float32 planes from a shard, or nullptr if it hasn't them
const float* planesOf(const ShardData& shard, const char* name){
    const ShardData::Attribute* attribute = shard.find(name);
    if(attribute == nullptr || attribute->type != 1 || attribute->components != 3 || attribute->planes.size() != 3*sizeof(float)*shard.particles){
        return nullptr;
    }
    return (const float*)attribute->planes.data();
}

//writes path through write(json): to path.tmp, moved into place once it's whole, so nothing reading it ever sees half of one. Says why not if it can't
template<typename Write>
bool writeFile(const std::string& path, std::string& why, Write write){
    std::string temporary = path + ".tmp";
    std::FILE* file = std::fopen(temporary.c_str(), "wb");
    if(file == nullptr){
        why = "can't write " + temporary + ": " + std::strerror(errno);
        return false;
    }
    std::vector<char> buffer(1 << 22);
    std::setvbuf(file, buffer.data(), _IOFBF, buffer.size());
    BinaryJson json(file);
    write(json);
    bool written = json.good() && std::fflush(file) == 0;
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

//a point attribute of 3 float32s per point, whose values, every point's tuple in order, the caller then writes with raw() before endVector()
void beginVector(BinaryJson& json, const char* name, uint64_t points){
    json.beginArray();
    json.beginArray();      //the definition: P is a position, v a direction, which tells Houdini how transforms move them
    json.string("scope");
    json.string("public");
    json.string("type");
    json.string("numeric");
    json.string("name");
    json.string(name);
    json.string("options");
    json.beginMap();
    json.string("type");
    json.beginMap();
    json.string("type");
    json.string("string");
    json.string("value");
    json.string(std::strcmp(name, "P") == 0 ? "point" : "vector");
    json.endMap();
    json.endMap();
    json.endArray();
    json.beginArray();      //the values
    json.string("size");
    json.integer(3);
    json.string("storage");
    json.string("fpreal32");
    json.string("defaults");
    json.beginArray();
    json.string("size");
    json.integer(1);
    json.string("storage");
    json.string("fpreal64");
    json.string("values");
    json.beginArray();
    json.real(0.0);
    json.endArray();
    json.endArray();
    json.string("values");
    json.beginArray();
    json.string("size");
    json.integer(3);
    json.string("storage");
    json.string("fpreal32");
    json.string("pagesize");
    json.integer(PAGE_SIZE);
    json.string("rawpagedata");
    json.beginFloats(3*points);
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
    for(const ShardData* shard : shards){
        const float* positions = planesOf(*shard, "P");
        if(positions == nullptr || planesOf(*shard, "v") == nullptr){
            why = "a shard without float32 P and v";
            return false;
        }
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
        for(const char* name : {"P", "v"}){
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
