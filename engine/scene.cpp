//Copyright 2023 Aberrant Behavior LLC

#include "scene.hpp"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

//Just enough JSON for scene files: objects, arrays, numbers, strings, true, false and null
struct Json{
    enum Kind{NUL, BOOLEAN, NUMBER, STRING, ARRAY, OBJECT};
    Kind kind = NUL;
    bool boolean = false;
    double number = 0.0;
    std::string text;
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> fields;     //in file order

    const Json* find(const std::string& key) const{
        for(const auto& field : fields){
            if(field.first == key){
                return &field.second;
            }
        }
        return nullptr;
    }
};

class JsonReader{
public:
    JsonReader(const std::string& text, const std::string& name)
    : text(text)
    , name(name)
    {}

    Json document(){
        Json value = parse();
        skipSpace();
        if(at < text.size()){
            fail("more after the end of the document");
        }
        return value;
    }

private:
    [[noreturn]] void fail(const std::string& what){
        int line = 1;
        for(size_t i = 0; i < at && i < text.size(); ++i){
            line += text[i] == '\n';
        }
        throw std::runtime_error(name + ":" + std::to_string(line) + ": " + what);
    }

    void skipSpace(){
        while(at < text.size() && (text[at] == ' ' || text[at] == '\t' || text[at] == '\n' || text[at] == '\r')){
            ++at;
        }
    }

    bool consume(const char* word){
        size_t length = std::char_traits<char>::length(word);
        if(text.compare(at, length, word) == 0){
            at += length;
            return true;
        }
        return false;
    }

    Json parse(){
        skipSpace();
        if(at >= text.size()){
            fail("unexpected end of file");
        }
        Json value;
        char c = text[at];
        if(c == '{'){
            value.kind = Json::OBJECT;
            ++at;
            skipSpace();
            if(at < text.size() && text[at] == '}'){
                ++at;
                return value;
            }
            while(true){
                skipSpace();
                if(at >= text.size() || text[at] != '"'){
                    fail("expected a quoted key");
                }
                std::string key = string();
                skipSpace();
                if(at >= text.size() || text[at] != ':'){
                    fail("expected ':' after \"" + key + "\"");
                }
                ++at;
                value.fields.emplace_back(key, parse());
                skipSpace();
                if(at < text.size() && text[at] == ','){
                    ++at;
                    continue;
                }
                if(at < text.size() && text[at] == '}'){
                    ++at;
                    return value;
                }
                fail("expected ',' or '}' in an object");
            }
        }
        if(c == '['){
            value.kind = Json::ARRAY;
            ++at;
            skipSpace();
            if(at < text.size() && text[at] == ']'){
                ++at;
                return value;
            }
            while(true){
                value.items.push_back(parse());
                skipSpace();
                if(at < text.size() && text[at] == ','){
                    ++at;
                    continue;
                }
                if(at < text.size() && text[at] == ']'){
                    ++at;
                    return value;
                }
                fail("expected ',' or ']' in an array");
            }
        }
        if(c == '"'){
            value.kind = Json::STRING;
            value.text = string();
            return value;
        }
        if(consume("true")){
            value.kind = Json::BOOLEAN;
            value.boolean = true;
            return value;
        }
        if(consume("false")){
            value.kind = Json::BOOLEAN;
            return value;
        }
        if(consume("null")){
            return value;
        }
        const char* start = text.c_str() + at;
        char* end;
        value.number = std::strtod(start, &end);
        if(end == start){
            fail(std::string("unexpected '") + c + "'");
        }
        value.kind = Json::NUMBER;
        at += end - start;
        return value;
    }

    std::string string(){    //at the opening quote
        std::string out;
        ++at;
        while(at < text.size() && text[at] != '"'){
            char c = text[at++];
            if(c == '\\' && at < text.size()){
                char escaped = text[at++];
                switch(escaped){
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u':   //only the ASCII range: scene files are paths and names
                        if(at + 4 > text.size()){
                            fail("a short \\u escape");
                        }
                        out += (char)std::strtol(text.substr(at, 4).c_str(), nullptr, 16);
                        at += 4;
                        break;
                    default: out += escaped;
                }
            }
            else{
                out += c;
            }
        }
        if(at >= text.size()){
            fail("a string that never ends");
        }
        ++at;
        return out;
    }

    const std::string& text;
    std::string name;
    size_t at = 0;
};

//reads a scene's values, complaining in terms of where they are in the file
class SceneReader{
public:
    explicit SceneReader(const std::string& name)
    : name(name)
    {}

    [[noreturn]] void fail(const std::string& where, const std::string& what) const{
        throw std::runtime_error(name + ": " + where + ": " + what);
    }

    //warns about any key of object not in known
    void checkKeys(const Json& object, const std::string& where, std::initializer_list<const char*> known) const{
        for(const auto& field : object.fields){
            bool found = false;
            for(const char* key : known){
                found = found || field.first == key;
            }
            if(!found){
                std::cerr<<name<<": "<<where<<": ignoring \""<<field.first<<"\", which flip2 doesn't know\n";
            }
        }
    }

    double number(const Json& object, const char* key, const std::string& where, double fallback) const{
        const Json* value = object.find(key);
        if(value == nullptr){
            return fallback;
        }
        if(value->kind != Json::NUMBER){
            fail(where + "." + key, "should be a number");
        }
        return value->number;
    }

    void vector3(const Json& object, const char* key, const std::string& where, double out[3], bool required = false) const{
        const Json* value = object.find(key);
        if(value == nullptr){
            if(required){
                fail(where, std::string("needs \"") + key + "\"");
            }
            return;
        }
        if(value->kind != Json::ARRAY || value->items.size() != 3){
            fail(where + "." + key, "should be 3 numbers, [x, y, z]");
        }
        for(int axis = 0; axis < 3; ++axis){
            if(value->items[axis].kind != Json::NUMBER){
                fail(where + "." + key, "should be 3 numbers, [x, y, z]");
            }
            out[axis] = value->items[axis].number;
        }
    }

    std::string text(const Json& object, const char* key, const std::string& where, const std::string& fallback) const{
        const Json* value = object.find(key);
        if(value == nullptr){
            return fallback;
        }
        if(value->kind != Json::STRING){
            fail(where + "." + key, "should be a string");
        }
        return value->text;
    }

    bool flag(const Json& object, const char* key, const std::string& where, bool fallback) const{
        const Json* value = object.find(key);
        if(value == nullptr){
            return fallback;
        }
        if(value->kind != Json::BOOLEAN){
            fail(where + "." + key, "should be true or false");
        }
        return value->boolean;
    }

    const Json* object(const Json& parent, const char* key, const std::string& where) const{
        const Json* value = parent.find(key);
        if(value != nullptr && value->kind != Json::OBJECT){
            fail(where + "." + key, "should be an object, {...}");
        }
        return value;
    }

    const Json* array(const Json& parent, const char* key, const std::string& where) const{
        const Json* value = parent.find(key);
        if(value != nullptr && value->kind != Json::ARRAY){
            fail(where + "." + key, "should be an array, [...]");
        }
        return value;
    }

    std::string name;
};

bool SceneShape::contains(const double point[3]) const{
    if(kind == SPHERE){
        double squared = 0.0;
        for(int axis = 0; axis < 3; ++axis){
            squared += (point[axis] - centre[axis])*(point[axis] - centre[axis]);
        }
        return squared < radius*radius;
    }
    for(int axis = 0; axis < 3; ++axis){
        if(point[axis] < min[axis] || point[axis] >= max[axis]){
            return false;
        }
    }
    return true;
}

void SceneShape::bounds(double low[3], double high[3]) const{
    for(int axis = 0; axis < 3; ++axis){
        low[axis] = kind == SPHERE ? centre[axis] - radius : min[axis];
        high[axis] = kind == SPHERE ? centre[axis] + radius : max[axis];
    }
}

Scene loadScene(const std::string& path){
    std::ifstream file(path);
    if(!file){
        throw std::runtime_error(path + ": can't open it");
    }
    std::stringstream contents;
    contents<<file.rdbuf();
    std::string text = contents.str();
    Json root = JsonReader(text, path).document();
    SceneReader read(path);
    if(root.kind != Json::OBJECT){
        read.fail("the file", "should be one JSON object, {...}");
    }
    read.checkKeys(root, "the scene", {"schema", "fps", "frames", "domain", "solver", "gravity", "particlesPerVoxel", "seed", "fluids", "emitters", "sinks", "forces", "partitions", "devices", "output"});
    std::string schema = read.text(root, "schema", "the scene", "flip2.scene/1");
    if(schema != "flip2.scene/1"){
        read.fail("schema", "is \"" + schema + "\"; this flip2 reads \"flip2.scene/1\"");
    }
    Scene scene;
    scene.path = path;
    scene.fps = read.number(root, "fps", "the scene", scene.fps);
    scene.frames = (int)read.number(root, "frames", "the scene", scene.frames);
    if(scene.fps <= 0.0 || scene.frames < 0){
        read.fail("the scene", "fps has to be positive and frames at least 0");
    }

    if(const Json* domain = read.object(root, "domain", "the scene")){
        read.checkKeys(*domain, "domain", {"min", "max", "voxelSize", "open"});
        double low[3], high[3];
        read.vector3(*domain, "min", "domain", low, true);
        read.vector3(*domain, "max", "domain", high, true);
        double voxelSize = read.number(*domain, "voxelSize", "domain", 0.0);
        if(voxelSize <= 0.0){
            read.fail("domain", "needs a positive \"voxelSize\"");
        }
        scene.nodeSize = 4.0*voxelSize;
        for(int axis = 0; axis < 3; ++axis){
            if(high[axis] <= low[axis]){
                read.fail("domain", "max has to be above min on every axis");
            }
            scene.domainMin[axis] = low[axis];
            //whole nodes, at least 2 per axis (a partition needs 2 node planes); a hair of rounding doesn't add a node
            double nodes = std::ceil((high[axis] - low[axis]) / scene.nodeSize - 1e-9);
            scene.nodes[axis] = (unsigned int)std::max(2.0, nodes);
        }
        if(const Json* open = read.array(*domain, "open", "domain")){
            const char* faces[6] = {"-x", "+x", "-y", "+y", "-z", "+z"};
            for(const Json& face : open->items){
                int bit = -1;
                for(int which = 0; which < 6; ++which){
                    bit = face.kind == Json::STRING && face.text == faces[which] ? which : bit;
                }
                if(bit < 0){
                    read.fail("domain.open", "lists faces, each \"-x\", \"+x\", \"-y\", \"+y\", \"-z\" or \"+z\"");
                }
                scene.openFaces |= 1u << bit;
            }
        }
    }

    if(const Json* solver = read.object(root, "solver", "the scene")){
        read.checkKeys(*solver, "solver", {"flipRatio", "cfl", "densityCorrectionTime", "pressureSolver", "advection", "dotProducts"});
        scene.flipRatio = read.number(*solver, "flipRatio", "solver", scene.flipRatio);
        scene.cfl = read.number(*solver, "cfl", "solver", scene.cfl);
        scene.densityCorrectionTime = read.number(*solver, "densityCorrectionTime", "solver", scene.densityCorrectionTime);
        scene.pressureSolver = read.text(*solver, "pressureSolver", "solver", scene.pressureSolver);
        scene.advection = read.text(*solver, "advection", "solver", scene.advection);
        scene.dotProducts = read.text(*solver, "dotProducts", "solver", scene.dotProducts);
        if(scene.pressureSolver != "multigrid" && scene.pressureSolver != "cg" && scene.pressureSolver != "jacobi" && scene.pressureSolver != "sor"){
            read.fail("solver.pressureSolver", "is multigrid, cg, jacobi or sor");
        }
        if(scene.advection != "rk3" && scene.advection != "euler"){
            read.fail("solver.advection", "is rk3 or euler");
        }
        if(scene.dotProducts != "exact" && scene.dotProducts != "blocks"){
            read.fail("solver.dotProducts", "is exact or blocks");
        }
        if(scene.cfl <= 0.0 || scene.flipRatio < 0.0 || scene.flipRatio > 1.0 || scene.densityCorrectionTime < 0.0){
            read.fail("solver", "cfl has to be positive, flipRatio between 0 and 1, and densityCorrectionTime at least 0");
        }
    }

    read.vector3(root, "gravity", "the scene", scene.gravity);
    scene.particlesPerVoxel = (int)read.number(root, "particlesPerVoxel", "the scene", scene.particlesPerVoxel);
    if(scene.particlesPerVoxel != 1 && scene.particlesPerVoxel != 8 && scene.particlesPerVoxel != 27){
        read.fail("particlesPerVoxel", "is 1, 8 or 27: a lattice of 1, 2 or 3 per side");
    }
    scene.seed = (unsigned long long)read.number(root, "seed", "the scene", (double)scene.seed);

    auto readShapes = [&](const char* key, std::vector<SceneShape>& shapes){
        const Json* list = read.array(root, key, "the scene");
        if(list == nullptr){
            return;
        }
        for(size_t index = 0; index < list->items.size(); ++index){
            const Json& item = list->items[index];
            std::string where = std::string(key) + "[" + std::to_string(index) + "]";
            if(item.kind != Json::OBJECT){
                read.fail(where, "should be an object, {...}");
            }
            read.checkKeys(item, where, {"shape", "min", "max", "center", "centre", "radius", "velocity"});
            SceneShape shape;
            std::string kind = read.text(item, "shape", where, "box");
            if(kind == "box"){
                shape.kind = SceneShape::BOX;
                read.vector3(item, "min", where, shape.min, true);
                read.vector3(item, "max", where, shape.max, true);
            }
            else if(kind == "sphere"){
                shape.kind = SceneShape::SPHERE;
                read.vector3(item, item.find("centre") ? "centre" : "center", where, shape.centre, true);
                shape.radius = read.number(item, "radius", where, 0.0);
                if(shape.radius <= 0.0){
                    read.fail(where, "a sphere needs a positive \"radius\"");
                }
            }
            else{
                read.fail(where + ".shape", "is box or sphere, not \"" + kind + "\"");
            }
            read.vector3(item, "velocity", where, shape.velocity);
            shapes.push_back(shape);
        }
    };
    readShapes("fluids", scene.fluids);
    readShapes("emitters", scene.emitters);
    readShapes("sinks", scene.sinks);
    if(scene.emitters.size() > 16 || scene.sinks.size() > 16){
        read.fail("the scene", "can have up to 16 emitters and 16 sinks");
    }

    if(const Json* forces = read.array(root, "forces", "the scene")){
        for(size_t index = 0; index < forces->items.size(); ++index){
            const Json& item = forces->items[index];
            std::string where = "forces[" + std::to_string(index) + "]";
            if(item.kind != Json::OBJECT){
                read.fail(where, "should be an object, {...}");
            }
            SceneForce force;
            std::string type = read.text(item, "type", where, "");
            if(type == "point" || type == "vortex"){
                force.kind = type == "point" ? SceneForce::POINT : SceneForce::VORTEX;
                read.checkKeys(item, where, {"type", "position", "axis", "strength", "radius", "falloff"});
                read.vector3(item, "position", where, force.position, true);
                read.vector3(item, "axis", where, force.axis);
                force.strength = read.number(item, "strength", where, force.strength);
                force.radius = read.number(item, "radius", where, force.radius);
                force.falloff = read.number(item, "falloff", where, force.falloff);
                double length = std::sqrt(force.axis[0]*force.axis[0] + force.axis[1]*force.axis[1] + force.axis[2]*force.axis[2]);
                if(force.kind == SceneForce::VORTEX && length == 0.0){
                    read.fail(where, "a vortex's \"axis\" can't be 0");
                }
                if(force.radius < 0.0 || force.falloff < 0.0){
                    read.fail(where, "radius and falloff can't be negative");
                }
            }
            else if(type == "turbulence"){
                force.kind = SceneForce::TURBULENCE;
                read.checkKeys(item, where, {"type", "strength", "scale", "speed", "seed"});
                force.strength = read.number(item, "strength", where, force.strength);
                force.scale = read.number(item, "scale", where, force.scale);
                force.speed = read.number(item, "speed", where, force.speed);
                force.seed = (unsigned int)read.number(item, "seed", where, force.seed);
                if(force.scale <= 0.0){
                    read.fail(where, "turbulence needs a positive \"scale\"");
                }
            }
            else if(type == "wind"){
                force.kind = SceneForce::WIND;
                read.checkKeys(item, where, {"type", "velocity", "drag", "depth"});
                read.vector3(item, "velocity", where, force.velocity, true);
                force.drag = read.number(item, "drag", where, force.drag);
                force.depth = read.number(item, "depth", where, force.depth);
                if(force.drag < 0.0 || force.depth < 0.0){
                    read.fail(where, "drag and depth can't be negative");
                }
            }
            else{
                read.fail(where + ".type", "is point, vortex, turbulence or wind, not \"" + type + "\"");
            }
            scene.forces.push_back(force);
        }
    }

    scene.partitions = (int)read.number(root, "partitions", "the scene", scene.partitions);
    scene.devices = (int)read.number(root, "devices", "the scene", scene.devices);
    if(scene.partitions < 1 || scene.devices < 0){
        read.fail("the scene", "partitions has to be at least 1 and devices at least 0");
    }
    if(const Json* output = read.object(root, "output", "the scene")){
        read.checkKeys(*output, "output", {"dir", "positions", "diagnostics"});
        scene.outputDirectory = read.text(*output, "dir", "output", scene.outputDirectory);
        scene.writePositions = read.flag(*output, "positions", "output", scene.writePositions);
        scene.diagnostics = read.text(*output, "diagnostics", "output", scene.diagnostics);
    }
    return scene;
}

//splitmix64: a well-mixed 64-bit hash, for jittering the seeds
static unsigned long long mixBits(unsigned long long x){
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30))*0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27))*0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

//a number in [0, 1) that depends only on the seed, the lattice cell and which coordinate it's for
static double jitter(unsigned long long seed, unsigned long long cell, int coordinate){
    return (mixBits(mixBits(seed) ^ (cell*3 + coordinate)) >> 11)*(1.0 / 9007199254740992.0);
}

void seedParticles(const Scene& scene, std::vector<double>& x, std::vector<double>& y, std::vector<double>& z, std::vector<float>& u, std::vector<float>& v, std::vector<float>& w){
    int perSide = scene.particlesPerVoxel == 27 ? 3 : scene.particlesPerVoxel == 8 ? 2 : 1;
    double spacing = scene.voxelSize() / perSide;
    unsigned long long lattice[3];  //lattice cells along each axis of the domain
    for(int axis = 0; axis < 3; ++axis){
        lattice[axis] = (unsigned long long)scene.nodes[axis]*4*perSide;
    }
    for(size_t index = 0; index < scene.fluids.size(); ++index){
        const SceneShape& shape = scene.fluids[index];
        double low[3], high[3];
        shape.bounds(low, high);
        unsigned long long first[3], last[3];   //the lattice cells whose centres can be inside it
        bool empty = false;
        for(int axis = 0; axis < 3; ++axis){
            double from = std::floor((low[axis] - scene.domainMin[axis]) / spacing - 0.5);
            double to = std::ceil((high[axis] - scene.domainMin[axis]) / spacing - 0.5);
            from = std::max(from, 0.0);
            to = std::min(to, (double)lattice[axis] - 1.0);
            empty = empty || to < from;
            first[axis] = (unsigned long long)std::max(from, 0.0);
            last[axis] = (unsigned long long)std::max(to, 0.0);
        }
        if(empty){
            continue;
        }
        for(unsigned long long k = first[2]; k <= last[2]; ++k){
            for(unsigned long long j = first[1]; j <= last[1]; ++j){
                for(unsigned long long i = first[0]; i <= last[0]; ++i){
                    double centre[3] = {scene.domainMin[0] + (i + 0.5)*spacing, scene.domainMin[1] + (j + 0.5)*spacing, scene.domainMin[2] + (k + 0.5)*spacing};
                    if(!shape.contains(centre)){
                        continue;
                    }
                    bool earlier = false;   //the cell belongs to the first shape holding its centre
                    for(size_t other = 0; other < index && !earlier; ++other){
                        earlier = scene.fluids[other].contains(centre);
                    }
                    if(earlier){
                        continue;
                    }
                    unsigned long long cell = i + lattice[0]*(j + lattice[1]*k);
                    x.push_back(scene.domainMin[0] + (i + jitter(scene.seed, cell, 0))*spacing);
                    y.push_back(scene.domainMin[1] + (j + jitter(scene.seed, cell, 1))*spacing);
                    z.push_back(scene.domainMin[2] + (k + jitter(scene.seed, cell, 2))*spacing);
                    u.push_back((float)shape.velocity[0]);
                    v.push_back((float)shape.velocity[1]);
                    w.push_back((float)shape.velocity[2]);
                }
            }
        }
    }
}
