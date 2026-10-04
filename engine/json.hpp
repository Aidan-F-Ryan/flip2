//Copyright 2023 Aberrant Behavior LLC

#ifndef JSON_HPP
#define JSON_HPP

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

//Just enough JSON for scene files and the cache's records: objects, arrays, numbers, strings, true, false and null
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

#endif
