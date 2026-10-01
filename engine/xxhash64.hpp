//Copyright 2023 Aberrant Behavior LLC

#ifndef XXHASH64_HPP
#define XXHASH64_HPP

#include <cstddef>
#include <cstdint>
#include <cstring>

//XXH64 (Yann Collet's xxHash, 64-bit), fed in pieces: what the cache's commit records name each shard file by, so a reader can tell a whole file from a
//torn one. Written from the published algorithm rather than taken from xxhash.h, which coeus doesn't have; a file's hash is the same as xxhsum -H64's
class Xxh64{
public:
    explicit Xxh64(uint64_t seed = 0)
    : lanes{seed + PRIME1 + PRIME2, seed + PRIME2, seed, seed - PRIME1}
    , seed(seed)
    {}

    void update(const void* data, size_t bytes){
        const unsigned char* in = (const unsigned char*)data;
        total += bytes;
        if(buffered + bytes < 32){
            std::memcpy(buffer + buffered, in, bytes);
            buffered += bytes;
            return;
        }
        if(buffered > 0){   //top up the stripe started last time
            size_t fill = 32 - buffered;
            std::memcpy(buffer + buffered, in, fill);
            stripe(buffer);
            in += fill;
            bytes -= fill;
            buffered = 0;
        }
        for(; bytes >= 32; in += 32, bytes -= 32){
            stripe(in);
        }
        std::memcpy(buffer, in, bytes);
        buffered = bytes;
    }

    uint64_t digest() const{
        uint64_t hash;
        if(total >= 32){
            hash = rotate(lanes[0], 1) + rotate(lanes[1], 7) + rotate(lanes[2], 12) + rotate(lanes[3], 18);
            for(uint64_t lane : lanes){
                hash = (hash ^ round(0, lane))*PRIME1 + PRIME4;
            }
        }
        else{
            hash = seed + PRIME5;
        }
        hash += total;
        const unsigned char* in = buffer;
        size_t left = buffered;
        for(; left >= 8; in += 8, left -= 8){
            hash = rotate(hash ^ round(0, read64(in)), 27)*PRIME1 + PRIME4;
        }
        if(left >= 4){
            hash = rotate(hash ^ (uint64_t)read32(in)*PRIME1, 23)*PRIME2 + PRIME3;
            in += 4;
            left -= 4;
        }
        for(; left > 0; ++in, --left){
            hash = rotate(hash ^ *in*PRIME5, 11)*PRIME1;
        }
        hash ^= hash >> 33;
        hash *= PRIME2;
        hash ^= hash >> 29;
        hash *= PRIME3;
        return hash ^ (hash >> 32);
    }

private:
    static constexpr uint64_t PRIME1 = 0x9E3779B185EBCA87ull;
    static constexpr uint64_t PRIME2 = 0xC2B2AE3D27D4EB4Full;
    static constexpr uint64_t PRIME3 = 0x165667B19E3779F9ull;
    static constexpr uint64_t PRIME4 = 0x85EBCA77C2B2AE63ull;
    static constexpr uint64_t PRIME5 = 0x27D4EB2F165667C5ull;

    static uint64_t rotate(uint64_t x, int by){
        return x << by | x >> (64 - by);
    }
    static uint64_t round(uint64_t lane, uint64_t input){
        return rotate(lane + input*PRIME2, 31)*PRIME1;
    }
    static uint64_t read64(const unsigned char* in){    //little-endian, as every machine flip2 runs on is
        uint64_t value;
        std::memcpy(&value, in, 8);
        return value;
    }
    static uint32_t read32(const unsigned char* in){
        uint32_t value;
        std::memcpy(&value, in, 4);
        return value;
    }
    void stripe(const unsigned char* in){
        for(int lane = 0; lane < 4; ++lane){
            lanes[lane] = round(lanes[lane], read64(in + 8*lane));
        }
    }

    uint64_t lanes[4];
    uint64_t seed;
    uint64_t total = 0;
    unsigned char buffer[32];
    size_t buffered = 0;
};

#endif
