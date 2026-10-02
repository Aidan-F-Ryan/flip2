//Copyright 2023 Aberrant Behavior LLC
#ifndef BGEO_HPP
#define BGEO_HPP
#include "cacheWriter.hu"   //ShardData
#include <string>
#include <vector>

//Houdini's geometry files (.bgeo): a frame's particles as a point cloud, P and v, laid out as Houdini 22 writes one (bgeo.cu)

//writes path, the particles of shards one after another, so in the order one partition would have held them. Says why not if it can't
bool writeParticlesBgeo(const std::string& path, const std::vector<const ShardData*>& shards, const std::string& software, std::string& why);

#endif
