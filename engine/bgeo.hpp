//Copyright 2023 Aberrant Behavior LLC
#ifndef BGEO_HPP
#define BGEO_HPP
#include "cacheWriter.hu"   //ShardData
#include "surface.hu"       //SurfaceMesh
#include <string>
#include <vector>

//Houdini's geometry files (.bgeo), laid out as Houdini 22 writes them (bgeo.cu): a frame's particles as a point cloud, P and v and what the shards carry
//of id and age, or its surface as quads

//writes path, the particles of shards one after another, so in the order one partition would have held them. Says why not if it can't
bool writeParticlesBgeo(const std::string& path, const std::vector<const ShardData*>& shards, const std::string& software, std::string& why);

//writes path, a frame's whitewater (its shards as writeParticlesBgeo has the particles'), with P, v, id and age as theirs, and life (the seconds it has
//left as foam), radius (the droplet's or the bubble's, metres), kind (0 spray, 1 foam, 2 a bubble) and pscale, the radius to draw it at: every point's
//the same, half the spacing the particles stand for
bool writeWhitewaterBgeo(const std::string& path, const std::vector<const ShardData*>& shards, float pscale, const std::string& software, std::string& why);

//writes path, mesh's quads facing outwards, their points with P and v
bool writeSurfaceBgeo(const std::string& path, const SurfaceMesh& mesh, const std::string& software, std::string& why);

#endif
