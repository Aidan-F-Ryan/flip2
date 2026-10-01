//Copyright 2023 Aberrant Behavior LLC

#ifndef VOLUMES_HPP
#define VOLUMES_HPP

#include "scene.hpp"
#include <memory>
#include <string>

//A level set from a VDB file, resampled into the engine's sparse bricks at voxelSize (see SceneField): grid names it, or "" takes the first float grid
//in the file, a level set before any other. Where its narrow band is narrower than the engine's 3 voxels, it's rebuilt wider first. With velocityGrid,
//that vector grid's velocities are sampled at the same points (staggered grids too). Throws std::runtime_error saying what's wrong, and in a build without
//OpenVDB, always
std::shared_ptr<const SceneField> loadLevelSet(const std::string& path, const std::string& grid, const std::string& velocityGrid, double voxelSize);

#endif
