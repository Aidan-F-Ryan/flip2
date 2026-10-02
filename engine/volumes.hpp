//Copyright 2023 Aberrant Behavior LLC

#ifndef VOLUMES_HPP
#define VOLUMES_HPP

#include "scene.hpp"
#include "surface.hu"
#include <memory>
#include <string>

//A level set from a VDB file, resampled into the engine's sparse bricks at voxelSize (see SceneField): grid names it, or "" takes the first float grid
//in the file, a level set before any other. Where its narrow band is narrower than the engine's 3 voxels, it's rebuilt wider first. With velocityGrid,
//that vector grid's velocities are sampled at the same points (staggered grids too). Throws std::runtime_error saying what's wrong, and in a build without
//OpenVDB, always
std::shared_ptr<const SceneField> loadLevelSet(const std::string& path, const std::string& grid, const std::string& velocityGrid, double voxelSize);

//writes path, a VDB file of the liquid's fields as Houdini's FLIP has them, for its whitewater: "surface", a level set (inside its narrow band, inside), and
//"vel", the velocity. Says why not if it can't, and in a build without OpenVDB, always can't
bool writeFluidFields(const std::string& path, const FluidFields& fields, std::string& why);

#endif
