//Copyright 2023 Aberrant Behavior LLC

//VDB level sets, resampled into the engine's sparse bricks (see volumes.hpp). Only this file sees OpenVDB, so the CUDA sources never include it

#include "volumes.hpp"
#include <stdexcept>

#ifdef FLIP2_WITH_OPENVDB

#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetRebuild.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <mutex>

static const int BRICK = 8;                     //SDF_BRICK, SDF_OUTSIDE and SDF_INSIDE (obstacles.hu)
static const int OUTSIDE = -1, INSIDE = -2;

//the world-space box around an index-space box of a grid, its corners through the grid's transform
static void worldBox(const openvdb::math::Transform& transform, const openvdb::Vec3d& low, const openvdb::Vec3d& high, double outLow[3], double outHigh[3]){
    for(int axis = 0; axis < 3; ++axis){
        outLow[axis] = INFINITY;
        outHigh[axis] = -INFINITY;
    }
    for(int corner = 0; corner < 8; ++corner){
        openvdb::Vec3d world = transform.indexToWorld(openvdb::Vec3d(corner & 1 ? high[0] : low[0], corner & 2 ? high[1] : low[1], corner & 4 ? high[2] : low[2]));
        for(int axis = 0; axis < 3; ++axis){
            outLow[axis] = std::min(outLow[axis], world[axis]);
            outHigh[axis] = std::max(outHigh[axis], world[axis]);
        }
    }
}

//a velocity grid's values for every sample of the bricks in use: each sample takes the velocity of the nearest point on the surface, as a deforming
//mesh's do, there along the level set's gradient. A velocity grid often only covers the level set's own narrow band, which can be narrower than the
//engine's, and past it reads 0. Sampler: BoxSampler, or StaggeredBoxSampler for a staggered grid
template<class Sampler>
static void sampleVelocities(const openvdb::Vec3SGrid& grid, const openvdb::FloatGrid& levelSet, const std::vector<size_t>& active, SceneField& field){
    const int samples = BRICK*BRICK*BRICK;
    double h = levelSet.voxelSize()[0];
    tbb::parallel_for(tbb::blocked_range<size_t>(0, active.size()), [&](const tbb::blocked_range<size_t>& range){
        openvdb::Vec3SGrid::ConstAccessor accessor = grid.getConstAccessor();   //a sampler keeps a reference to it, so it can't be a temporary
        openvdb::tools::GridSampler<openvdb::Vec3SGrid::ConstAccessor, Sampler> sampler(accessor, grid.transform());
        openvdb::FloatGrid::ConstAccessor distances = levelSet.getConstAccessor();
        openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler> distance(distances, levelSet.transform());
        for(size_t slot = range.begin(); slot != range.end(); ++slot){
            size_t brick = active[slot];
            size_t cell[3] = {brick % field.bricks[0], brick / field.bricks[0] % field.bricks[1], brick / ((size_t)field.bricks[0]*field.bricks[1])};
            for(int sample = 0; sample < samples; ++sample){
                int at[3] = {sample % BRICK, sample / BRICK % BRICK, sample / (BRICK*BRICK)};
                openvdb::Vec3d point;
                for(int axis = 0; axis < 3; ++axis){
                    point[axis] = field.origin[axis] + field.spacing*(cell[axis]*BRICK + at[axis]);
                }
                double d = distance.wsSample(point);
                openvdb::Vec3d gradient;
                for(int axis = 0; axis < 3; ++axis){
                    openvdb::Vec3d step(0.0);
                    step[axis] = h;
                    gradient[axis] = (distance.wsSample(point + step) - distance.wsSample(point - step)) / (2.0*h);
                }
                double length = gradient.length();
                openvdb::Vec3d surface = length > 0.0 ? point - (d / length)*gradient : point;
                openvdb::Vec3s velocity = sampler.wsSample(surface);
                for(int axis = 0; axis < 3; ++axis){
                    field.velocities[(slot*3 + axis)*samples + sample] = velocity[axis];
                }
            }
        }
    });
}

std::shared_ptr<const SceneField> loadLevelSet(const std::string& path, const std::string& gridName, const std::string& velocityName, double voxelSize){
    static std::once_flag initialized;
    std::call_once(initialized, []{ openvdb::initialize(); });
    openvdb::io::File file(path);
    try{
        file.open();
    }
    catch(const openvdb::Exception& error){
        throw std::runtime_error(path + ": can't read it as a VDB file: " + error.what());
    }
    //the grid: the one named, or the first float grid, a level set before any other
    openvdb::GridBase::Ptr chosen;
    if(!gridName.empty()){
        if(!file.hasGrid(gridName)){
            throw std::runtime_error(path + ": has no grid named \"" + gridName + "\"");
        }
        chosen = file.readGrid(gridName);
    }
    else{
        openvdb::GridPtrVecPtr grids = file.getGrids();     //kept: the loop mustn't outlive the vector a temporary would hold
        for(const openvdb::GridBase::Ptr& candidate : *grids){
            bool better = !chosen || (chosen->getGridClass() != openvdb::GRID_LEVEL_SET && candidate->getGridClass() == openvdb::GRID_LEVEL_SET);
            if(candidate->isType<openvdb::FloatGrid>() && better){
                chosen = candidate;
            }
        }
        if(!chosen){
            throw std::runtime_error(path + ": has no float grid, so no level set");
        }
    }
    openvdb::FloatGrid::Ptr grid = openvdb::gridPtrCast<openvdb::FloatGrid>(chosen);
    if(!grid){
        throw std::runtime_error(path + ": grid \"" + chosen->getName() + "\" holds " + chosen->valueType() + ", not the floats of a level set");
    }
    if(!grid->hasUniformVoxels()){
        throw std::runtime_error(path + ": grid \"" + grid->getName() + "\" has voxels of different lengths along each axis: resample it to cubes first");
    }
    if(grid->getGridClass() != openvdb::GRID_LEVEL_SET){
        std::cerr<<path<<": grid \""<<grid->getName()<<"\" isn't marked as a level set; its values are taken as signed distances, negative inside, anyway\n";
    }
    float band = (float)(3.0*voxelSize);
    double gridVoxel = grid->voxelSize()[0];
    if(grid->background() < band){  //its narrow band is narrower than the engine's: rebuilt wider, so the distances hold out to the engine's band
        grid = openvdb::tools::levelSetRebuild(*grid, 0.0f, (float)(band / gridVoxel) + 1.0f);
    }
    openvdb::CoordBBox box = grid->evalActiveVoxelBoundingBox();
    if(box.empty()){
        throw std::runtime_error(path + ": grid \"" + grid->getName() + "\" has no active voxels: it's empty");
    }
    auto field = std::make_shared<SceneField>();
    worldBox(grid->transform(), openvdb::Vec3d(box.min().asVec3d()) - 0.5, openvdb::Vec3d(box.max().asVec3d()) + 0.5, field->low, field->high);
    //the engine's lattice over its narrow band, padded as a mesh's is (MeshField::setUp)
    field->spacing = voxelSize;
    field->band = band;
    double pad = band + 2.0*voxelSize;
    double brickSize = BRICK*voxelSize;
    size_t numBricks = 1;
    for(int axis = 0; axis < 3; ++axis){
        field->origin[axis] = field->low[axis] - pad;
        field->bricks[axis] = (int)std::ceil((field->high[axis] - field->low[axis] + 2.0*pad) / brickSize) + 1;
        numBricks *= field->bricks[axis];
    }
    //the bricks that hold samples: any that a voxel of the band (within band of the surface) overlaps, which is any with a sample within band of it
    std::vector<char> near(numBricks, 0);
    for(openvdb::FloatGrid::ValueOnCIter value = grid->cbeginValueOn(); value; ++value){
        if(std::fabs(*value) >= band){
            continue;
        }
        openvdb::CoordBBox cells;
        value.getBoundingBox(cells);
        double low[3], high[3];
        worldBox(grid->transform(), openvdb::Vec3d(cells.min().asVec3d()) - 0.5, openvdb::Vec3d(cells.max().asVec3d()) + 0.5, low, high);
        int first[3], last[3];
        for(int axis = 0; axis < 3; ++axis){
            first[axis] = std::clamp((int)std::floor((low[axis] - field->origin[axis]) / brickSize), 0, field->bricks[axis] - 1);
            last[axis] = std::clamp((int)std::floor((high[axis] - field->origin[axis]) / brickSize), 0, field->bricks[axis] - 1);
        }
        for(int z = first[2]; z <= last[2]; ++z){
            for(int y = first[1]; y <= last[1]; ++y){
                for(int x = first[0]; x <= last[0]; ++x){
                    near[x + (size_t)field->bricks[0]*(y + (size_t)field->bricks[1]*z)] = 1;
                }
            }
        }
    }
    std::vector<size_t> active;
    field->table.assign(numBricks, OUTSIDE);
    for(size_t brick = 0; brick < numBricks; ++brick){
        if(near[brick]){
            field->table[brick] = (int)active.size();
            active.push_back(brick);
        }
    }
    //their samples, trilinear between the grid's voxels, clamped to the band; and the other bricks inside or out by the sign at their centres
    const int samples = BRICK*BRICK*BRICK;
    field->pool.resize(active.size()*samples);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numBricks), [&](const tbb::blocked_range<size_t>& range){
        openvdb::FloatGrid::ConstAccessor accessor = grid->getConstAccessor();    //a sampler keeps a reference to it, so it can't be a temporary
        openvdb::tools::GridSampler<openvdb::FloatGrid::ConstAccessor, openvdb::tools::BoxSampler> sampler(accessor, grid->transform());
        for(size_t brick = range.begin(); brick != range.end(); ++brick){
            size_t cell[3] = {brick % field->bricks[0], brick / field->bricks[0] % field->bricks[1], brick / ((size_t)field->bricks[0]*field->bricks[1])};
            int slot = field->table[brick];
            if(slot < 0){
                openvdb::Vec3d centre;
                for(int axis = 0; axis < 3; ++axis){
                    centre[axis] = field->origin[axis] + field->spacing*(cell[axis]*BRICK + 0.5*BRICK);
                }
                field->table[brick] = sampler.wsSample(centre) < 0.0f ? INSIDE : OUTSIDE;
                continue;
            }
            for(int sample = 0; sample < samples; ++sample){
                int at[3] = {sample % BRICK, sample / BRICK % BRICK, sample / (BRICK*BRICK)};
                openvdb::Vec3d point;
                for(int axis = 0; axis < 3; ++axis){
                    point[axis] = field->origin[axis] + field->spacing*(cell[axis]*BRICK + at[axis]);
                }
                field->pool[(size_t)slot*samples + sample] = std::clamp(sampler.wsSample(point), -band, band);
            }
        }
    });
    if(!velocityName.empty()){
        if(!file.hasGrid(velocityName)){
            throw std::runtime_error(path + ": has no grid named \"" + velocityName + "\" for velocities");
        }
        openvdb::Vec3SGrid::Ptr velocity = openvdb::gridPtrCast<openvdb::Vec3SGrid>(file.readGrid(velocityName));
        if(!velocity){
            throw std::runtime_error(path + ": grid \"" + velocityName + "\" isn't a grid of float vectors, so it can't be velocities");
        }
        field->velocities.resize(3*field->pool.size());
        if(velocity->getGridClass() == openvdb::GRID_STAGGERED){
            sampleVelocities<openvdb::tools::StaggeredBoxSampler>(*velocity, *grid, active, *field);
        }
        else{
            sampleVelocities<openvdb::tools::BoxSampler>(*velocity, *grid, active, *field);
        }
        for(size_t sample = 0; sample < field->pool.size(); ++sample){
            size_t brick = sample / samples, within = sample % samples;
            float speed = 0.0f;
            for(int axis = 0; axis < 3; ++axis){
                float v = field->velocities[(brick*3 + axis)*samples + within];
                speed += v*v;
            }
            field->fastest = std::max(field->fastest, std::sqrt(speed));
        }
    }
    file.close();
    return field;
}

#else

std::shared_ptr<const SceneField> loadLevelSet(const std::string& path, const std::string&, const std::string&, double){
    throw std::runtime_error(path + ": this build of flip2 has no OpenVDB, so it can't read VDB files; build it where OpenVDB is installed (libopenvdb-dev)");
}

#endif
