//Copyright 2023 Aberrant Behavior LLC

//Writes the VDB level sets the test scenes use, into the directory given (the repo's scenes/geo):
//  sphere.vdb      a sphere of radius 0.05 m centred at (0.125, 0.06, 0.125), the tanks' (tank-meshsphere.json), as grid "surface", 2.5 mm voxels
//  spinning.vdb    the same sphere, with a "vel" grid turning its surface about y once a second, like a ball spinning in place
//  ball.vdb        a sphere of radius 0.08 m centred at (0, 0.3, 0), drop-sphere.json's ball of water, 4 mm voxels
//  vectors.vdb     grids of vectors on 2 cm voxels over the box from -0.3 to 0.3 m on every axis, for the volume forces' scenes, whose answers are
//                  then known:
//                    "force"   (0.3, 0, 0) everywhere (force-volume.json)
//                    "vel"     (0.2, 0, 0) everywhere (velocity-volume.json)
//                    "linear"  (0.1 + z, 0.3 + x, -0.2 + y): each component by another axis, so a volume put in the wrong place, or with its axes
//                              crossed, pushes the wrong way; trilinear sampling gives it exactly, and it has no divergence, so pressure leaves
//                              it be (linear-volume.json)
//                    "patch"   (-0.4, 0.3, 0), but only in the slab from x = -0.25 to -0.07; the rest of the box isn't active, and has to do
//                              nothing (patch-volume.json)
//                    "strain"  (0.1 + x, 0.3 + y, -0.2 - 2z), staggered: each voxel holds each component where its lower face along that axis
//                              is, half a voxel back, as a simulation's velocity grid does. Each component changes along its own axis, so
//                              reading it at the voxel's centre instead would be off by half a voxel's worth (strain-volume.json)
//
//  makeTestLevelSets DIR

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <cmath>
#include <iostream>
#include <string>

int main(int argc, char** argv){
    if(argc != 2){
        std::cerr<<"usage: makeTestLevelSets DIR\n";
        return 2;
    }
    openvdb::initialize();
    std::string directory = std::string(argv[1]) + "/";
    const openvdb::Vec3f centre(0.125f, 0.06f, 0.125f);
    openvdb::FloatGrid::Ptr sphere = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(0.05f, centre, 0.0025f, 3.0f);
    sphere->setName("surface");
    openvdb::io::File(directory + "sphere.vdb").write({sphere});

    //its velocity, as a spinning ball's: omega x (x - centre), omega once a second about y, on the voxels of its band
    openvdb::Vec3SGrid::Ptr velocity = openvdb::Vec3SGrid::create(openvdb::Vec3s(0.0f));
    velocity->setTransform(sphere->transform().copy());
    velocity->setName("vel");
    velocity->setVectorType(openvdb::VEC_CONTRAVARIANT_RELATIVE);
    const openvdb::Vec3f omega(0.0f, 2.0f*(float)M_PI, 0.0f);
    openvdb::Vec3SGrid::Accessor write = velocity->getAccessor();
    for(openvdb::FloatGrid::ValueOnCIter value = sphere->cbeginValueOn(); value; ++value){
        openvdb::Vec3f at = sphere->indexToWorld(value.getCoord());
        write.setValue(value.getCoord(), omega.cross(at - centre));
    }
    openvdb::io::File(directory + "spinning.vdb").write({sphere, velocity});
    openvdb::FloatGrid::Ptr ball = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(0.08f, openvdb::Vec3f(0.0f, 0.3f, 0.0f), 0.004f, 3.0f);
    ball->setName("surface");
    openvdb::io::File(directory + "ball.vdb").write({ball});
    openvdb::GridPtrVec vectors;
    for(const char* name : {"force", "vel", "linear", "patch", "strain"}){
        openvdb::Vec3SGrid::Ptr grid = openvdb::Vec3SGrid::create(openvdb::Vec3s(0.0f));
        grid->setTransform(openvdb::math::Transform::createLinearTransform(0.02));     //voxel (i, j, k)'s centre is at 0.02 (i, j, k)
        grid->setName(name);
        grid->setVectorType(openvdb::VEC_CONTRAVARIANT_RELATIVE);
        std::string which = name;
        if(which == "force" || which == "vel"){
            grid->denseFill(openvdb::CoordBBox(openvdb::Coord(-15), openvdb::Coord(14)), which == "force" ? openvdb::Vec3s(0.3f, 0.0f, 0.0f) : openvdb::Vec3s(0.2f, 0.0f, 0.0f), true);
        }
        else if(which == "patch"){
            grid->denseFill(openvdb::CoordBBox(openvdb::Coord(-12, -15, -15), openvdb::Coord(-4, 14, 14)), openvdb::Vec3s(-0.4f, 0.3f, 0.0f), true);
        }
        else{
            openvdb::Vec3SGrid::Accessor accessor = grid->getAccessor();
            for(int z = -15; z <= 14; ++z){
                for(int y = -15; y <= 14; ++y){
                    for(int x = -15; x <= 14; ++x){
                        accessor.setValue(openvdb::Coord(x, y, z), which == "linear" ? openvdb::Vec3s(0.1f + 0.02f*z, 0.3f + 0.02f*x, -0.2f + 0.02f*y)
                                                                   : openvdb::Vec3s(0.1f + 0.02f*(x - 0.5f), 0.3f + 0.02f*(y - 0.5f), -0.2f - 2.0f*0.02f*(z - 0.5f)));
                    }
                }
            }
            if(which == "strain"){
                grid->setGridClass(openvdb::GRID_STAGGERED);
            }
        }
        vectors.push_back(grid);
    }
    openvdb::io::File(directory + "vectors.vdb").write(vectors);
    std::cout<<"wrote "<<directory<<"sphere.vdb, spinning.vdb, ball.vdb and vectors.vdb\n";
    return 0;
}
