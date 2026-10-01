//Copyright 2023 Aberrant Behavior LLC

//Writes the VDB level sets the test scenes use, into the directory given (the repo's scenes/geo):
//  sphere.vdb      a sphere of radius 0.05 m centred at (0.125, 0.06, 0.125), the tanks' (tank-meshsphere.json), as grid "surface", 2.5 mm voxels
//  spinning.vdb    the same sphere, with a "vel" grid turning its surface about y once a second, like a ball spinning in place
//  ball.vdb        a sphere of radius 0.08 m centred at (0, 0.3, 0), drop-sphere.json's ball of water, 4 mm voxels
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
    std::cout<<"wrote "<<directory<<"sphere.vdb, spinning.vdb and ball.vdb\n";
    return 0;
}
