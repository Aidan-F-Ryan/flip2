//Copyright 2023 Aberrant Behavior LLC

#ifndef TESTING_H
#define TESTING_H

#include "particles.hu"
#include <map>
#include <random>
#include <iostream>
#include <bitset>
#include <omp.h>
#include <string>
#include <cmath>

class ParticleSystemTester{
public:

    ParticleSystemTester(uint size)
    : particles(size)
    {}

    void setDomain(double nx, double ny, double nz, uint x, uint y, uint z, double cellSize){
        particles.setDomain(nx, ny, nz, x, y, z, cellSize);
    }

    void setFlipRatio(double ratio){
        particles.setFlipRatio(ratio);
    }

    void setDensityCorrectionTime(double seconds){
        particles.setDensityCorrectionTime(seconds);
    }

    void storeGridCellMap(){
        gridMap.clear();
        particles.gridCell.download(particles.stream);
        cudaStreamSynchronize(particles.stream);
        for(uint i = 0; i < particles.gridCell.size(); ++i){
            ++gridMap[particles.gridCell[i]];
        }
    }

    void validateGridCellOrdering(){
        particles.gridCell.download();
        uint prev = 0;
        std::map<uint, uint> tempGridMap;
        for(uint i = 0; i < particles.gridCell.size(); ++i){
            // std::cout<<i<<": "<<std::bitset<sizeof(uint)*8>(particles.gridCell[i])<<std::endl;
            if(prev > particles.gridCell[i]){
                std::cerr<<"gridCell not sorted "<<i<<" "<<particles.gridCell[i]<<"\n";
                exit(1);
            }
            // std::cout<<i<<" "<<particles.gridCell[i]<<"\n";
            ++tempGridMap[particles.gridCell[i]];
            prev = particles.gridCell[i];
        }
        for(auto iter = gridMap.begin(); iter != gridMap.end(); ++iter){
            if(iter->second != tempGridMap[iter->first]){
                std::cerr<<"gridCells different count "<<iter->first<<" "<<iter->second<<" "<<tempGridMap[iter->first]<<"\n";
            }
        }
    }
    
    //particles at random positions in the box lo..hi, in fractions of the domain (by default the dam break's column), and at rest, or with swirl
    //set, in one vortex filling the domain's xy cross section, swirl m/s at its fastest and not crossing the walls
    void randomizeParticlePositions(double3 lo = make_double3(1.0/3.0, 0.0, 1.0/3.0), double3 hi = make_double3(2.0/3.0, 2.0/3.0, 2.0/3.0), double swirl = 0.0){
        std::random_device rd;
        std::default_random_engine e2(rd());
        double width = particles.grid.sizeX*particles.grid.cellSize;
        double height = particles.grid.sizeY*particles.grid.cellSize;
        double depth = particles.grid.sizeZ*particles.grid.cellSize;
        std::uniform_real_distribution<> distX(particles.grid.negX + lo.x*width, particles.grid.negX + hi.x*width);
        std::uniform_real_distribution<> distY(particles.grid.negY + lo.y*height, particles.grid.negY + hi.y*height);
        std::uniform_real_distribution<> distZ(particles.grid.negZ + lo.z*depth, particles.grid.negZ + hi.z*depth);
        for(uint i = 0; i < particles.size; ++i){
            particles.px[i] = distX(e2);
            particles.py[i] = distY(e2);
            particles.pz[i] = distZ(e2);
            double x = M_PI*(particles.px[i] - particles.grid.negX)/width;
            double y = M_PI*(particles.py[i] - particles.grid.negY)/height;
            particles.vx[i] = swirl*sin(x)*cos(y);
            particles.vy[i] = -swirl*cos(x)*sin(y);
            particles.vz[i] = 0.0;
        }
        for(CudaVec<double>* position : {&particles.px, &particles.py, &particles.pz}){
            position->upload(particles.stream);
        }
        for(CudaVec<float>* velocity : {&particles.vx, &particles.vy, &particles.vz}){
            velocity->upload(particles.stream);
        }
        double voxelSize = particles.grid.cellSize / (2<<particles.refinementLevel);
        particles.setRestDensity(particles.size / ((hi.x - lo.x)*width*(hi.y - lo.y)*height*(hi.z - lo.z)*depth / (voxelSize*voxelSize*voxelSize)));   //what they start at
    }

    void runVerify(){
        particles.alignParticlesToGrid();
        storeGridCellMap();
        particles.sortParticles();
        validateGridCellOrdering();
        // particles.alignParticlesToSubCells();
        particles.generateVoxels();
        particles.particleVelToVoxels();

        particles.voxelsUx.download();
        particles.voxelsUy.download();
        particles.voxelsUz.download();
        particles.voxelIDsUsed.download();

        // for(uint i = 0; i < particles.voxelIDsUsed.size(); ++i){
        //     std::cout<<particles.voxelIDsUsed[i]<<": <"<<particles.voxelsUx[i]<<", "<<particles.voxelsUy[i]<<", "<<particles.voxelsUz[i]<<">\n";
        // }
        particles.pressureSolve();
        particles.updateVoxelVelocities();
        particles.advectParticles();
        particles.px.download();
        particles.py.download();
        particles.pz.download();

        for(uint i = 0; i < 10; ++i){
            std::cout<<particles.px[i]<<", "<<particles.py[i]<<", "<<particles.pz[i]<<"\n";
        }
    }
    void initialize(){
        particles.initialize();
    }
    void run(){
        particles.alignParticlesToGrid();
        particles.sortParticles();
        // particles.alignParticlesToSubCells();
        particles.generateVoxels();
        particles.particleVelToVoxels();
        particles.pressureSolve();
        particles.updateVoxelVelocities();
        particles.advectParticles();
    }

    void solveFrame(double fps){
        particles.solveFrame(fps);
    }

    void writePositionsToFile(const std::string& fileName){
        particles.writePositionsToFile(fileName);
    }

private:
    Particles particles;
    std::map<uint, uint> gridMap;
};

#endif