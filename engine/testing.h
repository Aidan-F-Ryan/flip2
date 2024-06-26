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

class ParticleSystemTester{
public:

    ParticleSystemTester(uint size)
    : particles(size)
    {}

    void setDomain(double nx, double ny, double nz, uint x, uint y, uint z, double cellSize){
        particles.setDomain(nx, ny, nz, x, y, z, cellSize);
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
    
    void randomizeParticlePositions(){
        std::random_device rd;
        std::default_random_engine e2(rd());
        double oneThirdYDimension = particles.grid.sizeY*particles.grid.cellSize / 3.0f;
        std::uniform_real_distribution<> distX(particles.grid.negX + oneThirdYDimension, particles.grid.negX + particles.grid.sizeX*particles.grid.cellSize - oneThirdYDimension);
        std::uniform_real_distribution<> distY(particles.grid.negY, particles.grid.negY + particles.grid.sizeY*particles.grid.cellSize - oneThirdYDimension);
        std::uniform_real_distribution<> distZ(particles.grid.negZ + oneThirdYDimension, particles.grid.negZ + particles.grid.sizeZ*particles.grid.cellSize - oneThirdYDimension);

        std::uniform_real_distribution<> vDist(-1.0f, 1.0f);
#pragma omp for
        for(uint i = 0; i < particles.size; ++i){
            particles.px[i] = distX(e2);
            particles.py[i] = distY(e2);
            particles.pz[i] = distZ(e2);
        }
        particles.px.upload(particles.stream);
        particles.py.upload(particles.stream);
        particles.pz.upload(particles.stream);
        particles.vx.zeroDeviceAsync(particles.stream);
        particles.vy.zeroDeviceAsync(particles.stream);
        particles.vz.zeroDeviceAsync(particles.stream);
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