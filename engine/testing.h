//Copyright Aberrant Behavior LLC 2023

#ifndef TESTING_H
#define TESTING_H

#include "particles.hu"
#include <map>
#include <random>
#include <iostream>

class ParticleSystemTester{
public:

    ParticleSystemTester(uint size)
    : particles(size)
    {}

    void setDomain(float nx, float ny, float nz, uint x, uint y, uint z, float cellSize){
        particles.setDomain(nx, ny, nz, x, y, z, cellSize);
    }

    void storeGridCellMap(){
        particles.gridCell.download(particles.stream);
        cudaStreamSynchronize(particles.stream);
        for(uint i = 0; i < particles.gridCell.size(); ++i){
            ++gridMap[particles.gridCell[i]];
        }
    }

    void validateGridCellOrdering(){
        particles.gridCell.download(particles.stream);
        cudaStreamSynchronize(particles.stream);
        uint prev = 0;
        std::map<uint, uint> tempGridMap;
        for(uint i = 0; i < particles.gridCell.size(); ++i){
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
        float oneThirdYDimension = particles.grid.sizeY*particles.grid.cellSize / 3.0f;
        std::uniform_real_distribution<> distX(particles.grid.negX + oneThirdYDimension, particles.grid.negX + particles.grid.sizeX*particles.grid.cellSize - oneThirdYDimension);
        std::uniform_real_distribution<> distY(particles.grid.negY + oneThirdYDimension, particles.grid.negY + particles.grid.sizeY*particles.grid.cellSize - oneThirdYDimension);
        std::uniform_real_distribution<> distZ(particles.grid.negZ + oneThirdYDimension, particles.grid.negZ + particles.grid.sizeZ*particles.grid.cellSize - oneThirdYDimension);

        std::uniform_real_distribution<> vDist(-1.0f, 1.0f);

        for(uint i = 0; i < particles.size; ++i){
            particles.px[i] = distX(e2);
            particles.py[i] = distY(e2);
            particles.pz[i] = distZ(e2);
            particles.vx[i] = vDist(e2);
            particles.vy[i] = vDist(e2);
            particles.vz[i] = vDist(e2);
        }
        
        particles.px.upload(particles.stream);
        particles.py.upload(particles.stream);
        particles.pz.upload(particles.stream);
        particles.vx.upload(particles.stream);
        particles.vy.upload(particles.stream);
        particles.vz.upload(particles.stream);
    }
    
    void runVerify(){
        particles.alignParticlesToGrid();
        storeGridCellMap();
        particles.sortParticles();
        validateGridCellOrdering();
        particles.alignParticlesToSubCells();
        particles.generateVoxels();
        particles.particleVelToVoxels();
        particles.voxelsUx.download();
        particles.voxelIDsX.download();
        particles.voxelsUy.download();
        particles.voxelIDsY.download();
        particles.voxelsUz.download();
        particles.voxelIDsZ.download();

        for(uint i = 0; i < particles.voxelIDsX.size(); ++i){
            std::cout<<particles.voxelIDsX[i]<<": "<<particles.voxelsUx[i]<<std::endl;
            std::cout<<particles.voxelIDsY[i]<<": "<<particles.voxelsUy[i]<<std::endl;
            std::cout<<particles.voxelIDsZ[i]<<": "<<particles.voxelsUy[i]<<std::endl;
        }
    }

    void run(){
        particles.alignParticlesToGrid();
        particles.sortParticles();
        particles.alignParticlesToSubCells();
        particles.generateVoxels();
        particles.particleVelToVoxels();
    }

private:
    Particles particles;
    std::map<uint, uint> gridMap;
};

#endif