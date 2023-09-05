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
        std::uniform_real_distribution<> distX(particles.grid.negX, particles.grid.negX + particles.grid.sizeX*particles.grid.cellSize);
        std::uniform_real_distribution<> distY(particles.grid.negY, particles.grid.negY + particles.grid.sizeY*particles.grid.cellSize);
        std::uniform_real_distribution<> distZ(particles.grid.negZ, particles.grid.negZ + particles.grid.sizeZ*particles.grid.cellSize);

        for(uint i = 0; i < particles.size; ++i){
            particles.px[i] = distX(e2);
            particles.py[i] = distY(e2);
            particles.pz[i] = distZ(e2);
        }
        
        particles.px.upload(particles.stream);
        particles.py.upload(particles.stream);
        particles.pz.upload(particles.stream);
    }
    
    void runVerify(){
        particles.alignParticlesToGrid();
        storeGridCellMap();
        particles.sortParticles();
        validateGridCellOrdering();
        particles.alignParticlesToSubCells();
    }

    void run(){
        particles.alignParticlesToGrid();
        particles.sortParticles();
        particles.alignParticlesToSubCells();
    }

private:
    Particles particles;
    std::map<uint, uint> gridMap;
};

#endif