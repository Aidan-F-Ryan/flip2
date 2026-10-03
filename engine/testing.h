//Copyright 2023 Aberrant Behavior LLC

#ifndef TESTING_H
#define TESTING_H

#include "simulation.hu"
#include <map>
#include <random>
#include <iostream>
#include <bitset>
#include <omp.h>
#include <string>
#include <cmath>

//runs a Simulation split into partitions partitions (1: the whole domain in one); the checks below look at the first partition
class ParticleSystemTester{
public:

    ParticleSystemTester(uint size, int partitions = 1, int devices = 0)     //devices 0: every GPU there is
    : simulation(size, partitions, devices)
    , particles(simulation.partition(0))
    {}

    ParticleSystemTester(uint size, std::unique_ptr<Transport> transport, int device)     //this process is transport's rank, on GPU device
    : simulation(size, std::move(transport), device)
    , particles(simulation.partition(0))
    {}

    void setDomain(double nx, double ny, double nz, uint x, uint y, uint z, double cellSize){
        simulation.setDomain(nx, ny, nz, x, y, z, cellSize);
    }

    void setFlipRatio(double ratio){
        simulation.forEachPartition([&](Particles& partition){ partition.setFlipRatio(ratio); });
    }

    void setApic(bool on){
        simulation.forEachPartition([&](Particles& partition){ partition.setApic(on); });
    }

    void setCfl(double voxels){
        simulation.forEachPartition([&](Particles& partition){ partition.setCfl(voxels); });
    }

    void setRungeKutta3(bool on){
        simulation.forEachPartition([&](Particles& partition){ partition.setRungeKutta3(on); });
    }

    void setDensityCorrectionTime(double seconds){
        simulation.forEachPartition([&](Particles& partition){ partition.setDensityCorrectionTime(seconds); });
    }

    void setPressureSolver(PressureSolver solver){
        simulation.forEachPartition([&](Particles& partition){ partition.setPressureSolver(solver); });
    }

    void setGravity(float3 gravity){
        simulation.forEachPartition([&](Particles& partition){ partition.setGravity(gravity); });
    }

    void setViscosity(double kinematic){
        simulation.forEachPartition([&](Particles& partition){ partition.setViscosity(kinematic); });
    }

    void setSurfaceTension(double overDensity){
        simulation.forEachPartition([&](Particles& partition){ partition.setSurfaceTension(overDensity); });
    }

    void setContactAngle(double degrees){
        simulation.forEachPartition([&](Particles& partition){ partition.setContactAngle(degrees); });
    }

    void setObstacles(const std::vector<SceneObstacle>& obstacles){
        simulation.forEachPartition([&](Particles& partition){ partition.setObstacles(obstacles); });
    }

    void setSources(const Sources& sources){
        simulation.forEachPartition([&](Particles& partition){ partition.setSources(sources); });
    }

    void setSourceMeshes(const std::vector<SceneObstacle>& meshes, const std::vector<FluidShape>& fluids){
        simulation.forEachPartition([&](Particles& partition){ partition.setSourceMeshes(meshes, fluids); });
    }

    void setForceFields(const std::vector<ForceField>& fields){
        simulation.forEachPartition([&](Particles& partition){ partition.setForceFields(fields); });
    }

    void setDotProductSums(DotProductSums sums){
        simulation.forEachPartition([&](Particles& partition){ partition.setDotProductSums(sums); });
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
    //set, in one vortex filling the domain's xy cross section, swirl m/s at its fastest and not crossing the walls. Every partition gets all of them, and
    //keeps its own when it initializes
    void randomizeParticlePositions(double3 lo = make_double3(1.0/3.0, 0.0, 1.0/3.0), double3 hi = make_double3(2.0/3.0, 2.0/3.0, 2.0/3.0), double swirl = 0.0){
        std::default_random_engine e2(1);   //a fixed seed: the same run gives the same result every time
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
        double voxelSize = particles.grid.cellSize / (2<<particles.refinementLevel);
        double restDensity = particles.size / ((hi.x - lo.x)*width*(hi.y - lo.y)*height*(hi.z - lo.z)*depth / (voxelSize*voxelSize*voxelSize));    //what they start at
        simulation.forEachPartition([&](Particles& partition){
            if(&partition != &particles){
                for(auto [mine, theirs] : {std::pair{&partition.px, &particles.px}, {&partition.py, &particles.py}, {&partition.pz, &particles.pz}}){
                    for(uint i = 0; i < particles.size; ++i){
                        (*mine)[i] = (*theirs)[i];
                    }
                }
                for(auto [mine, theirs] : {std::pair{&partition.vx, &particles.vx}, {&partition.vy, &particles.vy}, {&partition.vz, &particles.vz}}){
                    for(uint i = 0; i < particles.size; ++i){
                        (*mine)[i] = (*theirs)[i];
                    }
                }
            }
            for(CudaVec<double>* position : {&partition.px, &partition.py, &partition.pz}){
                position->upload(partition.stream);
            }
            for(CudaVec<float>* velocity : {&partition.vx, &partition.vy, &partition.vz}){
                velocity->upload(partition.stream);
            }
            partition.setRestDensity(restDensity);
        });
    }

    //every partition starts with these particles, x[i] to w[i] each, and keeps its own when it initializes; the density correction holds them at
    //restDensity particles per voxel
    void setParticles(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z, const std::vector<float>& u, const std::vector<float>& v,
                      const std::vector<float>& w, double restDensity){
        simulation.forEachPartition([&](Particles& partition){
            for(uint i = 0; i < partition.size; ++i){
                partition.px[i] = x[i];
                partition.py[i] = y[i];
                partition.pz[i] = z[i];
                partition.vx[i] = u[i];
                partition.vy[i] = v[i];
                partition.vz[i] = w[i];
            }
            for(CudaVec<double>* position : {&partition.px, &partition.py, &partition.pz}){
                position->upload(partition.stream);
            }
            for(CudaVec<float>* velocity : {&partition.vx, &partition.vy, &partition.vz}){
                velocity->upload(partition.stream);
            }
            partition.setRestDensity(restDensity);
        });
    }

    //APIC's velocity gradients for setParticles' particles, after setApic(true): nine arrays of every particle's, component c along axis a at [3c + a]
    void setAffine(const std::vector<std::vector<float>>& gradients){
        simulation.forEachPartition([&](Particles& partition){
            for(int k = 0; k < 9 && partition.size > 0; ++k){   //from pageable memory, so it's copied before this returns
                gpuErrchk(cudaMemcpyAsync(partition.affine[k].devPtr(), gradients[k].data(), sizeof(float)*partition.size, cudaMemcpyHostToDevice, partition.stream));
            }
        });
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
        simulation.initialize();
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
        simulation.solveFrame(fps);
    }

    void writePositionsToFile(const std::string& fileName){
        simulation.writePositionsToFile(fileName);
    }

    size_t particlesHere() const{
        return simulation.particlesHere();
    }

    void writeDiagnostics(const std::string& path, int frame){
        simulation.writeDiagnostics(path, frame);
    }

    void startCache(const std::string& directory, const CacheDescription& description, int committedBefore, std::function<void(const char*, int)> done){
        simulation.startCache(directory, description, committedBefore, std::move(done));
    }

    void writeCacheFrame(int frame){
        simulation.writeCacheFrame(frame);
    }

    std::string cacheError(){
        return simulation.cacheError();
    }

    std::string finishCache(){
        return simulation.finishCache();
    }

    void writeCheckpoint(int frame){
        simulation.writeCheckpoint(frame);
    }

    void resume(double time, unsigned long long substep){
        simulation.resume(time, substep);
    }

    bool anyRank(bool mine){
        return simulation.anyRank(mine);
    }

    void continueDiagnostics(const std::string& path, int lastFrame){
        simulation.continueDiagnostics(path, lastFrame);
    }

private:
    Simulation simulation;
    Particles& particles;   //the first partition, which the checks look at
    std::map<uint, uint> gridMap;
};

#endif