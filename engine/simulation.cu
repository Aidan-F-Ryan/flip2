//Copyright 2023 Aberrant Behavior LLC

#include "simulation.hu"
#include <thread>

Simulation::Simulation(uint numParticles, int numPartitions)
: totalParticles(numParticles)
, exchange(numPartitions, shared.stream)
, frameWriter(3*sizeof(float)*numParticles)
{
    for(int rank = 0; rank < numPartitions; ++rank){
        contexts.push_back(std::make_unique<LocalPartition>(exchange, rank));
        partitions.push_back(std::make_unique<Particles>(numParticles));   //each starts with room for every particle; initialize keeps its own
        exchange.attach(rank, partitions.back().get());
    }
}

void Simulation::setDomain(double nx, double ny, double nz, uint x, uint y, uint z, double cellSize){
    int count = numPartitions();
    std::vector<uint> planes(count + 1);
    for(int rank = 0; rank < count; ++rank){
        planes[rank] = (uint)((unsigned long long)z*rank/count) & ~1u;
    }
    planes[count] = z;
    for(int rank = 0; rank < count; ++rank){
        if(planes[rank + 1] < planes[rank] + 2){
            std::cerr<<"Simulation: "<<count<<" partitions is too many for "<<z<<" node planes; each needs at least 2\n";
            exit(1);
        }
    }
    for(int rank = 0; rank < count; ++rank){
        Particles& partition = *partitions[rank];
        partition.setPartition(rank, count, &exchange, contexts[rank].get(), planes, shared.stream);
        partition.setDomain(nx, ny, nz, x, y, z, cellSize);
    }
    if(count > 1){
        std::cout<<count<<" partitions, owning node planes";
        for(int rank = 0; rank < count; ++rank){
            std::cout<<" ["<<planes[rank]<<", "<<planes[rank + 1]<<")";
        }
        std::cout<<" along z\n";
    }
}

void Simulation::inLockstep(const std::function<void(Particles&)>& step){
    if(partitions.size() == 1){
        step(*partitions[0]);
        return;
    }
    std::vector<std::thread> threads;
    for(auto& partition : partitions){
        Particles* mine = partition.get();
        threads.emplace_back([&step, mine]{ step(*mine); });
    }
    for(std::thread& thread : threads){
        thread.join();
    }
}

void Simulation::initialize(){
    inLockstep([](Particles& partition){ partition.initialize(); });
}

void Simulation::solveFrame(double fps){
    inLockstep([fps](Particles& partition){ partition.solveFrame(fps); });
}

void Simulation::writePositionsToFile(const std::string& fileName){
    uint offset = 0;
    float* frame = frameWriter.deviceFrame(shared.stream);
    for(auto& partition : partitions){  //all on the shared stream, so the copy to the host comes after every partition's part
        partition->packPositions(frame + 3*offset);
        offset += partition->numParticles();
    }
    if(offset != totalParticles){
        std::cerr<<"Simulation: the partitions hold "<<offset<<" particles between them, not "<<totalParticles<<"\n";
        exit(1);
    }
    frameWriter.write(fileName, shared.stream);
}
