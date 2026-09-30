//Copyright 2023 Aberrant Behavior LLC

#include "simulation.hu"
#include <thread>

Simulation::Simulation(uint numParticles, int numPartitions, int numDevices)
: totalParticles(numParticles)
, exchange(numPartitions)
{
    int available;
    gpuErrchk(cudaGetDeviceCount(&available));
    this->numDevices = numDevices > 0 && numDevices < available ? numDevices : available;
    if(this->numDevices > numPartitions){
        this->numDevices = numPartitions;
    }
    for(int rank = 0; rank < numPartitions; ++rank){
        gpuErrchk(cudaSetDevice(rank*this->numDevices/numPartitions));    //neighbours share a GPU when there are more partitions than GPUs
        contexts.push_back(std::make_unique<LocalPartition>(exchange, rank));
        partitions.push_back(std::make_unique<Particles>(numParticles));   //each starts with room for every particle; initialize keeps its own
        exchange.attach(rank, partitions.back().get());
    }
    for(int device = 0; device < this->numDevices; ++device){  //direct copies between GPUs wherever they can; elsewhere the driver stages them through the host
        gpuErrchk(cudaSetDevice(device));
        for(int peer = 0; peer < this->numDevices; ++peer){
            int canAccess = 0;
            if(peer != device && cudaDeviceCanAccessPeer(&canAccess, device, peer) == cudaSuccess && canAccess){
                cudaError_t enabled = cudaDeviceEnablePeerAccess(peer, 0);
                if(enabled != cudaSuccess && enabled != cudaErrorPeerAccessAlreadyEnabled){
                    gpuErrchk(enabled);
                }
                cudaGetLastError();     //clears an already-enabled error
            }
        }
    }
    frameWriter = std::make_unique<FrameWriter>(3*sizeof(float)*numParticles);
    frameCopies.resize(numPartitions);
    for(int rank = 0; rank < numPartitions; ++rank){
        gpuErrchk(cudaSetDevice(partitions[rank]->device()));
        for(int buffer = 0; buffer < frameWriter->numBuffers(); ++buffer){
            cudaEvent_t copied;
            gpuErrchk(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
            frameCopies[rank].push_back(copied);
        }
    }
    gpuErrchk(cudaSetDevice(partitions[0]->device()));
}

Simulation::~Simulation(){
    frameWriter.reset();    //writes every frame, so nothing waits on the events any more
    for(int rank = numPartitions() - 1; rank >= 0; --rank){     //each partition's things, freed on its own GPU
        gpuErrchk(cudaSetDevice(partitions[rank]->device()));
        for(cudaEvent_t copied : frameCopies[rank]){
            cudaEventDestroy(copied);
        }
        contexts[rank].reset();
        partitions[rank].reset();
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
        gpuErrchk(cudaSetDevice(partition.device()));
        partition.setPartition(rank, count, &exchange, contexts[rank].get(), planes);
        partition.setDomain(nx, ny, nz, x, y, z, cellSize);
    }
    if(count > 1){
        std::cout<<count<<" partitions on "<<numDevices<<" GPU"<<(numDevices > 1 ? "s" : "")<<", owning node planes";
        for(int rank = 0; rank < count; ++rank){
            std::cout<<" ["<<planes[rank]<<", "<<planes[rank + 1]<<") on GPU "<<partitions[rank]->device()<<(rank + 1 < count ? "," : "");
        }
        std::cout<<" along z\n";
    }
}

void Simulation::forEachPartition(const std::function<void(Particles&)>& setting){
    for(auto& partition : partitions){
        gpuErrchk(cudaSetDevice(partition->device()));
        setting(*partition);
    }
}

void Simulation::inLockstep(const std::function<void(Particles&)>& step){
    if(partitions.size() == 1){
        gpuErrchk(cudaSetDevice(partitions[0]->device()));
        step(*partitions[0]);
        return;
    }
    std::vector<std::thread> threads;
    for(auto& partition : partitions){
        Particles* mine = partition.get();
        threads.emplace_back([&step, mine]{
            gpuErrchk(cudaSetDevice(mine->device()));  //every CUDA call on this thread goes to the partition's GPU
            step(*mine);
        });
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
    int buffer = frameWriter->acquire();
    float* frame = frameWriter->hostBuffer(buffer);
    std::vector<cudaEvent_t> copies;
    uint offset = 0;
    for(int rank = 0; rank < numPartitions(); ++rank){  //each partition copies its part in from its own GPU
        Particles& partition = *partitions[rank];
        if(offset + partition.numParticles() > totalParticles){
            std::cerr<<"Simulation: the partitions hold more than the "<<totalParticles<<" particles there are\n";
            exit(1);
        }
        gpuErrchk(cudaSetDevice(partition.device()));
        partition.copyPositionsToHost(frame + 3*(size_t)offset, frameCopies[rank][buffer]);
        copies.push_back(frameCopies[rank][buffer]);
        offset += partition.numParticles();
    }
    if(offset != totalParticles){
        std::cerr<<"Simulation: the partitions hold "<<offset<<" particles between them, not "<<totalParticles<<"\n";
        exit(1);
    }
    frameWriter->submit(buffer, fileName, copies);
}
