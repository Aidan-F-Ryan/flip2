//Copyright 2023 Aberrant Behavior LLC

#include "simulation.hu"
#include <thread>

Simulation::Simulation(uint numParticles, int numPartitions, int numDevices)
: totalParticles(numParticles)
, numRanks(numPartitions)
, hub(std::make_unique<LocalHub>(numPartitions))
{
    int available;
    gpuErrchk(cudaGetDeviceCount(&available));
    this->numDevices = numDevices > 0 && numDevices < available ? numDevices : available;
    if(this->numDevices > numPartitions){
        this->numDevices = numPartitions;
    }
    for(int rank = 0; rank < numPartitions; ++rank){
        gpuErrchk(cudaSetDevice(rank*this->numDevices/numPartitions));    //neighbours share a GPU when there are more partitions than GPUs
        ranks.push_back(rank);
        partitions.push_back(std::make_unique<Particles>(numParticles));   //each starts with room for every particle; initialize keeps its own
        hub->attach(rank, partitions.back()->device());
        transports.push_back(std::make_unique<LocalTransport>(*hub, rank));
        contexts.push_back(std::make_unique<PartitionExchange>(*transports.back(), *partitions.back()));
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
    startFrameWriter();
    gpuErrchk(cudaSetDevice(partitions[0]->device()));
}

Simulation::Simulation(uint numParticles, std::unique_ptr<Transport> transport, int device)
: totalParticles(numParticles)
, numRanks(transport->size())
, numDevices(1)
{
    gpuErrchk(cudaSetDevice(device));
    ranks.push_back(transport->rank());
    partitions.push_back(std::make_unique<Particles>(numParticles));   //starts with room for every particle; initialize keeps its own
    transport->attach(partitions.back()->stream);     //with this rank's GPU current: NCCL makes its communicator here
    transports.push_back(std::move(transport));
    contexts.push_back(std::make_unique<PartitionExchange>(*transports.back(), *partitions.back()));
    if(ranks[0] == 0){
        startFrameWriter();
    }
}

void Simulation::startFrameWriter(){
    frameWriter = std::make_unique<FrameWriter>(3*sizeof(float)*totalParticles);
    frameCopies.resize(partitions.size());
    for(size_t index = 0; index < partitions.size(); ++index){
        gpuErrchk(cudaSetDevice(partitions[index]->device()));
        for(int buffer = 0; buffer < frameWriter->numBuffers(); ++buffer){
            cudaEvent_t copied;
            gpuErrchk(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
            frameCopies[index].push_back(copied);
        }
    }
}

Simulation::~Simulation(){
    frameWriter.reset();    //writes every frame, so nothing waits on the events any more
    for(int index = numPartitions() - 1; index >= 0; --index){     //each partition's things, freed on its own GPU
        gpuErrchk(cudaSetDevice(partitions[index]->device()));
        if(index < (int)frameCopies.size()){
            for(cudaEvent_t copied : frameCopies[index]){
                cudaEventDestroy(copied);
            }
        }
        contexts[index].reset();
        partitions[index].reset();
    }
}

void Simulation::setDomain(double nx, double ny, double nz, uint x, uint y, uint z, double cellSize){
    std::vector<uint> planes(numRanks + 1);
    for(int rank = 0; rank < numRanks; ++rank){
        planes[rank] = (uint)((unsigned long long)z*rank/numRanks) & ~1u;
    }
    planes[numRanks] = z;
    for(int rank = 0; rank < numRanks; ++rank){
        if(planes[rank + 1] < planes[rank] + 2){
            std::cerr<<"Simulation: "<<numRanks<<" ranks is too many for "<<z<<" node planes; each needs at least 2\n";
            exit(1);
        }
    }
    for(int index = 0; index < numPartitions(); ++index){
        Particles& partition = *partitions[index];
        gpuErrchk(cudaSetDevice(partition.device()));
        partition.setPartition(ranks[index], numRanks, transports[index].get(), contexts[index].get(), planes);
        partition.setDomain(nx, ny, nz, x, y, z, cellSize);
    }
    if(numRanks > 1 && ranks[0] == 0){
        if(numPartitions() == numRanks){
            std::cout<<numRanks<<" partitions on "<<numDevices<<" GPU"<<(numDevices > 1 ? "s" : "")<<", owning node planes";
        }
        else{
            std::cout<<numRanks<<" ranks in separate processes, owning node planes";
        }
        for(int rank = 0; rank < numRanks; ++rank){
            std::cout<<" ["<<planes[rank]<<", "<<planes[rank + 1]<<")";
            if(numPartitions() == numRanks){
                std::cout<<" on GPU "<<partitions[rank]->device();
            }
            std::cout<<(rank + 1 < numRanks ? "," : "");
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
    if(numPartitions() == numRanks){    //every partition here: each copies its part straight into the frame, from its own GPU
        int buffer = frameWriter->acquire();
        float* frame = frameWriter->hostBuffer(buffer);
        std::vector<cudaEvent_t> copies;
        uint offset = 0;
        for(int index = 0; index < numPartitions(); ++index){
            Particles& partition = *partitions[index];
            if(offset + partition.numParticles() > totalParticles){
                std::cerr<<"Simulation: the partitions hold more than the "<<totalParticles<<" particles there are\n";
                exit(1);
            }
            gpuErrchk(cudaSetDevice(partition.device()));
            partition.copyPositionsToHost(frame + 3*(size_t)offset, frameCopies[index][buffer]);
            copies.push_back(frameCopies[index][buffer]);
            offset += partition.numParticles();
        }
        if(offset != totalParticles){
            std::cerr<<"Simulation: the partitions hold "<<offset<<" particles between them, not "<<totalParticles<<"\n";
            exit(1);
        }
        frameWriter->submit(buffer, fileName, copies);
        return;
    }
    //across processes: every rank sends rank 0 its positions, which it gathers in rank order and writes out
    Particles& partition = *partitions[0];
    Transport& transport = *transports[0];
    gpuErrchk(cudaSetDevice(partition.device()));
    uint mine = partition.numParticles();
    std::vector<uint> counts(numRanks);
    transport.allGatherHost(&mine, counts.data(), sizeof(mine));
    uint total = 0;
    for(uint count : counts){
        total += count;
    }
    if(total != totalParticles){
        std::cerr<<"Simulation: the ranks hold "<<total<<" particles between them, not "<<totalParticles<<"\n";
        exit(1);
    }
    const float* packed = partition.packPositions();
    std::vector<TransportSend> sends;
    std::vector<TransportReceive> receives;
    if(ranks[0] != 0){
        sends.push_back({0, packed, sizeof(float)*3*mine});
        transport.exchange(sends, receives, partition.stream);
        return;
    }
    gatheredFrame.resizeAsync(3*total, partition.stream);
    if(mine > 0){
        gpuErrchk(cudaMemcpyAsync(gatheredFrame.devPtr(), packed, sizeof(float)*3*mine, cudaMemcpyDeviceToDevice, partition.stream));
    }
    size_t offset = mine;
    for(int rank = 1; rank < numRanks; ++rank){
        receives.push_back({rank, gatheredFrame.devPtr() + 3*offset, sizeof(float)*3*counts[rank]});
        offset += counts[rank];
    }
    transport.exchange(sends, receives, partition.stream);
    int buffer = frameWriter->acquire();
    gpuErrchk(cudaMemcpyAsync(frameWriter->hostBuffer(buffer), gatheredFrame.devPtr(), sizeof(float)*3*total, cudaMemcpyDeviceToHost, partition.stream));
    gpuErrchk(cudaEventRecord(frameCopies[0][buffer], partition.stream));
    frameWriter->submit(buffer, fileName, {frameCopies[0][buffer]});
}
