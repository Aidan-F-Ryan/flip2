//Copyright 2023 Aberrant Behavior LLC

#include "simulation.hu"
#include <algorithm>
#include "diagnostics.hu"
#include <chrono>
#include <cstdio>
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
    frameWriter = std::make_unique<FrameWriter>(std::max<size_t>(3*sizeof(float)*totalParticles, 4096));   //it grows if emitters add particles
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
    cacheWriter.reset();    //writes and commits every frame, and
    frameWriter.reset();    //writes every frame, so nothing waits on the events any more
    for(int index = numPartitions() - 1; index >= 0; --index){     //each partition's things, freed on its own GPU
        gpuErrchk(cudaSetDevice(partitions[index]->device()));
        if(index < (int)frameCopies.size()){
            for(cudaEvent_t copied : frameCopies[index]){
                cudaEventDestroy(copied);
            }
        }
        if(index < (int)cacheCopies.size()){
            for(cudaEvent_t copied : cacheCopies[index]){
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
    this->planes = planes;
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

size_t Simulation::particlesHere() const{
    size_t total = 0;
    for(const auto& partition : partitions){
        total += partition->numParticles();
    }
    return total;
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
    auto start = std::chrono::steady_clock::now();
    inLockstep([fps](Particles& partition){ partition.solveFrame(fps); });
    lastFrameSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

//a JSON array of n numbers
template <typename T>
static std::string jsonArray(const T* values, int n, const char* format){
    std::string text = "[";
    char number[32];
    for(int i = 0; i < n; ++i){
        std::snprintf(number, sizeof(number), format, values[i]);
        text += (i ? "," : "") + std::string(number);
    }
    return text + "]";
}

void Simulation::writeDiagnostics(const std::string& path, int frame){
    std::vector<FrameDiagnostics> results(partitions.size());
    inLockstep([&](Particles& partition){   //every partition takes part in the sums, and each ends with the same totals
        for(size_t index = 0; index < partitions.size(); ++index){
            if(partitions[index].get() == &partition){
                results[index] = computeFrameDiagnostics(partition);
            }
        }
    });
    if(ranks[0] != 0){
        return;
    }
    const Particles& first = *partitions[0];
    if(!diagnostics.is_open()){
        diagnostics.open(path, std::ios::trunc);
        if(!diagnostics){
            std::cerr<<"Simulation: can't write diagnostics to "<<path<<"\n";
            exit(1);
        }
        cudaDeviceProp properties;
        gpuErrchk(cudaGetDeviceProperties(&properties, first.device()));
        int runtime = 0;
        cudaRuntimeGetVersion(&runtime);
        double voxelSize = first.grid.cellSize / (2<<first.refinementLevel);
        uint nodes[3] = {first.grid.sizeX, first.grid.sizeY, first.grid.sizeZ};
        char line[512];
        std::snprintf(line, sizeof(line), "{\"flip2\":\"diagnostics\",\"version\":1,\"ranks\":%d,\"gpu\":\"%s\",\"sm\":%d,\"cudaRuntime\":%d,\"nodes\":%s,\"cellSize\":%.9g,\"voxelSize\":%.9g}\n",
            numRanks, properties.name, properties.major*10 + properties.minor, runtime, jsonArray(nodes, 3, "%u").c_str(), (double)first.grid.cellSize, voxelSize);
        diagnostics<<line;
    }
    const FrameDiagnostics& d = results[0];
    double voxelSize = first.grid.cellSize / (2<<first.refinementLevel);
    double centroid[3];
    for(int axis = 0; axis < 3; ++axis){
        centroid[axis] = d.particles > 0 ? d.positionSum[axis] / d.particles : 0.0;
    }
    char head[1024];
    std::snprintf(head, sizeof(head), "{\"frame\":%d,\"time\":%.17g,\"substeps\":%u,\"dtMin\":%.17g,\"dtMax\":%.17g,\"wallSeconds\":%.6f,\"particles\":%llu,\"hash\":\"%016llx%016llx\",\"kineticEnergy\":%.17g,",
        frame, first.elapsedTime, first.substepsThisFrame, first.smallestDtThisFrame, first.largestDtThisFrame, frame > 0 ? lastFrameSeconds : 0.0, d.particles, d.hashHigh, d.hashLow, d.kineticEnergy);
    diagnostics<<head
               <<"\"momentum\":"<<jsonArray(d.momentum, 3, "%.17g")<<",\"angularMomentum\":"<<jsonArray(d.angularMomentum, 3, "%.17g")
               <<",\"centroid\":"<<jsonArray(centroid, 3, "%.17g")<<",\"lowest\":"<<jsonArray(d.lowest, 3, "%.17g")<<",\"highest\":"<<jsonArray(d.highest, 3, "%.17g");
    char tail[256];
    std::snprintf(tail, sizeof(tail), ",\"fastest\":%.9g,\"occupiedVoxels\":%llu,\"volume\":%.17g,", d.fastest, d.occupiedVoxels, d.occupiedVoxels*voxelSize*voxelSize*voxelSize);
    diagnostics<<tail<<"\"perVoxel\":"<<jsonArray(d.perVoxel, DIAGNOSTIC_BUCKETS, "%llu")<<",\"core\":"<<jsonArray(d.core, DIAGNOSTIC_BUCKETS, "%llu")<<"}\n";
    diagnostics.flush();
}

void Simulation::startCache(const std::string& directory, CacheDescription description, std::function<void(int)> committed){
    const Particles& first = *partitions[0];
    cudaDeviceProp properties;
    gpuErrchk(cudaGetDeviceProperties(&properties, first.device()));
    cudaRuntimeGetVersion(&description.cudaRuntime);
    description.gpu = properties.name;
    description.sm = properties.major*10 + properties.minor;
    description.worldSize = numRanks;
    description.partitionPlanes = planes;
    description.nodes[0] = first.grid.sizeX;
    description.nodes[1] = first.grid.sizeY;
    description.nodes[2] = first.grid.sizeZ;
    description.nodeSize = first.grid.cellSize;
    description.voxelSize = first.grid.cellSize / (2<<first.refinementLevel);
    description.domainMin[0] = first.grid.negX;
    description.domainMin[1] = first.grid.negY;
    description.domainMin[2] = first.grid.negZ;
    cacheWriter = std::make_unique<CacheWriter>(directory, description, ranks, std::move(committed));
    cacheCopies.resize(partitions.size());
    for(size_t index = 0; index < partitions.size(); ++index){
        gpuErrchk(cudaSetDevice(partitions[index]->device()));
        for(int buffer = 0; buffer < cacheWriter->numBuffers(); ++buffer){
            cudaEvent_t copied;
            gpuErrchk(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
            cacheCopies[index].push_back(copied);
        }
    }
    gpuErrchk(cudaSetDevice(partitions[0]->device()));
}

void Simulation::writeCacheFrame(int frame){
    int buffer = cacheWriter->acquire(6*particlesHere());   //emitters and sinks change the count from frame to frame
    float* host = cacheWriter->hostBuffer(buffer);
    std::vector<CacheShard> shards;
    std::vector<cudaEvent_t> copies;
    size_t offset = 0;
    for(int index = 0; index < numPartitions(); ++index){   //each copies its own planes straight from its GPU
        Particles& partition = *partitions[index];
        gpuErrchk(cudaSetDevice(partition.device()));
        partition.copyFrameColumnsToHost(host + offset, cacheCopies[index][buffer]);
        shards.push_back({ranks[index], partition.numParticles(), offset});
        copies.push_back(cacheCopies[index][buffer]);
        offset += 6*(size_t)partition.numParticles();
    }
    gpuErrchk(cudaSetDevice(partitions[0]->device()));
    cacheWriter->submit(buffer, frame, partitions[0]->elapsedTime, shards, copies);
}

std::string Simulation::cacheError(){
    return cacheWriter ? cacheWriter->error() : "";
}

std::string Simulation::finishCache(){
    if(!cacheWriter){
        return "";
    }
    cacheWriter->flush();
    return cacheWriter->error();
}

void Simulation::writePositionsToFile(const std::string& fileName){
    if(numPartitions() == numRanks){    //every partition here: each copies its part straight into the frame, from its own GPU
        size_t total = particlesHere();     //emitters and sinks change it from frame to frame
        int buffer = frameWriter->acquire(sizeof(float)*3*total);
        float* frame = frameWriter->hostBuffer(buffer);
        std::vector<cudaEvent_t> copies;
        size_t offset = 0;
        for(int index = 0; index < numPartitions(); ++index){
            Particles& partition = *partitions[index];
            gpuErrchk(cudaSetDevice(partition.device()));
            partition.copyPositionsToHost(frame + 3*offset, frameCopies[index][buffer]);
            copies.push_back(frameCopies[index][buffer]);
            offset += partition.numParticles();
        }
        frameWriter->submit(buffer, fileName, copies, sizeof(float)*3*total);
        return;
    }
    //across processes: every rank sends rank 0 its positions, which it gathers in rank order and writes out
    Particles& partition = *partitions[0];
    Transport& transport = *transports[0];
    gpuErrchk(cudaSetDevice(partition.device()));
    uint mine = partition.numParticles();
    std::vector<uint> counts(numRanks);
    transport.allGatherHost(&mine, counts.data(), sizeof(mine));
    size_t total = 0;
    for(uint count : counts){
        total += count;
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
    int buffer = frameWriter->acquire(sizeof(float)*3*total);
    gpuErrchk(cudaMemcpyAsync(frameWriter->hostBuffer(buffer), gatheredFrame.devPtr(), sizeof(float)*3*total, cudaMemcpyDeviceToHost, partition.stream));
    gpuErrchk(cudaEventRecord(frameCopies[0][buffer], partition.stream));
    frameWriter->submit(buffer, fileName, {frameCopies[0][buffer]}, sizeof(float)*3*total);
}
