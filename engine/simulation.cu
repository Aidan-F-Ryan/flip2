//Copyright 2023 Aberrant Behavior LLC

#include "simulation.hu"
#include <algorithm>
#include "diagnostics.hu"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

std::vector<uint> Simulation::planesFor(uint z, int numRanks){
    std::vector<uint> planes(numRanks + 1);
    for(int rank = 0; rank < numRanks; ++rank){
        planes[rank] = (uint)((unsigned long long)z*rank/numRanks) & ~1u;
    }
    planes[numRanks] = z;
    for(int rank = 0; rank < numRanks; ++rank){
        if(planes[rank + 1] < planes[rank] + 2){
            return {};
        }
    }
    return planes;
}

void Simulation::setDomain(double nx, double ny, double nz, uint x, uint y, uint z, double cellSize){
    std::vector<uint> planes = planesFor(z, numRanks);
    if(planes.empty()){
        std::cerr<<"Simulation: "<<numRanks<<" ranks is too many for "<<z<<" node planes; each needs at least 2\n";
        exit(1);
    }
    this->planes = planes;
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

std::string Simulation::solveError(){
    return partitions[0]->solveError();
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

void Simulation::writePhaseDiagnostics(const std::string& path, int frame){
    PhaseStatistics phases[2];  //liquid, air
    forEachPartition([&](Particles& partition){ partition.phaseStatistics(phases[0], phases[1]); });
    if(!phaseDiagnostics.is_open()){
        phaseDiagnostics.open(path, std::ios::trunc);
        if(!phaseDiagnostics){
            std::cerr<<"Simulation: can't write phase diagnostics to "<<path<<"\n";
            exit(1);
        }
    }
    const Particles& first = *partitions[0];
    char head[256];
    std::snprintf(head, sizeof(head), "{\"frame\":%d,\"time\":%.17g,\"substeps\":%u,\"floor\":%.9g,\"binHeight\":%.9g", frame, first.elapsedTime, first.substepsThisFrame,
        (double)first.grid.negY, first.grid.sizeY*(double)first.grid.cellSize / PHASE_HEIGHT_BINS);
    phaseDiagnostics<<head;
    const char* names[2] = {"liquid", "air"};
    for(int phase = 0; phase < 2; ++phase){
        const PhaseStatistics& p = phases[phase];
        double centroid[3];
        for(int axis = 0; axis < 3; ++axis){
            centroid[axis] = p.count > 0 ? p.positionSum[axis] / p.count : 0.0;
        }
        char numbers[256];
        std::snprintf(numbers, sizeof(numbers), ",\"%s\":{\"count\":%llu,\"escaped\":%llu,\"kineticEnergy\":%.9g,\"fastest\":%.9g,\"centroid\":", names[phase], p.count, p.escaped,
            p.kineticEnergy, p.fastest);
        phaseDiagnostics<<numbers<<jsonArray(centroid, 3, "%.9g")<<",\"heights\":"<<jsonArray(p.heights, PHASE_HEIGHT_BINS, "%u")<<"}";
    }
    phaseDiagnostics<<"}\n";
    phaseDiagnostics.flush();
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
    //the whitewater as it is: how many of each kind, a hash of every one's state, the same in any order, and how many maxParticles has kept from being made
    WhitewaterStatistics whitewater;
    if(partitions[0]->whitewater.on){
        for(const std::unique_ptr<Particles>& partition : partitions){
            gpuErrchk(cudaSetDevice(partition->device()));
            partition->whitewaterStatistics(whitewater);
        }
        gpuErrchk(cudaSetDevice(partitions[0]->device()));
        if(numPartitions() != numRanks){    //across processes each rank holds its own: all of them, added up
            unsigned long long mine[8] = {whitewater.count, whitewater.kinds[0], whitewater.kinds[1], whitewater.kinds[2], whitewater.hashHigh, whitewater.hashLow, whitewater.dropped, 0};
            std::memcpy(mine + 7, &whitewater.fastest, sizeof(float));
            std::vector<unsigned long long> all(8*(size_t)numRanks);
            transports[0]->allGatherHost(mine, all.data(), sizeof(mine));
            whitewater = WhitewaterStatistics();
            for(int rank = 0; rank < numRanks; ++rank){
                const unsigned long long* theirs = all.data() + 8*(size_t)rank;
                whitewater.count += theirs[0];
                for(int kind = 0; kind < 3; ++kind){
                    whitewater.kinds[kind] += theirs[1 + kind];
                }
                whitewater.hashHigh += theirs[4];
                whitewater.hashLow += theirs[5];
                whitewater.dropped += theirs[6];
                float fastest;
                std::memcpy(&fastest, theirs + 7, sizeof(float));
                whitewater.fastest = std::max(whitewater.fastest, fastest);
            }
        }
    }
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
    diagnostics<<tail<<"\"perVoxel\":"<<jsonArray(d.perVoxel, DIAGNOSTIC_BUCKETS, "%llu")<<",\"core\":"<<jsonArray(d.core, DIAGNOSTIC_BUCKETS, "%llu");
    if(first.whitewater.on){    //dropped: since the frame before, or since the bake started or was resumed
        char more[360];
        std::snprintf(more, sizeof(more), ",\"whitewater\":{\"particles\":%llu,\"spray\":%llu,\"foam\":%llu,\"bubbles\":%llu,\"hash\":\"%016llx%016llx\",\"fastest\":%.9g,\"dropped\":%llu}",
                      whitewater.count, whitewater.kinds[0], whitewater.kinds[1], whitewater.kinds[2], whitewater.hashHigh, whitewater.hashLow, (double)whitewater.fastest,
                      whitewater.dropped - whitewaterDroppedBefore);
        whitewaterDroppedBefore = whitewater.dropped;
        diagnostics<<more;
    }
    diagnostics<<"}\n";
    diagnostics.flush();
}

void Simulation::startCache(const std::string& directory, CacheDescription description, int committedBefore, std::function<void(const char*, int)> done){
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
    cacheWriter = std::make_unique<CacheWriter>(directory, description, ranks, committedBefore, std::move(done));
    cacheCopies.resize(partitions.size());
    whitewaterCopies.resize(partitions.size());
    for(size_t index = 0; index < partitions.size(); ++index){
        gpuErrchk(cudaSetDevice(partitions[index]->device()));
        for(int buffer = 0; buffer < cacheWriter->numBuffers(); ++buffer){
            for(std::vector<std::vector<cudaEvent_t>>* copies : {&cacheCopies, &whitewaterCopies}){
                cudaEvent_t copied;
                gpuErrchk(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
                (*copies)[index].push_back(copied);
            }
        }
    }
    gpuErrchk(cudaSetDevice(partitions[0]->device()));
}

void Simulation::whitewaterHere(unsigned long long& particles, unsigned long long& dropped){
    particles = 0;
    dropped = 0;
    for(const std::unique_ptr<Particles>& partition : partitions){
        gpuErrchk(cudaSetDevice(partition->device()));
        particles += partition->whitewaterCount();
        dropped += partition->whitewaterDropped();
    }
    gpuErrchk(cudaSetDevice(partitions[0]->device()));
}

//each partition here's whitewater particles, or nothing without whitewater
std::vector<uint> Simulation::countWhitewater(){
    std::vector<uint> counts;
    if(partitions[0]->whitewater.on){
        for(const std::unique_ptr<Particles>& partition : partitions){
            gpuErrchk(cudaSetDevice(partition->device()));
            counts.push_back(partition->whitewaterCount());
        }
        gpuErrchk(cudaSetDevice(partitions[0]->device()));
    }
    return counts;
}

void Simulation::writeCacheFrame(int frame){
    bool ids = cacheWriter->writesIds(), ages = cacheWriter->writesAges();
    std::vector<uint> counts;   //each partition's particles that go into the frame: with air, the liquid's alone
    size_t total = 0;
    for(const std::unique_ptr<Particles>& partition : partitions){
        gpuErrchk(cudaSetDevice(partition->device()));
        counts.push_back(partition->countFrameParticles());
        total += counts.back();
    }
    std::vector<uint> whitewaterCounts = countWhitewater();     //with whitewater, each partition's, which goes in after the particles
    size_t fluidBytes = (CacheWriter::frameBytes(total, ids, ages) + 7) / 8*8;
    size_t whitewaterTotal = 0;
    for(uint count : whitewaterCounts){
        whitewaterTotal += count;
    }
    int buffer = cacheWriter->acquire(fluidBytes + CacheWriter::whitewaterBytes(whitewaterTotal, false));  //emitters and sinks change the count from frame to frame
    char* host = cacheWriter->hostBuffer(buffer);
    std::vector<CacheShard> shards, whitewater;
    std::vector<cudaEvent_t> copies;
    size_t offset = 0;
    for(int index = 0; index < numPartitions(); ++index){   //each copies its own planes straight from its GPU
        Particles& partition = *partitions[index];
        gpuErrchk(cudaSetDevice(partition.device()));
        partition.copyFrameColumnsToHost((float*)(host + offset), ids, ages, cacheCopies[index][buffer]);
        shards.push_back({ranks[index], counts[index], offset});
        copies.push_back(cacheCopies[index][buffer]);
        offset += CacheWriter::frameBytes(counts[index], ids, ages);
    }
    offset = fluidBytes;
    for(size_t index = 0; index < whitewaterCounts.size(); ++index){
        Particles& partition = *partitions[index];
        gpuErrchk(cudaSetDevice(partition.device()));
        partition.copyWhitewaterToHost(host + offset, false, whitewaterCopies[index][buffer]);
        whitewater.push_back({ranks[index], whitewaterCounts[index], offset});
        copies.push_back(whitewaterCopies[index][buffer]);
        offset += CacheWriter::whitewaterBytes(whitewaterCounts[index], false);
    }
    gpuErrchk(cudaSetDevice(partitions[0]->device()));
    cacheWriter->submit(buffer, frame, partitions[0]->elapsedTime, shards, copies, whitewater);
}

void Simulation::writeCheckpoint(int frame){
    bool apic = partitions[0]->apic;
    std::vector<uint> whitewaterCounts = countWhitewater();
    size_t fluidBytes = (CacheWriter::checkpointBytes(particlesHere(), apic) + 7) / 8*8;
    size_t whitewaterTotal = 0;
    for(uint count : whitewaterCounts){
        whitewaterTotal += count;
    }
    int buffer = cacheWriter->acquire(fluidBytes + CacheWriter::whitewaterBytes(whitewaterTotal, true));
    char* host = cacheWriter->hostBuffer(buffer);
    std::vector<CacheShard> shards, whitewater;
    std::vector<cudaEvent_t> copies;
    size_t offset = 0;
    CheckpointState state;
    for(int index = 0; index < numPartitions(); ++index){
        Particles& partition = *partitions[index];
        gpuErrchk(cudaSetDevice(partition.device()));
        partition.copyCheckpointToHost(host + offset, cacheCopies[index][buffer]);
        shards.push_back({ranks[index], partition.numParticles(), offset});
        copies.push_back(cacheCopies[index][buffer]);
        offset += CacheWriter::checkpointBytes(partition.numParticles(), apic);
        state.acceleration = std::max(state.acceleration, partition.acceleration());    //this rank's partitions' largest; the record takes every rank's
    }
    offset = fluidBytes;
    for(size_t index = 0; index < whitewaterCounts.size(); ++index){
        Particles& partition = *partitions[index];
        gpuErrchk(cudaSetDevice(partition.device()));
        partition.copyWhitewaterToHost(host + offset, true, whitewaterCopies[index][buffer]);
        whitewater.push_back({ranks[index], whitewaterCounts[index], offset});
        copies.push_back(whitewaterCopies[index][buffer]);
        offset += CacheWriter::whitewaterBytes(whitewaterCounts[index], true);
    }
    gpuErrchk(cudaSetDevice(partitions[0]->device()));
    state.time = partitions[0]->elapsedTime;
    state.substep = partitions[0]->substepIndex;
    state.apic = apic;
    state.nextId = partitions[0]->nextId();
    state.whitewaterNextId = partitions[0]->nextWhitewaterId();
    cacheWriter->submitCheckpoint(buffer, frame, state, shards, copies, whitewater);
}

void Simulation::resume(double time, unsigned long long substep, double acceleration){
    inLockstep([time, substep, acceleration](Particles& partition){ partition.resume(time, substep, acceleration); });
}

bool Simulation::anyRank(bool mine){
    if(numPartitions() == numRanks){    //every rank is here, and sees what this process does
        return mine;
    }
    char says = mine ? 1 : 0;
    std::vector<char> all(numRanks);
    transports[0]->allGatherHost(&says, all.data(), 1);
    return std::any_of(all.begin(), all.end(), [](char each){ return each != 0; });
}

void Simulation::continueDiagnostics(const std::string& path, int lastFrame){
    if(ranks[0] != 0){
        return;
    }
    std::vector<std::string> kept;
    {
        std::ifstream in(path);
        std::string line;
        while(std::getline(in, line)){
            size_t at = line.find("\"frame\":");
            if(at == std::string::npos ? line.find("\"flip2\"") != std::string::npos : std::atoi(line.c_str() + at + 8) <= lastFrame){
                kept.push_back(line);
            }
        }
    }
    if(kept.empty()){   //nothing to carry on: writeDiagnostics starts it afresh
        return;
    }
    diagnostics.open(path, std::ios::trunc);
    if(!diagnostics){
        std::cerr<<"Simulation: can't write diagnostics to "<<path<<"\n";
        exit(1);
    }
    for(const std::string& line : kept){
        diagnostics<<line<<"\n";
    }
    diagnostics.flush();
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
