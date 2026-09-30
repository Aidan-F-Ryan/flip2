//Copyright 2023 Aberrant Behavior LLC

#include "localExchange.hu"
#include "particles.hu"

LocalExchange::LocalExchange(int numPartitions)
: numPartitions(numPartitions)
, partitions(numPartitions, nullptr)
, devices(numPartitions, 0)
, streams(numPartitions, nullptr)
, ready(numPartitions, nullptr)
, consumed(numPartitions, nullptr)
, pointers(numPartitions, nullptr)
, values(numPartitions, 0.0)
, scratchSpace(numPartitions, nullptr)
{
    if(numPartitions < 1 || numPartitions > MAX_PARTITIONS){
        std::cerr<<"LocalExchange: "<<numPartitions<<" partitions; between 1 and "<<MAX_PARTITIONS<<" are supported\n";
        exit(1);
    }
}

LocalExchange::~LocalExchange(){
    for(int rank = 0; rank < numPartitions; ++rank){
        if(partitions[rank] != nullptr){
            cudaSetDevice(devices[rank]);
            cudaEventDestroy(ready[rank]);
            cudaEventDestroy(consumed[rank]);
            cudaFree(scratchSpace[rank]);
        }
    }
}

void LocalExchange::attach(int rank, Particles* partition){
    partitions[rank] = partition;
    devices[rank] = partition->device();
    streams[rank] = partition->stream;
    gpuErrchk(cudaEventCreateWithFlags(&ready[rank], cudaEventDisableTiming));
    gpuErrchk(cudaEventCreateWithFlags(&consumed[rank], cudaEventDisableTiming));
    gpuErrchk(cudaMalloc((void**)&scratchSpace[rank], sizeof(long long)*SCRATCH_WORDS*MAX_PARTITIONS));
}

void LocalExchange::barrier(){
    std::unique_lock<std::mutex> lock(mutex);
    unsigned long long arrivedIn = generation;
    if(++arrived == numPartitions){
        arrived = 0;
        ++generation;
        released.notify_all();
    }
    else{
        released.wait(lock, [&]{ return generation != arrivedIn; });
    }
}

void LocalExchange::startReads(int rank, const std::vector<int>& peers){
    gpuErrchk(cudaEventRecord(ready[rank], streams[rank]));
    barrier();
    for(int peer : peers){
        gpuErrchk(cudaStreamWaitEvent(streams[rank], ready[peer], 0));
    }
}

void LocalExchange::copyFrom(int rank, void* destination, int from, const void* source, size_t bytes){
    if(bytes > 0){
        gpuErrchk(cudaMemcpyPeerAsync(destination, devices[rank], source, devices[from], bytes, streams[rank]));
    }
}

void LocalExchange::finishReads(int rank, const std::vector<int>& peers){
    gpuErrchk(cudaEventRecord(consumed[rank], streams[rank]));
    barrier();
    for(int peer : peers){
        gpuErrchk(cudaStreamWaitEvent(streams[rank], consumed[peer], 0));
    }
}

std::vector<int> LocalExchange::neighbours(int rank) const{
    std::vector<int> next;
    if(rank > 0){
        next.push_back(rank - 1);
    }
    if(rank + 1 < numPartitions){
        next.push_back(rank + 1);
    }
    return next;
}

std::vector<int> LocalExchange::everyone() const{
    std::vector<int> all;
    for(int rank = 0; rank < numPartitions; ++rank){
        all.push_back(rank);
    }
    return all;
}

// ---- kernels ----

//an owner adds a neighbour's ghost copies of its boundary plane's voxels into its own: the run of nodes holding particles, then the empty run
__global__ void addEdgeRuns(uint particleStart, uint particleCount, uint emptyStart, uint emptyCount, const int* ghosts, int* values){
    uint i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < particleCount){
        values[particleStart + i] += ghosts[i];
    }
    else if(i < particleCount + emptyCount){
        values[emptyStart + i - particleCount] += ghosts[i];
    }
}

//every partition's values, a slot each, added up in partition order
template <typename T>
__global__ void addSlots(const T* slots, int numPartitions, uint count, T* sum){
    for(uint i = threadIdx.x; i < count; i += blockDim.x){
        T total = slots[i];
        for(int partition = 1; partition < numPartitions; ++partition){
            total += slots[partition*LocalExchange::SCRATCH_WORDS + i];
        }
        sum[i] = total;
    }
}

// ---- one partition's side ----

LocalPartition::~LocalPartition(){
    for(int* buffer : reduceBuffers){
        cudaFree(buffer);
    }
}

//ghost voxels take their owners' values: the ghosts of each neighbour's boundary plane are its two runs there, in the same order
template <typename T>
void LocalPartition::fill(T* values, cudaStream_t stream){
    if(exchange.size() == 1){
        return;
    }
    exchange.publish(rank, values);
    std::vector<int> peers = exchange.neighbours(rank);
    exchange.startReads(rank, peers);
    Particles& me = exchange.partition(rank);
    for(int side = 0; side < 2; ++side){
        int owner = side == 0 ? rank - 1 : rank + 1;
        if(owner < 0 || owner >= exchange.size()){
            continue;
        }
        Particles& other = exchange.partition(owner);
        const T* theirs = (const T*)exchange.published(owner);
        const Particles::VoxelRun& particles = other.edgeParticles[1 - side];   //the plane on its side facing this partition
        const Particles::VoxelRun& empty = other.edgeEmpty[1 - side];
        exchange.copyFrom(rank, values + me.ghostParticles[side].start, owner, theirs + particles.start, sizeof(T)*particles.count);
        exchange.copyFrom(rank, values + me.ghostEmpty[side].start, owner, theirs + empty.start, sizeof(T)*empty.count);
    }
    exchange.finishReads(rank, peers);
}

void LocalPartition::fillGhosts(float* values, cudaStream_t stream){
    fill(values, stream);
}

void LocalPartition::fillGhosts(char* values, cudaStream_t stream){
    fill(values, stream);
}

int* LocalPartition::reduceBuffer(int side, uint count, cudaStream_t stream){
    if(reduceBufferSizes[side] < count){
        if(reduceBuffers[side] != nullptr){
            gpuErrchk(cudaFreeAsync(reduceBuffers[side], stream));
        }
        gpuErrchk(cudaMallocAsync((void**)&reduceBuffers[side], sizeof(int)*count, stream));
        reduceBufferSizes[side] = count;
    }
    return reduceBuffers[side];
}

//owners add in their neighbours' ghost copies of their voxels: a neighbour's ghosts of this partition's plane are one run, holding particles first
void LocalPartition::reduceGhosts(int* values, cudaStream_t stream){
    if(exchange.size() == 1){
        return;
    }
    exchange.publish(rank, values);
    std::vector<int> peers = exchange.neighbours(rank);
    exchange.startReads(rank, peers);
    Particles& me = exchange.partition(rank);
    for(int side = 0; side < 2; ++side){
        int holder = side == 0 ? rank - 1 : rank + 1;
        if(holder < 0 || holder >= exchange.size()){
            continue;
        }
        Particles& other = exchange.partition(holder);
        const Particles::VoxelRun& particles = other.ghostParticles[1 - side];  //its ghosts of this partition's plane on this side, then the empty ones
        uint count = particles.count + other.ghostEmpty[1 - side].count;
        if(count == 0){
            continue;
        }
        int* ghosts = reduceBuffer(side, count, stream);
        exchange.copyFrom(rank, ghosts, holder, (const int*)exchange.published(holder) + particles.start, sizeof(int)*count);
        const Particles::VoxelRun& mineParticles = me.edgeParticles[side];
        const Particles::VoxelRun& mineEmpty = me.edgeEmpty[side];
        addEdgeRuns<<<count / 128 + 1, 128, 0, stream>>>(mineParticles.start, mineParticles.count, mineEmpty.start, mineEmpty.count, ghosts, values);
        gpuErrchk(cudaPeekAtLastError());
    }
    exchange.finishReads(rank, peers);
}

double LocalPartition::maxOverPartitions(double value){
    if(exchange.size() == 1){
        return value;
    }
    exchange.publishValue(rank, value);
    exchange.barrier();
    double largest = exchange.publishedValue(0);
    for(int partition = 1; partition < exchange.size(); ++partition){
        largest = fmax(largest, exchange.publishedValue(partition));
    }
    exchange.barrier();     //everyone has read before anyone publishes again
    return largest;
}

bool LocalPartition::anyOverPartitions(bool value){
    return maxOverPartitions(value ? 1.0 : 0.0) > 0.0;
}

//each partition's values into a slot of every partition's scratch, then each adds the slots up itself, in partition order, so all get the same sum
void LocalPartition::sumOverPartitions(long long* values, uint count, cudaStream_t stream){
    if(exchange.size() == 1){
        return;
    }
    if(count > LocalExchange::SCRATCH_WORDS){
        std::cerr<<"sumOverPartitions: "<<count<<" values, more than the "<<LocalExchange::SCRATCH_WORDS<<" there's room for\n";
        exit(1);
    }
    exchange.publish(rank, values);
    std::vector<int> peers = exchange.everyone();
    exchange.startReads(rank, peers);
    long long* slots = exchange.scratch(rank);
    for(int partition = 0; partition < exchange.size(); ++partition){
        exchange.copyFrom(rank, slots + partition*LocalExchange::SCRATCH_WORDS, partition, exchange.published(partition), sizeof(long long)*count);
    }
    exchange.finishReads(rank, peers);     //nobody is still reading this partition's values when it overwrites them
    addSlots<<<1, 32, 0, stream>>>(slots, exchange.size(), count, values);
    gpuErrchk(cudaPeekAtLastError());
}

void LocalPartition::sumOverPartitions(double* value, cudaStream_t stream){
    if(exchange.size() == 1){
        return;
    }
    exchange.publish(rank, value);
    std::vector<int> peers = exchange.everyone();
    exchange.startReads(rank, peers);
    double* slots = (double*)exchange.scratch(rank);
    for(int partition = 0; partition < exchange.size(); ++partition){
        exchange.copyFrom(rank, slots + partition*LocalExchange::SCRATCH_WORDS, partition, exchange.published(partition), sizeof(double));
    }
    exchange.finishReads(rank, peers);
    addSlots<<<1, 32, 0, stream>>>(slots, exchange.size(), 1u, value);
    gpuErrchk(cudaPeekAtLastError());
}

//the first coarse grid's cells are 2x2x2 voxels, so a node plane holds 2 of its planes, and a partition's cells are one run of it
void LocalPartition::gatherFirstLevel(void* cells, size_t bytesPerCell, cudaStream_t stream){
    if(exchange.size() == 1){
        return;
    }
    exchange.publish(rank, cells);
    std::vector<int> peers = exchange.everyone();
    exchange.startReads(rank, peers);
    Particles& me = exchange.partition(rank);
    size_t cellsPerPlane = (size_t)(2*me.grid.sizeX)*(2*me.grid.sizeY);
    for(int partition = 0; partition < exchange.size(); ++partition){
        if(partition != rank){
            size_t first = 2*me.partitionPlanes[partition]*cellsPerPlane*bytesPerCell;
            size_t bytes = 2*(me.partitionPlanes[partition + 1] - me.partitionPlanes[partition])*cellsPerPlane*bytesPerCell;
            exchange.copyFrom(rank, (char*)cells + first, partition, (const char*)exchange.published(partition) + first, bytes);
        }
    }
    exchange.finishReads(rank, peers);
}
