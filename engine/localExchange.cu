//Copyright 2023 Aberrant Behavior LLC

#include "localExchange.hu"
#include "particles.hu"

static const uint SCRATCH_WORDS = 64;   //64-bit words of scratch per partition: enough for an exact sum's digits, or a double

LocalExchange::LocalExchange(int numPartitions, cudaStream_t stream)
: numPartitions(numPartitions)
, sharedStream(stream)
, partitions(numPartitions, nullptr)
, pointers(numPartitions, nullptr)
, values(numPartitions, 0.0)
, scratchSpace(numPartitions, nullptr)
{
    if(numPartitions < 1 || numPartitions > MAX_PARTITIONS){
        std::cerr<<"LocalExchange: "<<numPartitions<<" partitions; between 1 and "<<MAX_PARTITIONS<<" are supported\n";
        exit(1);
    }
    for(void*& space : scratchSpace){
        gpuErrchk(cudaMalloc(&space, SCRATCH_WORDS*sizeof(long long)));
    }
}

LocalExchange::~LocalExchange(){
    for(void* space : scratchSpace){
        cudaFree(space);
    }
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

// ---- kernels ----

template <typename T>
struct EveryPartition{      //one array per partition, passed by value
    const T* array[MAX_PARTITIONS];
};

//ghost voxels take their owners' values: a block per ghost node, whose voxels are one run in its owner's arrays, in the same order.
//A range is the ghost's first voxel, the owner's first voxel, how many, and which partition owns it
template <typename T>
__global__ void copyFromOwners(const uint* ranges, EveryPartition<T> owners, T* values){
    const uint* range = ranges + 4*blockIdx.x;
    const T* owner = owners.array[range[3]];
    for(uint i = threadIdx.x; i < range[2]; i += blockDim.x){
        values[range[0] + i] = owner[range[1] + i];
    }
}

//an owner adds one partition's ghost copies of its voxels into its own
__global__ void addFromGhosts(const uint* ranges, uint owner, const int* ghosts, int* values){
    const uint* range = ranges + 4*blockIdx.x;
    if(range[3] == owner){
        for(uint i = threadIdx.x; i < range[2]; i += blockDim.x){
            values[range[1] + i] += ghosts[range[0] + i];
        }
    }
}

__global__ void sumIntegersOverPartitions(EveryPartition<long long> partitions, int numPartitions, uint count, long long* sum){
    for(uint i = threadIdx.x; i < count; i += blockDim.x){
        long long total = 0;
        for(int partition = 0; partition < numPartitions; ++partition){
            total += partitions.array[partition][i];
        }
        sum[i] = total;
    }
}

__global__ void sumDoublesOverPartitions(EveryPartition<double> partitions, int numPartitions, double* sum){
    if(threadIdx.x == 0){
        double total = *partitions.array[0];
        for(int partition = 1; partition < numPartitions; ++partition){
            total += *partitions.array[partition];
        }
        *sum = total;
    }
}

// ---- one partition's side ----

template <typename T>
void LocalPartition::fill(T* values, cudaStream_t stream){
    if(exchange.size() == 1){
        return;
    }
    exchange.publish(rank, values);
    exchange.barrier();
    EveryPartition<T> owners;
    for(int partition = 0; partition < exchange.size(); ++partition){
        owners.array[partition] = (const T*)exchange.published(partition);
    }
    Particles& me = exchange.partition(rank);
    uint ghosts = me.numStoredNodes - me.numOwnNodes;
    if(ghosts > 0){
        copyFromOwners<<<ghosts, 128, 0, stream>>>(me.ghostRanges.devPtr(), owners, values);
        gpuErrchk(cudaPeekAtLastError());
    }
    exchange.barrier();
}

void LocalPartition::fillGhosts(float* values, cudaStream_t stream){
    fill(values, stream);
}

void LocalPartition::fillGhosts(char* values, cudaStream_t stream){
    fill(values, stream);
}

void LocalPartition::reduceGhosts(int* values, cudaStream_t stream){
    if(exchange.size() == 1){
        return;
    }
    exchange.publish(rank, values);
    exchange.barrier();
    for(int holder = 0; holder < exchange.size(); ++holder){   //integers: the order they're added in doesn't matter
        Particles& partition = exchange.partition(holder);
        uint ghosts = partition.numStoredNodes - partition.numOwnNodes;
        if(holder != rank && ghosts > 0){
            addFromGhosts<<<ghosts, 128, 0, stream>>>(partition.ghostRanges.devPtr(), rank, (const int*)exchange.published(holder), values);
            gpuErrchk(cudaPeekAtLastError());
        }
    }
    exchange.barrier();
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

void LocalPartition::sumOverPartitions(long long* values, uint count, cudaStream_t stream){
    if(exchange.size() == 1){
        return;
    }
    if(count > SCRATCH_WORDS){
        std::cerr<<"sumOverPartitions: "<<count<<" values, more than the "<<SCRATCH_WORDS<<" there's room for\n";
        exit(1);
    }
    exchange.publish(rank, values);
    exchange.barrier();
    EveryPartition<long long> partitions;
    for(int partition = 0; partition < exchange.size(); ++partition){
        partitions.array[partition] = (const long long*)exchange.published(partition);
    }
    long long* sum = (long long*)exchange.scratch(rank);
    sumIntegersOverPartitions<<<1, 32, 0, stream>>>(partitions, exchange.size(), count, sum);
    gpuErrchk(cudaPeekAtLastError());
    exchange.barrier();     //every partition has queued its reads of the others' values before any overwrites its own
    gpuErrchk(cudaMemcpyAsync(values, sum, sizeof(long long)*count, cudaMemcpyDeviceToDevice, stream));
}

void LocalPartition::sumOverPartitions(double* value, cudaStream_t stream){
    if(exchange.size() == 1){
        return;
    }
    exchange.publish(rank, value);
    exchange.barrier();
    EveryPartition<double> partitions;
    for(int partition = 0; partition < exchange.size(); ++partition){
        partitions.array[partition] = (const double*)exchange.published(partition);
    }
    double* sum = (double*)exchange.scratch(rank);
    sumDoublesOverPartitions<<<1, 32, 0, stream>>>(partitions, exchange.size(), sum);
    gpuErrchk(cudaPeekAtLastError());
    exchange.barrier();
    gpuErrchk(cudaMemcpyAsync(value, sum, sizeof(double), cudaMemcpyDeviceToDevice, stream));
}

//the first coarse grid's cells are 2x2x2 voxels, so a node plane holds 2 of its planes, and a partition's cells are one run of it
void LocalPartition::gatherFirstLevel(void* cells, size_t bytesPerCell, cudaStream_t stream){
    if(exchange.size() == 1){
        return;
    }
    exchange.publish(rank, cells);
    exchange.barrier();
    Particles& me = exchange.partition(rank);
    size_t cellsPerPlane = (size_t)(2*me.grid.sizeX)*(2*me.grid.sizeY);
    for(int partition = 0; partition < exchange.size(); ++partition){
        if(partition != rank){
            size_t first = 2*me.partitionPlanes[partition]*cellsPerPlane*bytesPerCell;
            size_t bytes = 2*(me.partitionPlanes[partition + 1] - me.partitionPlanes[partition])*cellsPerPlane*bytesPerCell;
            gpuErrchk(cudaMemcpyAsync((char*)cells + first, (const char*)exchange.published(partition) + first, bytes, cudaMemcpyDeviceToDevice, stream));
        }
    }
    exchange.barrier();
}
