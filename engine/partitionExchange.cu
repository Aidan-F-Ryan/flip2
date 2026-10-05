//Copyright 2023 Aberrant Behavior LLC

#include "partitionExchange.hu"
#include "particles.hu"

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

//every rank's count values, one run each in rank order, added up in that order
template <typename T>
__global__ void addInRankOrder(const T* gathered, int ranks, uint count, T* sum){
    for(uint i = threadIdx.x; i < count; i += blockDim.x){
        T total = gathered[i];
        for(int rank = 1; rank < ranks; ++rank){
            total += gathered[rank*count + i];
        }
        sum[i] = total;
    }
}

PartitionExchange::~PartitionExchange(){
    for(void* allocated : buffers){
        cudaFree(allocated);
    }
}

void* PartitionExchange::buffer(int which, size_t bytes, cudaStream_t stream){
    if(bufferBytes[which] < bytes){
        if(buffers[which] != nullptr){
            gpuErrchk(cudaFreeAsync(buffers[which], stream));
        }
        gpuErrchk(cudaMallocAsync(&buffers[which], bytes, stream));
        bufferBytes[which] = bytes;
    }
    return buffers[which];
}

//Ghost voxels take their owners' values. Each partition sends the neighbour on each side its boundary plane's two runs, and receives the neighbour's
//into its ghost runs, which are laid out the same way (checkGhostRuns made sure)
template <typename T>
void PartitionExchange::fill(T* values, cudaStream_t stream){
    if(transport.size() == 1){
        return;
    }
    std::vector<TransportSend> sends;
    std::vector<TransportReceive> receives;
    for(int side = 0; side < 2; ++side){
        int neighbour = side == 0 ? me.rank - 1 : me.rank + 1;
        if(neighbour < 0 || neighbour >= transport.size()){
            continue;
        }
        sends.push_back({neighbour, values + me.edgeParticles[side].start, sizeof(T)*me.edgeParticles[side].count});
        sends.push_back({neighbour, values + me.edgeEmpty[side].start, sizeof(T)*me.edgeEmpty[side].count});
        receives.push_back({neighbour, values + me.ghostParticles[side].start, sizeof(T)*me.ghostParticles[side].count});
        receives.push_back({neighbour, values + me.ghostEmpty[side].start, sizeof(T)*me.ghostEmpty[side].count});
    }
    transport.exchange(sends, receives, stream);
}

void PartitionExchange::fillGhosts(float* values, cudaStream_t stream){
    fill(values, stream);
}

void PartitionExchange::fillGhosts(char* values, cudaStream_t stream){
    fill(values, stream);
}

//Owners add in their neighbours' ghost copies of their voxels: each partition sends the neighbour on each side its ghosts of that neighbour's plane, one
//run, holding particles first, and adds what it receives into its own boundary plane's two runs
void PartitionExchange::reduceGhosts(int* values, cudaStream_t stream){
    if(transport.size() == 1){
        return;
    }
    std::vector<TransportSend> sends;
    std::vector<TransportReceive> receives;
    int* ghosts[2] = {nullptr, nullptr};
    for(int side = 0; side < 2; ++side){
        int neighbour = side == 0 ? me.rank - 1 : me.rank + 1;
        if(neighbour < 0 || neighbour >= transport.size()){
            continue;
        }
        uint sent = me.ghostParticles[side].count + me.ghostEmpty[side].count;
        uint received = me.edgeParticles[side].count + me.edgeEmpty[side].count;
        ghosts[side] = (int*)buffer(side, sizeof(int)*received, stream);
        sends.push_back({neighbour, values + me.ghostParticles[side].start, sizeof(int)*sent});
        receives.push_back({neighbour, ghosts[side], sizeof(int)*received});
    }
    transport.exchange(sends, receives, stream);
    for(int side = 0; side < 2; ++side){
        uint received = me.edgeParticles[side].count + me.edgeEmpty[side].count;
        if(ghosts[side] != nullptr && received > 0){
            addEdgeRuns<<<received / 128 + 1, 128, 0, stream>>>(me.edgeParticles[side].start, me.edgeParticles[side].count, me.edgeEmpty[side].start, me.edgeEmpty[side].count,
                ghosts[side], values);
            gpuErrchk(cudaPeekAtLastError());
        }
    }
}

double PartitionExchange::maxOverPartitions(double value){
    if(transport.size() == 1){
        return value;
    }
    std::vector<double> all(transport.size());
    transport.allGatherHost(&value, all.data(), sizeof(double));
    double largest = all[0];
    for(double other : all){
        largest = fmax(largest, other);
    }
    return largest;
}

bool PartitionExchange::anyOverPartitions(bool value){
    return maxOverPartitions(value ? 1.0 : 0.0) > 0.0;
}

template <typename T>
void PartitionExchange::sum(T* values, uint count, cudaStream_t stream){
    if(transport.size() == 1){
        return;
    }
    T* gathered = (T*)buffer(2, sizeof(T)*count*transport.size(), stream);
    transport.allGather(values, gathered, sizeof(T)*count, stream);
    addInRankOrder<<<1, 32, 0, stream>>>(gathered, transport.size(), count, values);
    gpuErrchk(cudaPeekAtLastError());
}

void PartitionExchange::sumOverPartitions(long long* values, uint count, cudaStream_t stream){
    sum(values, count, stream);
}

void PartitionExchange::sumOverPartitions(double* value, cudaStream_t stream){
    sum(value, 1u, stream);
}

//a dense grid over the domain with perNode cells along a node's side: a node plane holds perNode of its planes, and a partition's cells are one run of
//it. Each partition sends its run to every other and takes theirs
void PartitionExchange::gatherRuns(void* cells, size_t bytesPerCell, size_t perNode, cudaStream_t stream){
    if(transport.size() == 1){
        return;
    }
    size_t cellsPerPlane = (perNode*me.grid.sizeX)*(perNode*me.grid.sizeY);
    auto run = [&](int rank, size_t& first, size_t& bytes){
        first = perNode*me.partitionPlanes[rank]*cellsPerPlane*bytesPerCell;
        bytes = perNode*(me.partitionPlanes[rank + 1] - me.partitionPlanes[rank])*cellsPerPlane*bytesPerCell;
    };
    std::vector<TransportSend> sends;
    std::vector<TransportReceive> receives;
    size_t mineFirst, mineBytes;
    run(me.rank, mineFirst, mineBytes);
    for(int other = 0; other < transport.size(); ++other){
        if(other != me.rank){
            size_t first, bytes;
            run(other, first, bytes);
            sends.push_back({other, (const char*)cells + mineFirst, mineBytes});
            receives.push_back({other, (char*)cells + first, bytes});
        }
    }
    transport.exchange(sends, receives, stream);
}

//the first coarse grid's cells are 2x2x2 voxels
void PartitionExchange::gatherFirstLevel(void* cells, size_t bytesPerCell, cudaStream_t stream){
    gatherRuns(cells, bytesPerCell, 2, stream);
}

void PartitionExchange::gatherNodeCells(void* cells, size_t bytesPerCell, cudaStream_t stream){
    gatherRuns(cells, bytesPerCell, 1, stream);
}
