//Copyright 2023 Aberrant Behavior LLC

#include "localTransport.hu"
#include <cstring>
#include <iostream>

LocalHub::LocalHub(int size)
: numRanks(size)
, devices(size, 0)
, ready(size, nullptr)
, consumed(size, nullptr)
, posted(size, nullptr)
, matched(size*size, 0)
, hostValues(size, nullptr)
{}

LocalHub::~LocalHub(){
    for(int rank = 0; rank < numRanks; ++rank){
        if(ready[rank] != nullptr){
            cudaSetDevice(devices[rank]);
            cudaEventDestroy(ready[rank]);
            cudaEventDestroy(consumed[rank]);
        }
    }
}

void LocalHub::attach(int rank, int device){
    devices[rank] = device;
    gpuErrchk(cudaEventCreateWithFlags(&ready[rank], cudaEventDisableTiming));
    gpuErrchk(cudaEventCreateWithFlags(&consumed[rank], cudaEventDisableTiming));
}

void LocalHub::barrier(){
    std::unique_lock<std::mutex> lock(mutex);
    unsigned long long arrivedIn = generation;
    if(++arrived == numRanks){
        arrived = 0;
        ++generation;
        released.notify_all();
    }
    else{
        released.wait(lock, [&]{ return generation != arrivedIn; });
    }
}

void LocalTransport::exchange(const std::vector<TransportSend>& sends, const std::vector<TransportReceive>& receives, cudaStream_t stream){
    int ranks = hub.size();
    hub.posted[me] = &sends;
    gpuErrchk(cudaEventRecord(hub.ready[me], stream));
    hub.barrier();      //every rank's sends are posted, and written as far as their streams go
    std::vector<size_t> next(ranks, 0);     //per sender: where to look for its next send to this rank
    std::vector<int> fromCount(ranks, 0);
    for(const TransportReceive& receive : receives){
        const std::vector<TransportSend>& theirs = *hub.posted[receive.from];
        size_t& k = next[receive.from];
        while(k < theirs.size() && theirs[k].to != me){
            ++k;
        }
        if(k == theirs.size() || theirs[k].bytes != receive.bytes){
            std::cerr<<"rank "<<me<<": a receive of "<<receive.bytes<<" bytes from rank "<<receive.from<<" has no matching send\n";
            exit(1);
        }
        if(fromCount[receive.from]++ == 0){
            gpuErrchk(cudaStreamWaitEvent(stream, hub.ready[receive.from], 0));
        }
        if(receive.bytes > 0){
            gpuErrchk(cudaMemcpyPeerAsync(receive.data, hub.devices[me], theirs[k].data, hub.devices[receive.from], receive.bytes, stream));
        }
        ++k;
    }
    for(int from = 0; from < ranks; ++from){
        hub.matched[from*ranks + me] = fromCount[from];
    }
    gpuErrchk(cudaEventRecord(hub.consumed[me], stream));
    hub.barrier();      //every rank's copies are queued and its consumed event recorded
    std::vector<int> toCount(ranks, 0);
    for(const TransportSend& send : sends){
        ++toCount[send.to];
    }
    for(int to = 0; to < ranks; ++to){
        if(toCount[to] != hub.matched[me*ranks + to]){
            std::cerr<<"rank "<<me<<": "<<toCount[to]<<" sends to rank "<<to<<", which received "<<hub.matched[me*ranks + to]<<"\n";
            exit(1);
        }
        if(toCount[to] > 0){
            gpuErrchk(cudaStreamWaitEvent(stream, hub.consumed[to], 0));
        }
    }
    //no barrier needed here: the next exchange only writes posted and matched after its first barrier, which every rank reaches after reading these
}

void LocalTransport::allGatherHost(const void* mine, void* all, size_t bytes){
    hub.hostValues[me] = mine;
    hub.barrier();
    for(int rank = 0; rank < hub.size(); ++rank){
        memcpy((char*)all + rank*bytes, hub.hostValues[rank], bytes);
    }
    hub.barrier();      //every rank has read before anyone posts again
}
