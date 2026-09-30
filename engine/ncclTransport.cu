//Copyright 2023 Aberrant Behavior LLC

#include "ncclTransport.hu"
#include <cstring>
#include <iostream>

NcclTransport::NcclTransport(int rank, int size, const std::string& rendezvous, double timeoutSeconds)
: me(rank)
, ranks(size)
, bootstrap(std::make_unique<TcpTransport>(rank, size, rendezvous, timeoutSeconds))
{
    ncclUniqueId offered;
    memset(&offered, 0, sizeof(offered));
    if(me == 0){
        check(ncclGetUniqueId(&offered), "making NCCL's unique id");
    }
    std::vector<ncclUniqueId> all(size);
    bootstrap->allGatherHost(&offered, all.data(), sizeof(ncclUniqueId));     //rank 0's is the one
    id = all[0];
}

NcclTransport::~NcclTransport(){    //with this rank's GPU current
    if(communicator != nullptr){
        ncclCommDestroy(communicator);
    }
    if(staging != nullptr){
        cudaFree(staging);
    }
}

void NcclTransport::check(ncclResult_t result, const char* doing){
    if(result != ncclSuccess){
        std::cerr<<"rank "<<me<<": NCCL failed "<<doing<<": "<<ncclGetErrorString(result);
        if(communicator != nullptr){
            const char* detail = ncclGetLastError(communicator);
            if(detail != nullptr && detail[0] != '\0'){
                std::cerr<<" ("<<detail<<")";
            }
        }
        std::cerr<<std::endl;
        exit(1);
    }
}

void NcclTransport::attach(cudaStream_t stream){
    this->stream = stream;
    check(ncclCommInitRank(&communicator, ranks, id, me), "joining the communicator");     //on the current GPU; waits for every rank
    bootstrap.reset();
}

void NcclTransport::exchange(const std::vector<TransportSend>& sends, const std::vector<TransportReceive>& receives, cudaStream_t stream){
    //both sides know every size, so they skip the same empty messages and the rest still pair up in order
    check(ncclGroupStart(), "starting an exchange");
    for(const TransportSend& send : sends){
        if(send.bytes > 0){
            check(ncclSend(send.data, send.bytes, ncclUint8, send.to, communicator, stream), "sending");
        }
    }
    for(const TransportReceive& receive : receives){
        if(receive.bytes > 0){
            check(ncclRecv(receive.data, receive.bytes, ncclUint8, receive.from, communicator, stream), "receiving");
        }
    }
    check(ncclGroupEnd(), "finishing an exchange");
}

void NcclTransport::allGather(const void* mine, void* all, size_t bytes, cudaStream_t stream){
    check(ncclAllGather(mine, all, bytes, ncclUint8, communicator, stream), "gathering");
}

//through the GPU, on the rank's stream, so it keeps its place among the other NCCL calls; waits for all of it
void NcclTransport::allGatherHost(const void* mine, void* all, size_t bytes){
    size_t total = bytes*ranks;
    if(stagingBytes < total){
        if(staging != nullptr){
            gpuErrchk(cudaFree(staging));
        }
        gpuErrchk(cudaMalloc((void**)&staging, total));
        stagingBytes = total;
    }
    gpuErrchk(cudaMemcpyAsync(staging + me*bytes, mine, bytes, cudaMemcpyHostToDevice, stream));
    check(ncclAllGather(staging + me*bytes, staging, bytes, ncclUint8, communicator, stream), "gathering host values");     //in place
    gpuErrchk(cudaMemcpyAsync(all, staging, total, cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaStreamSynchronize(stream));
}
