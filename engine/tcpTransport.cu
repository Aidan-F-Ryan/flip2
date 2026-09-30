//Copyright 2023 Aberrant Behavior LLC

#include "tcpTransport.hu"
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#ifdef MSG_NOSIGNAL
static const int SEND_FLAGS = MSG_NOSIGNAL;     //a vanished peer is an error, not a SIGPIPE
#else
static const int SEND_FLAGS = 0;                //macOS: SO_NOSIGPIPE is set on the socket instead
#endif

[[noreturn]] static void fail(int rank, const std::string& what, bool withErrno = true){
    std::cerr<<"rank "<<rank<<": "<<what<<(withErrno ? std::string(": ") + strerror(errno) : std::string())<<std::endl;
    exit(1);
}

//blocking whole sends and receives, for the handshakes while connecting
static void sendAll(int rank, int socket, const void* data, size_t bytes){
    for(size_t done = 0; done < bytes;){
        ssize_t sent = send(socket, (const char*)data + done, bytes - done, SEND_FLAGS);
        if(sent < 0){
            if(errno == EINTR){
                continue;
            }
            fail(rank, "sending while connecting");
        }
        done += sent;
    }
}

static void receiveAll(int rank, int socket, void* data, size_t bytes){
    for(size_t done = 0; done < bytes;){
        ssize_t received = recv(socket, (char*)data + done, bytes - done, 0);
        if(received == 0){
            fail(rank, "a rank hung up while connecting", false);
        }
        if(received < 0){
            if(errno == EINTR){
                continue;
            }
            fail(rank, "receiving while connecting");
        }
        done += received;
    }
}

static int listenOn(int rank, unsigned short port, int backlog, unsigned short& boundPort){
    int listener = socket(AF_INET, SOCK_STREAM, 0);
    if(listener < 0){
        fail(rank, "making a socket");
    }
    int yes = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port);
    if(bind(listener, (sockaddr*)&address, sizeof(address)) != 0){
        fail(rank, "listening on port " + std::to_string(port));
    }
    if(listen(listener, backlog) != 0){
        fail(rank, "listening");
    }
    socklen_t length = sizeof(address);
    getsockname(listener, (sockaddr*)&address, &length);
    boundPort = ntohs(address.sin_port);
    return listener;
}

static int acceptWithin(int rank, int listener, double seconds, sockaddr_in& from){
    pollfd waiting = {listener, POLLIN, 0};
    int ready = poll(&waiting, 1, (int)(seconds*1000));
    if(ready <= 0){
        fail(rank, "no rank connected within " + std::to_string((int)seconds) + " s", ready < 0);
    }
    socklen_t length = sizeof(from);
    int accepted = accept(listener, (sockaddr*)&from, &length);
    if(accepted < 0){
        fail(rank, "accepting a rank");
    }
    return accepted;
}

static int connectWithin(int rank, const sockaddr_in& address, double seconds){     //the other side may not be listening yet: keep trying
    auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while(true){
        int connection = socket(AF_INET, SOCK_STREAM, 0);
        if(connection < 0){
            fail(rank, "making a socket");
        }
        if(connect(connection, (const sockaddr*)&address, sizeof(address)) == 0){
            return connection;
        }
        close(connection);
        if(std::chrono::steady_clock::now() > deadline){
            char ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &address.sin_addr, ip, sizeof(ip));
            fail(rank, std::string("couldn't connect to ") + ip + ":" + std::to_string(ntohs(address.sin_port)) + " within " + std::to_string((int)seconds) + " s");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

TcpTransport::TcpTransport(int rank, int size, const std::string& rendezvous, double timeoutSeconds)
: me(rank)
, ranks(size)
, timeout(timeoutSeconds)
, sockets(size, -1)
{
    size_t colon = rendezvous.rfind(':');
    if(colon == std::string::npos){
        fail(me, "the rendezvous address \"" + rendezvous + "\" isn't host:port", false);
    }
    std::string host = rendezvous.substr(0, colon);
    unsigned short rendezvousPort = (unsigned short)std::stoi(rendezvous.substr(colon + 1));
    struct Entry{
        unsigned int ip;        //network order
        unsigned int port;
    };
    std::vector<Entry> table(size, Entry{0, 0});    //where each rank listens for the ranks above it
    unsigned short myPort = 0;
    int listener = -1;
    if(me == 0){    //everyone checks in here, and gets everyone's address back
        listener = listenOn(me, rendezvousPort, size, myPort);
        for(int joined = 1; joined < size; ++joined){
            sockaddr_in from{};
            int connection = acceptWithin(me, listener, timeout, from);
            unsigned int hello[2];
            receiveAll(me, connection, hello, sizeof(hello));
            int other = (int)hello[0];
            if(other <= 0 || other >= size || sockets[other] >= 0){
                fail(me, "rank " + std::to_string(other) + " checked in, which is out of range or already here", false);
            }
            sockets[other] = connection;
            table[other] = {from.sin_addr.s_addr, hello[1]};
        }
        for(int other = 1; other < size; ++other){
            sendAll(me, sockets[other], table.data(), sizeof(Entry)*size);
        }
    }
    else{
        if(me < size - 1){  //the ranks above connect here
            listener = listenOn(me, 0, size, myPort);
        }
        addrinfo hints{};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        addrinfo* found = nullptr;
        if(getaddrinfo(host.c_str(), nullptr, &hints, &found) != 0 || found == nullptr){
            fail(me, "can't resolve the rendezvous host " + host, false);
        }
        sockaddr_in address = *(sockaddr_in*)found->ai_addr;
        freeaddrinfo(found);
        address.sin_port = htons(rendezvousPort);
        sockets[0] = connectWithin(me, address, timeout);
        unsigned int hello[2] = {(unsigned int)me, myPort};
        sendAll(me, sockets[0], hello, sizeof(hello));
        receiveAll(me, sockets[0], table.data(), sizeof(Entry)*size);
        for(int lower = 1; lower < me; ++lower){
            sockaddr_in lowerAddress{};
            lowerAddress.sin_family = AF_INET;
            lowerAddress.sin_addr.s_addr = table[lower].ip;
            lowerAddress.sin_port = htons((unsigned short)table[lower].port);
            sockets[lower] = connectWithin(me, lowerAddress, timeout);
            unsigned int mine = me;
            sendAll(me, sockets[lower], &mine, sizeof(mine));
        }
        for(int joined = me + 1; joined < size; ++joined){  //in whatever order they come
            sockaddr_in from{};
            int connection = acceptWithin(me, listener, timeout, from);
            unsigned int other;
            receiveAll(me, connection, &other, sizeof(other));
            if((int)other <= me || (int)other >= size || sockets[other] >= 0){
                fail(me, "rank " + std::to_string(other) + " connected, which is out of range or already here", false);
            }
            sockets[other] = connection;
        }
    }
    if(listener >= 0){
        close(listener);
    }
    for(int other = 0; other < size; ++other){
        if(sockets[other] >= 0){
            int yes = 1;
            setsockopt(sockets[other], IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));    //small messages go at once
#ifdef SO_NOSIGPIPE
            setsockopt(sockets[other], SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
            fcntl(sockets[other], F_SETFL, fcntl(sockets[other], F_GETFL) | O_NONBLOCK);
        }
    }
}

TcpTransport::~TcpTransport(){
    for(int connection : sockets){
        if(connection >= 0){
            close(connection);
        }
    }
    if(pinned != nullptr){
        cudaFreeHost(pinned);
    }
}

char* TcpTransport::staging(size_t bytes){
    if(pinnedBytes < bytes){
        if(pinned != nullptr){
            gpuErrchk(cudaFreeHost(pinned));
        }
        gpuErrchk(cudaHostAlloc((void**)&pinned, bytes, cudaHostAllocDefault));
        pinnedBytes = bytes;
    }
    return pinned;
}

//Sends and receives every message, in order per peer, working on whichever sockets are ready so that two ranks sending each other a lot can't block
//each other. No progress for the timeout means a rank has died or gone out of step
void TcpTransport::transfer(std::vector<Message>& sends, std::vector<Message>& receives){
    std::vector<std::vector<Message*>> outgoing(ranks), incoming(ranks);
    for(Message& message : sends){
        if(message.peer == me || message.peer < 0 || message.peer >= ranks){
            fail(me, "a send to rank " + std::to_string(message.peer), false);
        }
        message.header = message.bytes;
        message.done = 0;
        outgoing[message.peer].push_back(&message);
    }
    for(Message& message : receives){
        if(message.peer == me || message.peer < 0 || message.peer >= ranks){
            fail(me, "a receive from rank " + std::to_string(message.peer), false);
        }
        message.done = 0;
        incoming[message.peer].push_back(&message);
    }
    std::vector<size_t> nextOut(ranks, 0), nextIn(ranks, 0);
    size_t remaining = sends.size() + receives.size();
    std::vector<pollfd> waiting;
    std::vector<int> waitingPeer;
    const size_t HEADER = sizeof(unsigned long long);
    while(remaining > 0){
        waiting.clear();
        waitingPeer.clear();
        for(int peer = 0; peer < ranks; ++peer){
            short events = (nextOut[peer] < outgoing[peer].size() ? POLLOUT : 0) | (nextIn[peer] < incoming[peer].size() ? POLLIN : 0);
            if(events != 0){
                waiting.push_back({sockets[peer], events, 0});
                waitingPeer.push_back(peer);
            }
        }
        int ready = poll(waiting.data(), waiting.size(), (int)(timeout*1000));
        if(ready < 0 && errno == EINTR){
            continue;
        }
        if(ready <= 0){
            fail(me, "nothing moved for " + std::to_string((int)timeout) + " s: a rank died or fell out of step", ready < 0);
        }
        for(size_t i = 0; i < waiting.size(); ++i){
            int peer = waitingPeer[i];
            int connection = sockets[peer];
            if(waiting[i].revents & POLLOUT){
                while(nextOut[peer] < outgoing[peer].size()){
                    Message& message = *outgoing[peer][nextOut[peer]];
                    bool blocked = false;
                    while(message.done < HEADER + message.bytes){
                        const char* from = message.done < HEADER ? (const char*)&message.header + message.done : message.data + (message.done - HEADER);
                        size_t left = message.done < HEADER ? HEADER - message.done : message.bytes - (message.done - HEADER);
                        ssize_t sent = send(connection, from, left, SEND_FLAGS);
                        if(sent < 0){
                            if(errno == EAGAIN || errno == EWOULDBLOCK){
                                blocked = true;
                                break;
                            }
                            if(errno == EINTR){
                                continue;
                            }
                            fail(me, "sending to rank " + std::to_string(peer));
                        }
                        message.done += sent;
                    }
                    if(blocked){
                        break;
                    }
                    ++nextOut[peer];
                    --remaining;
                }
            }
            if(waiting[i].revents & (POLLIN | POLLHUP | POLLERR)){
                while(nextIn[peer] < incoming[peer].size()){
                    Message& message = *incoming[peer][nextIn[peer]];
                    bool blocked = false;
                    while(message.done < HEADER + message.bytes){
                        char* to = message.done < HEADER ? (char*)&message.header + message.done : message.data + (message.done - HEADER);
                        size_t left = message.done < HEADER ? HEADER - message.done : message.bytes - (message.done - HEADER);
                        ssize_t received = recv(connection, to, left, 0);
                        if(received == 0){
                            fail(me, "rank " + std::to_string(peer) + " hung up", false);
                        }
                        if(received < 0){
                            if(errno == EAGAIN || errno == EWOULDBLOCK){
                                blocked = true;
                                break;
                            }
                            if(errno == EINTR){
                                continue;
                            }
                            fail(me, "receiving from rank " + std::to_string(peer));
                        }
                        message.done += received;
                        if(message.done == HEADER && message.header != message.bytes){
                            fail(me, "rank " + std::to_string(peer) + " sent " + std::to_string(message.header) + " bytes where " + std::to_string(message.bytes) + " were expected", false);
                        }
                    }
                    if(blocked){
                        break;
                    }
                    ++nextIn[peer];
                    --remaining;
                }
            }
        }
    }
}

void TcpTransport::exchange(const std::vector<TransportSend>& sends, const std::vector<TransportReceive>& receives, cudaStream_t stream){
    size_t total = 0;
    for(const TransportSend& send : sends){
        total += send.bytes;
    }
    for(const TransportReceive& receive : receives){
        total += receive.bytes;
    }
    char* host = staging(total > 0 ? total : 1);
    std::vector<Message> out, in;
    size_t offset = 0;
    for(const TransportSend& send : sends){
        if(send.bytes > 0){
            gpuErrchk(cudaMemcpyAsync(host + offset, send.data, send.bytes, cudaMemcpyDeviceToHost, stream));
        }
        out.push_back({send.to, host + offset, send.bytes, 0});
        offset += send.bytes;
    }
    for(const TransportReceive& receive : receives){
        in.push_back({receive.from, host + offset, receive.bytes, 0});
        offset += receive.bytes;
    }
    gpuErrchk(cudaStreamSynchronize(stream));     //what's sent is in host memory
    transfer(out, in);
    for(size_t i = 0; i < receives.size(); ++i){
        if(receives[i].bytes > 0){
            gpuErrchk(cudaMemcpyAsync(receives[i].data, in[i].data, receives[i].bytes, cudaMemcpyHostToDevice, stream));
        }
    }
    gpuErrchk(cudaStreamSynchronize(stream));     //before the staging is used again
}

void TcpTransport::allGatherHost(const void* mine, void* all, size_t bytes){
    memcpy((char*)all + me*bytes, mine, bytes);
    std::vector<Message> out, in;
    for(int other = 0; other < ranks; ++other){
        if(other != me){
            out.push_back({other, (char*)mine, bytes, 0});
            in.push_back({other, (char*)all + other*bytes, bytes, 0});
        }
    }
    transfer(out, in);
}
