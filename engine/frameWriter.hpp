//Copyright 2023 Aberrant Behavior LLC

#ifndef FRAMEWRITER_HPP
#define FRAMEWRITER_HPP

#include "typedefs.h"
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

//Writes frames to disk on a background thread so the simulation never waits on the disk. The caller takes a free pinned host buffer big enough for the
//frame with acquire(), has each partition copy its part into it from its own GPU, recording an event when its copy lands, then submits the buffer with
//those events; the thread writes the buffer out once every event has completed. The simulation only blocks if every buffer is still being written. A
//frame can be any size: a buffer grows when one doesn't fit. The destructor finishes all submitted frames
class FrameWriter{
public:
    FrameWriter(size_t bytesPerFrame, int numBuffers = 3){
        for(int i = 0; i < numBuffers; ++i){
            float* hostBuffer;
            gpuErrchk(cudaHostAlloc((void**)&hostBuffer, bytesPerFrame, cudaHostAllocPortable));     //portable: pinned for every GPU's copies
            hostBuffers.push_back(hostBuffer);
            capacities.push_back(bytesPerFrame);
            freeBuffers.push_back(i);
        }
        thread = std::thread(&FrameWriter::run, this);
    }

    ~FrameWriter(){
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        changed.notify_all();
        thread.join();
        for(float* hostBuffer : hostBuffers){
            cudaFreeHost(hostBuffer);
        }
    }

    int numBuffers() const{
        return (int)hostBuffers.size();
    }

    int acquire(size_t bytes){  //a free buffer's index, once one is free, holding at least bytes
        int buffer;
        {
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock, [this]{ return !freeBuffers.empty(); });
            buffer = freeBuffers.back();
            freeBuffers.pop_back();
        }
        if(capacities[buffer] < bytes){     //nobody else touches a buffer between acquire and submit
            gpuErrchk(cudaFreeHost(hostBuffers[buffer]));
            gpuErrchk(cudaHostAlloc((void**)&hostBuffers[buffer], bytes, cudaHostAllocPortable));
            capacities[buffer] = bytes;
        }
        return buffer;
    }

    float* hostBuffer(int buffer){
        return hostBuffers[buffer];
    }

    //writes bytes of buffer to fileName once every event in copies has completed. The events mustn't be recorded again until the buffer is free
    void submit(int buffer, const std::string& fileName, const std::vector<cudaEvent_t>& copies, size_t bytes){
        {
            std::lock_guard<std::mutex> lock(mutex);
            jobs.push({buffer, fileName, copies, bytes});
        }
        changed.notify_all();
    }

private:
    struct Job{
        int buffer;
        std::string fileName;
        std::vector<cudaEvent_t> copies;
        size_t bytes;
    };

    void run(){
        while(true){
            Job job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [this]{ return stopping || !jobs.empty(); });
                if(jobs.empty()){
                    return;     //stopping, and nothing left to write
                }
                job = jobs.front();
                jobs.pop();
            }
            for(cudaEvent_t copy : job.copies){
                gpuErrchk(cudaEventSynchronize(copy));
            }
            FILE* file = std::fopen(job.fileName.c_str(), "wb");
            if(file == nullptr || (job.bytes > 0 && std::fwrite(hostBuffers[job.buffer], 1, job.bytes, file) != job.bytes)){
                std::fprintf(stderr, "FrameWriter: failed writing %s\n", job.fileName.c_str());
            }
            if(file != nullptr){
                std::fclose(file);
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                freeBuffers.push_back(job.buffer);
            }
            changed.notify_all();
        }
    }

    std::vector<float*> hostBuffers;
    std::vector<size_t> capacities;
    std::vector<int> freeBuffers;
    std::queue<Job> jobs;
    std::mutex mutex;
    std::condition_variable changed;
    bool stopping = false;
    std::thread thread;
};

#endif
