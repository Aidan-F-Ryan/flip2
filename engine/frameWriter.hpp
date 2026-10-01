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

//Writes frames to disk on a background thread so the simulation never waits on the disk. The caller takes a free pinned host buffer with acquire(), has
//each partition copy its part into it from its own GPU, recording an event when its copy lands, then submits the buffer with those events; the thread
//writes the buffer out once every event has completed. The simulation only blocks if every buffer is still being written. The destructor finishes all
//submitted frames
class FrameWriter{
public:
    FrameWriter(size_t bytesPerFrame, int numBuffers = 3)
    : bytes(bytesPerFrame)
    {
        for(int i = 0; i < numBuffers; ++i){
            float* hostBuffer;
            gpuErrchk(cudaHostAlloc((void**)&hostBuffer, bytes, cudaHostAllocPortable));     //portable: pinned for every GPU's copies
            hostBuffers.push_back(hostBuffer);
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

    int acquire(){  //a free buffer's index, once one is free
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [this]{ return !freeBuffers.empty(); });
        int buffer = freeBuffers.back();
        freeBuffers.pop_back();
        return buffer;
    }

    float* hostBuffer(int buffer){
        return hostBuffers[buffer];
    }

    //writes buffer to fileName once every event in copies has completed. The events mustn't be recorded again until the buffer is free
    void submit(int buffer, const std::string& fileName, const std::vector<cudaEvent_t>& copies){
        {
            std::lock_guard<std::mutex> lock(mutex);
            jobs.push({buffer, fileName, copies});
        }
        changed.notify_all();
    }

private:
    struct Job{
        int buffer;
        std::string fileName;
        std::vector<cudaEvent_t> copies;
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
            if(file == nullptr || std::fwrite(hostBuffers[job.buffer], 1, bytes, file) != bytes){
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

    size_t bytes;
    std::vector<float*> hostBuffers;
    std::vector<int> freeBuffers;
    std::queue<Job> jobs;
    std::mutex mutex;
    std::condition_variable changed;
    bool stopping = false;
    std::thread thread;
};

#endif
