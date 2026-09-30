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

//Writes frames to disk on a background thread so the simulation never waits on the disk. The caller packs a frame into deviceFrame(stream) on its stream, then
//write() queues an async copy into one of a few pinned host buffers and returns; the thread writes each buffer out once its copy has landed. The copy runs on
//the writer's own stream once the pack is done, so the simulation's next kernels run alongside it rather than behind it, and the next pack waits for it to
//finish reading. The simulation only blocks if every buffer is still being written. The destructor finishes all queued frames.
class FrameWriter{
public:
    FrameWriter(size_t bytesPerFrame, int numBuffers = 3)
    : bytes(bytesPerFrame)
    {
        gpuErrchk(cudaMalloc((void**)&deviceBuffer, bytes));
        gpuErrchk(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
        gpuErrchk(cudaEventCreateWithFlags(&packed, cudaEventDisableTiming));
        gpuErrchk(cudaEventCreateWithFlags(&deviceFree, cudaEventDisableTiming));
        for(int i = 0; i < numBuffers; ++i){
            float* hostBuffer;
            cudaEvent_t event;
            gpuErrchk(cudaMallocHost((void**)&hostBuffer, bytes));
            gpuErrchk(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            hostBuffers.push_back(hostBuffer);
            copied.push_back(event);
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
        cudaStreamSynchronize(copyStream);
        for(size_t i = 0; i < hostBuffers.size(); ++i){
            cudaFreeHost(hostBuffers[i]);
            cudaEventDestroy(copied[i]);
        }
        cudaEventDestroy(packed);
        cudaEventDestroy(deviceFree);
        cudaStreamDestroy(copyStream);
        cudaFree(deviceBuffer);
    }

    //the device array to pack the next frame into, once the last frame's copy out of it is done: stream waits for that, not the host
    float* deviceFrame(cudaStream_t stream){
        gpuErrchk(cudaStreamWaitEvent(stream, deviceFree, 0));
        return deviceBuffer;
    }

    void write(const std::string& fileName, cudaStream_t stream){
        int buffer;
        {
            std::unique_lock<std::mutex> lock(mutex);
            changed.wait(lock, [this]{ return !freeBuffers.empty(); });
            buffer = freeBuffers.back();
            freeBuffers.pop_back();
        }
        gpuErrchk(cudaEventRecord(packed, stream));
        gpuErrchk(cudaStreamWaitEvent(copyStream, packed, 0));
        gpuErrchk(cudaMemcpyAsync(hostBuffers[buffer], deviceBuffer, bytes, cudaMemcpyDeviceToHost, copyStream));
        gpuErrchk(cudaEventRecord(copied[buffer], copyStream));
        gpuErrchk(cudaEventRecord(deviceFree, copyStream));
        {
            std::lock_guard<std::mutex> lock(mutex);
            jobs.push({buffer, fileName});
        }
        changed.notify_all();
    }

private:
    struct Job{
        int buffer;
        std::string fileName;
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
            gpuErrchk(cudaEventSynchronize(copied[job.buffer]));
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
    float* deviceBuffer;
    cudaStream_t copyStream;
    cudaEvent_t packed;         //on the simulation's stream: the frame is in deviceBuffer
    cudaEvent_t deviceFree;     //on copyStream: deviceBuffer has been copied out
    std::vector<float*> hostBuffers;
    std::vector<cudaEvent_t> copied;
    std::vector<int> freeBuffers;
    std::queue<Job> jobs;
    std::mutex mutex;
    std::condition_variable changed;
    bool stopping = false;
    std::thread thread;
};

#endif
