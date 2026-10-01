//Copyright 2023 Aberrant Behavior LLC

#include "cudaVec.hu"
#include "typedefs.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>
#include "cudaDeviceManager.hu"

template <typename T>
CudaVec<T>::CudaVec()
: d_vec(nullptr)
, numElements(0){   //size() is 0 until it's resized

}

template <typename T>
CudaVec<T>::CudaVec(const uint& size){
    resize(size);
}

template <typename T>
void CudaVec<T>::resize(const uint& size){
    if(d_vec != nullptr){
        GPU_MEMORY_ALLOCATED -= sizeof(T) * vec.size();
    }
    numElements = size;
    vec.resize(size);
    cuMalloc();
}

template <typename T>
void CudaVec<T>::resizeAsync(const uint& size, const cudaStream_t& stream){
    if(d_vec != nullptr){
        GPU_MEMORY_ALLOCATED -= sizeof(T) * vec.size();
    }
    numElements = size;
    cuMallocAsync(stream);
}

template <typename T>
T& CudaVec<T>::operator[](const uint& i){
    return vec[i];
}

template <typename T>
const T& CudaVec<T>::operator[](const uint& i) const{
    return vec[i];
}

template <typename T>
void CudaVec<T>::upload(){
    gpuErrchk( cudaMemcpy(d_vec, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice) );
}

template <typename T>
void CudaVec<T>::download(){
    if(numElements != vec.size()){
        vec.resize(numElements);
    }
    gpuErrchk( cudaMemcpy(vec.data(), d_vec, vec.size()*sizeof(T), cudaMemcpyDeviceToHost) );
}

template <typename T>
void CudaVec<T>::upload(cudaStream_t stream){
    if(numElements != vec.size()){
        std::cerr<<"ERROR: ATTEMPT TO UPLOAD CPU VECTOR TO ASYNC ALLOCATED GPU VECTOR\nCPU VEC SIZE: "<<vec.size()<<"\nGPU_VEC SIZE: "<<numElements<<"\n";
        exit(1);
    }
    gpuErrchk( cudaMemcpyAsync(d_vec, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice, stream) );
}

template <typename T>
void CudaVec<T>::download(cudaStream_t stream){
    if(vec.size() != numElements){
        vec.resize(numElements);
    }
    gpuErrchk( cudaMemcpyAsync(vec.data(), d_vec, vec.size()*sizeof(T), cudaMemcpyDeviceToHost, stream) );
}

template <typename T>
T* CudaVec<T>::devPtr(){
    return d_vec;
}

template <typename T>
const T* CudaVec<T>::devPtr() const{
    return d_vec;
}

template<typename T>
void CudaVec<T>::cuMalloc(){
    if(d_vec != nullptr){
        gpuErrchk( cudaFree(d_vec) );
    }
    gpuErrchk( cudaMalloc((void**)&d_vec, sizeof(T)*vec.size()) );
    GPU_MEMORY_ALLOCATED += sizeof(T) * vec.size();
}

template<typename T>
void CudaVec<T>::cuMallocAsync(const cudaStream_t& stream){
    if(d_vec != nullptr){
        gpuErrchk( cudaFreeAsync(d_vec, stream) );
    }
    gpuErrchk( cudaMallocAsync((void**)&d_vec, sizeof(T)*numElements, stream) );
    GPU_MEMORY_ALLOCATED += sizeof(T) * numElements;
}

template<typename T>
void CudaVec<T>::swapDevicePtr(T* devPtr){
    if(d_vec != nullptr && d_vec != devPtr){
        gpuErrchk( cudaFree(d_vec) );
        GPU_MEMORY_ALLOCATED -= sizeof(T) * numElements;
    }
    d_vec = devPtr;
}

template<typename T>
void CudaVec<T>::swapDevicePtrAsync(T* devPtr, cudaStream_t stream){
    if(d_vec != nullptr && d_vec != devPtr){
        gpuErrchk( cudaFreeAsync(d_vec, stream) );
        GPU_MEMORY_ALLOCATED -= sizeof(T) * numElements;
    }
    d_vec = devPtr;
}

template<typename T>
void CudaVec<T>::adoptAsync(T* devPtr, uint size, cudaStream_t stream){
    if(d_vec != nullptr && d_vec != devPtr){
        gpuErrchk( cudaFreeAsync(d_vec, stream) );
    }
    GPU_MEMORY_ALLOCATED += (long long)sizeof(T)*size - (long long)sizeof(T)*numElements;
    d_vec = devPtr;
    numElements = size;
}

template <typename T>
void CudaVec<T>::clear(){
    if(d_vec != nullptr){
        GPU_MEMORY_ALLOCATED -= sizeof(T) * numElements;
        cudaFree(d_vec);
        d_vec = nullptr;
    }
    vec.clear();
    numElements = 0;
}

template <typename T>
void CudaVec<T>::clearAsync(cudaStream_t stream){
    if(d_vec != nullptr){
        GPU_MEMORY_ALLOCATED -= sizeof(T) * numElements;
        cudaFreeAsync(d_vec, stream);
        d_vec = nullptr;
    }
    vec.clear();
    numElements = 0;
}

template<typename T>
void CudaVec<T>::print() const{
    for(uint i = 0; i < vec.size(); ++i){
        std::cout<<+vec[i]<<", ";
    }
    std::cout<<std::endl;
}

template<typename T>
void CudaVec<T>::zeroDevice(){
    zeroArray<<<numElements / BLOCKSIZE + 1, BLOCKSIZE>>>(numElements, d_vec);
}

template<typename T>
void CudaVec<T>::zeroDeviceAsync(cudaStream_t stream){
    zeroArray<<<numElements / BLOCKSIZE + 1, BLOCKSIZE, 0, stream>>>(numElements, d_vec);
}

__device__ char abs(const char& in){
    return in;
}

__device__ uint abs(const uint& in){
    return in;
}

template<typename T>
__global__ void getMaxFromArray(bool absolute, uint numElements, T* array, T* out){
    __shared__ T shared[WORKSIZE];
    uint index = threadIdx.x + blockIdx.x*WORKSIZE;
    if(index < numElements){
        shared[threadIdx.x] = array[index];
    }
    else{
        shared[threadIdx.x] = 0;
    }
    if(index + blockDim.x < numElements){
        shared[threadIdx.x + blockDim.x] = array[index + blockDim.x];
    }
    else{
        shared[threadIdx.x + blockDim.x] = 0;
    }

    for(int i = blockDim.x; i >= 1; i >>= 1){
        __syncthreads();
        if(threadIdx.x < i){
            if(!absolute)
                shared[threadIdx.x] = shared[threadIdx.x] > shared[threadIdx.x + i] ? shared[threadIdx.x] : shared[threadIdx.x + i];
            else
                shared[threadIdx.x] = abs(shared[threadIdx.x]) > abs(shared[threadIdx.x + i]) ? shared[threadIdx.x] : shared[threadIdx.x + i];
        }
    }

    __syncthreads();

    if(threadIdx.x == 0){
        out[blockIdx.x] = shared[threadIdx.x];
    }
}

template <typename T>
T CudaVec<T>::getMax(cudaStream_t stream, bool abs){
    return getMax(stream, abs, numElements);
}

//getMax's one pass for floats' magnitudes. Non-negative floats order like their bit patterns (NaN above infinity), so each block takes an integer max of
//|x|'s bits and one atomicMax a block combines them: exact, whatever order the blocks land in
__global__ void largestMagnitudeBits(uint numElements, const float* values, unsigned int* largest){
    __shared__ unsigned int warpLargest[BLOCKSIZE / 32];
    unsigned int bits = 0;
    for(uint i = threadIdx.x + blockIdx.x*blockDim.x; i < numElements; i += blockDim.x*gridDim.x){
        bits = max(bits, __float_as_uint(values[i]) & 0x7fffffffu);
    }
    for(int offset = 16; offset > 0; offset /= 2){
        bits = max(bits, __shfl_down_sync(0xffffffffu, bits, offset));
    }
    if(threadIdx.x % 32 == 0){
        warpLargest[threadIdx.x / 32] = bits;
    }
    __syncthreads();
    if(threadIdx.x < 32){
        bits = threadIdx.x < blockDim.x / 32 ? warpLargest[threadIdx.x] : 0;
        for(int offset = 16; offset > 0; offset /= 2){
            bits = max(bits, __shfl_down_sync(0xffffffffu, bits, offset));
        }
        if(threadIdx.x == 0){
            atomicMax(largest, bits);
        }
    }
}

static void addLargestMagnitude(const float* values, uint numElements, unsigned int* largest, cudaStream_t stream){
    if(numElements > 0){
        uint blocks = numElements / BLOCKSIZE + 1;
        largestMagnitudeBits<<<blocks < 1024 ? blocks : 1024, BLOCKSIZE, 0, stream>>>(numElements, values, largest);
        gpuErrchk(cudaPeekAtLastError());
    }
}

static float readLargestMagnitude(unsigned int* largest, cudaStream_t stream){
    unsigned int bits;
    gpuErrchk(cudaMemcpyAsync(&bits, largest, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));  //to pageable memory: back once it's landed
    gpuErrchk(cudaFreeAsync(largest, stream));
    float magnitude;
    memcpy(&magnitude, &bits, sizeof(float));
    return magnitude;
}

float largestMagnitude(std::initializer_list<const CudaVec<float>*> arrays, cudaStream_t stream){
    unsigned int* largest;
    gpuErrchk(cudaMallocAsync((void**)&largest, sizeof(unsigned int), stream));
    gpuErrchk(cudaMemsetAsync(largest, 0, sizeof(unsigned int), stream));
    for(const CudaVec<float>* array : arrays){
        addLargestMagnitude(array->d_vec, array->numElements, largest, stream);
    }
    return readLargestMagnitude(largest, stream);
}

template <typename T>
T CudaVec<T>::getMax(cudaStream_t stream, bool abs, uint numElements){
    if constexpr(std::is_same<T, float>::value){
        if(abs){
            unsigned int* largest;
            gpuErrchk(cudaMallocAsync((void**)&largest, sizeof(unsigned int), stream));
            gpuErrchk(cudaMemsetAsync(largest, 0, sizeof(unsigned int), stream));
            addLargestMagnitude(d_vec, numElements, largest, stream);
            return readLargestMagnitude(largest, stream);
        }
    }
    T* outArray;
    T* outArray2;
    T out;
    cudaMallocAsync((void**)&outArray, sizeof(T) * (numElements / WORKSIZE + 1), stream);
    gpuErrchk(cudaPeekAtLastError());
    getMaxFromArray<<<numElements / WORKSIZE + 1, BLOCKSIZE, 0, stream>>>(abs, numElements, d_vec, outArray);
    gpuErrchk(cudaPeekAtLastError());
    cudaMallocAsync((void**)&outArray2, sizeof(T) * ((numElements / WORKSIZE + 1) / WORKSIZE + 1), stream);
    // cudaStreamSynchronize(stream);
    gpuErrchk(cudaPeekAtLastError());
    for(int i = numElements / WORKSIZE + 1; i > 1; i = i / WORKSIZE + 1){
        getMaxFromArray<<<i / WORKSIZE + 1, BLOCKSIZE, 0, stream>>>(abs, i, outArray, outArray2);
        // cudaStreamSynchronize(stream);
        gpuErrchk(cudaPeekAtLastError());
        T* temp = outArray;
        outArray = outArray2;
        outArray2 = temp;
    }
    cudaMemcpyAsync(&out, outArray, sizeof(T), cudaMemcpyDeviceToHost, stream);
    gpuErrchk(cudaPeekAtLastError());
    cudaStreamSynchronize(stream);
    cudaFreeAsync(outArray, stream);
    cudaFreeAsync(outArray2, stream);
    // cudaStreamSynchronize(stream);
    gpuErrchk(cudaPeekAtLastError());
    return out;
}

template<typename T>
CudaVec<T>::~CudaVec(){
    if(d_vec != nullptr){
        GPU_MEMORY_ALLOCATED -= sizeof(T) * vec.size();
        gpuErrchk( cudaFree(d_vec) );
    }
}

template class CudaVec<float>;
template class CudaVec<uint>;
template class CudaVec<char>;
template class CudaVec<double>;

template <typename T>
std::atomic<long long> CudaVec<T>::GPU_MEMORY_ALLOCATED{0};