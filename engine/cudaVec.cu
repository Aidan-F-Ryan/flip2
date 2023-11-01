//Copyright 2023 Aberrant Behavior LLC

#include "cudaVec.hu"
#include "typedefs.h"
#include <iostream>

template <typename T>
CudaVec<T>::CudaVec()
: d_vec(nullptr){

}

template <typename T>
CudaVec<T>::CudaVec(const uint& size){
    resize(size);
}

template <typename T>
void CudaVec<T>::resize(const uint& size){
    numElements = size;
    vec.resize(size);
    cuMalloc();
}

template <typename T>
void CudaVec<T>::resizeAsync(const uint& size, const cudaStream_t& stream){
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

template <typename T>
void CudaVec<T>::clear(){
    if(d_vec != nullptr){
        cudaFree(d_vec);
        d_vec = nullptr;
    }
    GPU_MEMORY_ALLOCATED -= sizeof(T) * numElements;
    vec.clear();
    numElements = 0;
}

template <typename T>
void CudaVec<T>::clearAsync(cudaStream_t stream){
    if(d_vec != nullptr){
        cudaFreeAsync(d_vec, stream);
        d_vec = nullptr;
    }
    GPU_MEMORY_ALLOCATED -= sizeof(T) * numElements;
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

template<typename T>
CudaVec<T>::~CudaVec(){
    if(d_vec != nullptr){
        gpuErrchk( cudaFree(d_vec) );
    }
}

template class CudaVec<float>;
template class CudaVec<uint>;
template class CudaVec<char>;

template <typename T>
uint CudaVec<T>::GPU_MEMORY_ALLOCATED = 0;