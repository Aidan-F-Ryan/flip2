#include "cudaVec.hu"
#include <iostream>

template <typename T>
CudaVec<T>::CudaVec()
: d_vec(0){

}

template <typename T>
CudaVec<T>::CudaVec(const uint& size){
    resize(size);
}

template <typename T>
void CudaVec<T>::resize(const uint& size){
    vec.resize(size);
    cuMalloc();
}

template <typename T>
T& CudaVec<T>::operator[](const uint& i){
    return vec[i];
}

template <typename T>
void CudaVec<T>::upload(){
    cudaMemcpy(d_vec, vec.data(), vec.size()*sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void CudaVec<T>::download(){
    cudaMemcpy(vec.data(), d_vec, vec.size()*sizeof(T), cudaMemcpyDeviceToHost);
}

template <typename T>
T* CudaVec<T>::devPtr(){
    return d_vec;
}

template<typename T>
void CudaVec<T>::cuMalloc(){
    if(d_vec != nullptr){
        cudaFree(d_vec);
    }
    cudaMalloc((void**)&d_vec, sizeof(T)*vec.size());
}

template<typename T>
void CudaVec<T>::swapDevicePtr(T* devPtr){
    if(d_vec != nullptr){
        cudaFree(d_vec);
    }
    d_vec = devPtr;
}

template<typename T>
void CudaVec<T>::print(){
    for(uint i = 0; i < vec.size(); ++i){
        std::cout<<+vec[i]<<", ";
    }
    std::cout<<std::endl;
}

template class CudaVec<float>;
template class CudaVec<uint>;
template class CudaVec<char>;