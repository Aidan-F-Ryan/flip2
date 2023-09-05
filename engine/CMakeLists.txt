set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

set(cudaLibraries
    particles
    cudaVec
)

add_subdirectory(algorithms)

foreach(lib ${cudaLibraries})
    add_library(${lib} ${lib}.cu)
    set_target_properties(${lib} PROPERTIES CUDA_ARCHITECTURES "61;86;")
    target_link_libraries(${lib} radixSortKernels parallelPrefixSumKernels reductionKernels)
endforeach()

add_executable(main main.cu)
target_link_libraries(main ${cudaLibraries} kernels parallelPrefixSumKernels radixSortKernels reductionKernels)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "61;86;")