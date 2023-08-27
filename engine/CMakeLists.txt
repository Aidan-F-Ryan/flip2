cmake_minimum_required(VERSION 3.21)
project(flip2 LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")

set(cudaLibraries
    cudaVec
)

foreach(lib ${cudaLibraries})
    add_library(${lib} ${lib}.cu)
    set_target_properties(${lib} PROPERTIES CUDA_ARCHITECTURES "61;")
endforeach()

add_executable(main main.cu)
target_link_libraries(main ${cudaLibraries})
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "61;")