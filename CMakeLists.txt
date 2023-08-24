cmake_minimum_required(VERSION 3.21)
project(flip2 LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

set(cudaLibraries
    cudaVec
)

foreach(lib ${cudaLibraries})
    add_library(${lib} ${lib}.cu)
endforeach()

add_executable(main main.cu)
target_link_libraries(main ${cudaLibraries})