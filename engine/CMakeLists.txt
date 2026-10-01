#Copyright 2023 Aberrant Behavior LLC

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

find_package(OpenMP)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo -fPIC")

if(OPENMP_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"-fopenmp\"")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(cudaLibraries
    # particles
    cudaVec
)

add_subdirectory(algorithms)

foreach(lib ${cudaLibraries})
    add_library(${lib} ${lib}.cu)
    set_target_properties(${lib} PROPERTIES CUDA_ARCHITECTURES "86")
    target_link_libraries(${lib} radixSortKernels parallelPrefixSumKernels reductionKernels)
endforeach()

find_package(Threads REQUIRED)

add_library(particles SHARED
    particles.cu
    localTransport.cu
    partitionExchange.cu
    tcpTransport.cu
    simulation.cu
    diagnostics.cu
    forces.cu
    sources.cu
)

target_link_libraries(particles ${cudaLibraries}  parallelPrefixSumKernels radixSortKernels reductionKernels particleToGridFunctions perVoxelParticleListFunctions voxelSolveFunctions conjugateGradientFunctions multigridFunctions Threads::Threads)
set_target_properties(particles PROPERTIES CUDA_ARCHITECTURES "86")

# NCCL joins ranks in separate processes, one GPU each (FLIP2_TRANSPORT=nccl). Optional: found through NCCL_HOME (a CMake or environment variable: an
# NCCL install, build directory or pip's nvidia/nccl) or the system's paths; without it, flip2 builds as before and ranks join over TCP
find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${NCCL_HOME} $ENV{NCCL_HOME} PATH_SUFFIXES include)
find_library(NCCL_LIBRARY nccl HINTS ${NCCL_HOME} $ENV{NCCL_HOME} PATH_SUFFIXES lib lib64)
if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    message(STATUS "NCCL: ${NCCL_LIBRARY}")
    target_sources(particles PRIVATE ncclTransport.cu)
    target_include_directories(particles PUBLIC ${NCCL_INCLUDE_DIR})
    target_link_libraries(particles ${NCCL_LIBRARY})
    target_compile_definitions(particles PUBLIC FLIP2_WITH_NCCL)
else()
    message(STATUS "NCCL not found (set NCCL_HOME to use it): ranks can only join over TCP")
endif()

add_executable(main main.cu)
target_link_libraries(main particles gomp)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "86")

# flip2: the command line, which runs a scene file (scene.hpp)
add_executable(flip2 flip2.cu scene.cpp)
target_link_libraries(flip2 particles gomp)
set_target_properties(flip2 PROPERTIES CUDA_ARCHITECTURES "86")