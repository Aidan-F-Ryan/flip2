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
    set_target_properties(${lib} PROPERTIES CUDA_ARCHITECTURES "${flip2Architectures}")
    target_link_libraries(${lib} radixSortKernels parallelPrefixSumKernels reductionKernels)
endforeach()

find_package(Threads REQUIRED)

# the engine, as a static library: flip2 is one self-contained program (with its third-party libraries beside it in a release, tools/package-linux.sh)
add_library(particles STATIC
    particles.cu
    localTransport.cu
    partitionExchange.cu
    tcpTransport.cu
    simulation.cu
    diagnostics.cu
    forces.cu
    sources.cu
    obstacles.cu
    pockets.cu
    levelset.cu
    viscosity.cu
    freesurface.cu
    twophase.cu
    whitewater.cu
    cacheWriter.cu
)

target_link_libraries(particles ${cudaLibraries}  parallelPrefixSumKernels radixSortKernels reductionKernels particleToGridFunctions perVoxelParticleListFunctions voxelSolveFunctions conjugateGradientFunctions multigridFunctions Threads::Threads)
set_target_properties(particles PROPERTIES CUDA_ARCHITECTURES "${flip2Architectures}")

# blosc compresses the cache's shards (cacheWriter.cu). Optional: without it they're written raw
find_path(BLOSC_INCLUDE_DIR blosc.h)
find_library(BLOSC_LIBRARY blosc)
if(BLOSC_INCLUDE_DIR AND BLOSC_LIBRARY)
    message(STATUS "blosc: ${BLOSC_LIBRARY}")
    target_include_directories(particles PRIVATE ${BLOSC_INCLUDE_DIR})
    target_link_libraries(particles ${BLOSC_LIBRARY})
    target_compile_definitions(particles PRIVATE FLIP2_WITH_BLOSC)
else()
    message(STATUS "blosc not found (install libblosc-dev): the cache's shards are written uncompressed")
endif()

# what cache.json names the build: FLIP2_BUILD_ID if it's given, or git describe of the source tree when configured, or "unknown"
set(FLIP2_BUILD_ID "" CACHE STRING "the build's name in the caches it writes")
set(buildId "${FLIP2_BUILD_ID}")
if(NOT buildId)
    execute_process(COMMAND git describe --always --dirty WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} OUTPUT_VARIABLE buildId OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(NOT buildId)
    set(buildId "unknown")
endif()
target_compile_definitions(particles PUBLIC FLIP2_BUILD_ID="${buildId}")

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
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "${flip2Architectures}")

# flip2: the command line, which runs a scene file (scene.hpp)
add_executable(flip2 flip2.cu scene.cpp volumes.cpp bgeo.cu surface.cu)
target_link_libraries(flip2 particles gomp)
set_target_properties(flip2 PROPERTIES CUDA_ARCHITECTURES "${flip2Architectures}")
if(BLOSC_INCLUDE_DIR AND BLOSC_LIBRARY)     # and Houdini's compressed .bgeo.sc exports (bgeo.cu)
    target_include_directories(flip2 PRIVATE ${BLOSC_INCLUDE_DIR})
    target_link_libraries(flip2 ${BLOSC_LIBRARY})
    target_compile_definitions(flip2 PRIVATE FLIP2_WITH_BLOSC)
endif()
string(REPLACE ";" "," architectureNames "${flip2Architectures}")     #a CMake list would split the definition
target_compile_definitions(flip2 PRIVATE FLIP2_ARCHITECTURES="${architectureNames}")
# installed (tools/package-linux.sh), flip2 finds the libraries it was built with in ../lib beside it, whatever the machine has
set_target_properties(flip2 PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
target_link_options(flip2 PRIVATE "LINKER:--disable-new-dtags")    #an RPATH, not a RUNPATH, so the libraries those libraries load are found there too
install(TARGETS flip2 RUNTIME DESTINATION bin)

# OpenVDB lets scenes name VDB level sets (volumes.cpp). Optional: without it, flip2 builds as before, and a scene naming one fails, saying why. Found
# directly rather than through OpenVDB's own find module, which also wants Boost's iostreams library, which linking to its shared library doesn't need;
# its headers do need Boost's (libboost-dev)
find_path(OPENVDB_INCLUDE_DIR openvdb/openvdb.h)
find_path(BOOST_HEADERS boost/numeric/conversion/conversion_traits.hpp)
find_library(OPENVDB_LIBRARY openvdb)
find_library(TBB_LIBRARY tbb)
find_library(IMATH_LIBRARY NAMES Imath Imath-3_1)    #the half floats its headers use, when it was built with Imath's
if(NOT IMATH_LIBRARY)
    set(IMATH_LIBRARY "")
endif()
if(OPENVDB_INCLUDE_DIR AND NOT BOOST_HEADERS)
    message(STATUS "OpenVDB found, but not Boost's headers, which its headers include (install libboost-dev): scenes can't name VDB level sets")
elseif(OPENVDB_INCLUDE_DIR AND OPENVDB_LIBRARY AND TBB_LIBRARY)
    message(STATUS "OpenVDB: ${OPENVDB_LIBRARY}")
    target_include_directories(flip2 SYSTEM PRIVATE ${OPENVDB_INCLUDE_DIR})
    target_compile_definitions(flip2 PRIVATE FLIP2_WITH_OPENVDB)
    target_link_libraries(flip2 ${OPENVDB_LIBRARY} ${TBB_LIBRARY} ${IMATH_LIBRARY})
    # writes the level sets the tests use (scenes/geo/*.vdb)
    add_executable(makeTestLevelSets ../tools/makeTestLevelSets.cpp)
    target_include_directories(makeTestLevelSets SYSTEM PRIVATE ${OPENVDB_INCLUDE_DIR})
    target_link_libraries(makeTestLevelSets ${OPENVDB_LIBRARY} ${TBB_LIBRARY} ${IMATH_LIBRARY})
else()
    message(STATUS "OpenVDB not found: scenes can't name VDB level sets")
endif()