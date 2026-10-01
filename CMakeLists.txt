cmake_minimum_required(VERSION 3.21)
project(flip2 LANGUAGES CUDA CXX)

# The GPUs the build runs on, as CMake's CUDA_ARCHITECTURES: 86 (RTX 30xx, A-series Ampere) while developing, or "release" for every one a release
# supports: machine code for Turing (75: RTX 20xx, T4), Ampere (80: A100; 86), Ada (89: RTX 40xx, L4, L40), Hopper (90: H100), Blackwell (100: B200;
# 120: RTX 50xx), each also running on its family's later minor versions, plus PTX from 120 that the driver compiles for GPUs newer still. CUDA 13
# builds nothing older than Turing
set(FLIP2_CUDA_ARCHITECTURES "86" CACHE STRING "the GPUs to build for: a CUDA_ARCHITECTURES list, or release")
if(FLIP2_CUDA_ARCHITECTURES STREQUAL "release")
    set(flip2Architectures "75-real;80-real;86-real;89-real;90-real;100-real;120")
else()
    set(flip2Architectures "${FLIP2_CUDA_ARCHITECTURES}")
endif()
message(STATUS "flip2 builds for CUDA architectures ${flip2Architectures}")

add_subdirectory(engine)