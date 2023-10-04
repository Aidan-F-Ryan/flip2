#Copyright 2023 Aberrant Behavior LLC

set(cudaLibraries 
    parallelPrefixSumKernels
    radixSortKernels
    reductionKernels
    particleToGridFunctions
    perVoxelParticleListFunctions
    voxelSolveFunctions
)

foreach(lib ${cudaLibraries})
    add_library(${lib} ${lib}.cu)
    set_target_properties(${lib} PROPERTIES CUDA_ARCHITECTURES "61;86;")
endforeach()