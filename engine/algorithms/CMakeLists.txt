set(cudaLibraries 
    parallelPrefixSumKernels
    radixSortKernels
    kernels
    reductionKernels
)

foreach(lib ${cudaLibraries})
    add_library(${lib} ${lib}.cu)
    set_target_properties(${lib} PROPERTIES CUDA_ARCHITECTURES "61;86;")
endforeach()