#Copyright Aberrant Behavior LLC 2023

set(cudaLibraries 
    parallelPrefixSumKernels
    radixSortKernels
    reductionKernels
)

foreach(lib ${cudaLibraries})
    add_library(${lib} ${lib}.cu)
    set_target_properties(${lib} PROPERTIES CUDA_ARCHITECTURES "61;86;")
endforeach()

add_library(kernels kernels.cu)
target_link_libraries(kernels
    parallelPrefixSumKernels
    radixSortKernels
    reductionKernels
)