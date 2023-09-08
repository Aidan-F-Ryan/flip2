#ifndef TYPEDEFS_H
#define TYPEDEFS_H

// #define BLOCKSIZE 32
#define BLOCKSIZE 256
#define WORKSIZE (BLOCKSIZE<<1)

#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



typedef unsigned int uint;

#endif