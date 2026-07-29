// refer 2.4 section documentation file : PMPP(programming massively parallel
// processors)/2.Heterogeneous_data_parallel_computing/2.4_Device_global_memory_and_data_transfer.md
// for full and clearinfo on all these experimentations ,,, ,
#include <cuda_runtime.h>
#include <iostream>

// Experimenting on cudaMalloc and cudaFree commands :

// Test 1: 1 Million Floats

int main() {
  //   float *A_d;
  //   int size = 1000000 * sizeof(float); // 4 MB
  //   cudaMalloc((void **)&A_d, size);
  //   std::cin.get();
  //   cudaFree(A_d);
  // }

  // Test 2: 4 Million Floats
  // int main() {
  //   float *A_d;
  //   int size = 4000000 * sizeof(float); // 16 MB
  //   cudaMalloc((void **)&A_d, size);
  //   std::cin.get();
  //   cudaFree(A_d);

  /*
  =============================================================================
  EXPERIMENT RESULTS: THE HIDDEN CUDA CONTEXT OVERHEAD

  During testing on an RTX 3060 GPU, monitoring nvidia-smi revealed:
  - Test 1 (4 MB requested) consumed 108 MB of VRAM.
  - Test 2 (16 MB requested) consumed 120 MB of VRAM.

  The Math:
  - 108 MB (Total) - 4 MB (Array) = 104 MB Overhead
  - 120 MB (Total) - 16 MB (Array) = 104 MB Overhead

  Conclusion:
  The constant 104 MB overhead is the "CUDA Context". The very first time a
  program calls a CUDA API function (like cudaMalloc), the CUDA runtime
  initializes itself on the device. This context creation is a mandatory,
  one-time "flat fee" per process.
  =============================================================================
  */

  // Test 3 : cudaMemcpy

  // int size = 1000000 * sizeof(float); //~4 MiB (or) 1 million floats ,,, ,
  // using new instead of malloc ,,, ,
  //  float* A_h=new float[1000000];
  // setting all to 1 ,,, ,
  //     float *A_h = (float *)malloc(size);
  //   for (int i = 0; i < 1000000; ++i) {
  //     A_h[i] = 1.0f;
  //   }

  //  using 4 million floats
  int size = 4000000 * sizeof(float); //~16Mib (or) 4 million floats ,,, ,

  float *A_h = (float *)malloc(size);
  for (int i = 0; i < 4000000; ++i) {
    A_h[i] = 1.0f;
  }
  std::cout << "[";
  // I filled A_hwith 1 , not 1.0f or 1.0 ,
  //  so lets see whether it will take as 1.0f natively or not ,
  // by printing the values ,
  // should print 1.0 's    ,,, ,
  for (int i = 0; i < 100; ++i) {
    std::cout << A_h[i] << " ";
  }
  std::cout << "]\n";

  // allocate same space in gpu vram before doing cudaMemcpy ,,, ,
  float *A_d;
  cudaMalloc((void **)&A_d, size);

  // right after creating this would be garbage. To print and see the
  // values/memory in gpu:
  // 1) From the CPU (main function): first , need to cudaMemcpy it back to a
  // CPU array and use std::cout as the CPU cannot read VRAM directly , 2) From
  // the GPU : CUDA actually has a built-in printf() , but can be used inside a
  // __global__ kernel , it can print the VRAM data directly to the terminal ,
  float *B_h = (float *)malloc(size);
  cudaMemcpy(B_h, A_d, size, cudaMemcpyDeviceToHost);
  // should print some garbage or luckily zeroes     ,,, ,
  std::cout << "[";
  for (int i = 0; i < 100; ++i) {
    std::cout << B_h[i] << " ";
  }
  std::cout << "]\n";

  // now lets transfer A_h data (already filled with 1) using cudaMemcpy
  cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  // now lets print the values in A_h by printing thosevalues transferrign to
  // cpu should print 1.0 's    ,,, ,
  float *C_h = (float *)malloc(size);
  cudaMemcpy(C_h, A_d, size, cudaMemcpyDeviceToHost);
  std::cout << "[";
  for (int i = 0; i < 100; ++i) {
    std::cout << C_h[i] << " ";
  }
  std::cout << "]\n";

  std::cin.get();
  cudaFree(A_d);
  free(A_h);
  free(B_h);
  free(C_h);
  std::cin.get();
}

/*
=============================================================================
EXPERIMENT RESULTS: CUDA MEMCPY & CPU RAM (htop)

In this experiment, we tracked the actual CPU RAM using `htop` while running
data transfers to prove the CPU-side impact of CUDA.

1. Test 1: 1 Million Floats (4 MB per array)
   - A_h (4 MB) + B_h (4 MB) + C_h (4 MB) = 12 MB of CPU Arrays
   - Total CPU RAM used right before pressing Enter = ~108.6 MB
   - After freeing the arrays, memory dropped to ~96.6 MB.
   - 108.6 MB - 12 MB = ~96.6 MB of hidden background memory!

2. Test 2: 4 Million Floats (16 MB per array)
   - A_h (16 MB) + B_h (16 MB) + C_h (16 MB) = 48 MB of CPU Arrays
   - Total CPU RAM used right before pressing Enter = ~144.7 MB
   - After freeing the arrays, memory dropped to ~96.7 MB.
   - 144.7 MB - 48 MB = ~96.7 MB of hidden background memory!

WHAT IS THE ~96 MB OVERHEAD?
This is the "CUDA Context".

What the official NVIDIA docs say:
- CUDA Driver API (cuCtxCreate): A CUDA context encapsulates all resources
  (modules, streams, memory allocations) for a process on the GPU.
  (https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
- NVIDIA MPS Documentation: "each CUDA process using a GPU allocates separate
  storage and scheduling resources on the GPU."
  (https://docs.nvidia.com/deploy/mps/index.html)
- CUDA Driver API (cuCtxDestroy): Destroying a context "destroys and cleans up
  all resources associated with the context" including CUmodule, CUfunction,
  CUstream, CUevent, and all memory allocations.

NOTE: NVIDIA does NOT publish the exact MB size of this context overhead.
The ~96-100 MB on GPU (VRAM) and ~96-100 MB on CPU (RAM) are our own empirical
measurements on an RTX 3060 (GA106, 28 SMs). This number varies by GPU
architecture, SM count, and driver version.

This proves that the "irreducible tax" of the CUDA Context exists simultaneously
in both the CPU RAM and GPU VRAM for every single CUDA process!
=============================================================================
*/
