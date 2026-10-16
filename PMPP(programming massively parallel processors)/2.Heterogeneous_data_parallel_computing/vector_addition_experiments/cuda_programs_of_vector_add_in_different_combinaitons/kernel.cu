#include <cuda_runtime.h>

// The actual GPU Kernel
__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

// The "Launcher" function (This can be called from standard C++)
void launch_vecAdd(float* d_A, float* d_B, float* d_C, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // The <<< >>> syntax only works in .cu files
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    // Wait for the GPU to finish before returning to CPU code
    cudaDeviceSynchronize();
}
