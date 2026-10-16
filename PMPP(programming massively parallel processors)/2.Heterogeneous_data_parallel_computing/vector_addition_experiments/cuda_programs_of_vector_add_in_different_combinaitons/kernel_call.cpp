#include <iostream>
#include <cuda_runtime.h> // Still needed for cudaMalloc/cudaMemcpy

// Tell the compiler that this function exists somewhere else (in the .cu file)
void launch_vecAdd(float* d_A, float* d_B, float* d_C, int n);

int main() {
    int n = 5000;
    size_t size = n * sizeof(float);

    // 1. Host Memory
    float *h_A = new float[n], *h_B = new float[n], *h_C = new float[n];
    for(int i=0; i<n; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    // 2. Device Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 3. Move to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 4. CALL THE LAUNCHER (This bridges the .cpp and .cu files)
    launch_vecAdd(d_A, d_B, d_C, n);

    // 5. Move back to CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result: " << h_C[0] << std::endl; // Should be 3.0

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return 0;
}

/*Compilation commands :
nvcc -c kernel.cu -o kernel.o
g++ -c main.cpp -o main.o -I/usr/local/cuda/include
g++ main.o kernel.o -o my_cuda_app -L/usr/local/cuda/lib64 -lcudart
./my_cuda_app

*/