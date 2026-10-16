#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// 1. THE KERNEL (Runs on GPU)
// Each thread calculates one element of the sum
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {
    // Calculate the global index for this thread
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    // Check if the index is within bounds
    if (i < n) {
        C[i] = A[i] + B[i];
    }
    
}

// 2. THE HOST WRAPPER (Runs on CPU)
// Handles memory allocation, data transfer, and kernel launch
float* vecAdd(float* A, float* B, float* C, int n) {
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    // Allocate memory on the GPU (Device)
    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    // Copy data from Host (CPU) to Device (GPU)
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // Launch the Kernel
    // We use blocks of 256 threads. ceil(n/256) gives the number of blocks needed.
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);

    // Copy result back from Device (GPU) to Host (CPU)
    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    return C;
}

// 3. MAIN FUNCTION
int main() {
    int n = 10000;
    size_t size = n * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize input vectors
    for(int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Call the host wrapper
    float* h_c=vecAdd(h_A, h_B, h_C, n);
std::cout<<"output_vector : [ ";
for(int i=0;i<n;i++){
    std::cout<<h_c[i]<<(n-1==i?" ":",");
}
std::cout<<"]" <<std::endl;
    
    

   
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

//compilation commands :
//nvcc vector_add.cu -o vector_add && ./vector_add