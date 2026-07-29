#include <cuda_runtime.h>
#include <iostream>

int main() {
    float *C_d;
    // Let's set it to a known garbage value
    C_d = (float*)0xDEADBEEF;
    std::cout << "Before cudaMalloc: " << C_d << std::endl;
    
    cudaError_t err = cudaMalloc((void**)&C_d, 1e15);
    std::cout << "cudaMalloc error: " << cudaGetErrorString(err) << std::endl;
    std::cout << "After cudaMalloc: " << C_d << std::endl;
    
    cudaError_t err2 = cudaFree(C_d);
    std::cout << "cudaFree error: " << cudaGetErrorString(err2) << std::endl;
    return 0;
}
