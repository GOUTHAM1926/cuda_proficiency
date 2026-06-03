#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("=== GPU: %s ===\n\n", prop.name);
    
    // Memory hierarchy
    printf("--- MEMORY HIERARCHY ---\n");
    printf("Global Memory (VRAM):     %zu MB  (%zu bytes)\n", prop.totalGlobalMem / (1024*1024), prop.totalGlobalMem);
    printf("L2 Cache Size:            %d KB   (%d bytes)\n", prop.l2CacheSize / 1024, prop.l2CacheSize);
    printf("Shared Memory per Block:  %zu KB  (%zu bytes)\n", prop.sharedMemPerBlock / 1024, prop.sharedMemPerBlock);
    printf("Shared Memory per SM:     %zu KB  (%zu bytes)\n", prop.sharedMemPerMultiprocessor / 1024, prop.sharedMemPerMultiprocessor);
    printf("Registers per Block:      %d\n", prop.regsPerBlock);
    printf("Registers per SM:         %d\n", prop.regsPerMultiprocessor);
    
    printf("\n--- SM INFO ---\n");
    printf("Number of SMs:            %d\n", prop.multiProcessorCount);
    
    printf("\n--- THREAD/BLOCK/GRID LIMITS ---\n");
    printf("Max Threads per Block:    %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per SM:       %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max Blocks per SM:        %d\n", prop.maxBlocksPerMultiProcessor);
    printf("Max Grid Size (x):        %d\n", prop.maxGridSize[0]);
    printf("Max Grid Size (y):        %d\n", prop.maxGridSize[1]);
    printf("Max Grid Size (z):        %d\n", prop.maxGridSize[2]);
    printf("Max Block Dim (x):        %d\n", prop.maxThreadsDim[0]);
    printf("Max Block Dim (y):        %d\n", prop.maxThreadsDim[1]);
    printf("Max Block Dim (z):        %d\n", prop.maxThreadsDim[2]);
    printf("Warp Size:                %d\n", prop.warpSize);
    
    printf("\n--- COMPUTE CAPABILITY ---\n");
    printf("Compute Capability:       %d.%d\n", prop.major, prop.minor);
    
    printf("\n--- DERIVED VALUES ---\n");
    printf("Max Warps per SM:         %d  (maxThreadsPerSM / warpSize)\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
    printf("Total CUDA Cores (approx): %d  (SMs × 128 for Ampere)\n", prop.multiProcessorCount * 128);
    
    return 0;
}
