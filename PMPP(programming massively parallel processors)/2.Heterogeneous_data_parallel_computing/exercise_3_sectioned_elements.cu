#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// -----------------------------------------------------------------------------
// Kernel implementing Option (D) for Exercise 3:
// i = blockIdx.x * blockDim.x * 2 + threadIdx.x
// Section 1: C[i]
// Section 2: C[i + blockDim.x]
// -----------------------------------------------------------------------------
__global__ void vecAdd_sectioned_two_elements(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Section 1: Each thread in the block calculates one element
    if (i < n) {
        C[i] = A[i] + B[i];
    }

    // Section 2: All threads in the block move to the next section
    if (i + blockDim.x < n) {
        C[i + blockDim.x] = A[i + blockDim.x] + B[i + blockDim.x];
    }
}

// -----------------------------------------------------------------------------
// Alternative implementation using a for-loop:
// Iterates through sections (s = 0, 1)
// -----------------------------------------------------------------------------
__global__ void vecAdd_sectioned_loop(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    #pragma unroll
    for (int s = 0; s < 2; s++) {
        int idx = i + s * blockDim.x;
        if (idx < n) {
            C[idx] = A[idx] + B[idx];
        }
    }
}

// -----------------------------------------------------------------------------
// Diagnostic kernel: records which thread wrote to which index
// and counts write operations to detect any duplicate overwrites
// -----------------------------------------------------------------------------
__global__ void track_sectioned_writes(int* write_counter, int* written_by_thread, int* written_in_section, int n) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Section 1
    if (i < n) {
        atomicAdd(&write_counter[i], 1);
        written_by_thread[i] = global_tid;
        written_in_section[i] = 1;
    }

    // Section 2
    if (i + blockDim.x < n) {
        atomicAdd(&write_counter[i + blockDim.x], 1);
        written_by_thread[i + blockDim.x] = global_tid;
        written_in_section[i + blockDim.x] = 2;
    }
}

int main() {
    printf("====================================================================\n");
    printf("  PMPP Chapter 2 - Exercise 3 Empirical Validation (Option D)\n");
    printf("  Formula: i = blockIdx.x * blockDim.x * 2 + threadIdx.x\n");
    printf("====================================================================\n\n");

    // -------------------------------------------------------------------------
    // TEST 1: Small Trace to visualize Section 1 and Section 2 mapping
    // n = 16, blockDim = 4 threads/block
    // Total elements per block = 2 * 4 = 8
    // Grid size = 16 / 8 = 2 blocks
    // -------------------------------------------------------------------------
    printf("[TEST 1] Small Trace: n = 16, blockDim = 4 threads/block\n");
    const int small_n = 16;
    const int small_block_dim = 4;
    const int small_grid_dim = (small_n + (small_block_dim * 2) - 1) / (small_block_dim * 2);

    int *d_write_counter, *d_written_by, *d_section;
    int h_write_counter[small_n], h_written_by[small_n], h_section[small_n];

    CHECK_CUDA(cudaMalloc((void**)&d_write_counter, small_n * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_written_by, small_n * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_section, small_n * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_write_counter, 0, small_n * sizeof(int)));

    track_sectioned_writes<<<small_grid_dim, small_block_dim>>>(d_write_counter, d_written_by, d_section, small_n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_write_counter, d_write_counter, small_n * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_written_by, d_written_by, small_n * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_section, d_section, small_n * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Launched: Grid = %d blocks, Block = %d threads (Total threads = %d)\n",
           small_grid_dim, small_block_dim, small_grid_dim * small_block_dim);
    printf("----------------------------------------------------------------------------------------\n");
    printf(" Index | Block | Section | Local Thread (threadIdx.x) | Global Thread ID | Write Count\n");
    printf("----------------------------------------------------------------------------------------\n");
    int small_collisions = 0;
    int small_unwritten = 0;
    for (int idx = 0; idx < small_n; idx++) {
        int g_tid = h_written_by[idx];
        int b_idx = g_tid / small_block_dim;
        int t_idx = g_tid % small_block_dim;
        printf("  %4d |   %3d |    %4d |                         %2d |               %2d |          %2d\n",
               idx, b_idx, h_section[idx], t_idx, g_tid, h_write_counter[idx]);
        if (h_write_counter[idx] > 1) small_collisions++;
        if (h_write_counter[idx] == 0) small_unwritten++;
    }
    printf("----------------------------------------------------------------------------------------\n");
    printf("Summary Test 1: Collisions = %d, Unwritten elements = %d\n\n",
           small_collisions, small_unwritten);

    CHECK_CUDA(cudaFree(d_write_counter));
    CHECK_CUDA(cudaFree(d_written_by));
    CHECK_CUDA(cudaFree(d_section));

    // -------------------------------------------------------------------------
    // TEST 2: Full Math Validation on large vector (n = 100,001, blockDim = 256)
    // -------------------------------------------------------------------------
    printf("[TEST 2] Large Scale Math Validation: n = 100,001, blockDim = 256 threads/block\n");
    const int large_n = 100001;
    const int large_block_dim = 256;
    const int elements_per_block = large_block_dim * 2; // 512 elements per block
    const int large_grid_dim = (large_n + elements_per_block - 1) / elements_per_block;

    size_t bytes = large_n * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_C_ref = (float*)malloc(bytes);

    for (int k = 0; k < large_n; k++) {
        h_A[k] = (float)k * 0.5f;
        h_B[k] = (float)(large_n - k) * 0.25f;
        h_C_ref[k] = h_A[k] + h_B[k];
        h_C[k] = 0.0f;
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Run Sectioned Kernel
    vecAdd_sectioned_two_elements<<<large_grid_dim, large_block_dim>>>(d_A, d_B, d_C, large_n);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    int errors_kernel_1 = 0;
    for (int k = 0; k < large_n; k++) {
        if (fabsf(h_C[k] - h_C_ref[k]) > 1e-5f) {
            errors_kernel_1++;
        }
    }
    printf("Kernel 1 (Two if-blocks) Validation Errors: %d out of %d elements\n", errors_kernel_1, large_n);

    // Run Sectioned Loop Kernel
    CHECK_CUDA(cudaMemset(d_C, 0, bytes));
    vecAdd_sectioned_loop<<<large_grid_dim, large_block_dim>>>(d_A, d_B, d_C, large_n);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    int errors_kernel_2 = 0;
    for (int k = 0; k < large_n; k++) {
        if (fabsf(h_C[k] - h_C_ref[k]) > 1e-5f) {
            errors_kernel_2++;
        }
    }
    printf("Kernel 2 (Loop implementation) Validation Errors: %d out of %d elements\n", errors_kernel_2, large_n);

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    printf("\nAll validation checks completed successfully.\n");
    return 0;
}
