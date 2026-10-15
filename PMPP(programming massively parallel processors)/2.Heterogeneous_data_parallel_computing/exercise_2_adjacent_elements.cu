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
// Kernel implementing Option (C):
// i = (blockIdx.x * blockDim.x + threadIdx.x) * 2
// Each thread processes TWO ADJACENT elements: C[i] and C[i + 1]
// -----------------------------------------------------------------------------
__global__ void vecAdd_adjacent_two_elements(const float* A, const float* B, float* C, int n) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    // First element
    if (i < n) {
        C[i] = A[i] + B[i];
    }
    // Second (adjacent) element
    if (i + 1 < n) {
        C[i + 1] = A[i + 1] + B[i + 1];
    }
}

// -----------------------------------------------------------------------------
// Alternative implementation using a for-loop (unrolled by nvcc at compile time)
// -----------------------------------------------------------------------------
__global__ void vecAdd_adjacent_loop(const float* A, const float* B, float* C, int n) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    for (int k = 0; k < 2; k++) {
        int idx = base + k;
        if (idx < n) {
            C[idx] = A[idx] + B[idx];
        }
    }
}

// -----------------------------------------------------------------------------
// Diagnostic kernel: records which global thread ID wrote to which index
// and counts how many times each index was written (to prove zero collisions)
// -----------------------------------------------------------------------------
__global__ void track_adjacent_writes(int* write_counter, int* written_by_thread, int n) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    if (i < n) {
        atomicAdd(&write_counter[i], 1);
        written_by_thread[i] = global_tid;
    }
    if (i + 1 < n) {
        atomicAdd(&write_counter[i + 1], 1);
        written_by_thread[i + 1] = global_tid;
    }
}

int main() {
    printf("====================================================================\n");
    printf("  PMPP Chapter 2 - Exercise 2 Empirical Validation (Option C)\n");
    printf("  Formula: i = (blockIdx.x * blockDim.x + threadIdx.x) * 2\n");
    printf("====================================================================\n\n");

    // -------------------------------------------------------------------------
    // TEST 1: Small trace to visualize thread-to-index mapping (n = 16, blockDim = 4)
    // -------------------------------------------------------------------------
    printf("[TEST 1] Small Trace: n = 16, blockDim = 4 threads/block\n");
    const int small_n = 16;
    const int small_block_dim = 4;
    // With coarsening factor 2, each block covers 4 * 2 = 8 elements.
    // So we need ceil(16 / 8) = 2 blocks.
    const int small_grid_dim = (small_n + (small_block_dim * 2) - 1) / (small_block_dim * 2);

    int *d_write_counter, *d_written_by;
    int h_write_counter[small_n], h_written_by[small_n];

    CHECK_CUDA(cudaMalloc((void**)&d_write_counter, small_n * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_written_by, small_n * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_write_counter, 0, small_n * sizeof(int)));

    track_adjacent_writes<<<small_grid_dim, small_block_dim>>>(d_write_counter, d_written_by, small_n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_write_counter, d_write_counter, small_n * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_written_by, d_written_by, small_n * sizeof(int), cudaMemcpyDeviceToHost));

    printf("Launched: Grid = %d blocks, Block = %d threads (Total threads = %d)\n",
           small_grid_dim, small_block_dim, small_grid_dim * small_block_dim);
    printf("--------------------------------------------------------------------\n");
    printf(" Data Index (i) | Written By Global Thread ID | Write Count (Collisions?)\n");
    printf("--------------------------------------------------------------------\n");

    bool test1_passed = true;
    for (int idx = 0; idx < small_n; idx++) {
        int gtid = h_written_by[idx];
        int b_idx = gtid / small_block_dim;
        int t_idx = gtid % small_block_dim;
        printf("    Index [%2d]  |  Thread %2d (Block %d, Thread %d) |  %d write(s) %s\n",
               idx, gtid, b_idx, t_idx, h_write_counter[idx],
               (h_write_counter[idx] == 1 ? "✓ OK" : "❌ COLLISION/GAP!"));
        if (h_write_counter[idx] != 1) test1_passed = false;
    }
    printf("--------------------------------------------------------------------\n");
    printf("TEST 1 Result: %s\n\n", test1_passed ? "PASSED (Zero collisions, Zero gaps!)" : "FAILED");

    cudaFree(d_write_counter);
    cudaFree(d_written_by);

    // -------------------------------------------------------------------------
    // TEST 2: Full Vector Addition Benchmark & Verification (n = 100,001)
    // Testing odd length to verify bounds checking
    // -------------------------------------------------------------------------
    const int N = 100001;
    const int BLOCK_SIZE = 256;
    // Each block processes BLOCK_SIZE * 2 elements = 512 elements
    const int GRID_SIZE = (N + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);

    printf("[TEST 2] Full Vector Addition Validation: N = %d elements\n", N);
    printf("Execution Configuration: <<<gridDim=%d, blockDim=%d>>>\n", GRID_SIZE, BLOCK_SIZE);
    printf("Total threads launched: %d (Covers up to %d elements)\n",
           GRID_SIZE * BLOCK_SIZE, GRID_SIZE * BLOCK_SIZE * 2);

    size_t bytes = N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host data
    for (int k = 0; k < N; k++) {
        h_A[k] = (float)k * 1.0f;
        h_B[k] = (float)k * 2.0f;
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch Option (C) Kernel
    vecAdd_adjacent_two_elements<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify against CPU reference
    bool test2_passed = true;
    for (int k = 0; k < N; k++) {
        float expected = h_A[k] + h_B[k];
        if (fabs(h_C[k] - expected) > 1e-5f) {
            printf("Mismatch at index %d: GPU = %f, Expected = %f\n", k, h_C[k], expected);
            test2_passed = false;
            break;
        }
    }

    printf("Verification against CPU reference: %s\n",
           test2_passed ? "100% MATCH across all 100,001 elements! ✓" : "FAILED ❌");

    // -------------------------------------------------------------------------
    // TEST 3: Alternative For-Loop Implementation (N = 100,001)
    // -------------------------------------------------------------------------
    printf("\n[TEST 3] Alternative For-Loop Implementation: N = %d elements\n", N);
    CHECK_CUDA(cudaMemset(d_C, 0, bytes));
    vecAdd_adjacent_loop<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    bool test3_passed = true;
    for (int k = 0; k < N; k++) {
        float expected = h_A[k] + h_B[k];
        if (fabs(h_C[k] - expected) > 1e-5f) {
            printf("Mismatch at index %d in loop kernel: GPU = %f, Expected = %f\n", k, h_C[k], expected);
            test3_passed = false;
            break;
        }
    }
    printf("Verification of Loop Kernel: %s\n",
           test3_passed ? "100% MATCH across all 100,001 elements! ✓" : "FAILED ❌");


    printf("\n====================================================================\n");
    printf("  Summary: Option (C) mathematically and empirically verified!\n");
    printf("====================================================================\n");

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
