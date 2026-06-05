// 02_bench.cu
// PURPOSE: Measure real wall-clock timings on the RTX 3060 for:
//   TEST 1  Pure data flow (copy global->shared->global, NO math): sync vs async
//   TEST 2  Realistic workload (load tile + compute on it): sync vs async double-buffered
//
// The point of TEST 1 is to show the raw transfer path cost.
// The point of TEST 2 is to show *why* async matters: the background copy of the
//   next tile overlaps with compute on the current tile (latency hiding), and the
//   copy no longer burns registers.
//
// Build: nvcc -arch=sm_86 -O3 02_bench.cu -o bench
// Run:   ./bench
//
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

constexpr int TILE   = 256;          // floats per tile (1 KB)  -> one per thread
constexpr int NTILES = 4096;         // tiles processed per block (sequential)
constexpr int BLOCKS = 1024;
constexpr int ITERS  = 50;           // timing repeats
constexpr int COMPUTE_ITERS = 64;    // synthetic math to create overlap opportunity

__device__ __forceinline__ float crunch(float x){
    #pragma unroll
    for(int i=0;i<COMPUTE_ITERS;i++) x = fmaf(x, 1.0000001f, 0.5f);
    return x;
}

// ---------------- TEST 1: pure copy, no math ----------------
__global__ void purecopy_sync(const float* __restrict__ g, float* out){
    __shared__ float s[TILE];
    int t = threadIdx.x;
    long base = (long)blockIdx.x * NTILES * TILE;
    float acc = 0.f;
    for(int k=0;k<NTILES;k++){
        s[t] = g[base + (long)k*TILE + t];   // LDG + STS  (through register)
        __syncthreads();
        acc += s[t];
        __syncthreads();
    }
    out[blockIdx.x*TILE + t] = acc;
}

__global__ void purecopy_async(const float* __restrict__ g, float* out){
    __shared__ float s[TILE];
    int t = threadIdx.x;
    long base = (long)blockIdx.x * NTILES * TILE;
    float acc = 0.f;
    for(int k=0;k<NTILES;k++){
        __pipeline_memcpy_async(&s[t], &g[base + (long)k*TILE + t], sizeof(float)); // LDGSTS
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        acc += s[t];
        __syncthreads();
    }
    out[blockIdx.x*TILE + t] = acc;
}

// ---------------- TEST 2: load + compute ----------------
// Sync: load tile, sync, compute, sync, repeat. Copy CANNOT overlap compute.
__global__ void work_sync(const float* __restrict__ g, float* out){
    __shared__ float s[TILE];
    int t = threadIdx.x;
    long base = (long)blockIdx.x * NTILES * TILE;
    float acc = 0.f;
    for(int k=0;k<NTILES;k++){
        s[t] = g[base + (long)k*TILE + t];
        __syncthreads();
        acc += crunch(s[t]);
        __syncthreads();
    }
    out[blockIdx.x*TILE + t] = acc;
}

// Async double-buffered: prefetch tile k+1 while computing tile k (overlap).
__global__ void work_async(const float* __restrict__ g, float* out){
    __shared__ float s[2][TILE];
    int t = threadIdx.x;
    long base = (long)blockIdx.x * NTILES * TILE;
    float acc = 0.f;

    // prime: issue load of tile 0 into buffer 0
    __pipeline_memcpy_async(&s[0][t], &g[base + t], sizeof(float));
    __pipeline_commit();

    for(int k=0;k<NTILES;k++){
        int cur = k & 1;
        int nxt = (k+1) & 1;
        if(k+1 < NTILES){                       // issue load of next tile into other buffer
            __pipeline_memcpy_async(&s[nxt][t], &g[base + (long)(k+1)*TILE + t], sizeof(float));
            __pipeline_commit();
        }
        __pipeline_wait_prior(k+1<NTILES ? 1 : 0); // wait until current tile's copy is done
        __syncthreads();
        acc += crunch(s[cur][t]);               // compute overlaps the next tile's copy
        __syncthreads();
    }
    out[blockIdx.x*TILE + t] = acc;
}

template<class F>
float time_kernel(F launch){
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    launch();                       // warmup
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    for(int i=0;i<ITERS;i++) launch();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms=0; CK(cudaEventElapsedTime(&ms,a,b));
    return ms/ITERS;
}

int main(){
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("GPU: %s  CC %d.%d\n", p.name, p.major, p.minor);

    size_t N = (size_t)BLOCKS * NTILES * TILE;
    float *g, *out;
    CK(cudaMalloc(&g, N*sizeof(float)));
    CK(cudaMalloc(&out, (size_t)BLOCKS*TILE*sizeof(float)));
    CK(cudaMemset(g, 1, N*sizeof(float)));

    double gb = (double)N*sizeof(float)/1e9;
    printf("Data read per launch: %.2f GB\n\n", gb);

    float t1s = time_kernel([&]{ purecopy_sync <<<BLOCKS,TILE>>>(g,out); });
    float t1a = time_kernel([&]{ purecopy_async<<<BLOCKS,TILE>>>(g,out); });
    printf("TEST 1  PURE COPY (no math)\n");
    printf("  sync  (LDG+STS via reg): %8.3f ms   %6.1f GB/s\n", t1s, gb/(t1s/1e3));
    printf("  async (LDGSTS,no reg)  : %8.3f ms   %6.1f GB/s\n", t1a, gb/(t1a/1e3));
    printf("  -> speedup x%.2f\n\n", t1s/t1a);

    float t2s = time_kernel([&]{ work_sync <<<BLOCKS,TILE>>>(g,out); });
    float t2a = time_kernel([&]{ work_async<<<BLOCKS,TILE>>>(g,out); });
    printf("TEST 2  LOAD + COMPUTE, HIGH occupancy (1024 blocks: lots of warps already hide latency)\n");
    printf("  sync  (load, then compute) : %8.3f ms\n", t2s);
    printf("  async (compute || prefetch): %8.3f ms\n", t2a);
    printf("  -> speedup x%.2f\n\n", t2s/t2a);

    // TEST 3: starve the SMs of warps. RTX 3060 has 28 SMs; launch 28 blocks so each
    // SM holds ~1 block. With few warps there is little TLP to hide global-load latency,
    // so explicit async prefetch (overlap) has something real to hide.
    int LOW = 28;
    float t3s = time_kernel([&]{ work_sync <<<LOW,TILE>>>(g,out); });
    float t3a = time_kernel([&]{ work_async<<<LOW,TILE>>>(g,out); });
    printf("TEST 3  LOAD + COMPUTE, LOW occupancy (%d blocks: little TLP to hide latency)\n", LOW);
    printf("  sync  (load, then compute) : %8.3f ms\n", t3s);
    printf("  async (compute || prefetch): %8.3f ms\n", t3a);
    printf("  -> speedup x%.2f\n", t3s/t3a);

    CK(cudaFree(g)); CK(cudaFree(out));
    return 0;
}
