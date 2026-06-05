// 01_sass_proof.cu
// PURPOSE: Prove the data path at the *instruction* level.
//   - Without cp.async  -> compiler emits  LDG (global->register) + STS (register->shared)
//   - With cp.async     -> compiler emits  LDGSTS (global->shared, registers BYPASSED)
//        * 4-byte async  -> LDGSTS .ACCESS  (data still cached in L1)
//        * 16-byte async -> LDGSTS .BYPASS  (L1 not polluted)
//
// Build:  nvcc -arch=sm_86 -O3 -c 01_sass_proof.cu -o 01.o
// Inspect SASS: cuobjdump -sass 01.o
//
#include <cuda_pipeline.h>   // __pipeline_memcpy_async / commit / wait

// ---------- (A) "Normal" path: data is staged through a REGISTER ----------
__global__ void copy_sync(const float* __restrict__ g, float* out) {
    __shared__ float s[1024];
    int t = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + t;
    s[t] = g[idx];                 // <-- one source line, TWO instructions: LDG then STS
    __syncthreads();
    out[idx] = s[t] * 2.0f;        // touch result so nothing is optimized away
}

// ---------- (B) cp.async, 4 bytes: registers bypassed, L1 ACCESS mode ----------
__global__ void copy_async4(const float* __restrict__ g, float* out) {
    __shared__ float s[1024];
    int t = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + t;
    __pipeline_memcpy_async(&s[t], &g[idx], sizeof(float)); // 4B -> LDGSTS.E (ACCESS)
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    out[idx] = s[t] * 2.0f;
}

// ---------- (C) cp.async, 16 bytes: registers bypassed, L1 BYPASS mode ----------
__global__ void copy_async16(const float4* __restrict__ g, float4* out) {
    __shared__ float4 s[256];
    int t = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + t;
    __pipeline_memcpy_async(&s[t], &g[idx], sizeof(float4)); // 16B -> LDGSTS.E.BYPASS
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    float4 v = s[t];
    v.x *= 2.0f;
    out[idx] = v;
}
