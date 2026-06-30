// 03_gemm.cu  -- a BIG, real workload: tiled single-precision matrix multiply C
// = A*B
//
// This is THE classic place cp.async earns its keep. Both kernels stage tiles
// of A and B into shared memory and do the same math; they differ ONLY in HOW
// the tile gets to shared:
//
//   gemm_sync  : As[ty][tx] = A[...];  -> LDG (global->register) + STS
//   (register->shared) gemm_async : __pipeline_memcpy_async(...) -> LDGSTS
//   (global->shared, registers bypassed),
//                DOUBLE-BUFFERED so the NEXT tile streams in while the CURRENT
//                tile is being multiplied. That overlap is the whole point.
//
// Build: nvcc -arch=sm_86 -O3 03_gemm.cu -o gemm
// Run:   ./gemm
//
#include <cstdio>
#include <cstdlib>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e) {                                                                   \
      printf("CUDA err %s:%d %s\n", __FILE__, __LINE__,                        \
             cudaGetErrorString(e));                                           \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

constexpr int N = 4096; // 4096 x 4096 matrices
constexpr int TILE =
    32; // 32x32 tile -> 1024 threads/block (one C element per thread)
constexpr int ITERS = 10;

// ---------- classic shared-memory tiled GEMM (sync) ----------
__global__ void gemm_sync(const float *__restrict__ A,
                          const float *__restrict__ B, float *__restrict__ C,
                          int n) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;
  float acc = 0.f;
  int numTiles = n / TILE;
  for (int m = 0; m < numTiles; m++) {
    As[ty][tx] =
        A[row * n + (m * TILE + tx)]; // LDG + STS  (staged via register)
    Bs[ty][tx] = B[(m * TILE + ty) * n + col]; // LDG + STS
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; k++)
      acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  C[row * n + col] = acc;
}

// ---------- double-buffered cp.async tiled GEMM ----------
__global__ void gemm_async(const float *__restrict__ A,
                           const float *__restrict__ B, float *__restrict__ C,
                           int n) {
  __shared__ float As[2][TILE][TILE];
  __shared__ float Bs[2][TILE][TILE];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;
  float acc = 0.f;
  int numTiles = n / TILE;

  // prime: stream tile 0 of A and B into buffer 0 (LDGSTS, no registers used)
  __pipeline_memcpy_async(&As[0][ty][tx], &A[row * n + tx], sizeof(float));
  __pipeline_memcpy_async(&Bs[0][ty][tx], &B[ty * n + col], sizeof(float));
  __pipeline_commit();

  for (int m = 0; m < numTiles; m++) {
    int cur = m & 1, nxt = (m + 1) & 1;
    if (m + 1 < numTiles) { // issue NEXT tile's copy now...
      int mm = m + 1;
      __pipeline_memcpy_async(&As[nxt][ty][tx], &A[row * n + (mm * TILE + tx)],
                              sizeof(float));
      __pipeline_memcpy_async(&Bs[nxt][ty][tx], &B[(mm * TILE + ty) * n + col],
                              sizeof(float));
      __pipeline_commit();
    }
    __pipeline_wait_prior(
        m + 1 < numTiles ? 1 : 0); // ...wait only for CURRENT tile
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; k++)
      acc += As[cur][ty][k] * Bs[cur][k][tx]; // compute || next copy
    __syncthreads();
  }
  C[row * n + col] = acc;
}

template <class F> float timeit(F launch) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  launch();
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a));
  for (int i = 0; i < ITERS; i++)
    launch();
  CK(cudaEventRecord(b));
  CK(cudaEventSynchronize(b));
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, a, b));
  return ms / ITERS;
}

int main() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  printf("GPU: %s  CC %d.%d\n", p.name, p.major, p.minor);
  printf("GEMM: %dx%d * %dx%d, tile %d\n\n", N, N, N, N, TILE);

  size_t bytes = (size_t)N * N * sizeof(float);
  float *A, *B, *Cs, *Ca;
  CK(cudaMalloc(&A, bytes));
  CK(cudaMalloc(&B, bytes));
  CK(cudaMalloc(&Cs, bytes));
  CK(cudaMalloc(&Ca, bytes));
  // fill with a known pattern via a tiny init kernel substitute: memset won't
  // give floats, so use host fill then copy.
  float *h = (float *)malloc(bytes);
  for (size_t i = 0; i < (size_t)N * N; i++)
    h[i] = (float)((i % 13) - 6) * 0.1f;
  CK(cudaMemcpy(A, h, bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(B, h, bytes, cudaMemcpyHostToDevice));

  dim3 blk(TILE, TILE), grd(N / TILE, N / TILE);
  double gflop = 2.0 * (double)N * N * N / 1e9;

  float ts = timeit([&] { gemm_sync<<<grd, blk>>>(A, B, Cs, N); });
  float ta = timeit([&] { gemm_async<<<grd, blk>>>(A, B, Ca, N); });

  // correctness: compare the two C matrices
  float *hs = (float *)malloc(bytes), *ha = (float *)malloc(bytes);
  CK(cudaMemcpy(hs, Cs, bytes, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ha, Ca, bytes, cudaMemcpyDeviceToHost));
  double maxdiff = 0;
  for (size_t i = 0; i < (size_t)N * N; i++) {
    double d = fabs(hs[i] - ha[i]);
    if (d > maxdiff)
      maxdiff = d;
  }

  printf("sync  tiled GEMM : %8.3f ms   %7.1f GFLOP/s\n", ts,
         gflop / (ts / 1e3));
  printf("async tiled GEMM : %8.3f ms   %7.1f GFLOP/s\n", ta,
         gflop / (ta / 1e3));
  printf("-> speedup x%.2f   (max |C_sync - C_async| = %.2e, should be ~0)\n",
         ts / ta, maxdiff);

  free(h);
  free(hs);
  free(ha);
  CK(cudaFree(A));
  CK(cudaFree(B));
  CK(cudaFree(Cs));
  CK(cudaFree(Ca));
  return 0;
}
