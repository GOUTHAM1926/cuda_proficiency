// In this file , i am gonna code vector addition code(both cuda kernel and
// host/cpu function) in this file itself , so that'sy this is a ".cu" cuda file
// , as cpp compielrs cant compile these extension files ,,, ,
#include <cuda_runtime.h>

#include <iostream>
__global__ void launch_vec_add(float *A_d, float *B_d, float *C_d, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    C_d[i] = A_d[i] + B_d[i];
  }
}
int main() {
  int n = 100;
  float *A_h = new float[n];

  float *B_h = new float[n];
  for (int i = 0; i < n; ++i) {
    A_h[i] = 1.0f;
  }
  for (int i = 0; i < n; ++i) {
    B_h[i] = 2.0f;
  }
  // std::cout << "A_h =[";
  // for (int i = 0; i < n; ++i) {
  //   std::cout << A_h[i] << (i == n - 1 ? " " : " , ");
  // };
  // std::cout << "]\n";
  // std::cout << "B_h = [";
  // for (int i = 0; i < n; ++i) {
  //   std::cout << B_h[i] << (i == n - 1 ? " " : " , ");
  // };
  // std::cout << "]\n";

  float *A_d, *B_d;
  cudaError_t err1 = cudaMalloc((void **)&A_d, n * sizeof(float));
  std::cout << "cudaMalloc of A_d status/error : " << cudaGetErrorString(err1)
            << std::endl;
  cudaError_t err2 = cudaMalloc((void **)&B_d, n * sizeof(float));
  std::cout << "cudaMalloc of B_d status/error : " << cudaGetErrorString(err2)
            << std::endl;
  cudaError_t err3 =
      cudaMemcpy(A_d, A_h, n * sizeof(float), cudaMemcpyHostToDevice);
  std::cout << "cudaMemcpy of A_h to A_d status/error : "
            << cudaGetErrorString(err3) << std::endl;
  cudaError_t err4 =
      cudaMemcpy(B_d, B_h, n * sizeof(float), cudaMemcpyHostToDevice);
  std::cout << "cudaMemcpy of B_h to B_d status/error : "
            << cudaGetErrorString(err4) << std::endl;
  float *C_d;
  cudaMalloc((void **)&C_d, n * sizeof(float));

  launch_vec_add<<<(n + 255) / 256, 256>>>(A_d, B_d, C_d, n);
  float *C_h = new float[n];
  cudaMemcpy(C_h, C_d, n * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "C_h = [";
  for (int i = 0; i < n; i++) {
    std::cout << C_h[i] << (i == n - 1 ? " " : " , ");
  }
  std::cout << "]\n";
  std::cin.get();
}
