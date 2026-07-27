/*
=============================================================================
A NOTE ON COMPILING C++ AS CUDA:
Because CUDA C is simply an extension of standard C++, a `.cu` file can
seamlessly contain standard, native C++ host code right alongside actual
GPU kernel code. The NVIDIA compiler (nvcc) acts as a universal compiler
that can flawlessly compile standard CPU C++ code and GPU code together!
=============================================================================
*/

#include <iostream>
#include <vector>
using namespace std;
// void vecadd(float *A, float *B, float *C, int N) {
//   for (int i = 0; i < N; ++i) {
//     C[i] = A[i] + B[i];
//   }
// }
// int main() {
//   int N = 10;
//   float *A = new float[N];
//   float *B = new float[N];

//   // WARNING: 'new float[N]' (and C's malloc) does NOT guarantee
//   zero-initialization!
//   // If it prints zeroes right now, it is purely luck because the OS
//   // handed us a fresh, scrubbed memory page. On recycled memory,
//   // this will contain random garbage data.
//   //
//   // THE FIX: Notice the tiny syntax difference:
//   // new float[N]   -> FAST, but leaves random garbage data.
//   // new float[N]() -> The empty '()' forces C++ to scrub and zero-initialize
//   the memory! float *C = new float[N];

//   for (int i = 0; i < N; i++) {
//     A[i] = 1.0f;
//     B[i] = 2.0f;
//   }
//   std::cout << " vector-C before vecadd : \n";
//   std::cout << "[";
//   for (int i = 0; i < N; ++i) {
//     std::cout << C[i] << " ";
//   }
//   std::cout << "]\n";
//   vecadd(A, B, C, N);
//   std::cout << "[";
//   for (int i = 0; i < N; i++) {
//     std::cout << C[i] << " ";
//   }
//   std::cout << "]\n";
//   std::cout << "vector-addition completed";
//   delete[] A;
//   delete[] B;
//   delete[] C;
//   return 0;
// }

// Pass std::vector by reference.
// A and B can be "const since we don't change them.
// C is a normal reference so we can modify its elements.
// void vecadd(const std::vector<float> &A, const std::vector<float> &B,
//             std::vector<float> &C, int N) {
//   for (int i = 0; i < N; i++) {
//     C[i] = A[i] + B[i];
//   }
// }

// int main() {
//   int N = 10;

//   // Initialize vectors of size N. A is filled with 1.0, B with 2.0
//   std::vector<float> A(N, 1.0f);
//   std::vector<float> B(N, 2.0f);

//   // NOTE: If you only provide the size (N) without a second argument,
//   // std::vector automatically initializes all elements to zero (0.0)!
//   std::vector<float> C(N);
//   std::cout << "vector C before vecadd : \n";
//   std::cout << "[";
//   for (int i = 0; i < N; i++) {
//     std::cout << C[i] << " ";
//   }
//   std::cout << "]\n";
//   vecadd(A, B, C, N);

//   std::cout << "[";
//   for (int i = 0; i < N; i++) {
//     std::cout << C[i] << " ";
//   }
//   std::cout << "]\n";
//   std::cout << "vector-addition completed\n";

//   return 0;
// }

// Pass by Value (WARNING: THIS HAS A HUGE BUG!)
// Because C is passed by value (no '&'), this function receives a deep COPY of
// C. It successfully calculates A + B and stores it in the copy of C. But as
// soon as the function ends, that copy is deleted and thrown away! The original
// vector C inside main() is never touched and remains all zeroes.
void vecadd(std::vector<float> A, std::vector<float> B, std::vector<float> C,
            int N) {
  for (int i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }
}
int main() {
  int N = 10;
  std::vector<float> A(N, 1.0f);
  std::vector<float> B(N, 2.0f);
  std::vector<float> C(N);
  std::cout << " vector-c before vecadd:\n ";
  std::cout << "[";
  for (int i = 0; i < N; ++i) {
    std::cout << C[i] << " ";
  }
  std::cout << "]\n";
  vecadd(A, B, C, N);
  std::cout << "vector-c after vecadd: \n";
  std::cout << "[";
  for (int i = 0; i < N; ++i) {
    std::cout << C[i] << " ";
  }
  std::cout << "]\n";
  std::cout << "vector addition completed";
  return 0;
}

/*
=============================================================================
A NOTE ON COMPILING NATIVE HOST KERNELS:
The standard C++ functions are native CUDA host kernels.
This means this exact .cpp file can be compiled perfectly by either a standard
C++ compiler(g++) or the NVIDIA CUDA compiler(nvcc)!

(assuming file is in currently working directory(pwd))
To compile with standard C++ compiler :
g++ vector_addition.cpp && ./a.out

To compile with NVIDIA CUDA compiler:
nvcc vector_addition.cpp && ./a.out
=============================================================================
*/
