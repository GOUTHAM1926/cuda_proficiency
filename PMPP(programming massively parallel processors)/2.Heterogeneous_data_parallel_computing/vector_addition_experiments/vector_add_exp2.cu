// in this file , i want to implement one vecadd kernel that can be implemented
// for both cpu and gpu ,,, , (nested function call of device-kernel inside
// global-kernel for gpu and calling directly for cpu ,,, , )
#include <cuda_runtime.h>
#include <iostream>

// method-1
__device__ __host__ void vecadd_gpu_cpu(float *A, float *B, float *C, int N) {
#ifdef __CUDA_ARCH__
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    C[i] = A[i] + B[i];
  }
#else
  for (int i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }
#endif
}

__global__ void vecadd_gpu_main(float *A, float *B, float *C, int N) {
  vecadd_gpu_cpu(A, B, C, N);
}

// //method-2 of doing the same (Actual crct way than above's mangled/little
// confusing method-1)
// __device__ __host__ void vecadd_gpu_cpu(float* A ,float*B,float*C,int i){
//     //as we arent usign any gpu intrinsic or special gpu funcitons specific
//     to only gpu  and arent available/compatible for cpu , no need to use
// //ifdef __CUDA_ARCH_ macro's and all , as its used for masking/blocking gpu
// not seen by cpu and vice versa
// //refer this documentation's end , for clear info :
// https://github.com/GOUTHAM1926/cuda_proficiency/blob/main/PMPP(programming%20massively%20parallel%20processors)/2.Heterogeneous_data_parallel_computing/2.9_Exercises_Questions.md

// C[i]=A[i]+B[i];
// }
// __global__ void vecadd_gpu_main(float*A,float*B,float*C,int N){
//     int i=blockIdx.x*blockDim.x+threadIdx.x;
//     if(i<N){
//         vecadd_gpu_cpu(A,B,C,i);
//     }
// }
//--->now this mehod-2 works for gpu side , but for cpu side ,  there is a trap
//: as if we notice the parameter list , the 4rth argument is "int i"(the single
// index to calculate) , not "int N"(the total size of vector) , In main() at
// line calling cpu vecadd funciton is : "vecadd_gpu_cpu(A_h, B_h, C_h, N);" for
// mthod-1 its ok , but for method-2, , for cpu sid e, if we pass "N", instead
// of "i" it will write to C_h[N] , which is out-of-bounds(Array is indexed from
// 0 to N-1) and none of the valid elements "0" through "N-1" will be computed
// on the CPU ,,, , and to make MEthod-2 work for cpu too , in "main()" funciton
// we must call it inside a loop like : for(int
// i=0;i<N;++i){vecadd_gpu_cpu(A_h,B_h,C_h,i);} ---> with that loop , method-2
// is clean ,,, ,

int main() {
  float *A_h, *B_h, *C_h, *A_d, *B_d, *C_d;
  int N = 100;
  A_h = new float[N];
  for (int i = 0; i < N; ++i) {
    A_h[i] = 1.0f;
  }
  B_h = new float[N];
  for (int i = 0; i < N; ++i) {
    B_h[i] = 2.0f;
  }
  C_h = new float[N]();
  cudaError_t err1 = cudaMalloc((void **)&A_d, N * sizeof(float));
  std::cout << "Status of cudaMalloc of A_d : " << cudaGetErrorString(err1)
            << " in file " << __FILE__ << " at line no : " << __LINE__ << "\n";

  cudaError_t err2 = cudaMalloc((void **)&B_d, N * sizeof(float));
  std::cout << "Status of cudaMalloc of B_d : " << cudaGetErrorString(err2)
            << " in file " << __FILE__ << " at line no : " << __LINE__ << "\n";

  cudaError_t err3 = cudaMalloc((void **)&C_d, N * sizeof(float));
  std::cout << "Status of cudaMalloc of C_d : " << cudaGetErrorString(err3)
            << " in file " << __FILE__ << " at line no : " << __LINE__ << "\n";

  cudaError_t err4 =
      cudaMemcpy(A_d, A_h, N * sizeof(float), cudaMemcpyHostToDevice);
  std::cout << "Status of cudaMemcpy of A_d : " << cudaGetErrorString(err4)
            << " in file " << __FILE__ << " at line no : " << __LINE__ << "\n";

  cudaError_t err5 =
      cudaMemcpy(B_d, B_h, N * sizeof(float), cudaMemcpyHostToDevice);
  std::cout << "Status of cudaMemcpy of B_d : " << cudaGetErrorString(err5)
            << " in file " << __FILE__ << " at line no : " << __LINE__ << "\n";

  vecadd_gpu_main<<<((N + 255) / 256), 256>>>(A_d, B_d, C_d, N);
  // to check whether the kernel launched perfectly or any errors(like if
  // launched with many threads than limit and all etc.,) , and as this <<< >>>
  // isn't a funciton-call unlike cudamalloc/cudamemcpy to return status to be
  // caught error from cudaError_t , and as its just syntax to launch kernel ,
  // we have to use "cudaGetLastError()" function to actually catch/get the
  // error/status of kernel launch i.e., we cant do like cudaError_t error =
  // vecadd_gpu_main<<< M,N>>>(A,B,C,D) ; , becoz this "<<< >>>" only launches a
  // kernel , and won't return anything like function-calls(cudaMalloc or
  // cudaMemcpy) so , to catch these all launching error's , we need to use
  // "cudaGetLastError()" like i coded below ,,, ,
  // so simply cudaGetLastError() for Launch-Time Errors: Kernel launches
  // <<<grid, block>>> are asynchronous statements, not regular C++ function
  // calls, so they cannot return a cudaError_t directly.
  // cudaGetLastError() immediately queries the driver on the host for
  // configuration errors (e.g., launching more than 1024 threads per block,
  // invalid shared memory, invalid stream).
  cudaError_t err_launch = cudaGetLastError();
  std::cout << "Kernel Launch status : " << cudaGetErrorString(err_launch)
            << "\n";
  // this "cudaGetLastError()" catches only launching errors(host-still there
  // launching/enqueing) , other error's (for ex : wrote C[1000000] by-mistake
  // in kernel computation) , these other missing error's (after host alreayd
  // went , after issuing) , and all other error's if anything missed shouldbe
  // caught using "cudaDeviceSynchronize()" like i coded below and so simply :
  // cudaDeviceSynchronize() for Execution-Time Errors:Runtime computation
  // faults (e.g., illegal memory access, out-of-bounds pointer dereference,
  // misaligned memory) occur on the GPU SMs asynchronously while the host has
  // already moved ahead.
  // cudaDeviceSynchronize() blocks the CPU until the GPU finishes all queued
  // tasks and surfaces any latent execution errors.
  //  GPU stalls is spot on: cudaDeviceSynchronize() completely drains the GPU
  //  pipeline and halts host-device concurrency. It is good for debugging, but
  //  in production pipelines, synchronization should be kept to a minimum (or
  //  piggybacked onto naturally synchronizing transfers like cudaMemcpy) ,,, ,
  cudaError_t err_all = cudaDeviceSynchronize();
  std::cout << "kernel execution status : " << cudaGetErrorString(err_all)
            << "\n";
  // this usually stall gpu till all the issued/enqueued process get's compelted
  // , so usually this costs a full gpus tall , and this is fine during
  // development , but in production-codes and performance-specific codes ,
  // usage of this should be thought of once in clear before and should be
  // analzyed on the performance costs ,

  cudaError_t err6 =
      cudaMemcpy(C_h, C_d, N * sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "Status of cudaMemcpy of C_d : " << cudaGetErrorString(err6)
            << " in file " << __FILE__ << " at line no : " << __LINE__ << "\n";

  std::cout << "Vector C after gpu add : [ ";
  for (int i = 0; i < N; ++i) {
    std::cout << C_h[i] << ((i == N - 1) ? " " : " , ");
  }
  std::cout << " ]\n ";

  for (int i = 0; i < N; ++i) {
    A_h[i] = 2.0f;
  }
  vecadd_gpu_cpu(A_h, B_h, C_h, N);
  std::cout << "Vector C after cpu add : [ ";
  for (int i = 0; i < N; ++i) {
    std::cout << C_h[i] << ((i == N - 1) ? " " : " , ");
  }
  std::cout << " ]\n ";
  std::cin.get();
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  delete[] A_h;
  delete[] B_h;
  delete[] C_h;
}

// ============================================================================
// PERFORMANCE DIFFERENCE BETWEEN METHOD-1 AND METHOD-2 :
// ============================================================================
//
// there is ZERO performance difference between method-1 and method-2 ,,, ,
//
// why ? because a __device__ function call is NOT a kernel launch ,,, ,
// it is just a plain ordinary function call happening within the same thread ,
// and the compiler (nvcc/ptxas) INLINES it completely ,,, ,
// so at the SASS (final GPU machine code) level , the function call disappears
// entirely ,,, ,
//
// after inlining , both methods produce the exact same machine code :
//
//   method-1 after inlining (what each thread actually executes) :
//     i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < N) { C[i] = A[i] + B[i]; }
//
//   method-2 after inlining (what each thread actually executes) :
//     i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < N) { C[i] = A[i] + B[i]; }
//
// identical ,,, , same registers , same memory accesses , same instruction
// count , same SASS output ,,, ,
//
// the only difference between method-1 and method-2 is how we (the programmer)
// organize the source code for readability and CPU-side reuse ,,, ,
//
// IMPORTANT DISTINCTION :
// -----------------------
//  calling vecadd_gpu_cpu() inside vecadd_gpu_main() is a __device__ helper
//  function call , NOT a kernel launch ,,, , each thread does NOT "launch a
//  kernel" ,,, , each thread simply executes the inlined arithmetic
//  instructions directly ,,, ,
//
//  this is completely different from Dynamic Parallelism (Chapter 21) , where a
//  __global__ kernel launches another __global__ kernel from the GPU , creating
//  an entirely new child grid of threads with real overhead (device runtime ,
//  scheduling , synchronization) ,,, ,
//
//  | action                                     | what it is              |
//  overhead                    |
//  |--------------------------------------------|-------------------------|-----------------------------|
//  | vecadd_gpu_main<<<grid,block>>>(...)        | kernel launch (__global__)
//  | real overhead (microseconds) | | vecadd_gpu_cpu(A, B, C, i) | device
//  function call (__device__) | zero (inlined away)      | |
//  childKernel<<<1,1>>>(tid) inside __global__ | dynamic parallelism     |
//  heavy overhead (new grid)    |
//
// total kernel launches in both method-1 and method-2 : exactly 1
// total function calls at machine level : 0 (inlined away)
//
// ============================================================================
// COMPILATION FLAGS : -O3 AND DEVICE OPTIMIZATION
// ============================================================================
//
// command used : nvcc -O3 vector_add_exp2.cu -o vector_add_exp2
//
// nvcc is a COMPILER DRIVER (not a single compiler) ,,, , it splits the .cu
// file into two paths :
//
// 1. HOST PATH (CPU code like main() and the #else branch of vecadd_gpu_cpu) :
//    nvcc forwards host code to g++ (or whatever host compiler is configured) ,
//    and when we pass -O3 to nvcc , it forwards -O3 directly to g++ ,,, ,
//    so -O3 controls the CPU-side optimization level only :
//      -O0 = no optimization (fastest compile , easiest to debug)
//      -O1 = basic optimizations
//      -O2 = standard optimizations
//      -O3 = aggressive optimizations (loop unrolling , vectorization ,
//      inlining)
//
// 2. DEVICE PATH (GPU kernel code) :
//    GPU code goes through nvcc's own pipeline : CUDA C -> PTX -> SASS (via
//    ptxas) the device compiler (ptxas) has its OWN separate optimization flag
//    :
//    --ptxas-options or -Xptxas ,,, ,
//    BY DEFAULT , ptxas already runs at -O3 level optimization without us
//    specifying anything ,,, , so GPU code is always aggressively optimized
//    unless we explicitly turn it down ,,, ,
//
//    to control device optimization separately :
//      nvcc -Xptxas -O0 file.cu -o output   // lower device optimization (for
//      debugging GPU code) nvcc -Xptxas -O3 file.cu -o output   // explicitly
//      set device optimization to max
//
// summary :
//   | flag                        | affects                        | default |
//   |-----------------------------|--------------------------------|---------------|
//   | nvcc -O3                    | host compiler (g++) only       | -O0 | |
//   nvcc -Xptxas -O3            | device compiler (ptxas) only   | -O3
//   (already) | | nvcc -O3 -Xptxas -O3        | both separately | both at max |
//
// so when we compiled with "nvcc -O3 vector_add_exp2.cu" , the -O3 only made
// the main() function and host-side vecadd_gpu_cpu (the #else CPU branch) more
// optimized ,,, , the GPU kernel was already compiled at maximum optimization
// by default ,,, ,
//
// ============================================================================
// MISTAKE : -ptxas vs -Xptxas (AND WHAT ptxas EVEN MEANS)
// ============================================================================
//
// i tried this command :
//   nvcc vector_add_exp2.cu -O3 -ptxas -O3 -o vector_add_exp2
//
// and got :
//   nvcc fatal : Unknown option '-ptxas'
//
// the correct command is :
//   nvcc vector_add_exp2.cu -O3 -Xptxas -O3 -o vector_add_exp2
//
// WHY ?
// because nvcc is a COMPILER DRIVER ,,, , it does not do everything itself ,
// it delegates work to multiple downstream tools ,,, , and to pass flags to
// those downstream tools , nvcc uses the "-X" prefix pattern :
//
//   -X<tool_name> <flag_to_forward>
//
// this means : "hey nvcc , don't interpret this flag yourself , just forward
// it to <tool_name>" ,,, ,
//
// writing just "-ptxas" makes nvcc think it is a standalone nvcc flag , which
// does not exist , so it throws "Unknown option" ,,, ,
//
// ----------------------------------------------------------------------------
// WHAT IS ptxas ?
// ----------------------------------------------------------------------------
//
// ptxas = PTX Assembler ,,, ,
//
// remember from 2.7_Compilation.md , the device compilation pipeline is :
//
//   CUDA C source code
//       |
//       v
//   [nvcc frontend] ---> PTX (intermediate representation , like Java bytecode)
//       |
//       v
//   [ptxas] ---> SASS (final GPU machine code , actual binary instructions)
//       |
//       v
//   executable / fatbinary
//
// ptxas is the tool that takes PTX (Parallel Thread Execution) intermediate
// code and ASSEMBLES it into SASS (Streaming ASSembler) machine code for the
// specific GPU architecture ,,, , that is why it is called PTX Assembler ,
// it assembles PTX into real GPU binary ,,, ,
//
// so when we write "-Xptxas -O3" , we are telling nvcc :
// "forward -O3 to the PTX Assembler , so it optimizes the GPU machine code
// generation at maximum level" ,,, ,
//
// ----------------------------------------------------------------------------
// ALL -X FORWARDING FLAGS IN nvcc :
// ----------------------------------------------------------------------------
//
// nvcc has three main -X forwarding flags , one for each downstream tool :
//
// 1. -Xcompiler <flag>
//    forwards <flag> to the HOST COMPILER (g++ on linux , MSVC on windows)
//    this is the tool that compiles all host-side C/C++ code (main() , CPU
//    functions , everything outside __global__ and __device__)
//
//    example :
//      nvcc -Xcompiler -Wall file.cu      // forward -Wall (enable all
//      warnings) to g++ nvcc -Xcompiler -march=native file.cu  // forward
//      -march=native to g++ nvcc -Xcompiler "-O3,-ffast-math" file.cu  //
//      forward multiple flags (comma separated)
//
//    note : -O3 passed directly to nvcc (without -Xcompiler) also gets
//    forwarded to g++ , so "nvcc -O3" and "nvcc -Xcompiler -O3" do the
//    same thing for the host compiler ,,, , but -Xcompiler is needed for
//    flags that nvcc does not recognize on its own (like -Wall , -fPIC ,
//    -march=native , etc.) ,,, ,
//
// 2. -Xptxas <flag>
//    forwards <flag> to ptxas (PTX Assembler , the device code optimizer
//    and assembler that converts PTX to SASS)
//
//    example :
//      nvcc -Xptxas -O0 file.cu    // disable device optimization (useful for
//      debugging GPU code) nvcc -Xptxas -v file.cu     // make ptxas print
//      verbose info (register usage , spills , etc.) nvcc -Xptxas -dlcm=ca
//      file.cu  // set device L1 cache mode
//
// 3. -Xlinker <flag>
//    forwards <flag> to the HOST LINKER (ld on linux)
//    the linker is the tool that takes all compiled object files (.o) and
//    libraries (.so , .a) and links them together into the final executable
//
//    example :
//      nvcc -Xlinker -rpath=/usr/local/cuda/lib64 file.cu  // set runtime
//      library search path nvcc -Xlinker --verbose file.cu    // make the
//      linker print verbose linking info
//
// summary of the forwarding pattern :
//
//   | flag        | full name       | what tool it forwards to          | what
//   that tool does                          |
//   |-------------|-----------------|-----------------------------------|----------------------------------------------|
//   | -Xcompiler  | cross-compiler  | g++ (host C/C++ compiler)         |
//   compiles host-side CPU code                  | | -Xptxas     | cross-ptxas
//   | ptxas (PTX Assembler)             | assembles PTX into SASS (GPU machine
//   code)   | | -Xlinker    | cross-linker    | ld (host linker) | links object
//   files into final executable     |
//
// the "X" stands for "cross" as in "cross-forward this flag to another tool"
// ,,, , nvcc acts as a traffic controller , routing flags to the right
// downstream tool based on the -X prefix ,,, ,
//
