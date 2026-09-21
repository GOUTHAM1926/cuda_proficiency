 // refer 2.4 section documentaiton "PMPP(programming massively parallel
// processors)/2.Heterogeneous_data_parallel_computing/2.4_Device_global_memory_and_data_transfer.md"
// for clear info of these experiments ,,, ,
#include <cuda_runtime.h>
#include <iostream>
int main() {
  //   int a = printf("hello");
  //   printf("%d", a); //just for testing that printf actually retuns a int and
  //   printing in terminal is just an another task ofthat funciton and for
  //   clear info refer 2.4 section documentation file for more info ,
  int size = 1000000 * sizeof(float);

  // Experiment - 1 ---> Pass a null pointer to cudaMalloc an knowing the
  // importance of passing a nullptr to cudaMalloc , refer 2.4 section
  // ddocumentation file for more info , ---> cudaGetErrorString will give error
  // message as "no error" and its enum status = 0 ,,, ,
  float *A_d = nullptr;
  cudaError_t err = cudaMalloc((void **)&A_d, size);
  std::cout << cudaGetErrorString(err) << " :in " << __FILE__ << "at line"
            << __LINE__ << "\n";
  //"__FILE__"  and "__LINE__" are preprocessor macros and are not specific to C
  // or C++ ,so they work perfectly fine in both C or C++ , and can be used here
  // to detect in which file and in which line the error persists ,,, ,
  // should print "no error" as passing null pointer to
  // cudaMalloc doesnt matter as cudaMalloc just overwrites
  // with new gpu address ,
  //  doesnt matter what was there at that location earlier and passing
  //  null-pointer is the safe thing actually and good practice of code actually
  //  and for dangers regarding passing not-null and just defined pointers to
  //  cudamalloc aredocumented in that 2.4 section documentation file, refer
  //  that once ,
  //  ,
  // but if i pass null ptr to cudaMemcpy then only  its a problmand will show
  // error , lets try that ,

  // Experiment - 2 ---> Pass a null pointer to cudaMemcpy ,,, , ---> should
  // give "invalid argument" error ,,, ,
  float *B_d = nullptr;
  cudaError_t err1 = cudaMemcpy(B_d, A_d, size, cudaMemcpyDeviceToHost);
  if (err1 != cudaSuccess) {
    std::cout << cudaGetErrorString(err1) << " :in " << __FILE__ << " at line "
              << __LINE__ << "\n";
    // exit(EXIT_FAILURE); // EXIT_FAILURE is a preprocessor macro used to quit
    //  the
    //   process at that instant , which means to exit , the
    //   exit status 1 , for more info refer 2.4 section
    //   documentation file once , and if not commented this , then as before
    //   that std::cin.get() itself , thisis been executed and as we passed a
    //   null pointer to this cudaMemcpy API , pakka it will give error and it
    //   will quit before asking you to click enter for passing
    //   through std::cin.get() or to execute next lines code instructions , and
    //   it will print "1" ifu run "echo $?" command in terminal , for knowing
    //   what this "echo $?" command does , refer 2.4 section documentation file
    //   once ,,, ,
  }

  // Experiment -3 ---> allocating a huge absurd amount of memory that's not
  // physically available here to get that "out of memory error" ---> enum
  // status 2 ,,, ,
  long size1 = 1e15;
  // float *C_d = nullptr;
  float *C_d; // using this uninitialzied pointer for expriment-4 ,,, ,
  cudaError_t err2 = cudaMalloc((void **)&C_d, size1);
  std::cout << cudaGetErrorString(err2) << " in " << __FILE__ << " at line "
            << __LINE__ << "\n";
  // if (err2 != cudaSuccess) {
  //   std::cout << cudaGetErrorString(err2) << " in " << __FILE__ << " at line
  //   "
  //             << __LINE__ << "\n";
  //   exit(EXIT_FAILURE); // if uncommented this , then at this line it will
  //   kill
  //                       // this current process instantly and wont wait till
  //                       // clicking "Enter" for passing below std::cin.get()
  //                       ,
  //                       // as alloting that much big amount of memory that
  //                       // physically nto available in this RTX 3060 VRAM
  //                       (12.3
  //                       // GiB available) will result in "out of memory"
  //                       error ,
  //                       // so passing this if-block condition and quitting
  //                       this
  //                       // process instantly here at this line itself ,,, ,
  // }
  std::cin.get();
  cudaFree(A_d);
  // Experiemntation -4 ---> freeing a non-existing memory pointer ---> i
  // thought this would give error byfreeing an uninitialized pointer but it
  // gave "no error" , and i even tested that too and nvidia guys did a good
  // thing here , explained in 2.4 section documentaiton in detail ,
  //  C_d is a pointer which holds garbage value and doesnt point to any
  //  existing memory loacation in gpu ,,, ,
  cudaError_t err3 = cudaFree(C_d);
  std::cout << cudaGetErrorString(err3) << " in " << __FILE__ << " at line "
            << __LINE__ << "\n";
  // no need to free C_d as nothing is allcoated in gpu to
  // free/deaalocate and that C_d itself lives on cpu "stack" memory and will be
  // automatically destroyed once main() function ends , and cudaFree() is to
  // deallocate/free the data on the GPU Heap ,,, , and even if we do that
  // cudaFree(C_d) , its like cudaFree(nullptr) and as its a nullptr, its
  // completely safe but redundant ,
  //                                                             <---------why i
  //                                                             did experiment
  //                                                             : 4
  //   // and then i thought ---> if i wouldnt have initialzied that C_d
  // to a nullptr and just defined it as "float *C_d;"  and used cudaFree(C_d);
  // here na, then it might have been a problem as it will clear some other
  // location as that C_d pointer stores some garbage value which locates to
  // some other location in memory(sometimes might be the loaction used by
  // previous operations , which causes double-free bug or else if its a garbage
  // value ,not any valid location, then thats "invalid-free" bug which will
  // result in "Invalid", refer 2.4 section documentaion file for full info

  //---> but actually it gave " no error" as nvidia-guys had some safety code in
  // their api's , to know why it didnt give any error , i did another
  // experiment to debug :

  // Experiemnt - 5 ---> to find whats happening inside these cuda api's and why
  // it didnt give any error :
  float *C_d1 =
      (float *)0xDEADBEEF; // I forced it to hold a FATAL garbage value
  std::cout << "Before: " << C_d1 << std::endl; // Printed: 0xdeadbeef

  cudaError_t err4 = cudaMalloc((void **)&C_d1, 1e15); // Out of memory!
  std::cout << cudaGetErrorString(err4) << " in " << __FILE__ << " at line "
            << __LINE__ << "\n";
  std::cout << "After: " << C_d1 << std::endl; // Printed: 0 (nullptr!)
  cudaError_t err5 = cudaFree(C_d1);
  std::cout << cudaGetErrorString(err5) << " in " << __FILE__ << " at line "
            << __LINE__ << "\n"; // Returns: "no error"

  // when cudaMalloc fails (like throwing an "out of memory" error), the very
  // last thing the nvidia driver does before returning the error code is it
  // actively wipes your pointer and sets it to 0 (nullptr) and  i think nvidia
  // engineers knew that programmers forget to initialize pointers , so , if
  // cudaMalloc fails , they overwrite this garbage address with
  // nullptr so that if  accidentally called cudaFree(C_d) later , it acts like
  // cudaFree(nullptr) and safely returns "no error" instead of crashing
  // whole GPU context , omg , its nice right !!! !  ,,, ,

  // and also if u have doubt on whether to define a pointer as "float*
  // A_d" or " * A_d" (all are crct and same only though) ---> refer 2.4 section
  // documentation for clear explanations ,,, , "float *A_d"

  // Experiment - 5 ---> to print all the error messages the exists in a log
  // file , looping from 0 to 999  and printing cudaGetErrorString() for each
  // number to see which ones are defined ,,, ,
  for (int i = 0; i < 1500; ++i) {
    // std::cout << cudaGetErrorString(static_cast<cudaError_t>(i))
    //           << " for number " << i << "\n"; ---> this will print error msgs
    //           for all numbers , no filtering out the numbers where the error
    //           message is nothing ,,, ,
    // CRITICAL C++ RULE: cudaGetErrorString returns a 'const char*' (a memory
    // pointer). If you use '!=' on it, C++ compares the MEMORY ADDRESSES, not
    // the text! We MUST cast it to std::string so C++ compares the actual
    // letters ,
    std::string error_msg = cudaGetErrorString(static_cast<cudaError_t>(i));
    if (error_msg != "unrecognized error code") {
      std::cout << error_msg << " for number " << i << "\n";
    }
  }
}
