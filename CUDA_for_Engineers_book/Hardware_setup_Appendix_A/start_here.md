# GPU Computing Ecosystems

This document highlights the differences between the primary high-performance GPU programming stacks: NVIDIA's CUDA, AMD's ROCm, and Intel's oneAPI.

## Comparison Summary

| Feature | NVIDIA | AMD | Intel |
| :--- | :--- | :--- | :--- |
| **Primary Stack** | CUDA | ROCm | oneAPI |
| **Programing Language** | CUDA C++ | HIP / C++ | SYCL / C++ |
| **Portability** | Fixed (NVIDIA only) | High (Can run on NVIDIA/AMD) | Universal (CPU, GPU, FPGA) |
| **Library Maturity** | Industry-leading | Very Strong (AI-focused) | Growing (Science-focused) |
| **Best For** | Max performance, AI Training | Cost-effective AI, Open source | Cross-platform, Enterprise |

## Deep Dive

### NVIDIA: CUDA (Proprietary)
CUDA is the industry standard for high-performance computing, but it is locked to NVIDIA hardware. It offers the most mature ecosystem with highly optimized libraries like cuBLAS and cuDNN.

### AMD: ROCm & HIP (Open/Portable)
AMD uses ROCm as its primary stack. The **HIP (Heterogeneous-computing Interface for Portability)** C++ runtime allows developers to write code that can run on both AMD and NVIDIA GPUs. The `HIPIFY` tool can automatically port most CUDA codebases to HIP.

### Intel: oneAPI & SYCL (Universal)
Intel's oneAPI aims for cross-architecture unity (CPUs, GPUs, and FPGAs). It uses **SYCL**, a standard-based C++ programming model that is fully hardware-agnostic.

---
*Created as part of the CUDA Proficiency learning path.*