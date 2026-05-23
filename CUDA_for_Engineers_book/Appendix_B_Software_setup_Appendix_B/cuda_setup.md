# CUDA Software Setup — Appendix B Notes

## My System State (Verified)

| Property | Value |
|---|---|
| **GPU** | NVIDIA GeForce RTX 3060 (12 GB, 3584 CUDA Cores) |
| **Compute Capability** | 8.6 (Ampere) |
| **Driver Version** | 580.126.09 |
| **CUDA Toolkit (active)** | 13.0 (V13.0.88, released Aug 2025) |
| **CUDA Toolkit (also installed)** | 12.8 |
| **Symlink `/usr/local/cuda`** | → `/usr/local/cuda-13.0` |
| **`nvcc` version** | release 13.0, V13.0.88 |
| **OS** | Linux (Ubuntu) |

---

## Part 1 — What the Book Says (Old Methods, CUDA 7.5 Era)

The book "CUDA for Engineers" was written around the CUDA 7.5 era (circa 2015). Here's what it describes and **why each step is outdated**:

### 1.1 Book's Linux Setup Steps

| Step | Book's Method (CUDA 7.5) | What Changed |
|---|---|---|
| **Prerequisites** | `gcc --version` → if missing: `sudo apt-get install gcc` | Still valid. GCC is still needed. But modern systems use `build-essential` package which includes gcc, g++, make, etc. |
| **Download** | Go to `https://developer.nvidia.com/cuda-downloads`, pick OS, download `.deb` | URL still works! But the page now looks completely different. Many more options (Ubuntu 20.04/22.04/24.04, ARM support, network vs local `.deb`). The book shows CUDA 7.5 with only Ubuntu 14.04, OpenSUSE, RHEL, CentOS, SLES, SteamOS, Fedora |
| **Install** | Click `.deb` → "Open with Ubuntu Software Center" → Install button | Ubuntu Software Center **no longer exists**. Modern Ubuntu uses `dpkg -i` from terminal or Snap Store. The GUI package installer method is obsolete |
| **Repository setup** | `sudo apt-get update` then `sudo apt-get install cuda` | Still works conceptually but now you need to install the NVIDIA keyring package first |
| **PATH setup** | Edit `~/.profile`, add `export PATH=/usr/local/cuda/bin:$PATH` | `~/.profile` still works but most people use `~/.bashrc` or `~/.zshrc` now. The path itself is the same |
| **Library path** | Edit `~/.bashrc`, add `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` | Still valid! Same location. Modern installs also set `CUDA_HOME=/usr/local/cuda` |
| **Verify** | `printenv \| grep cuda` and `nvcc --version` | Still 100% valid verification methods |
| **Install samples** | Run `cuda-install-samples-7.5.sh ~` → copies to `~/NVIDIA_CUDA-7.5_Samples` | **COMPLETELY REMOVED.** This install script no longer exists in CUDA 13.0. Samples are now on GitHub |
| **Build samples** | `cd ~/NVIDIA_CUDA-7.5_Samples && make` | **COMPLETELY DIFFERENT.** Now you clone from GitHub and build with CMake, not raw Make |
| **Test run** | `cd 1_Utilities/deviceQuery && ./deviceQuery` | deviceQuery still exists as a sample on GitHub, but you must build it yourself now |

### 1.2 Book's "Initial Test Run" Was Simple Because

The book could say "just run `./deviceQuery`" because:

- NVIDIA shipped **pre-compiled binaries** with the toolkit
- The `cuda-install-samples-X.Y.sh` script copied everything to your home directory
- You just ran `make` in the samples directory and everything compiled

This convenience was progressively removed starting around CUDA 11.6+ (samples moved to GitHub) and fully eliminated in CUDA 13.0 (demo_suite removed too).

---

## Part 2 — The `demo_suite` Story: What It Was, and Why CUDA 13.0 Removed It

### 2.1 What Was the `demo_suite`?

The `demo_suite` was a collection of **pre-compiled, ready-to-run executable binaries** that NVIDIA shipped inside every CUDA Toolkit installation at:

```
/usr/local/cuda-X.Y/extras/demo_suite/
```

On your system, the CUDA 12.8 demo_suite still exists and contains:

```
/usr/local/cuda-12.8/extras/demo_suite/
├── bandwidthTest          # Tests host-to-device and device-to-host memory transfer speeds
├── busGrind               # Stress-tests the PCIe bus connection between CPU and GPU
├── deviceQuery            # Lists all CUDA GPU properties (compute capability, memory, cores, etc.)
├── nbody                  # N-body gravitational simulation — visual GPU compute demo
├── oceanFFT               # Ocean surface simulation using FFT — visual GPU compute demo
├── randomFog              # 3D fog rendering using random number generation on GPU
├── vectorAdd              # Simple vector addition — the "hello world" of CUDA
├── nbody_data_files/      # Shader/data files for nbody visualization
├── oceanFFT_data_files/   # Shader/data files for oceanFFT visualization
└── randomFog_data_files/  # Shader/data files for randomFog visualization
```

### 2.2 What Each Executable Does (Detailed)

#### `deviceQuery`

- **Purpose**: Queries the CUDA runtime API to list every property of every GPU in your system
- **What it shows**: GPU name, compute capability, total memory, CUDA cores, clock rates, max thread dimensions, warp size, memory bus width, L2 cache size, ECC support, unified addressing, etc.
- **Why it's useful**: The **single most important verification test** — if this runs and says "Result = PASS", your CUDA installation is working
- **Output on my system (via CUDA 12.8 binary)**:

  ```
  Device 0: "NVIDIA GeForce RTX 3060"
    CUDA Driver Version / Runtime Version          13.0 / 12.8
    CUDA Capability Major/Minor version number:    8.6
    Total amount of global memory:                 11901 MBytes
    (28) Multiprocessors, (128) CUDA Cores/MP:     3584 CUDA Cores
    GPU Max Clock rate:                            1807 MHz (1.81 GHz)
    Memory Clock rate:                             7501 Mhz
    Memory Bus Width:                              192-bit
    ...
    Result = PASS
  ```

#### `bandwidthTest`

- **Purpose**: Measures the data transfer speed between CPU (host) memory and GPU (device) memory
- **What it measures**: Host-to-Device bandwidth, Device-to-Host bandwidth, Device-to-Device bandwidth, in MB/s
- **Why it's useful**: Tells you if your PCIe connection is performing at expected speeds. A PCIe 3.0 x16 should give ~12 GB/s, PCIe 4.0 x16 should give ~25 GB/s
- **Modern replacement**: NVIDIA now recommends **NVBandwidth** (`https://github.com/NVIDIA/nvbandwidth`) which is more accurate for modern multi-GPU and NVLink topologies

#### `busGrind`

- **Purpose**: Stress-tests the PCIe bus by hammering it with continuous data transfers
- **Why it's useful**: Finds PCIe stability issues, helps diagnose hardware problems

#### `vectorAdd`

- **Purpose**: The simplest possible CUDA program — adds two arrays element-by-element on the GPU
- **Why it's useful**: If even this fails, something fundamental is broken

#### `nbody`

- **Purpose**: Simulates gravitational attraction between thousands of particles (N-body problem)
- **Why it's useful**: Visual, impressive demo showing GPU's parallel computing power. Uses OpenGL for rendering
- **Requires**: Working OpenGL/display. Won't run on headless servers

#### `oceanFFT`

- **Purpose**: Simulates ocean surface waves using Fast Fourier Transform (FFT) computed on the GPU
- **Why it's useful**: Demonstrates cuFFT library usage and GPU-accelerated scientific computing
- **Requires**: Working OpenGL/display

#### `randomFog`

- **Purpose**: Renders a 3D fog effect using GPU-generated random numbers (cuRAND library)
- **Why it's useful**: Demonstrates GPU random number generation and 3D rendering
- **Requires**: Working OpenGL/display

### 2.3 How to Run Them (From CUDA 12.8 on Your System)

Since you still have CUDA 12.8 installed, you can run these right now:

```bash
# deviceQuery — the most important test
/usr/local/cuda-12.8/extras/demo_suite/deviceQuery

# bandwidthTest — memory transfer speed test
/usr/local/cuda-12.8/extras/demo_suite/bandwidthTest

# vectorAdd — simplest CUDA program
/usr/local/cuda-12.8/extras/demo_suite/vectorAdd

# nbody — visual gravitational simulation (needs display)
/usr/local/cuda-12.8/extras/demo_suite/nbody

# oceanFFT — ocean wave simulation (needs display)
/usr/local/cuda-12.8/extras/demo_suite/oceanFFT

# randomFog — 3D fog rendering (needs display)
/usr/local/cuda-12.8/extras/demo_suite/randomFog

# busGrind — PCIe stress test
/usr/local/cuda-12.8/extras/demo_suite/busGrind
```

> **Note**: You don't need to `cd` into the directory. Just use the full path. The `./` before the name is only needed if you `cd` into the directory first. Using the full path like `/usr/local/cuda-12.8/extras/demo_suite/deviceQuery` works without `./`.

### 2.4 Why NVIDIA Removed `demo_suite` in CUDA 13.0

According to the **CUDA Toolkit 13.0 Release Notes** (August 2025), NVIDIA officially **removed the pre-built CUDA Demo Suite applications**. Here's the deep analysis of WHY:

#### Reason 1: Maintenance Burden of Pre-compiled Binaries

- Pre-compiled binaries must be built for **every supported architecture** (x86_64, ARM64, etc.)
- They must be tested against **every supported GPU architecture** (Turing, Ampere, Ada Lovelace, Hopper, Blackwell)
- They must be linked against **every supported OS's libraries** (different glibc versions, OpenGL implementations)
- This is a huge engineering cost for demos that most people never run

#### Reason 2: Binary Size Bloat

- The demo_suite on your CUDA 12.8 takes up ~7.3 MB of space
- Multiply that across every architecture and OS combination
- NVIDIA wants to keep the toolkit download lean

#### Reason 3: Moving to a "Source-First" Philosophy

- NVIDIA moved all samples to GitHub (`https://github.com/NVIDIA/cuda-samples`) starting with CUDA 11.6
- They want developers to **build from source** so they can:
  - Target their specific GPU architecture (compile with `-arch=sm_86` for RTX 3060)
  - Modify and learn from the source code
  - Always have the latest, most up-to-date samples
- Pre-compiled binaries compiled with older settings may not be optimal for newer GPUs

#### Reason 4: Some Demos Were Outdated

- `bandwidthTest` was removed from cuda-samples entirely as of version 12.9 because it gives **inaccurate results on modern systems**
- NVBandwidth is the proper modern tool
- Several demo_suite tools relied on deprecated OpenGL interop patterns

#### Reason 5: CUDA 13.0 Was a Major "Cleanup" Release

CUDA 13.0 removed several things at once:

- **Demo Suite** (pre-compiled test binaries)
- **NVIDIA Visual Profiler (nvvp)** and **nvprof** (legacy profilers)
- **libnvvp** directory (Visual Profiler Java application)
- **`computeprof`** binary (old profiler launcher)
- **Windows display driver bundling**
- **Ubuntu 20.04 support**
- **Intel ICC 2021.7 and MSVC 2017 compiler support**

And added new things:

- **Tile Programming Model** (new way to write CUDA besides SIMT)
- **`ctadvisor`** (Compile Time Advisor — new tool in `/usr/local/cuda-13.0/bin/`)
- **Zstd fatbin compression** (replacing LZ4)
- **Nsight Compute 2025.3.1** and **Nsight Systems 2025.3.2** (upgraded profilers)
- **Unified ARM platform support**

### 2.5 Proof: Directory Diff Between CUDA 12.8 and 13.0 on Your System

```
=== /usr/local/cuda-12.8/extras/ ===        === /usr/local/cuda-13.0/extras/ ===
├── CUPTI/                                   ├── CUPTI/                Still here
├── Debugger/                                ├── Debugger/             Still here
├── demo_suite/          ← EXISTS            (nothing)                 REMOVED!

=== Top-level differences ===
CUDA 12.8 HAS but 13.0 DOESN'T:
  - libnvvp/               (NVIDIA Visual Profiler — legacy profiler REMOVED)
  - computeprof (in bin/)  (legacy profiler launcher REMOVED)

CUDA 13.0 HAS but 12.8 DOESN'T:
  + doc/                   (new documentation directory)
  + ctadvisor (in bin/)    (new Compile Time Advisor tool)

BOTH HAVE (but upgraded versions):
  nsight-compute-2025.1.0  →  nsight-compute-2025.3.1
  nsight-systems-2024.6.2  →  nsight-systems-2025.3.2
  gds-12.8                 →  gds-13.0
```

---

## Part 3 — Modern CUDA Setup on Linux (The Right Way in 2025/2026)

### 3.1 Prerequisites

```bash
# Install build essentials (gcc, g++, make, etc.)
sudo apt-get update
sudo apt-get install build-essential linux-headers-$(uname -r)

# Verify gcc
gcc --version

# Install CMake (needed for building samples now)
sudo apt-get install cmake
cmake --version   # Should be 3.20+

# Install git (needed to clone samples)
sudo apt-get install git
```

### 3.2 Installing the CUDA Toolkit (Modern Method)

Go to: `https://developer.nvidia.com/cuda-downloads`

1. Select: **Linux** → **x86_64** → **Ubuntu** → **22.04** (or your version) → **deb (network)**
2. The page generates commands like:

```bash
# Download the NVIDIA keyring package (authentication)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb

# Install the keyring
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update package lists
sudo apt-get update

# Install CUDA toolkit
sudo apt-get install cuda-toolkit-13-0

# Or install everything (toolkit + driver)
sudo apt-get install cuda
```

> **Book said**: "Right-click on the .deb → Open with Ubuntu Software Center → Install"
> **Modern way**: Use terminal commands above. Ubuntu Software Center no longer exists.

### 3.3 Environment Variables Setup (Modern)

Add to your `~/.bashrc` (or `~/.zshrc` if using zsh):

```bash
# CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Then reload:

```bash
source ~/.bashrc
```

> **Book said**: Edit `~/.profile` for PATH, edit `~/.bashrc` for LD_LIBRARY_PATH (two separate files)
> **Modern way**: Put everything in `~/.bashrc`. Also add `CUDA_HOME` which the book didn't mention but many tools need.

### 3.4 Verification (Same as Book + Modern Additions)

```bash
# 1. Check environment variables are set (same as book)
printenv | grep -i cuda

# Expected output:
# LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:
# CUDA_HOME=/usr/local/cuda-13.0
# PATH=...:/usr/local/cuda-13.0/bin:...

# 2. Check nvcc compiler (same as book)
nvcc --version

# Expected output:
# Cuda compilation tools, release 13.0, V13.0.88

# 3. Check GPU status (book didn't mention this!)
nvidia-smi

# Shows: GPU name, driver version, CUDA version, temperature, memory usage, running processes

# 4. Quick GPU compute capability check (NEW in modern CUDA)
__nvcc_device_query
# Outputs just the compute capability number (e.g., "86" for compute 8.6)
```

---

## Part 4 — Modern Initial Test Runs (Replacing the Book's `demo_suite` Approach)

Since CUDA 13.0 has no `demo_suite`, here are ALL the ways to verify your CUDA installation:

### 4.1 Quick Tests (No Building Required)

These work immediately after installation:

```bash
# Test 1: nvidia-smi — verifies driver and GPU detection
nvidia-smi
# If this works → driver is installed, GPU is detected

# Test 2: nvcc — verifies compiler is in PATH
nvcc --version
# If this works → CUDA toolkit is properly installed and in PATH

# Test 3: __nvcc_device_query — verifies GPU compute capability
__nvcc_device_query
# Outputs "86" for RTX 3060 (compute 8.6)

# Test 4: nvidia-smi detailed query
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader
# Output: NVIDIA GeForce RTX 3060, 580.126.09, 8.6, 12288 MiB, 11408 MiB
```

### 4.2 Use the CUDA 12.8 `demo_suite` (You Still Have It!)

Since CUDA 12.8 is still installed on your system, you can use its demo_suite even while running CUDA 13.0:

```bash
# These binaries still work because they're linked against the CUDA runtime
# The driver (580.126.09) supports both CUDA 12.8 and 13.0

/usr/local/cuda-12.8/extras/demo_suite/deviceQuery
/usr/local/cuda-12.8/extras/demo_suite/bandwidthTest
/usr/local/cuda-12.8/extras/demo_suite/vectorAdd
/usr/local/cuda-12.8/extras/demo_suite/nbody
/usr/local/cuda-12.8/extras/demo_suite/oceanFFT
/usr/local/cuda-12.8/extras/demo_suite/randomFog
/usr/local/cuda-12.8/extras/demo_suite/busGrind
```

> This works because NVIDIA drivers are **backward compatible** — a driver that supports CUDA 13.0 can also run CUDA 12.8 binaries.

### 4.3 Build and Run Samples from GitHub (Modern Way)

This is what NVIDIA now officially recommends:

```bash
# Step 1: Clone the CUDA samples repository
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples

# Step 2: Check out the branch matching your CUDA version
git checkout v13.0
# (or use 'main' branch for latest)

# Step 3: Build using CMake
mkdir build && cd build
cmake ..
make -j$(nproc)    # Uses all CPU cores for faster compilation

# Step 4: Run deviceQuery
./Samples/1_Utilities/deviceQuery/deviceQuery

# Step 5: Run other samples
./Samples/1_Utilities/bandwidthTest/bandwidthTest    # May not exist in v13.0+
./Samples/0_Introduction/vectorAdd/vectorAdd
```

> **Book said**: Run `cuda-install-samples-7.5.sh ~` then `cd ~/NVIDIA_CUDA-7.5_Samples && make`
> **Modern way**: `git clone` → `cmake` → `make`. No install script, no copying to home directory.

### 4.4 Write Your Own Minimal Test (Fastest Modern Verification)

Create a simple CUDA program to verify everything works:

```cuda
// save as cuda_test.cu
#include <stdio.h>

__global__ void hello_kernel() {
    printf("Hello from GPU thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

int main() {
    // Check device
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("CUDA devices found: %d\n", deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %zu MB\n", prop.totalGlobalMem / (1024*1024));
    printf("CUDA Cores: %d MPs x cores/MP\n", prop.multiProcessorCount);

    // Launch a kernel
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();

    printf("\nCUDA test PASSED!\n");
    return 0;
}
```

Compile and run:

```bash
nvcc cuda_test.cu -o cuda_test
./cuda_test
```

### 4.5 NVBandwidth (Modern Replacement for `bandwidthTest`)

```bash
# Clone and build NVBandwidth
git clone https://github.com/NVIDIA/nvbandwidth.git
cd nvbandwidth

# Install dependency
sudo apt-get install libboost-program-options-dev

# Build
cmake .
make

# Run
./nvbandwidth
```

---

## Part 5 — Summary: Book's Methods vs Modern Methods

| Task | Book's Way (CUDA 7.5) | Modern Way (CUDA 13.0) |
|---|---|---|
| **Install CUDA** | Download .deb → Ubuntu Software Center | `wget keyring.deb` → `dpkg -i` → `apt install cuda` |
| **Set PATH** | Edit `~/.profile` | Edit `~/.bashrc`, also set `CUDA_HOME` |
| **Set LD_LIBRARY_PATH** | Edit `~/.bashrc` | Same (still in `~/.bashrc`) |
| **Verify install** | `printenv \| grep cuda` + `nvcc --version` | Same + `nvidia-smi` + `__nvcc_device_query` |
| **Get samples** | `cuda-install-samples-7.5.sh ~` | `git clone https://github.com/NVIDIA/cuda-samples.git` |
| **Build samples** | `cd ~/NVIDIA_CUDA-7.5_Samples && make` | `mkdir build && cd build && cmake .. && make` |
| **Quick device test** | `demo_suite/deviceQuery` (pre-compiled) | `nvidia-smi -q` or build deviceQuery from GitHub |
| **Bandwidth test** | `demo_suite/bandwidthTest` (pre-compiled) | Build NVBandwidth from GitHub |
| **Visual demos** | `demo_suite/nbody`, `oceanFFT`, `randomFog` | Build from cuda-samples GitHub |
| **Profile code** | nvprof / NVIDIA Visual Profiler | Nsight Systems (`nsys`) + Nsight Compute (`ncu`) |

---

## Part 6 — Key Takeaway

> **NVIDIA is NOT "not providing" things for CUDA 13.0.** They're providing **everything** — but as **source code on GitHub** instead of pre-compiled binaries in the toolkit.
>
> The shift is: **"We'll give you ready-made binaries"** → **"We'll give you source code, you build what you need."**
>
> This is actually BETTER because:
>
> 1. Samples stay updated independently of toolkit releases
> 2. You can modify and learn from the source
> 3. Binaries are optimized for YOUR specific GPU when you compile them
> 4. Toolkit download size is smaller
>
> But it does mean you need **Git** and **CMake** as additional prerequisites, which the old book didn't require.
