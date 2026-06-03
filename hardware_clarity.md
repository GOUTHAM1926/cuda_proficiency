# GPU Hardware Clarity — Common Doubts Answered

> **Target GPU: NVIDIA GeForce RTX 3060**
>
> - Architecture: Ampere (Compute Capability 8.6)
> - VRAM: 12 GB GDDR6
> - SMs: 28
> - CUDA Cores per SM: 128 (Total: 3584)
> - Tensor Cores per SM: 4
> - L2 Cache: 2.25 MB
> - SRAM per SM: 128 KB (split between L1 cache and shared memory)
> - Max Threads per Block: 1024
> - Max Threads per SM: 1536
> - Max Blocks per SM: 16
> - Max Warps per SM: 48
> - Warp Schedulers per SM: 4
> - Max Grid Size (x): 2^31 - 1 = 2,147,483,647 blocks

These specifications were obtained by running [`device_query.cu`](device_query.cu) which uses `cudaGetDeviceProperties()`.

---

## What Is Global Memory / VRAM / DRAM?

These are all the SAME thing — the big main memory on the GPU card.

```
Global Memory (VRAM):  12 GB on the RTX 3060
```

This is the GDDR6 memory chips physically soldered onto the GPU card. NVIDIA calls it "global memory" in CUDA. Hardware people call it VRAM or DRAM. Accessible by ALL threads in ALL blocks across ALL SMs. Large but slow.

---

## What Is L2 Cache? Why Does It Exist?

```
L2 Cache: 2.25 MB on the RTX 3060
```

L2 cache sits BETWEEN the SMs and VRAM. It's shared by ALL 28 SMs.

### Why not just go directly from VRAM to L1?

Because L1 is PER SM. If SM #0 and SM #15 both need the same data, without L2 they'd BOTH have to go all the way to VRAM (slow). With L2, the data gets cached in one central place. SM #0 reads it from VRAM → gets cached in L2. SM #15 then finds it in L2 (fast) instead of going to VRAM again.

L2 is the "shared middle layer" that prevents every SM from independently hammering the slow VRAM.

### What if data exceeds 2.25 MB?

It does NOT fail. L2 is a CACHE, not storage. It only holds the most recently accessed data. When full, it evicts the oldest/least-used data to make room for new data.

```
Thread reads address A → not in L2 → fetched from VRAM → cached in L2
Thread reads address B → not in L2 → fetched from VRAM → cached in L2
...L2 fills up...
Thread reads address X → not in L2 → fetched from VRAM →
    L2 EVICTS oldest entry → caches X
Thread reads address A again →
    Maybe still in L2 (cache hit = fast!)
    Maybe evicted (cache miss = go to VRAM again, slow)
```

If your working set fits in 2.25 MB → lots of cache hits (fast). If huge → lots of cache misses (slow, goes to VRAM).

### Why only 2.25 MB? Why not more?

Fast memory (SRAM) is expensive to build — takes more silicon area, more power, more money. NVIDIA chose 2.25 MB as a tradeoff for this GPU tier. Other GPUs have different amounts:

```
RTX 3060 (Ampere):   2.25 MB L2
RTX 3090 (Ampere):   6 MB L2
RTX 4090 (Ada):      72 MB L2     ← massive jump!
A100 (datacenter):   40 MB L2
H100 (datacenter):   50 MB L2
```

---

## The 128 KB SRAM Per SM — L1 Cache and Shared Memory

Each of the 28 SMs has its own separate 128 KB SRAM. SM #0's SRAM is completely independent from SM #15's SRAM.

```
GPU Die:
┌─────────┬─────────┬─────────┬─────────┐
│  SM #0  │  SM #1  │  SM #2  │  SM #3  │
│ 128KB   │ 128KB   │ 128KB   │ 128KB   │
│ SRAM    │ SRAM    │ SRAM    │ SRAM    │
├─────────┴─────────┴─────────┴─────────┤
│          L2 Cache (2.25 MB)            │  ← shared by ALL SMs
├────────────────────────────────────────┤
│          VRAM (12 GB GDDR6)            │  ← shared by ALL SMs
└────────────────────────────────────────┘
```

### This SRAM is NOT just "L1 cache." It is SPLIT into two parts

```
128 KB SRAM on each SM
┌──────────────────────────────────────────┐
│   Part 1: SHARED MEMORY                  │
│   - Programmer-controlled (YOU write code)│
│   - Visible to threads in SAME BLOCK only │
│   - Max configurable: 100 KB per SM       │
│   - Default max per block: 48 KB          │
│──────────────────────────────────────────│
│   Part 2: L1 CACHE                        │
│   - Hardware-controlled (automatic)       │
│   - You write NO code for it              │
│   - Hardware decides what to cache         │
│   - Gets whatever KB is left after shared │
└──────────────────────────────────────────┘
```

### Shared memory — who sees it?

Shared memory is PER BLOCK, not per SM. Each block gets its own private chunk. Block 0 cannot see Block 2's shared memory, even if both are on the same SM.

```
SM #0:
  Block 0:  shared_mem_0[...]  ← ONLY Block 0's threads can access this
  Block 2:  shared_mem_2[...]  ← ONLY Block 2's threads can access this

  Block 0's Thread 5 tries to read shared_mem_2? IMPOSSIBLE.
```

### L1 cache — who sees it?

L1 cache is transparent. No thread writes code saying "read from L1 cache." Instead:

```
// Your code just reads global memory normally:
float x = global_array[i];

// Behind the scenes, hardware does:
// 1. Check L1 cache → found? Return it (fast)
// 2. Not in L1? Check L2 → found? Return it, also cache in L1
// 3. Not in L2? Go to VRAM → return it, cache in L2 AND L1
```

L1 automatically caches global memory reads for ALL threads on that SM regardless of which block they're in.

### Shared memory vs L1 cache — key differences

| | Shared Memory | L1 Cache |
|---|---|---|
| **Who controls it?** | Programmer (you write code to use it) | Hardware (automatic, behind the scenes) |
| **Who sees it?** | Threads in the SAME BLOCK only | All threads on that SM (transparent) |
| **You write code?** | Yes, declare `__shared__` arrays | No, happens automatically |
| **When data lands here** | Only when YOU explicitly copy data | Automatically when you read global memory |

### What does "100 KB shared memory per SM" vs "48 KB per block" mean?

- **48 KB** = default maximum that ONE block can request
- **100 KB** = total shared memory budget for the ENTIRE SM (split among all blocks on that SM)

### If each block uses 48 KB, how many blocks fit?

```
Total shared memory on SM:  100 KB
Each block requests:         48 KB
100 / 48 = 2.08 → only 2 blocks fit (2 × 48 = 96 KB used)
```

So yes, if every block uses 48 KB shared memory, only 2 blocks can run per SM.

### What about the remaining 28 KB (128 - 100)?

That 28 KB is used as **L1 cache**. It is NOT idle. It is NOT L2 cache. The hardware automatically uses it to cache global memory reads. The split is flexible:

```
Config 1: 100 KB shared memory + 28 KB L1 cache  = 128 KB
Config 2: 64 KB shared memory  + 64 KB L1 cache  = 128 KB
Config 3: 48 KB shared memory  + 80 KB L1 cache  = 128 KB
Config 4: 0 KB shared memory   + 128 KB L1 cache = 128 KB
```

If your kernel uses lots of shared memory, L1 gets smaller. If your kernel uses none, L1 gets the full 128 KB.

---

## Threads, Blocks, Grids, and SMs

### Definitions

- **Thread** = smallest unit. One thread executes one copy of your function.
- **Block** = group of threads (up to 1024). All threads in a block run on the SAME SM and can cooperate via shared memory and `__syncthreads()`.
- **Grid** = ALL blocks launched for one kernel. 1 kernel = 1 grid. Can have up to 2^31-1 blocks.
- **SM** = Streaming Multiprocessor. Hardware processing unit. The RTX 3060 GPU has 28 SMs.

### Can an SM run more than one block?

Yes! An SM can run MULTIPLE blocks at the same time. Not just one. The limit is on total threads per SM, not on blocks per SM.

- Max threads **per block** = 1024 (hard limit — a single block cannot exceed this)
- Max threads **per SM** = 1536 (total across ALL blocks on that SM)

So if each block has 256 threads, one SM can run 1536/256 = 6 blocks simultaneously. If each block has 1024 threads, the SM can only run 1 block (since 2 × 1024 = 2048 exceeds 1536).

### Threads per SM: 1536 vs 1024 — both are correct but mean different things

- **1024** = maximum threads in ONE block. A single block cannot have more than 1024 threads.
- **1536** = maximum threads an SM can manage at the same time. This can come from MULTIPLE blocks.

```
Valid configurations on one SM:
  1 block × 1024 threads = 1024 total   (OK, under 1536)
  3 blocks × 512 threads = 1536 total   (OK, exactly at limit)
  6 blocks × 256 threads = 1536 total   (OK)

  NOT possible:
  2 blocks × 1024 threads = 2048        (exceeds 1536 per SM!)
  1 block × 1536 threads                (exceeds 1024 per block!)
```

### What about 1 thread per block — can we launch 1024 blocks in an SM?

No. Even with 1-thread blocks, you hit the **max blocks per SM limit of 16**. So with 1-thread blocks, you'd only get 16 blocks on one SM, using only 16 of the 1536 thread slots. That would be extremely wasteful — 1520 thread slots sitting empty.

### Where do blocks live?

A block is always assigned to exactly ONE SM. It runs entirely there and never moves. When it finishes, the SM is free to take the next waiting block.

A block CANNOT be bigger than what an SM can handle (max 1024 threads per block).

```
SM #0                    SM #1                    SM #2
┌──────────────┐        ┌──────────────┐        ┌──────────────┐
│  Block 0     │        │  Block 3     │        │  Block 1     │
│  Block 2     │        │  Block 7     │        │  Block 5     │
│  Block 6     │        │              │        │  Block 9     │
└──────────────┘        └──────────────┘        └──────────────┘
```

The GPU's hardware scheduler decides which SM gets which block. You don't control this.

### How many blocks per SM?

Multiple limits compete — the TIGHTEST one wins:

```
1. Max blocks per SM:        16
2. Max threads per SM:       1536
3. Shared memory per SM:     100 KB
4. Registers per SM:         65536

Example cases:
  Block = 256 threads, 4KB shared mem:
    Block limit:  16 blocks
    Thread limit: 1536/256 = 6 blocks
    → TIGHTEST = 6 blocks (threads are bottleneck)

  Block = 48KB shared mem, 256 threads:
    Shared limit: 100/48 = 2 blocks
    Thread limit: 6 blocks
    → TIGHTEST = 2 blocks (shared memory is bottleneck)

  Block = 32 threads, 0 shared mem:
    Block limit:  16 blocks
    Thread limit: 1536/32 = 48 (but capped at 16)
    → TIGHTEST = 16 blocks (block count is the bottleneck)
```

### Grid size (2^31-1) vs SM occupancy (16) — NOT contradictory

It is often noted that we can initialize 2^31-1 blocks, yet max blocks per SM is only 16. Both are correct but mean different things:

- **2^31-1** = how many blocks you can launch in ONE kernel (grid limit)
- **16** = how many blocks ONE SM can hold at any moment (occupancy limit)

```
You launch 2,000,000 blocks.
GPU has 28 SMs × 16 blocks = 448 blocks at a time.

Round 1: 448 blocks run. 1,999,552 wait in queue.
As blocks finish, new ones are assigned.
Eventually all 2,000,000 complete.
```

---

## What Does `__syncthreads()` Do?

It's a barrier that works ONLY within one block. When a thread hits `__syncthreads()`, it stops and waits until EVERY thread in the same block has also reached that line. Then they all continue together.

```
Block 0 (4 threads):
  Thread 0:  step_A()  →  __syncthreads()  →  step_B()
  Thread 1:  step_A()  →  __syncthreads()  →  step_B()
  Thread 2:  step_A()  →  __syncthreads()  →  step_B()
  Thread 3:  step_A()  →  __syncthreads()  →  step_B()

  Thread 2 finishes step_A first. WAITS at barrier.
  Thread 0 finishes. WAITS.
  Thread 3 finishes. WAITS.
  Thread 1 finishes last. NOW all 4 are at barrier.
  → All 4 proceed to step_B together.
```

**Why is this useful?** If thread 0 writes a result to shared memory in step_A, and thread 3 needs to read that result in step_B, the barrier guarantees thread 0 has finished writing before thread 3 tries to read it.

**Critical rule:** `__syncthreads()` ONLY synchronizes threads WITHIN the same block. It does NOT work across different blocks. Block 0 and Block 1 cannot use `__syncthreads()` to coordinate.

### How do you synchronize ACROSS blocks then?

Two options:

**Option A: Launch a second kernel.** When a kernel finishes, ALL blocks are guaranteed complete. So launch kernel 1, let it finish, then launch kernel 2 that reads kernel 1's results. The kernel boundary is the synchronization point.

**Option B: Use atomic operations.** An atomic operation is a special hardware instruction that lets multiple threads safely write to the SAME global memory location without corrupting the data. For example, `atomicAdd(&total, value)` means "add a value to the total, and the hardware guarantees that no two threads corrupt each other even if they do it at the same time."

```
Example — each block adds its partial sum to a global total:
  Thread 0 of Block 0:  atomicAdd(&global_total, block_0_sum)
  Thread 0 of Block 1:  atomicAdd(&global_total, block_1_sum)
  Thread 0 of Block 2:  atomicAdd(&global_total, block_2_sum)
  ...
  Hardware ensures no corruption, even though all write to &global_total
```

Atomic operations are simpler (one kernel instead of two), but can be slow if thousands of threads hit the same address.

---

## 48 Warps But Only 4 Warp Schedulers — What Does "Concurrent" Mean?

The RTX 3060 has 48 max warps per SM and 4 warp schedulers per SM. This means 48 × 32 = 1536 threads concurrently per SM. But if only 4 warps execute per cycle, what does "concurrently" actually mean?

**4 warp schedulers = at any SINGLE clock cycle, only 4 warps execute instructions.** One warp per scheduler.

**The other 44 warps are WAITING:**

- Waiting for data from global memory (takes 200-400 clock cycles!)
- Waiting at a `__syncthreads()` barrier
- Waiting for a computation result
- Waiting for their turn

### This is called LATENCY HIDING

```
Clock cycle 1:
  Scheduler 0 → Warp 0: ADD instruction
  Scheduler 1 → Warp 1: ADD instruction
  Scheduler 2 → Warp 2: LOAD from global memory (will take 300 cycles!)
  Scheduler 3 → Warp 3: MUL instruction

Clock cycle 2:
  Warp 2 is STALLED (waiting for memory)
  → Scheduler instantly switches to Warp 4 (ready!)
  The switch costs ZERO cycles. No penalty.

...300 cycles later...
  Warp 2's data arrives → Warp 2 is "ready" again
  Next free scheduler picks it up and continues
```

**"Concurrent" means:** all 48 warps are loaded and managed by the SM. At any instant, 4 execute and 44 wait. But the scheduler switches SO FAST (every clock cycle) that all 48 make progress over time. It does NOT mean all 48 execute at the same clock cycle.

### 28 SMs × 1536 threads = 43,008 "concurrent" threads

```
Resident (concurrent):    28 × 1536 = 43,008 threads  (loaded, managed, ready)
Actually executing:       28 × 4 × 32 = 3,584 threads (at any one clock cycle)
```

If you launch MORE than 43,008 threads worth of blocks, the extra blocks wait in a hardware queue until running blocks finish.

---

## Tensor Cores vs CUDA Cores for Matmul

A common question arises: if we have 128 CUDA cores but only 4 tensor cores per SM, aren't 4 tensor cores too little for huge matrix multiplications?

**A tensor core is NOT like a CUDA core.** Don't think of "4 tensor cores" as "4 tiny things."

**One CUDA core:** does 1 multiply-add per clock cycle. `a*b + c` = 1 operation.

**One tensor core:** does a 4×4 matrix multiply-accumulate in ONE clock cycle. That's 64 multiply-add operations in a single cycle.

```
Per clock cycle:
  128 CUDA cores:   128 × 1 = 128 FMA operations
  4 Tensor cores:   4 × 64  = 256 FMA operations  (at FP16)

  Tensor cores are 2× faster minimum for matrix ops.
```

Tensor cores also work on lower precision (FP16, BF16, INT8, TF32) which further multiplies throughput. So 4 tensor cores is NOT a bottleneck — each one is a dense matrix engine. That's why deep learning frameworks route all matmul through tensor cores automatically.

---

## Complete Memory Hierarchy — Data Flow

```
                    SIZE            SPEED          WHO ACCESSES IT
                    ────            ─────          ────────────────
  Registers         256 KB total    Fastest        One thread only
       ↑
  L1 / Shared Mem   128 KB per SM   Very fast      Same SM (L1=auto, shared=same block)
       ↑
  L2 Cache          2.25 MB total   Fast           ALL SMs
       ↑
  VRAM (Global)     12 GB           Slow           ALL SMs
```

**Path A — Normal global memory read (automatic):**

```
Thread does: float x = array[i];
Hardware does: Register ← L1 ← L2 ← VRAM (caches along the way)
```

**Path B — Explicit shared memory (programmer-controlled):**

```
1. shared_mem[tid] = array[i]     // load from global → shared memory
2. __syncthreads()                 // wait for all threads to load
3. float y = shared_mem[other_tid] // read from shared memory (super fast)
```

---

## Full RTX 3060 Specifications Table

| Property | Value | Meaning |
|---|---|---|
| **VRAM (Global Memory)** | 12 GB | Main GPU memory, slow but large |
| **L2 Cache** | 2.25 MB (2304 KB) | Shared cache across all SMs |
| **SRAM per SM** | 128 KB | Split between L1 cache and shared memory |
| **Shared Mem per SM** | 100 KB max | Total shared memory budget for all blocks on SM |
| **Shared Mem per Block** | 48 KB default | Max a single block can use by default |
| **SMs** | 28 | Independent processing units |
| **CUDA Cores per SM** | 128 | Simple arithmetic units (3584 total) |
| **Tensor Cores per SM** | 4 | Matrix multiply engines (64 ops/cycle each) |
| **Max Threads per Block** | 1024 | Hard limit on one block |
| **Max Threads per SM** | 1536 | Total across all blocks on one SM |
| **Max Blocks per SM** | 16 | Max blocks simultaneously on one SM |
| **Max Blocks per Grid (x)** | 2^31 - 1 | Max blocks in one kernel launch |
| **Max Warps per SM** | 48 | Resident warps (1536/32) |
| **Warp Schedulers per SM** | 4 | Actually executing per cycle |
| **Warp Size** | 32 | Threads per warp (fixed) |
| **Registers per SM** | 65536 | Shared among all threads on SM |
| **Compute Capability** | 8.6 | Ampere architecture |

---

## How to Find These Specifications on Your Own System

A common question is: what are the commands to find L2 cache, L1 cache, shared memory sizes on a GPU?

**For VRAM:** `nvidia-smi` is enough — it shows total GPU memory.

**For L2 cache, shared memory, thread limits, and everything else:** There's no simple `nvidia-smi` command. You need to write a CUDA program using `cudaGetDeviceProperties()`. That's what [`device_query.cu`](device_query.cu) in the cuda_proficiency root does. The key fields are:

```c
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

prop.totalGlobalMem              // VRAM size in bytes
prop.l2CacheSize                 // L2 cache size in bytes
prop.sharedMemPerBlock           // Max shared memory per block
prop.sharedMemPerMultiprocessor   // Max shared memory per SM
prop.maxThreadsPerBlock           // Max threads per block (1024)
prop.maxThreadsPerMultiProcessor  // Max threads per SM (1536)
prop.maxBlocksPerMultiProcessor   // Max blocks per SM (16)
prop.multiProcessorCount          // Number of SMs (28)
prop.warpSize                     // Warp size (32)
prop.regsPerMultiprocessor        // Registers per SM
prop.maxGridSize[0]               // Max grid size x dimension
```

**For L1 cache size:** There is no direct `cudaGetDeviceProperties` field for L1 cache size. L1 shares the 128 KB SRAM with shared memory, so L1 size = 128 KB minus however much shared memory your kernel uses. You know the total SRAM (128 KB for Ampere) from the architecture documentation.

Compile and run: `nvcc device_query.cu -o device_query && ./device_query`
