# Asynchronous Copy (`cp.async`): Loading Global → Shared Memory Without Registers

> **Topic:** How modern NVIDIA GPUs (Ampere architecture and newer) copy data from global
> memory (VRAM) into shared memory **without staging it through the thread's registers**,
> using the `cp.async` mechanism.
>
>
> Experimented on **NVIDIA RTX 3060 (Compute Capability 8.6)**.
>
> Companion documentation to [hardware_clarity.md](hardware_clarity.md). All test code and raw results are in
> [cp_async_experimentation/](cp_async_experimentation/).

---

## 1. Overview

On a GPU, a very common pattern is: *load a block of data from slow global memory (VRAM)
into fast on-chip shared memory, then have the threads compute on it.*

The surprising fact is that, in **traditional CUDA**, this "copy" is not direct. Before the
data can reach shared memory, each thread is forced to first pull it into its own **registers**
and then push it back out — an extra, wasteful stop along the way.

`cp.async` (introduced with NVIDIA's **Ampere** architecture, **Compute Capability 8.0+**)
provides a dedicated hardware path that copies global → shared **directly, bypassing the
registers**, and does so **asynchronously** (in the background) so the thread can keep working.

This document explains the old path, the new path, the layers of the API, the size/alignment
rules, what actually gets bypassed, and proves all of it on real hardware.

---

## 2. Background: GPU threads need registers to do math

One hardware rule underpins everything below:

- The **ALUs (CUDA cores / math units) can only operate on values held in registers.** They
  cannot read directly from shared memory or global memory.
- To compute on any value, a thread must first load it into a register (`LDS` = Load from
  Shared, `LDG` = Load from Global).

So registers are **unavoidable for the math step**. The question `cp.async` answers is about
the **copy step** (global → shared) — *that* step is where registers were being wasted.

---

## 3. The traditional path (without `cp.async`): the extra register visit

Consider the standard CUDA line that stages data into shared memory:

```cuda
__shared__ float tile[256];
tile[threadIdx.x] = global_array[threadIdx.x];   // looks like one direct step
```

It *looks* like `VRAM → Shared Memory`. **It is not.** A thread can only move data by holding
it in its "hands" first — and a thread's hands are its **registers**. The compiler turns that
single line into **two** machine instructions:

1. **`LDG` (Load Global):** `VRAM → L2 cache → L1 cache → Register`
2. **`STS` (Store Shared):** `Register → Shared Memory`

So the real data flow of a traditional copy is:

```
VRAM → L2 → L1 → REGISTER → Shared Memory
```

**Why this is wasteful:**

- The data lands in a private register **only to be immediately pushed back out** to shared
  memory — a pointless round trip.
- It **costs clock cycles** and **burns power**.
- It **consumes registers.** A thread has a limited register budget; using registers just as
  a copy middleman reduces how many threads can run at once (lower **occupancy**).

And later, when the thread actually computes on that data, it makes **another** register
trip (this one is necessary):

```
Shared Memory → REGISTER → Math Cores (ALU)
```

So traditionally the data passes through registers **twice** — once needlessly (the copy),
once necessarily (the math).

---

## 4. The `cp.async` path (Ampere+): direct global → shared

Starting with the **Ampere** architecture — specifically **Compute Capability 8.0 and newer**
— the SM has a dedicated unit that moves data **independently of the threads' registers**.
Conceptually you tell the hardware:

> "Take this chunk of memory from VRAM and place it **directly** into shared memory. Tell me
> when you're done."

**Copy flow with `cp.async`:**

```
VRAM → L2 → (L1) → Shared Memory       ← registers are BYPASSED
```

Two distinct benefits:

1. **Registers are bypassed during the copy.** The data goes straight into the shared-memory
   banks; threads never hold it. Those registers stay free for real math → better occupancy.
2. **The copy is asynchronous (runs in the background).** Instead of stalling while the load
   arrives, the thread *issues* the copy and immediately continues with other useful work.
   When it needs the data, it calls a wait instruction (`cp.async.wait_group` / `wait_all`)
   to confirm the background copy finished.

The math step is unchanged — you still load shared → register → ALU — but now the register is
used **only when the value is genuinely needed for computation**, not as a copy middleman.

---

## 5. The layers: C++ API → PTX → machine instruction

`cp.async` is not a single thing you write; it is a hardware capability exposed at several
levels. They are **layers of the same mechanism**, not alternatives:

```
cuda::memcpy_async(...)          ← high-level, portable C++ API
__pipeline_memcpy_async(...)     ← lower-level intrinsic  (<cuda_pipeline.h>)
        │  compiler lowers to ↓
cp.async.{ca,cg}.shared.global   ← PTX instruction (the virtual ISA)
        │  ptxas assembles to ↓
LDGSTS.E[.BYPASS]                ← the real SASS machine instruction (what runs on the GPU)
```

- **`cuda::memcpy_async`** — the recommended high-level API for in-kernel global→shared
  copies. On Ampere+ it compiles down to `cp.async`/`LDGSTS`. On older GPUs it falls back to
  the traditional `LDG`+`STS` path but keeps the async programming model (so the code stays
  portable).
- **`cp.async`** — the PTX (assembly-level) instruction.
- **`LDGSTS`** — short for **L**oa**D** **G**lobal **ST**ore **S**hared: the single native
  machine instruction that performs the direct copy.

> **Do not confuse this with `cudaMemcpyAsync`.** `cudaMemcpyAsync` is a *host-side* function
> that copies data **host ↔ device** over PCIe on a stream. It has nothing to do with shared
> memory or registers. `cuda::memcpy_async` (note the `::`) is the *device-side*,
> inside-the-kernel global→shared copy described here.

---

## 6. Copy size and alignment

`cp.async` / `LDGSTS` only operates on copy chunks of **4, 8, or 16 bytes** (= 32, 64, or 128
bits). These are the only legal transfer sizes per instruction.

### What "size" means

The size is **how many bytes are moved by one copy instruction**. For example, one float is 4
bytes, a `float2` (or `double`) is 8 bytes, and a `float4` (or four packed floats) is 16
bytes. You can copy a contiguous chunk in one shot — e.g. 16 bytes = four floats together.

### What "alignment" means

**Alignment refers to the memory address, not the datatype.** A copy of size *N* bytes must
start at an address that is a **multiple of *N***:

- A **16-byte** copy must start at an address divisible by 16 (**16-byte aligned**).
- An **8-byte** copy must start at an address divisible by 8.
- A **4-byte** copy must start at an address divisible by 4.

Think of memory as a long shelf divided into 16-byte slots. A 16-byte copy must begin exactly
at the start of a slot — you cannot start in the middle and straddle two slots. Aligned memory
lets the hardware grab the whole chunk in a single clean transaction.

Two points worth clarifying:

- **Moving data collectively:** one instruction moves a *contiguous block* (e.g. 4 floats as
  16 bytes), not one tiny value at a time.
- **Relationship to datatypes:** indirect. The datatype determines how many bytes are moved
  and whether the address is naturally aligned (a `float4` is naturally 16-byte aligned), but
  the rule itself is about **byte counts and addresses**, not the type name.

### Small types (1-byte `char`, 2-byte `half`)

You cannot issue a `cp.async` for a single 1- or 2-byte value — the minimum is 4 bytes. You
**pack** several small values into a 4/8/16-byte chunk and copy the chunk (e.g. 8 `half`s = 16
bytes → one `LDGSTS.E.BYPASS.128`). A lone small value that cannot be batched falls back to
the traditional `LDG`+`STS`.

---

## 7. L1 cache behavior: ACCESS vs BYPASS

The **size also decides whether the L1 cache is used** during the async copy:

| Copy chunk | Registers? | L1 cache? | Mode | SASS instruction | Data path |
|---|---|---|---|---|---|
| **4 bytes** | bypassed | **used** | L1 ACCESS | `LDGSTS.E` | VRAM → L2 → L1 → Shared |
| **8 bytes** | bypassed | **used** | L1 ACCESS | `LDGSTS.E` | VRAM → L2 → L1 → Shared |
| **16 bytes** | bypassed | **skipped** | L1 BYPASS | `LDGSTS.E.BYPASS.128` | VRAM → L2 → Shared |

In **L1 BYPASS** mode (16-byte copies), the data goes straight from L2 into shared memory
without polluting L1 — useful because that data is being parked in shared memory anyway, so
caching it in L1 would just evict other useful data.

**Crucial point:** registers are bypassed for **all three sizes**. Only the **L1** behavior
changes with size.

---

## 8. What is the main goal — bypassing registers or L1?

**The primary goal of `cp.async` is to bypass the registers** (and to run the copy
asynchronously). That is what:

- frees up registers for real computation (raising occupancy), and
- removes the wasteful copy-time register round trip, and
- lets the copy overlap with compute.

**Bypassing L1 is a secondary, optional, size-dependent bonus** — it only happens for 16-byte
copies, and its purpose is to avoid polluting L1 with staging data. If you only ever
remembered one sentence:

> **`cp.async` exists to copy global → shared *without using registers*. Skipping L1 is just
> an extra perk available at the 16-byte size.**

---

## 9. Proofs measured on the RTX 3060

Four independent confirmations. Code: [async_copy_investigation/](async_copy_investigation/).

### Proof 1 — The actual machine instructions (SASS)

[`01_sass_proof.cu`](async_copy_investigation/01_sass_proof.cu), via `cuobjdump -sass`:

```
copy_sync   : LDG.E.CONSTANT R2, [R2.64]      ; global → register R2
              STS [R9.X4], R2                  ; register R2 → shared   (data sits in R2!)

copy_async4 : LDGSTS.E            [R7], [R2.64]   ; 4-byte  global → shared, NO data register
copy_async16: LDGSTS.E.BYPASS.128 [R7], [R2.64]   ; 16-byte global → shared, NO register, L1 BYPASSED
```

The traditional copy is two instructions with the data parked in register R2; the async copy
is a single `LDGSTS` with **no destination data register**. The 16-byte form carries `.BYPASS`
(also skips L1).

### Proof 2 — Profiler instruction counts (Nsight Compute, `ncu`)

Identical copy workload, counted by the hardware:

| Metric | sync | async |
|---|---|---|
| global loads (`LDG`) | 33,554,432 | **0** |
| shared stores (`STS`) | 33,554,432 | **0** |
| `LDGSTS` | 0 | **33,554,432** |
| registers / thread | 40 | 38 |

A perfect mirror image — every traditional load+store becomes one `LDGSTS`, with zero LDG/STS.

### Proof 3 — Timings (and an important caveat)

[`02_bench.cu`](async_copy_investigation/02_bench.cu):

```
PURE COPY (no math):           sync 13.4 ms / 322 GB/s   async 15.0 ms / 287 GB/s   ×0.89  (async slower)
LOAD+COMPUTE, high occupancy:  sync 14.3 ms              async 15.6 ms              ×0.92  (async slower)
LOAD+COMPUTE, low occupancy:   sync  1.21 ms             async  1.11 ms             ×1.09  (async wins)
```

`cp.async` is **not automatically faster at raw copying.** When a kernel is bandwidth-bound
with full occupancy, the GPU already hides load latency using other warps, so async's small
overhead can make it slightly slower. The wins come from **freed registers (higher occupancy)**
and **overlapping the copy with compute** — visible here in the low-occupancy case.

### Proof 4 — A large real workload: 4096×4096 tiled matrix multiply

[`03_gemm.cu`](async_copy_investigation/03_gemm.cu) (~137 GFLOP):

```
sync  tiled GEMM :  144.5 ms    951 GFLOP/s
async tiled GEMM :  132.7 ms   1036 GFLOP/s   →  ×1.09 faster, results BIT-IDENTICAL (maxdiff = 0)
```

The async GEMM double-buffers: it streams the **next** tile into shared via `LDGSTS` while
multiplying the **current** tile — overlap that the traditional version cannot do. Same exact
result, ~9% faster.

### Inspecting the data flow yourself

- **Definitive:** `cuobjdump -sass file.o` → look for `LDG`+`STS` vs `LDGSTS`.
- **Counting:** `ncu --metrics sm__sass_inst_executed_op_ldgsts.sum,...op_global_ld.sum,...op_shared_st.sum`
- **Registers/occupancy:** `nvcc --ptxas-options=-v`.

---

## 10. When does `cp.async` actually help?

- **Best:** memory-latency-bound or register-pressured kernels, and **software-pipelined**
  kernels that overlap loading the next tile with computing the current one (e.g. GEMM,
  convolutions, stencils).
- **Little or no benefit (or slightly slower):** simple, already-bandwidth-bound streaming
  copies at full occupancy, where there is no compute to overlap and warps already hide
  latency.
- **Use 16-byte copies** (e.g. `float4`) when possible — fewer instructions and L1 isn't
  polluted with staging data.

---

## 11. Common misconceptions & doubts (FAQ)

These are frequent points of confusion when first learning `cp.async`.

**Q: Doesn't data flow VRAM → Shared Memory → Registers (registers only when doing math)?**
No. In the *traditional* path the copy itself goes `VRAM → L2 → L1 → Register → Shared` — the
register is used even just to copy. The math step then adds a second `Shared → Register → ALU`
trip. `cp.async` removes the *copy-time* register trip.

**Q: Is the NVIDIA "Asynchronous Data Copies" page about `cudaMemcpyAsync`?**
No. That page describes the device-side `cuda::memcpy_async` (in-kernel global→shared), which
lowers to `cp.async`/`LDGSTS`. `cudaMemcpyAsync` is a different, host-side host↔device API.

**Q: Are `cuda::memcpy_async` and `cp.async` two different techniques?**
No — same mechanism, different layers: C++ API → PTX (`cp.async`) → SASS (`LDGSTS`).

**Q: Are the sizes 4/8/16 *bits*?**
No — **bytes** (4/8/16 bytes = 32/64/128 bits).

**Q: What exactly is "byte alignment"? Is it about datatypes, or about sending data together?**
It is about the **memory address**. A copy of *N* bytes must begin at an address that is a
multiple of *N* (e.g. a 16-byte copy starts at an address divisible by 16). It relates to
moving a **contiguous chunk in one transaction** (yes, "data collectively"), and a datatype
matters only because it sets how many bytes you move and whether the address is naturally
aligned (a `float4` is naturally 16-byte aligned). The rule itself is about **byte counts and
addresses**, not the type name.

**Q: Can small datatypes (1-byte `char`, 2-byte `half`) use `cp.async`?**
Not individually — the minimum chunk is 4 bytes. Pack several small values into a 4/8/16-byte
chunk and copy the chunk. A lone unbatched small value falls back to `LDG`+`STS`.

**Q: Do small (4/8-byte) async copies still go through registers?**
No. **Registers are bypassed for all three sizes (4, 8, 16 bytes).** Small copies still use
**L1** (L1 ACCESS mode), but never a register.

**Q: Is the main aim to bypass L1 or to bypass registers?**
The main aim is to **bypass registers** (and copy asynchronously). Skipping L1 is a secondary
bonus that only applies to 16-byte copies (L1 BYPASS mode).

**Q: Does `cp.async` always make a program faster?**
No. It can be neutral or slightly slower for simple bandwidth-bound copies at full occupancy.
Its real value is register/occupancy savings and overlapping copy with compute.

---

## 12. Summary

- The traditional global→shared copy makes an **extra, wasteful stop in the registers**:
  `VRAM → L2 → L1 → Register → Shared`, with math adding `Shared → Register → ALU`.
- **`cp.async` (Ampere+, hardware `LDGSTS`) copies global → shared without registers:**
  `VRAM → L2 → (L1) → Shared`. Registers are used only when data is genuinely needed for math.
- **Primary purpose: bypass registers + copy asynchronously.** Bypassing L1 is a secondary,
  size-dependent bonus.
- **Sizes: 4 / 8 / 16 bytes**, each requiring matching address alignment. Registers are
  bypassed for all three; only **16-byte** copies also bypass **L1**.
- Verified by official NVIDIA documentation and proven on the **RTX 3060** four ways: SASS
  instructions, profiler counts, timings, and a 4096×4096 GEMM.

> **One-line takeaway:** `cp.async` always skips registers; whether it also skips L1 depends
> on size — only the 16-byte copy skips L1, while 4/8-byte copies still pass through L1.

---

## 13. Sources & links

**Official NVIDIA documentation**

- *Asynchronous Data Copies* — CUDA C++ Programming Guide:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html>
  > "Copying 4 or 8 bytes always happens in the so called L1 ACCESS mode, in which case data
  > is also cached in the L1, while copying 16-bytes enables the L1 BYPASS mode, in which case
  > the L1 is not polluted."
- *PTX ISA — `cp.async`*:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async>
  (src in global state space, dst in shared; `.cg` / L1-bypass allowed only for cp-size 16.)
- NVIDIA blog — *Controlling Data Movement to Boost Performance on the Ampere Architecture*:
  <https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/>
  > "the thread block no longer stages data through registers"; async memcpy "does not use any
  > registers, which means less register pressure and better occupancy."

**Test code & results (this repo)**

- [async_copy_investigation/01_sass_proof.cu](async_copy_investigation/01_sass_proof.cu) — SASS proof (`LDG`+`STS` vs `LDGSTS`)
- [async_copy_investigation/02_bench.cu](async_copy_investigation/02_bench.cu) — pure-copy + load+compute timings
- [async_copy_investigation/03_gemm.cu](async_copy_investigation/03_gemm.cu) — 4096×4096 tiled GEMM, sync vs async
- [async_copy_investigation/FINDINGS.md](async_copy_investigation/FINDINGS.md) — condensed results + reproduce commands

**Hardware:** NVIDIA GeForce RTX 3060 (Ampere, Compute Capability 8.6). **Toolkit:** CUDA 11.5
(`nvcc`); profiled with Nsight Compute.
