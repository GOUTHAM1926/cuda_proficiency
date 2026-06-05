# cp.async data-path investigation — RTX 3060 (Ampere, CC 8.6)

Goal: verify, against NVIDIA docs AND on real hardware, whether `cp.async` copies
global→shared memory **without staging through registers**.

## Verdict
The colleague is **correct on the main point**: with `cp.async`, registers are
bypassed. There is one nuance the diagram got slightly wrong about **L1**.

| Claim | Verdict |
|---|---|
| Without cp.async: VRAM→L2→L1→**Register**→Shared (LDG then STS) | ✅ TRUE |
| With cp.async: registers are bypassed | ✅ TRUE (proven 3 ways) |
| With cp.async: VRAM→L2→**Shared** (L1 also skipped) | ⚠️ only for 16-byte / `.cg`. Small (4/8B) async copies still go through L1 |
| Math phase: Shared→Register→ALU (can't compute straight from shared) | ✅ TRUE |

## Proof 1 — instruction-level (SASS, `cuobjdump -sass 01.o`)
```
copy_sync      : LDG.E.CONSTANT R2, [R2.64]   ; global -> register R2
                 STS [R9.X4], R2              ; register R2 -> shared   (2 instr, reg in the middle)

copy_async4    : LDGSTS.E         [R7], [R2.64]   ; 4B  global -> shared, NO data reg (L1 ACCESS)
copy_async16   : LDGSTS.E.BYPASS.128 [R7], [R2.64]; 16B global -> shared, NO data reg, L1 BYPASS
```
`cp.async` becomes a single **LDGSTS** instruction with no destination data register.
The 16-byte form literally carries `.BYPASS` (skips L1); the 4-byte form does not.

## Proof 2 — profiler instruction counts (`ncu`, pure-copy kernel, identical work)
| metric | sync | async |
|---|---|---|
| global loads (LDG) | 33,554,432 | **0** |
| shared stores (STS) | 33,554,432 | **0** |
| LDGSTS | 0 | **33,554,432** |
| registers/thread | 40 | 38 |

Sync: every element = 1 LDG + 1 STS (staged in a register). Async: 0 LDG/STS, all LDGSTS.

## Proof 3 — timings (`./bench`). async is NOT automatically faster.
```
TEST 1 pure copy (no math):           sync 13.4 ms / 322 GB/s   async 15.0 ms / 287 GB/s   x0.89
TEST 2 load+compute, HIGH occupancy:  sync 14.3 ms              async 15.6 ms              x0.92
TEST 3 load+compute, LOW occupancy:   sync  1.21 ms             async  1.11 ms             x1.09
```
- When **bandwidth-bound with full occupancy**, other warps already hide load latency
  (TLP), so async's overhead makes it slightly slower.
- When **latency-bound (few warps / low occupancy)**, async double-buffering overlaps the
  next tile's copy with current compute and wins.
- The durable wins of cp.async are: **freed registers → higher occupancy** and
  **overlap of copy with compute**, not a faster raw copy.

## Proof 4 — BIG real workload: 4096x4096 tiled SGEMM (`./gemm`)
```
sync  tiled GEMM :  144.5 ms    951 GFLOP/s
async tiled GEMM :  132.7 ms   1036 GFLOP/s   -> x1.09, results bit-identical (maxdiff 0)
```
SASS of the GEMM kernels: `gemm_sync` contains only `LDG.E.CONSTANT` + `STS`;
`gemm_async` contains `LDGSTS.E` (the double-buffered tile prefetches). Double-buffering
costs 2x shared mem (16384 vs 8192 bytes/block) but overlaps the next tile's copy with the
current tile's multiply -> ~9% faster on a 137 GFLOP job, exactly correct numerically.

## Doc sources (exact quotes)
- CUDA C++ Programming Guide, Asynchronous Data Copies / Advanced Kernel Programming:
  "Copying 4 or 8 bytes always happens in the so called L1 ACCESS mode, in which case
  data is also cached in the L1, while copying 16-bytes enables the L1 BYPASS mode, in
  which case the L1 is not polluted."
- NVIDIA blog "Controlling Data Movement to Boost Performance on the Ampere Architecture":
  "the thread block no longer stages data through registers" / async memcpy "does not use
  any registers, which means less register pressure and better occupancy."
- PTX ISA, `cp.async`: src is in global state space, dst in shared; `.cg` (L2 only, bypass
  L1) is allowed only when cp-size is 16 bytes.

## How to reproduce
```
nvcc -arch=sm_86 -O3 --ptxas-options=-v -c 01_sass_proof.cu -o 01.o
cuobjdump -sass 01.o | grep -E "Function|LDG|STS|LDGSTS"
nvcc -arch=sm_86 -O3 02_bench.cu -o bench && ./bench
/usr/local/cuda-13.0/bin/ncu --section-folder /usr/local/cuda-13.0/nsight-compute-2025.3.1/sections \
  --metrics sm__sass_inst_executed_op_global_ld.sum,sm__sass_inst_executed_op_shared_st.sum,sm__sass_inst_executed_op_ldgsts.sum,launch__registers_per_thread \
  -k "regex:purecopy" ./bench
```
