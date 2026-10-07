# 📝 Memory Note for Next CUDA Session (Continue from Oct 7, 2026)

> **Last Session:** 2026-10-07 (Chat #19)  
> **Conversation ID:** `af0ddda6-31e4-4b38-9794-be45ba0632f0`  
> **File completed:** `PMPP/.../2.Heterogeneous_data_parallel_computing/2.6_Calling_kernel_functions.md`

---

## Where We Left Off

**Book:** PMPP (Programming Massively Parallel Processors) — Hwu/Kirk/El Hajj  
**Section:** Finished §2.6 — Calling Kernel Functions. Ready for the next section.  
**What we documented in §2.6:**
- `<<<...>>>` kernel launch syntax (execution configuration parameters)
- Ceiling division deep-dive: `ceil(n/256.0)` vs integer division trap (`n/256`), the `(n+255)/256` integer trick
- Complete `vecAdd` host code (Figure 2.13) with line-by-line walkthrough
- Why blocks can execute in any arbitrary order (block independence, no inter-block sync)
- Transparent scalability — same code, different speeds on different GPUs, with real GPU comparisons (GTX 1650 / RTX 3060 / RTX 4090 / A100)
- ALL block size factors pulled from Chapters 4 & 5: warp alignment (multiple of 32), occupancy (thread slots, block slots), register pressure ("performance cliffs"), shared memory limits, algorithm-specific needs
- The overhead warning: vector addition is too simple for GPU speedup due to low compute-to-transfer ratio
- Figures 2.12 and 2.13 screenshots saved

## What's Next

**Check the book for what section comes after §2.6.** The user will provide the textbook screenshots for the next section (likely §2.7 or the start of Chapter 3). Continue documenting in the same in-depth style.

## Quick Reference

- **Index file:** `all_cuda_chats_index.md` (updated, chat #19 added)
- **All 19 chat IDs are logged** in the index
- **Hardware:** RTX 3060, Ampere CC 8.6, 28 SMs, 12 GiB VRAM
- **Total workspace docs:** ~500+ KB across PMPP Ch.1 (complete), Ch.2 §2.1–§2.6 (in progress)
