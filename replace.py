import re
import sys

def replace_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    start_marker = "## Warp Execution: Warp Divergence and Loop Unrolling"
    end_marker = "## Tensor Cores vs CUDA Cores for Matmul"
    
    new_section = """## Warp Execution: Loop Unrolling and Warp Divergence

To truly understand how GPU hardware optimizes code, we must look at how the compiler and the SM execute instructions across threads. Two of the most critical concepts are **Loop Unrolling** and **Warp Divergence**.

### What is Loop Unrolling?

**Loop Unrolling** is an extremely powerful optimization technique performed entirely by the **compiler at compile-time** (before the code ever runs on the GPU). 

A normal loop is computationally expensive because of hidden "bookkeeping" overhead. Every single time a loop iterates, the hardware must execute several invisible instructions:
1. Update the loop counter (e.g., `d--`).
2. Check the condition to see if the loop should continue (e.g., `d >= 0`).
3. Execute a **branch instruction** to jump the program counter back to the top of the loop.

These instructions do no useful math — they just keep the loop machinery turning. 

**How Unrolling Works:**
If you write a loop where the number of passes is **constant and known in advance** (e.g., decoding a tensor that always has exactly 3 dimensions), the compiler recognizes this at compile-time. It completely deletes the loop machinery and writes the body of the loop out in a straight line, repeating it the exact number of times needed.

```cpp
// 1. You write a fixed loop:
for (int d = dims - 1; d >= 0; d--) {
    coord[d] = linear_idx % sizes[d];
    linear_idx = linear_idx / sizes[d];
}

// 2. The compiler deletes the loop and "unrolls" it into this:
coord[2] = linear_idx % sizes[2]; linear_idx = linear_idx / sizes[2];
coord[1] = linear_idx % sizes[1]; linear_idx = linear_idx / sizes[1];
coord[0] = linear_idx % sizes[0]; linear_idx = linear_idx / sizes[0];
```

**Why is it an optimization?**
By removing the counter, the condition check, and the branch jumps, 100% of the clock cycles are now spent doing actual math. Straight-line code flows perfectly through the hardware without stalling, allowing for maximum execution speed.

### What is Warp Divergence?

To understand divergence, you must first know what a **Warp** is. On the GPU, threads do not execute independently. The hardware physically groups threads into bundles of 32 called **warps**. All 32 threads in a warp run in **lockstep** — they share a single Program Counter and are forced to execute the *exact same instruction at the exact same time*.

**Warp divergence** occurs when the 32 threads in a warp are forced onto different execution paths. 

Imagine you write a `while` loop that stops processing when a specific condition is met, and that condition depends on the specific data each thread is handling (a *data-dependent loop*):

```cpp
// A data-dependent loop
while (data_value != 0) {
    // do work
    data_value = data_value / 2;
}
```

If Thread 0 needs 5 passes through the loop, but Thread 1 only needs 1 pass, what does the hardware do? 
Since they are bound together in the same warp, they *cannot* separate. The hardware **serializes** the execution. It forces the warp to keep looping until the *slowest* thread (Thread 0) is completely finished. During passes 2, 3, 4, and 5, Thread 1 is "masked off" — it just sits idle, wasting clock cycles. 

This is warp divergence: 32 threads that should have worked perfectly in parallel end up taking turns or waiting on each other, drastically reducing your hardware efficiency.

### The Connection (The Link)

The relationship between these two concepts is fundamental: **You cannot have one without the other.** 

1. **If you have Warp Divergence, Loop Unrolling is Impossible:**
If you write a loop whose pass count depends on thread-specific data, the compiler *cannot* unroll it at compile-time because it doesn't know how many times it needs to repeat the code. Because the loop bounds are variable, the hardware must evaluate them at runtime, leading directly to warp divergence.

2. **If you fix the bounds, you kill Divergence and unlock Unrolling:**
By designing your algorithms to use fixed, constant loop bounds, you guarantee that every single thread in the warp runs the loop the exact same number of times. This completely eliminates warp divergence (no thread ever waits for another). Because the loop is fixed, the compiler gains the mathematical certainty it needs to flatten your code into hyper-fast, unrolled instructions. 

In short: **Constant loop bounds are the key.** They keep your warps synchronized (killing divergence) while simultaneously allowing the compiler to rip out the overhead (enabling loop unrolling).

---

"""
    
    # We find the section and replace it.
    idx_start = content.find(start_marker)
    idx_end = content.find(end_marker)
    
    if idx_start != -1 and idx_end != -1:
        new_content = content[:idx_start] + new_section + content[idx_end:]
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Updated {filepath}")
    else:
        print(f"Markers not found in {filepath}")

replace_in_file('/home/blu-bridge016/cuda_proficiency/hardware_clarity.md')
replace_in_file('/home/blu-bridge016/cuda_proficiency/CUDA_for_Engineers/Appendix_A_Hardware_setup_Appendix_A/start_here.md')

