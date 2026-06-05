import os

def update_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    start_str = "**Let's Look at a Direct Example Connecting Both:**"
    
    # We will slice from start_str onwards, assuming the example is the last part of this section.
    # Actually, in start_here.md there is "---" and "## Tensor Cores vs CUDA Cores for Matmul" afterwards.
    
    start_idx = content.find(start_str)
    if start_idx == -1:
        print(f"Start string not found in {filepath}")
        return
        
    # Find the end of the example. In my previous append, the example ended with "...unlocks the compiler optimization (unrolling).**"
    end_str = "unlocks the compiler optimization (unrolling).**"
    end_idx = content.find(end_str, start_idx)
    
    if end_idx == -1:
        print(f"End string not found in {filepath}")
        return
        
    end_idx += len(end_str)
    
    new_example = """**Let's Look at a Direct Example Connecting Both:**

*(For a deeper mathematical dive into this specific problem, see Section **"4. Mathematical Proofs & Loop Edge-Cases"** in `data_accessing_techniques.md`).*

**The Context (The Problem):**
Imagine we have a 3D tensor of shape `[2, 3, 4]`. We want to convert a flat 1D `linear_idx` (like 14 or 2) into 3D coordinates `(depth, row, col)`. To do this, we repeatedly divide by the dimension size and take the remainder. 

There are two ways to write the loop that does this:

**1. The "Bad" Way (Stopping when `linear_idx` becomes 0):**
```cpp
while (linear_idx != 0) {
    // Process one dimension
    coord = linear_idx % size;
    linear_idx = linear_idx / size;
}
```
* **Why it seems logical at first:** If our `linear_idx` becomes 0 after peeling off the innermost dimensions, why keep doing math? It seems faster to just stop early.
* **The Reality:** 
  * **Thread A** is given `linear_idx = 14`. It takes **3 passes** for the number to reach 0.
  * **Thread B** is given `linear_idx = 2`. It takes **1 pass** for the number to reach 0.
* **The Result on Divergence:** Because Thread B finishes in 1 pass while Thread A takes 3, Thread B is forced to sit completely idle (wasting cycles) while Thread A finishes passes 2 and 3. The warp has **diverged**.
* **The Result on Unrolling:** The compiler looks at this code and says, "I don't know how many times this loop will run—it depends entirely on what `linear_idx` the thread is given at runtime." Therefore, it **cannot** unroll the code. All loop bookkeeping overhead remains.

**2. The "Good" Way (Stopping when all dimensions are completed):**
```cpp
// For a 3D tensor, dims = 3.
for (int d = dims - 1; d >= 0; d--) {
    // Process one dimension
    coord[d] = linear_idx % sizes[d];
    linear_idx = linear_idx / sizes[d];
}
```
* **Why it is mathematically correct:** Even if `linear_idx` reaches 0 early (like for Thread B), we *must* continue the loop to ensure the outermost dimensions explicitly get set to `0` (e.g., coordinate `[0, 0, 2]`). If we stop early, those outer coordinates are left uninitialized (garbage memory)!
* **The Result on Divergence:** Because `dims` is always 3 for a 3D tensor, **Thread A** and **Thread B** both run the loop exactly **3 times**, regardless of what their `linear_idx` is. Both threads execute in perfect lockstep. No one finishes early, no one waits. Warp divergence is **completely killed**.
* **The Result on Unrolling:** The compiler looks at this code and says, "Ah, this loop always runs exactly 3 times for every single thread, guaranteed." It safely deletes the `for` loop machinery and writes the math instructions out 3 times in a straight line (unrolling it). 

This is the link: **by forcing the loop to run for all dimensions instead of stopping at 0, we ensure mathematical correctness, kill warp divergence, and unlock loop unrolling all at the same time.**"""

    new_content = content[:start_idx] + new_example + content[end_idx:]
    
    with open(filepath, 'w') as f:
        f.write(new_content)
    print(f"Successfully updated {filepath}")

update_file('/home/blu-bridge016/cuda_proficiency/hardware_clarity.md')
update_file('/home/blu-bridge016/cuda_proficiency/CUDA_for_Engineers/Appendix_A_Hardware_setup_Appendix_A/start_here.md')

