import re
import sys

def add_example(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    target_text = """In short: **Constant loop bounds are the key.** They keep your warps synchronized (killing divergence) while simultaneously allowing the compiler to rip out the overhead (enabling loop unrolling)."""
    
    replacement_text = """In short: **Constant loop bounds are the key.** They keep your warps synchronized (killing divergence) while simultaneously allowing the compiler to rip out the overhead (enabling loop unrolling).

**Let's Look at a Direct Example Connecting Both:**

Imagine we are converting a flat 1D linear index into a 3D tensor coordinate.

**The "Bad" Way (Data-Dependent):**
```cpp
while (linear_idx != 0) {
    // Process one dimension
    coord = linear_idx % size;
    linear_idx = linear_idx / size;
}
```
* **Thread A** has `linear_idx = 14`. It takes **3 passes** for the number to reach 0.
* **Thread B** has `linear_idx = 2`. It takes **1 pass** for the number to reach 0.
* **The Result on Divergence:** Thread B finishes early and is forced to sit completely idle (wasting cycles) while Thread A finishes passes 2 and 3. The warp has **diverged**.
* **The Result on Unrolling:** The compiler looks at this code and says, "I don't know how many times this loop will run—it depends on the value of `linear_idx` at runtime." Therefore, it **cannot** unroll the code. The loop overhead remains.

**The "Good" Way (Fixed Bounds):**
```cpp
for (int d = 3 - 1; d >= 0; d--) {
    // Process one dimension
    coord = linear_idx % size;
    linear_idx = linear_idx / size;
}
```
* **Thread A** and **Thread B** both run the loop exactly **3 times**, regardless of their `linear_idx` values.
* **The Result on Divergence:** Both threads execute in perfect lockstep. No one finishes early, no one waits. Warp divergence is **killed**.
* **The Result on Unrolling:** The compiler looks at this code and says, "This loop always runs exactly 3 times, guaranteed." It safely deletes the `for` loop machinery and copies the code inside 3 times (unrolling it). 

This is the link: **writing fixed loops simultaneously solves the hardware problem (divergence) and unlocks the compiler optimization (unrolling).**"""

    if target_text in content:
        new_content = content.replace(target_text, replacement_text)
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Updated {filepath}")
    else:
        print(f"Target text not found in {filepath}")

add_example('/home/blu-bridge016/cuda_proficiency/hardware_clarity.md')
add_example('/home/blu-bridge016/cuda_proficiency/CUDA_for_Engineers/Appendix_A_Hardware_setup_Appendix_A/start_here.md')

