# Historical and technical overview of why the computing industry shifted from single-core processors to parallel processors (multicore CPUs and GPUs)  

# The Transition from Sequential to Parallel Computing  

1. The "Virtuous Cycle" (1980s - 2003)- How it worked :
For two decades, software ran faster simply because hardware companies (Intel, AMD) made faster chips with higher clock speeds and during the two decades of growth,these single-CPU microprocessors brought GFLOPS, or giga(10^9) floating-point operations per second, to the desktop and TFLOPS, or tera(10^12) floating-point operations per second, to data centers.
---> The Result: Developers didn't have to change their code; it just "naturally" got faster every year. This fueled better user interfaces and more complex features.

2. The "Wall" in 2003 (Energy & Heat)- The Problem : Around 2003, the drive for faster clock speeds hit a physical limit. Increasing speed further caused energy consumption and heat dissipation issues (the chips would literally melt).

---> The End of the Era: Sequential (one-step-at-a-time) instructions could no longer be made significantly faster.

1. The Solution-(Multiple Cores)- The Shift: Instead of making one "super-fast" CPU, manufacturers started putting multiple "processor cores" on a single chip.

---> The Catch: To benefit from this, an application must be divided into multiple instruction sequences (threads) that run simultaneously.

1. The "Concurrency Revolution- "The New Reality : Sequential programs now only run on one of the many cores, meaning they don't get faster with new hardware.
---> The Need: Significant performance gains are now only possible through parallel programs, where multiple threads cooperate to finish a task. This has moved parallel programming from a "niche" skill for elite scientists to a "must-have" skill for all developers.

# In-Depth Simple Visualization

Consider the analogy of a Kitchen:

---> Before 2003: A single chef worked faster and faster every year. Eventually, speed increases reached physical limits (excessive heat dissipation).
---> After 2003: Increasing the speed of a single chef was abandoned. Instead, four chefs were assigned to the kitchen. The Challenge: If a recipe (the software code) is written for only one person to follow sequentially, the remaining three chefs remain completely idle. The recipe must be restructured so all four chefs work in parallel simultaneously.
