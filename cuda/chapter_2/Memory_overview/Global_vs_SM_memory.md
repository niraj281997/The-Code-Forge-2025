# 🔹 Global Memory
Global memory lives in device DRAM (the big memory chips sitting on the GPU card, e.g., 8 GB, 16 GB, etc.).
It is off-chip, not inside the Streaming Multiprocessor (SM).
Access latency is very high (400–800 cycles), though caches (L2 and optionally L1) can help.
Accessible by all SMs and all threads.

# 🔹 SM (Streaming Multiprocessor) Memory
Each SM has only on-chip memory resources:
Registers (private to threads, fastest).
Shared memory (shared by threads in the same block).
L1/Texture/Constant caches (depending on architecture).
These are tiny compared to global DRAM, but super fast.

# 🧠 So:
Global memory is not part of the SM.
SMs just request data from global memory through the memory hierarchy (L2 → DRAM).
Think of it like this:
SM = CPU core
Registers + shared memory = CPU core’s registers + L1 cache
Global memory = RAM outside the core

# CUDA Memory Hierarchy (Text Representation)

        ┌─────────────────────────────┐
        │        Registers            │   (per thread, fastest, on-chip)
        └─────────────┬───────────────┘
                      │
        ┌─────────────▼───────────────┐
        │       Shared Memory         │   (per block, on-chip)
        └─────────────┬───────────────┘
                      │
        ┌─────────────▼───────────────┐
        │          L1 Cache           │   (per SM, on-chip)
        └─────────────┬───────────────┘
                      │
        ┌─────────────▼───────────────┐
        │          L2 Cache           │   (shared across all SMs)
        └─────────────┬───────────────┘
                      │
        ┌─────────────▼───────────────┐
        │     Global Memory (DRAM)    │   (off-chip, accessible by all SMs)
        └─────────────────────────────┘



# ✅ This makes it crystal clear that:
  Registers / Shared / L1 are inside the SM.
  
  L2 and Global DRAM are outside the SM.
# 🧩 Maxwell Architecture
![Maxwell Architecture](../SupportFile/MaxwellArch.png)
