
# CUDA Registers and Register Spilling

## 1. What Are Registers?
- Registers are the **fastest memory** on a GPU.
- They are **private to each thread** and live **on-chip** inside an SM (Streaming Multiprocessor).
- Used for temporary variables like loop counters or intermediate calculations.
- Access latency: **~1 cycle**.

Example:
```cpp
__global__ void kernel() {
    int x = threadIdx.x;   // lives in a register
    int y = x * 2;         // also in a register
}
```

## 2. Register Allocation
- Each SM has a **fixed number of registers** (e.g., 65,536).
- Registers are divided among all active threads on the SM.

Example:
- 65,536 registers per SM  
- Kernel uses 128 registers per thread  
- Max threads per SM = 65,536 ÷ 128 = **512 threads**  

## 3. What If a Thread Needs More Registers?
- Register usage is decided **at compile time** by the CUDA compiler (PTXAS).
- If the code needs more variables than available registers, the compiler **spills** them.

## 4. Register Spilling → Local Memory
- Spilled registers go into **local memory**.
- Local memory is:
  - Thread-private
  - Stored in **global DRAM** (slow)
- Latency: **hundreds of cycles** (~400–600).

Compiler output example:
```
ptxas info    : Used 128 registers, 64 bytes spill stores, 64 bytes spill loads
```

## 5. Why Not Spill Into Shared Memory?
- Shared memory is on-chip but shared across threads.
- Spilled registers must remain **private to each thread**.
- Therefore, spills always go to local (global-backed) memory.

## 6. Performance Implications
- High register usage → lower occupancy.
- Register spilling → global memory traffic.
- Both hurt performance.

## 7. How to Control Register Usage
- Check usage:
  ```bash
  nvcc -Xptxas -v kernel.cu
  ```
- Limit registers:
  ```bash
  nvcc -maxrregcount=32 kernel.cu
  ```
- Use launch bounds:
  ```cpp
  __launch_bounds__(128, 2)
  ```
- Optimize code to reuse variables.

## 8. Memory Hierarchy Recap
Registers (fastest) → spill → Local memory (global DRAM)  
Shared memory (on-chip, per block)  
Global memory (slow, shared across threads)

## 🔑 Key Takeaways
- Registers are fastest but limited.  
- If registers are exceeded, values spill into local memory (global DRAM).  
- Spilling is **slow** and should be avoided.  
- Balance register usage vs occupancy for performance tuning.
