## 🧠 CUDA Knowledge Point: Warps & Warp Divergence

### 🔹 What is a Warp?

- A **warp** is a group of **32 threads** that execute instructions in lockstep on NVIDIA GPUs.
- CUDA uses the **SIMT (Single Instruction, Multiple Threads)** model, where all threads in a warp follow the same instruction stream.
- Warps are the fundamental unit of execution and scheduling on the GPU.

---

### 🔸 Warp Scheduling & Execution

- Threads are grouped into blocks, and blocks are divided into warps:
  - Example: A block with 128 threads → 128 / 32 = 4 warps
- The GPU schedules warps, not individual threads.
- Warps are executed independently and may be interleaved to hide memory latency.

---
## ⚠️ Warp Divergence: Example & Performance Impact

### 🔸 What is Warp Divergence?

Warp divergence occurs when threads in the same warp follow different execution paths due to conditional branching. This forces the GPU to serialize execution, reducing parallelism and performance.

---

### 🔍 Example of Warp Divergence

```cpp
__global__ void DivergentKernel(int* data) {
    int tid = threadIdx.x;
    if (tid % 2 == 0) {
        data[tid] = tid * 2;  // Even threads
    } else {
        data[tid] = tid * 3;  // Odd threads
    }
}
```
| Concept           | Description |
|------------------|-------------|
| **Warp Divergence** | Occurs when threads in the same warp follow **different control paths** due to branching (`if`, `switch`, etc.) |
| **Impact**        | Divergent paths are executed **serially**, reducing parallelism and throughput |
| **Example**       | ```cpp if (threadIdx.x % 2 == 0) {...} else {...} ``` — even and odd threads diverge |

---

### ✅ Best Practices to Minimize Warp Divergence

- Align control flow so threads in a warp follow the same path.
- Avoid branching based on `threadIdx.x` unless necessary.
- Use **predicated instructions** or **warp-level primitives** like `__shfl_sync`, `__ballot_sync`, etc.
- Size thread blocks in **multiples of 32** to avoid partially filled warps.

---

### 🔧 Advanced Warp-Level Programming

- Use **warp-level primitives** for collective operations like reductions, scans, and broadcasts.
- Examples:
  - `__shfl_down_sync()` for intra-warp reductions
  - `__ballot_sync()` for voting across threads
- These primitives help optimize performance by reducing global memory access and synchronization overhead.

---

### 📌 Warp Summary Table

| Topic             | Key Insight |
|------------------|-------------|
| Warp Size         | 32 threads per warp |
| Execution Model   | SIMT (Single Instruction, Multiple Threads) |
| Divergence Cause  | Conditional branching within a warp |
| Performance Tip   | Keep threads in a warp on the same execution path |
| Warp Primitives   | Use `__shfl_sync`, `__ballot_sync`, etc. for efficient intra-warp communication |

---

> Efficient warp usage is the backbone of high-performance CUDA code. Think of warps as lanes on a highway—keep them flowing in sync to avoid traffic jams.
