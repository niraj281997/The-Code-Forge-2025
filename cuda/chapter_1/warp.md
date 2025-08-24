## 🧠 CUDA Knowledge Point: Warps & Warp Divergence

### 🔹 What is a Warp?

- A **warp** is a group of **32 threads** that execute the same instruction simultaneously.
- CUDA uses the **SIMT (Single Instruction, Multiple Threads)** model.
- Threads are scheduled and executed **warp-wide**, not individually.

---

### ⚠️ Warp Divergence

| Concept           | Description |
|------------------|-------------|
| **Warp Divergence** | Occurs when threads in the same warp follow **different execution paths** due to control flow (e.g., `if`, `switch`) |
| **Impact**        | Divergent paths are executed **serially**, reducing parallelism and performance |
| **Example**       | ```cpp if (threadIdx.x % 2 == 0) {...} else {...} ``` — even and odd threads diverge |

---

### ✅ Best Practices to Minimize Warp Divergence

- Align control flow so threads in a warp follow the same path.
- Avoid branching based on `threadIdx.x` unless necessary.
- Use **predicated instructions** or **warp-level primitives** when possible.
- Size thread blocks in **multiples of 32** to avoid partially filled warps.

---

### 📌 Warp Summary Table

| Topic             | Key Insight |
|------------------|-------------|
| Warp Size         | 32 threads per warp |
| Execution Model   | SIMT (Single Instruction, Multiple Threads) |
| Divergence Cause  | Conditional branching within a warp |
| Performance Tip   | Keep threads in a warp on the same execution path |

---

> Efficient CUDA code keeps warps coherent. Divergence is like traffic congestion—avoidable with smart planning.
