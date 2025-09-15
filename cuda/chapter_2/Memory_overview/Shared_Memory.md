# CUDA Shared Memory — Deep Dive

> A focused guide to what CUDA shared memory is, why you need it, how it works, common pitfalls, performance trade-offs, and practical patterns with working examples.

---

## Table of contents

1. What is CUDA shared memory?
2. Why use shared memory (when it helps)
3. How shared memory works (hardware + software)
4. Declaring and using shared memory (static & dynamic)
5. Bank architecture and bank conflicts — detailed
6. Common patterns and examples (tiled matrix multiply, padding)
7. Synchronization, hazards, and `__syncthreads()`
8. Occupancy, sizing, and trade-offs
9. Profiling and debugging tips
10. Quick checklist / heuristics
11. Example kernels (copy-paste ready)

---

## 1. What is CUDA shared memory?

CUDA shared memory is a region of on-chip memory that threads within the same thread block can read and write. It is much lower latency and higher bandwidth than global memory. Shared memory is allocated per block and disappears when the block finishes. Use it when multiple threads in a block need to collaborate or when you can reuse global data locally to avoid repeated global loads.

## 2. Why use shared memory (when it helps)

* **Data reuse**: bring a tile of global memory into on-chip memory and reuse it across many thread operations.
* **Inter-thread communication**: threads in a block can share intermediate results without writing back to global memory.
* **Latency hiding and throughput**: shared memory is on-chip and significantly faster than global memory; for computations with high reuse it reduces global memory bandwidth pressure.

Use shared memory when the compute-to-memory ratio is high and the data you want to reuse fits in the per-block shared memory budget.

## 3. How shared memory works (hardware + software)

* Shared memory is physically on the GPU chip and is partitioned by Streaming Multiprocessor (SM). Each SM has a limited shared memory capacity.
* When you launch a kernel, each block that runs on an SM gets its own portion of the SM's shared memory.
* The lifetime of shared memory is the lifetime of the block. Two different blocks (even on the same SM) cannot access each other's shared memory.
* Hardware exposes shared memory to threads via many banks (modern NVidia GPUs use 32 banks). Each bank can service a limited width per cycle. Access patterns that map multiple threads to the same bank at conflicting addresses cause serialization (bank conflicts).

## 4. Declaring and using shared memory (static & dynamic)

### Static (compile-time) size

```cpp
__global__ void kernel_static(){
    __shared__ float tile[256]; // size known at compile-time
    int tid = threadIdx.x;
    tile[tid] = some_value;
    __syncthreads();
    // use tile
}
```

### Dynamic (size provided at launch)

```cpp
// kernel
extern __shared__ float dyn[]; // size specified at kernel launch (in bytes)

// launch (in host C++ code)
int sharedBytes = blockDim.x * sizeof(float);
myKernel<<<grid, block, sharedBytes>>>(...);
```

Dynamic shared memory is useful when the tile size depends on runtime inputs.

## 5. Bank architecture and bank conflicts — detailed

* Shared memory is split into banks. The most common configuration for recent NVIDIA GPUs is 32 banks.
* Each bank services contiguous addresses in a round-robin fashion: successive 32-bit words typically go to successive banks.
* If each thread in a warp accesses different banks, the accesses can be serviced in parallel. If multiple threads access different addresses that map to the same bank, those accesses are serialized — this is a bank conflict.
* If all threads access the **same** address within a bank, hardware can perform a broadcast and there is no conflict.

**Key consequences**

* Poorly chosen indexing or strides (for example, accessing a column of a matrix stored in row-major order) frequently leads to bank conflicts.
* Multi-byte types and 64-bit/128-bit access patterns change how banks are addressed — on some GPUs there are two addressing modes (4-byte or 8-byte bank size) that affect conflict behavior.

**How to avoid bank conflicts**

* **Padding**: add an extra column in a 2D shared-memory tile. A common trick: `__shared__ float tile[TILE][TILE+1];` The `+1` breaks the stride that would map threads to the same bank.
* **Reorder computations**: use warp-friendly indexing so that consecutive threads access consecutive addresses.
* **Use broadcasts**: when all threads read the same address, prefer reading once and broadcasting.

## 6. Common patterns and examples

### Tiled matrix multiplication (classic use-case)

You load tiles of A and B into shared memory, synchronize, compute partial products, and accumulate. This reduces global memory traffic dramatically when each tile is reused many times.

High-level pattern:

1. Each block computes a TILE x TILE sub-block of output C.
2. For each phase `p`: load A tile and B tile from global memory into shared memory.
3. `__syncthreads()`
4. Multiply-accumulate across the tile and repeat.
5. Write final result to global memory.

Padding the shared-memory tiles with an extra column (`TILE+1`) is a cheap way to prevent bank conflicts when threads read across rows/columns.

## 7. Synchronization, hazards, and `__syncthreads()`

* `__syncthreads()` is a barrier for all threads in a block — every thread must reach each `__syncthreads()` call, otherwise behavior is undefined and deadlocks can occur.
* Avoid placing `__syncthreads()` inside conditionals where only some threads take the branch. If conditional sync is necessary, ensure that the condition is uniform across the entire block (e.g., based on `blockIdx`), or restructure the algorithm.
* Do not assume `__syncthreads()` synchronizes across blocks — it only synchronizes threads inside the same block.

## 8. Occupancy, sizing, and trade-offs

* Shared memory is a limited resource per SM. If each block uses a large chunk of shared memory, fewer blocks can run concurrently on an SM which can reduce occupancy (active warps) and hurt latency hiding.
* Optimizing workload requires balancing:

  * Block size (threads per block)
  * Shared memory per block
  * Register usage per thread

Sometimes using slightly less shared memory to increase the number of resident blocks/warps gives better performance than fully maximizing data reuse.

## 9. Profiling and debugging tips

* Use Nsight Compute and Nsight Systems to inspect shared memory usage and bank conflict metrics.
* Look for indicators in profiling results: warp stalls caused by shared memory bank conflicts or MIO stalls.
* Micro-benchmark simple access patterns to learn how a particular GPU architecture behaves.

## 10. Quick checklist / heuristics

* Use shared memory when:

  * You have reuse of data across threads in a block.
  * The data fits comfortably in the per-block shared memory budget.
* Avoid shared memory when:

  * Data cannot be reused enough to justify copying it from global memory.
  * You need cross-block communication.
* When using shared memory:

  * Align and pad to avoid bank conflicts.
  * Keep `__syncthreads()` usage uniform and safe.
  * Profile — guesses are cheap, measurements are essential.

## 11. Example kernels (ready to copy)

### 1) Small example: dynamic shared memory

```cpp
// Host launch
int B = 128;
int sharedBytes = B * sizeof(float);
myKernel<<<grid, B, sharedBytes>>>(...);

// Device
__global__ void myKernel(float *data){
    extern __shared__ float s[]; // dynamic shared memory
    int tid = threadIdx.x;
    s[tid] = data[blockIdx.x * blockDim.x + tid];
    __syncthreads();
    // use s[]
}
```

### 2) Tiled matrix multiply (with padding to reduce bank conflicts)

```cpp
#define TILE 16

__global__ void matMulShared(const float *A, const float *B, float *C, int N){
    __shared__ float sA[TILE][TILE+1]; // +1 breaks the stride that can cause bank conflicts
    __shared__ float sB[TILE][TILE+1];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float sum = 0.0f;
    int numPhases = (N + TILE - 1) / TILE;

    for (int ph = 0; ph < numPhases; ++ph){
        int aCol = ph * TILE + tx;
        int bRow = ph * TILE + ty;

        if (row < N && aCol < N) sA[ty][tx] = A[row * N + aCol]; else sA[ty][tx] = 0.0f;
        if (bRow < N && col < N) sB[ty][tx] = B[bRow * N + col]; else sB[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k){
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = sum;
}
```

---

### Final notes

* Shared memory is a powerful optimization tool. Use it when data reuse and inter-thread cooperation give a measurable benefit. Avoid it when it complicates code and gives little practical speedup.
* Always profile and iterate; microbenchmarks and Nsight tools are your friends.

---

*End of deep dive.*
