#include <cuda_runtime.h>
#include <cstdio>      // for printf
#include <iostream>    // optional

__global__ void f_vector_add(const int *a, const int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int size = 1 << 10;               // 1024 elements
    const int threadsPerBlock = 256;
    int numBlocks = size / threadsPerBlock; // 1024 / 256 = 4 → OK for exact division
   const size_t memsize = size * sizeof(int);

    // Host allocations
    int *a_h = new int[size];
    int *b_h = new int[size];
    int *c_h = new int[size];

    // Initialize host data
    for (int i = 0; i < size; ++i) {
        a_h[i] = std::rand() % 10;
        b_h[i] = std::rand() % 10;
    }

    // Device allocations
    int *a_k, *b_k, *c_k;
    cudaMalloc(&a_k, memsize);
    cudaMalloc(&b_k, memsize);
    cudaMalloc(&c_k, memsize);

    // Host -> Device copies (correct direction)
    cudaMemcpy(a_k, a_h, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(b_k, b_h, memsize, cudaMemcpyHostToDevice);

    // Launch
    f_vector_add<<<numBlocks, threadsPerBlock>>>(a_k, b_k, c_k, size);

    // Ensure kernel completed (useful while learning/debugging)
    cudaDeviceSynchronize();

    // Device -> Host copy
    cudaMemcpy(c_h, c_k, memsize, cudaMemcpyDeviceToHost);

    // Print a few results
    for (int i = 0; i < 10; ++i) {
        printf("%d + %d = %d\n", a_h[i], b_h[i], c_h[i]);
    }

    // Cleanup
    cudaFree(a_k); cudaFree(b_k); cudaFree(c_k);
    delete[] a_h; delete[] b_h; delete[] c_h;

    return 0;
}

