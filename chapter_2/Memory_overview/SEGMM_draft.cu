#include<cuda_runtime.h>
#include<stdio.h>
__global__ void sgemm_gpu_kernel(float *a, float *b, int N, int M, int K)
{
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(row < N && col < M)
	{
		float sum = 0;

	}
	return;
}

void f_random_initialize(float *p, int size)
{
	for(int i =0; i < size; i++)
	{
		*(p + i) = rand()%10;
	}
}



int main() 
{
    float *a, *b, *c;
    float *a_d, *b_d, *c_d;

    int N = 1024, M = 1024, K = 1024;

    // Host allocations
    a = (float*) malloc(N * K * sizeof(float));
    b = (float*) malloc(K * M * sizeof(float));
    c = (float*) malloc(N * M * sizeof(float));

    // Device allocations
    cudaMalloc((void**)&a_d, N * K * sizeof(float));
    cudaMalloc((void**)&b_d, K * M * sizeof(float));
    cudaMalloc((void**)&c_d, N * M * sizeof(float));

    f_random_initialize(a, N * K);    
    f_random_initialize(b, K * M);    
    f_random_initialize(c, N * M);    
	
    cudaMemcpy()

    // Cleanup
    free(a); free(b); free(c);
    cudaFree(a_d); cudaFree(b_d); cudaFree(c_d);

    return 0;
}
