#include<cuda_runtime.h>
#include<iostream>
using namespace std;
#include<stdio.h>

class Node
{
	private : 
		int a;
		int b;
	public :
		__device__ void funtion()
		{
			printf("Hello World\n");
			return;
		}
};

__global__ void print_from_gpu()
{

	printf("Thread Id : %d , Block Id : %d\n", threadIdx,blockIdx);
	Node a;
	a.funtion();
}

int main()
{

	
	print_from_gpu<<<10,10>>>();
	cudaDeviceSynchronize();
	return 0;
}
