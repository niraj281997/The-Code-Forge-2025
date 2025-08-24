#include<stdio.h>
#include<cuda_runtime.h>
typedef struct Coefficients_AOS {
  int r;
  int b;
  int g;
  int hue;
  int saturation;
  int maxVal;
  int minVal;
  int finalVal; 
} Node;

__global__ void f_aos(Node * data, int N)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i<N)
	{
		int grayscale = (data[i].r + data[i].g + data[i].b)/data[i].maxVal;
  		int hue_sat = data[i].hue * data[i].saturation / data[i].minVal;
		 data[i].finalVal = grayscale*hue_sat; 
	}
}

int main()
{	
	int N =10;
	Node* d_ptr;
	cudaMalloc((void**) &d_ptr ,N *sizeof(Node));
	f_aos<<<1,N>>>(d_ptr,N);
	cudaFree(d_ptr);
	return 0;
}
