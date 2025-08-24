#include<stdio.h>
#include<cuda_runtime.h>

typedef struct node
{
	int *iptr;
	int *fptr;
}Node;

__global__ void f_Structure_Of_Array_SOA(Node *ptr,int N)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N)
       	{
       		 ptr->iptr[tid] = tid;
       		 ptr->fptr[tid] = tid * 2;
    	}
}

__global__ void f_print_SOA(Node* obj, int N)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if(id<N)
	{
		printf("iptr = %d, fptr =  %d\n",obj->iptr[id] , obj->fptr[id]);
	}
	return;

}
void f_init()
{
	int N, *p1,*p2;
	printf("Enter the number of element :");
	scanf("%d",&N);
	Node obj;
	
	cudaMalloc((void**)&p1,N*sizeof(int));
	cudaMalloc((void**)&p2,N*sizeof(int));
	
	obj.iptr = p1;
	obj.fptr = p2;
	
	Node *d_obj;
    	cudaMalloc((void**)&d_obj, sizeof(Node));

    	// copy host struct (with device pointers inside) -> device struct
   	 cudaMemcpy(d_obj, &obj, sizeof(Node), cudaMemcpyHostToDevice);

	f_Structure_Of_Array_SOA<<<1,N>>>(d_obj,N);

	cudaDeviceSynchronize();
	f_print_SOA<<<1,N>>>(d_obj,N);
	cudaDeviceSynchronize();

	cudaFree(p1);
	cudaFree(p2);
	return;
}

int main()
{
	f_init();	

	return 0;
}
