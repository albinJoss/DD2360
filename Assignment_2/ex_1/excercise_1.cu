#include <stdio.h>
#define N 256
#define TPB 256

__global__ void hello_World()
{
    printf("Hello world! My threadId is %d\n", threadIdx.x);        //Print the threadId
}

int main()
{
    hello_World<<<N/TPB,TPB>>>();   //Launch the kernel
    cudaDeviceSynchronize();        //Wait for all threads to return
    return 0;
}