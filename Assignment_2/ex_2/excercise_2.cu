#include <stdio.h>
//#include <sys/time.h>
#include <time.h>
#include <limits.h>

#define TPB 256

void memsetF(float *x, float *y, float *d_x, float *d_y, int array_size)           //Sets the arrays using memcpy to speed up the execution
{
    for(int i = 0; i < array_size; ++i)
    {
        x[i] = 2.0f;
        y[i] = 8.0f;
    }
    cudaMemcpy(d_x, x, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, array_size * sizeof(float), cudaMemcpyHostToDevice);
}

/*double cpuSecond()              //Timing from within the program
{
    struct timeval tp;
    gettimeofday(&tp, null);
    return ((double) tp.tv_sec + (double)tp.tv_usec*1.e-6);
}*/

__global__ 
void saxpy_for_gpu(float *x, float *y, const float a, int array_size)     //Executes SAXPY on the GPU parallel
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < array_size)
    {
        y[i] += a * x[i];
    }
}


void saxpy_for_cpu(float *x, float *y, const float a, int array_size)       //Executes saxpy on the CPU sequentially
{
    for(int i = 0; i < array_size; ++i)
    {
        y[i] += a * x[i];
    }
}

int main(int argc, char *argv[])                          //Driver code
{
    int array_size = argc == 1 ? 10000 : atoi(argv[1]);
    int num_blocks = (array_size + TPB - 1) / TPB;
    float *x, *y, *d_x, *d_y, *cuda_y;
    x = (float*)malloc(array_size * sizeof(float));
    y = (float*)malloc(array_size * sizeof(float));
    cuda_y = (float*)malloc(array_size * sizeof(float));
    cudaMalloc(&d_x, array_size * sizeof(float));
    cudaMalloc(&d_y, array_size * sizeof(float));
    memsetF(x, y, d_x, d_y, array_size);
    const float a = 4.0f;
    float margin = 0.01f;
    double iStart, iElaps;
    int correctness = 1;
    clock_t starting_time;
  
      
        
    //iStart = cpuSecond();
    starting_time = clock();
    saxpy_for_cpu(x, y, a, array_size);
    //iElaps = cpuSecond() - iStart;
    printf("Computing SAXPY with %d elements on the CPU. Done in %lf seconds!\n", array_size, (double) (clock() - starting_time) / CLOCKS_PER_SEC);

    //iStart = cpuSecond();
    starting_time = clock();
    saxpy_for_gpu<<<num_blocks, TPB>>>(d_x, d_y, a, array_size);
    cudaDeviceSynchronize();
    //iElaps = cpuSecond() - iStart;
    printf("Computing SAXPY with %d elements on the GPU. Done in %lf seconds!\n", array_size, (double) (clock() - starting_time) / CLOCKS_PER_SEC);
        
    cudaMemcpy(cuda_y, d_y, array_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        
    for(int j = 0; j < array_size; ++j)
    {
        
        if(abs(y[j] != cuda_y[j]) > margin)
        {
            correctness = 0;
        }
    }
        
        
    if(correctness == 1)
    {
        printf("Comparing the output for each implementation... Correct!\n");
    }
    else
    {
        printf("Comparing the output for each implementation... Incorrect!\n");
    }

    
  

    free(x);
    free(y);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}