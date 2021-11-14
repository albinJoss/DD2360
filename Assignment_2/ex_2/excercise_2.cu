#include <stdio.h>
//#include <sys/time.h>
#include <time.h>
#include <limits.h>
#define FINAL_SIZE 50000
#define CHANGE 4000
#define TPB 256

#define ARRAY_SIZE  10000

void memsetF(float *x, float *y, float *d_x, float *d_y)           //Sets the arrays using memcpy to speed up the execution
{
    for(int i = 0; i < ARRAY_SIZE; ++i)
    {
        x[i] = 2.0f;
        y[i] = 8.0f;
    }
    cudaMemcpy(d_x, x, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
}

int check_correctness(float *y, float *d_y)       //check if the arrays match each other
{
    
    float margin = 0.01f;
    for(int j = 0; j < ARRAY_SIZE; ++j)
    {
        
        if(abs(y[j] != d_y[j]) > margin)
        {
            return 0;
        }
    }
    return 1;
} 

/*double cpuSecond()              //Timing from within the program
{
    struct timeval tp;
    gettimeofday(&tp, null);
    return ((double) tp.tv_sec + (double)tp.tv_usec*1.e-6);
}*/

__global__ 
void saxpy_for_gpu(float *x, float *y, const float a)     //Executes SAXPY on the GPU parallel
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < ARRAY_SIZE)
    {
        y[i] += a * x[i];
    }
}


void saxpy_for_cpu(float *x, float *y, const float a)       //Executes saxpy on the CPU sequentially
{
    for(int i = 0; i < ARRAY_SIZE; ++i)
    {
        y[i] += a * x[i];
    }
}

int main()                          //Driver code
{
    int num_blocks = (ARRAY_SIZE + TPB - 1) / TPB;
    float *x, *y, *d_x, *d_y, *cuda_y;
    x = (float*)malloc(ARRAY_SIZE * sizeof(float));
    y = (float*)malloc(ARRAY_SIZE * sizeof(float));
    cuda_y = (float*)malloc(ARRAY_SIZE * sizeof(float));
    cudaMalloc(&d_x, ARRAY_SIZE * sizeof(float));
    cudaMalloc(&d_y, ARRAY_SIZE * sizeof(float));
    memsetF(x, y, d_x, d_y);
    const float a = 4.0f;
    double iStart, iElaps;
    int correctness = 1;
    clock_t starting_time;
  
      
        
        //iStart = cpuSecond();
        starting_time = clock();
        saxpy_for_cpu(x, y, a);
        //iElaps = cpuSecond() - iStart;
        printf("Computing SAXPY with %d elements on the CPU. Done in %lf seconds!\n", ARRAY_SIZE, (double) (clock() - starting_time) / CLOCKS_PER_SEC);

        //iStart = cpuSecond();
        starting_time = clock();
        saxpy_for_gpu<<<num_blocks, TPB>>>(d_x, d_y, a);
        cudaDeviceSynchronize();
        //iElaps = cpuSecond() - iStart;
        printf("Computing SAXPY with %d elements on the GPU. Done in %lf seconds!\n", ARRAY_SIZE, (double) (clock() - starting_time) / CLOCKS_PER_SEC);
        
        cudaMemcpy(cuda_y, d_y, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        
        correctness = check_correctness(y, cuda_y);
        
        
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