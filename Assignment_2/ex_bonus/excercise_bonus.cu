#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <curand.h>

#define SEED 921
#define NUM_ITER 1073741824 //Total number of iterations
#define NUM_BLOCKS 512

__global__
void calculateDouble(curandState *states,  int num_iterations, int *count)
{
    double x, y, z;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(SEED, i, 0, &states[i]);

    for(int j = 0; j < num_iterations; ++j)
    {
        x = curand_uniform(&states[i]);
        y = curand_uniform(&states[i]);

        z = sqrt((x * x) + (y * y));

        if(z <=1.0)
        {
            count[i] += 1;
        }
    }
}

__global__
void calculateSingle(curandState *states,  int num_iterations, int *count)
{
    float x, y, z;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(SEED, i, 0, &states[i]);

    for(int j = 0; j < num_iterations; ++j)
    {
        x = curand_uniform(&states[i]);
        y = curand_uniform(&states[i]);

        z = sqrt((x*x) + (y * y));

        if(z <=1.0)
        {
            count[i] += 1;
        }
    }
}

int main(int argc, char *argv[])
{
    clock_t starting_time, end_time;
    int num_blocks, single, TPB; 
    int *d_count, *count;
    unsigned long long num_iterations, iterations_per_thread;
    double pi_double;
    float pi_single;
    unsigned long long sum = 0;
    curandState *dev_random;
    
    switch(argc)            //Change the number of iterations, blocks and decide if we're doing single or double precision
    { 
        case 1:
            single = 0;
            num_iterations = NUM_ITER;
            num_blocks = NUM_BLOCKS;
            break;

        case 2:
            single = 0;
            num_iterations = NUM_ITER;
            num_blocks = atoi(argv[1]);
            break;
        
        case 3:
            single = 0;
            num_iterations = atoi(argv[2]);
            num_blocks = atoi(argv[1]);
            break;

        default:
            single = atoi(argv[3]);
            num_iterations = atoi(argv[2]);
            num_blocks = atoi(argv[1]);
            break;
    }

    TPB =  131072 /  num_blocks;
    iterations_per_thread = num_iterations / (num_blocks * TPB); 
    printf("%llu\n", iterations_per_thread);
    cudaMalloc((void**)&dev_random, TPB * num_blocks * sizeof(curandState));
    cudaMalloc(&d_count, num_blocks * TPB * sizeof(int));
    count = (int*)malloc(num_blocks * TPB * sizeof(int));
    //cudaMemcpy(d_count, count, num_blocks * TPB, cudaMemcpyHostToDevice);

    
    if(single == 0)
    {
        starting_time = clock();
        calculateDouble<<<num_blocks, TPB>>>(dev_random, iterations_per_thread, d_count);
        cudaDeviceSynchronize();
        end_time = clock();
        cudaMemcpy(count, d_count, num_blocks * TPB * sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 0; i < num_blocks * TPB; ++i)
        {
            
            sum += count[i];
        }
        printf("%llu\n", sum);
        pi_double = ((double)sum / ((double) num_iterations)) * 4.0;
        printf("Done with the simulation on the for double precision, using %ld iterations and a block size of %d. It took %lf seconds on the GPU and a total of %lf seconds to do and pi was estimated to %.15f\n", (long int)num_iterations, num_blocks, (double) (end_time - starting_time) / CLOCKS_PER_SEC, (double) (clock() - starting_time) / CLOCKS_PER_SEC, pi_double);
    }
    else
    {
        starting_time = clock();
        calculateSingle<<<num_blocks, TPB>>>(dev_random, iterations_per_thread, d_count);;
        cudaDeviceSynchronize();
        end_time = clock();
        cudaMemcpy(count, d_count, num_blocks * TPB * sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 0; i < num_blocks * TPB; ++i)
        {
            sum += count[i];
        }
        pi_single = ((float)sum /((float) iterations_per_thread *  TPB * num_blocks)) * 4.0;
        printf("Done with the simulation on the for single precision, using %ld iterations and a block size of %d. It took %lf seconds on the GPU and a total of %lf seconds to do and pi was estimated to %.15f\n", (long int)num_iterations, num_blocks, (double) (end_time - starting_time) / CLOCKS_PER_SEC, (double) (clock() - starting_time) / CLOCKS_PER_SEC, pi_single);
    }
    cudaFree(dev_random);
    cudaFree(d_count);
    free(count);
    return 0;
}