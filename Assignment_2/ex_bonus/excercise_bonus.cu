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
void calculateDouble(curandState *states, unsigned long long int num_iterations, long int *count)
{
    double x, y, z;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(SEED, i, 0, &states[i]);

    for(unsigned long long int j = 0; j < num_iterations; ++j)
    {
        x = curand_uniform_double(&states[i]);
        y = curand_uniform_double(&states[i]);

        z = sqrt((x * x) + (y * y));

        if(z <=1.0)
        {
            count[i] += 1;
        }
    }
}

__global__
void calculateSingle(curandState *states, unsigned long long int num_iterations, long int *count)
{
    float x, y, z;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(SEED, i, 0, &states[i]);

    for(unsigned long long int j = 0; j < num_iterations; ++j)
    {
        x = curand_uniform(&states[i]);
        y = curand_uniform(&states[i]);

        z = sqrtf((x*x) + (y * y));

        if(z <=1.0)
        {
            count[i] += 1;
        }
    }
}

int main(int argc, char *argv[])
{
    clock_t starting_time, end_time;
    int block_size, single, num_blocks; 
    long int *d_count, *count;
    unsigned long long int num_iterations, iterations_per_thread;
    double pi_double;
    float pi_single;
    unsigned long long sum = 0;
    curandState *dev_random;
    
    switch(argc)            //Change the number of iterations, blocks and decide if we're doing single or double precision
    { 
        case 1:
            single = 0;
            num_iterations = NUM_ITER;
            block_size = NUM_BLOCKS;
            break;

        case 2:
            single = 0;
            num_iterations = NUM_ITER;
            block_size = atoi(argv[1]);
            break;
        
        case 3:
            single = 0;
            num_iterations = atoll(argv[2]);
            block_size = atoi(argv[1]);
            break;

        default:
            single = atoi(argv[3]);
            num_iterations = atoll(argv[2]);
            block_size = atoi(argv[1]);
            break;
    }

    num_blocks =  10496/ block_size;
    iterations_per_thread = num_iterations / (block_size * num_blocks); 
    
    cudaMalloc((void**)&dev_random, num_blocks * block_size * sizeof(curandState));
    cudaMalloc(&d_count, block_size * num_blocks * sizeof(long int));
    count = (long int*)malloc(block_size * num_blocks * sizeof(long int));
    

    
    if(single == 0)
    {
        starting_time = clock();
        calculateDouble<<<num_blocks, block_size>>>(dev_random, iterations_per_thread, d_count);
        cudaDeviceSynchronize();
        end_time = clock();
        cudaMemcpy(count, d_count, block_size * num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 0; i < block_size * num_blocks; ++i)
        {
            
            sum += count[i];
        }
        
        pi_double = ((double)sum / ((double) num_iterations)) * 4.0;
        printf("Done with the simulation on the for double precision, using %llu iterations and a block size of %d. It took %lf seconds on the GPU and a total of %lf seconds to do and pi was estimated to %.15f\n", num_iterations, block_size, (double) (end_time - starting_time) / CLOCKS_PER_SEC, (double) (clock() - starting_time) / CLOCKS_PER_SEC, pi_double);
    }
    else
    {
        starting_time = clock();
        calculateSingle<<<num_blocks, block_size>>>(dev_random, iterations_per_thread, d_count);;
        cudaDeviceSynchronize();
        end_time = clock();
        cudaMemcpy(count, d_count, block_size * num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 0; i < block_size * num_blocks; ++i)
        {
            sum += count[i];
        }
        pi_single = ((float)sum /((float) iterations_per_thread *  num_blocks * block_size)) * 4.0;
        printf("Done with the simulation on the for single precision, using %llu iterations and a block size of %d. It took %lf seconds on the GPU and a total of %lf seconds to do and pi was estimated to %.15f\n", num_iterations, block_size, (double) (end_time - starting_time) / CLOCKS_PER_SEC, (double) (clock() - starting_time) / CLOCKS_PER_SEC, pi_single);
    }
    
    cudaFree(dev_random);
    cudaFree(d_count);
    free(count);
    return 0;
}