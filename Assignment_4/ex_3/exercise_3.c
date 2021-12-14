#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <errno.h>
#include <time.h>

typedef struct Saxpy
{
    float *x;
    float *y;
    float *d_x;
    float *d_y;
    float a;
    size_t array_size;
} Saxpy;

void memsetF(Saxpy *args)
{
    args->x = (float *)malloc(args->array_size * sizeof(float));
    args->y = (float *)malloc(args->array_size * sizeof(float));
	args->d_x = (float*)malloc(args->array_size * sizeof(float));
	args->d_y = (float*)malloc(args->array_size * sizeof(float));

    args->a = 4.0f;

    for (int i = 0; i < args->array_size; ++i)
    {
        args->x[i] = 2.0f;
        args->y[i] = 8.0f;
    }
}

void sequentially(Saxpy *args)
{
    for (int i = 0; i < args->array_size; ++i)
    {
        args->y[i] += args->a * args->x[i];
    }
}

void acc(size_t array_size, float *d_x, float *d_y)
{
	float a = 4.0f;

	d_x = (float*)malloc(array_size * sizeof(float));
	d_y = (float*)malloc(array_size * sizeof(float));

	for (size_t i = 0; i < array_size; i++) {
		d_x[i] = 2.0f;
		d_y[i] = 8.0f;
	}

	#pragma acc parallel
	for (size_t i = 0; i < array_size; i++)
	{
		d_y[i] += a * d_x[i];
	}



	free(d_x);
	free(d_y);
}

void correctness(Saxpy *args)
{
    int correctness = 1;
    float margin = 0.1f;
    for (int j = 0; j < args->array_size; ++j)
    {

        if (abs(args->y[j] - args->d_y[j]) > margin)
        {
            correctness = 0;
        }
    }

    if (correctness == 1)
    {
        printf("Comparing the output for each implementation... Correct!\n");
    }
    else
    {
        printf("Comparing the output for each implementation... Incorrect!\n");
    }
}

int main(int argc, char *argv[])
{

    size_t arr = argc > 1 ? atoi(argv[1]) : 10000;
    clock_t start;
    float* x = 0;
	float* y = 0;
	float a = 1.5f;
	printf("Set first\n");
	float* d_x = 0;
	float* d_y = 0;
	// printf("Set everything!\n");
	start = clock();

	acc(arr, d_x, d_y);

	double end = (double)(clock() - start) / CLOCKS_PER_SEC;;
	printf("Computing SAXPY with %d elements using OpenACC. Done in %lf seconds!\n", arr, end);

	Saxpy *args = malloc(sizeof(Saxpy *));
	args->array_size = arr;
	memsetF(args);
	printf("Time for the OpenACC\n");
	start = clock();

    
	sequentially(args);
	end = (double)(clock() - start) / CLOCKS_PER_SEC;;
	printf("Computing SAXPY with %d elements on the CPU. Done in %lf seconds!\n", args->array_size, end);
	
	memcpy(args->d_y, d_y, sizeof(float *) * args->array_size);
   	correctness(args);
	
	free(args->x);
	free(args->y);
	free(args->d_x);
	free(args->d_y);
	free(args);
    return 0;
}