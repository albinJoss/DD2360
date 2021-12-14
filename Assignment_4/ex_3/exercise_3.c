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
    size_t *array_size;
} Saxpy;

// void memsetF(Saxpy *args)
// {
//     args->x = (float *)malloc(args->array_size * sizeof(float));

//     args->y = (float *)malloc(args->array_size * sizeof(float));

//     args->d_x = (float *)malloc(args->array_size * sizeof(float));

//     args->d_y = (float *)malloc(args->array_size * sizeof(float));

//     args->a = 4.0f;

//     for (int i = 0; i < args->array_size; ++i)
//     {
//         args->x[i] = 2.0f;
//         args->y[i] = 8.0f;

//         args->d_x[i] = 2.0f;
//         args->d_y[i] = 8.0f;
//     }
// }

void sequentially(Saxpy *args)
{
    for (int i = 0; i < args->array_size; ++i)
    {
        args->y[i] += args->a * args->x[i];
    }
}

void acc(size_t array_size)
{
    float* g_x = 0;
	float* g_y = 0;
	float a = 1.6f;

	g_x = (float*)malloc(array_size * sizeof(float));
	g_y = (float*)malloc(array_size * sizeof(float));

	for (size_t i = 0; i < array_size; i++) {
		g_x[i] = (float)i;
		g_y[i] = 1.0f;
	}

	// printf("Computing SAXPY with OepnACC...\n");
	

	#pragma acc parallel loop copyin(g_x[0:array_size]) copyout(g_y[0:array_size])
	for (size_t i = 0; i < array_size; i++)
	{
		g_y[i] += a * g_x[i];
	}

	free(g_x);
	free(g_y);
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
        // printf("Comparing the output for each implementation... Correct!\n");
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

	// x = (float*)malloc(arr * sizeof(float));
	// y = (float*)malloc(arr * sizeof(float));

	// for (size_t i = 0; i <arr; i++) {
	// 	x[i] = (float)i;
	// 	y[i] = 1.0f;
	// }

	// printf("Computing SAXPY sequentially...\n");
	start = clock();

	// for (size_t i = 0; i < arr; i++)
	// {
	// 	y[i] += a * x[i];
	// }

	 double end = (double)(clock() - start) / CLOCKS_PER_SEC;;
	// printf("Done! Time elapsed: (ms) = %d\t (s): %f\n\n", (int)end, end / 1e6);

	// free(x);
	// free(y);

	
	start = clock();

    acc(arr);

	end = (double)(clock() - start) / CLOCKS_PER_SEC;;
	// printf("Done! Time elapsed: (ms) = %d\t (s): %f\n\n", (int)end, end / 1e6);
    printf("%lf\n", end);

    return 0;
}