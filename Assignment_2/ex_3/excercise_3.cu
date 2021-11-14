#include <stdio.h>
#include <math.h>
#include <windows.h>
#include <wchar.h>

#define DIMENSIONS 3
#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 100
#define BLOCK_SIZE 256

typedef struct 
{
    float3 Position;
    float3 Velocity;
} Particle;

__global__
void timestep_for_gpu(Particle *particles, int time)
{
    
}

void timestep_for_cpu(Particle *particles, int time)
{

}

int main(int argc, char *argv[])
{
    int particles, iterations, block_size;
    const float dt = 1.0f;
    switch(argc)
    {
        case 1:
            particles = NUM_PARTICLES;
            iterations = NUM_ITERATIONS;
            block_size = atoi(argv[1]);
            break;
        
        case 2:
            particles = NUM_PARTICLES;
            iterations = atoi(argv[2]);
            block_size = atoi(argv[1]);
            break;

        case 3:
            particles = atoi(argv[3]);
            iterations = atoi(argv[2]);
            block_size = atoi(argv[1]);
            break;

        default:
            particles = NUM_PARTICLES;
            iterations = NUM_ITERATIONS;
            block_size = BLOCK_SIZE;
            break;

    }
}