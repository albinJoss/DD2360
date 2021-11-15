#include <stdio.h>
#include <math.h>
#include <windows.h>
#include <wchar.h>


#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 100
#define BLOCK_SIZE 256

typedef struct 
{
    float3 Position;
    float3 Velocity;
} Particle;

__global__
void timestep_for_gpu(Particle *particles, int num_particles, int iteration)
{
    const float dt = 1.0f;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < num_particles)
    {
        particles[i].Velocity.x = (iteration % 3 == 0) ? (particles[i].Velocity.x - 0.01) : (particles[i].Velocity.x + 0.01);                                    //Update velocity for the particle
        particles[i].Velocity.y = (iteration % 4 == 0) ? (particles[i].Velocity.y - 0.01) : (particles[i].Velocity.y + 0.01);
        particles[i].Velocity.z = (iteration % 5 == 0) ? (particles[i].Velocity.z - 0.01) : (particles[i].Velocity.z + 0.01);

        particles[i].Position.x += particles[i].Velocity.x * dt;        //Update the position of the particle based on the updated velocity
        particles[i].Position.y += particles[i].Velocity.y * dt;
        particles[i].Position.z += particles[i].Velocity.z * dt;
    }
}

void timestep_for_cpu(Particle *particles, int num_particles, int iteration)
{
    const float dt = 1.0f;
    for(int i = 0; i < num_particles; ++i)
    {
       particles[i].Velocity.x = (iteration % 3 == 0) ? (particles[i].Velocity.x - 0.01) : (particles[i].Velocity.x + 0.01);                                    //Update velocity for the particle
        particles[i].Velocity.y = (iteration % 4 == 0) ? (particles[i].Velocity.y - 0.01) : (particles[i].Velocity.y + 0.01);
        particles[i].Velocity.z = (iteration % 5 == 0) ? (particles[i].Velocity.z - 0.01) : (particles[i].Velocity.z + 0.01);

        particles[i].Position.x += particles[i].Velocity.x * dt;        //Update position for the particle based on the updated velocity
        particles[i].Position.y += particles[i].Velocity.y * dt;
        particles[i].Position.z += particles[i].Velocity.z * dt;
    }
}

int main(int argc, char *argv[])
{
    printf("Enter\n");
    int num_particles, num_iterations, block_size, iteration, grid_size;
    int correctness = 1;
    Particle *particles, *d_particles, *cuda_particles;
    clock_t starting_time;
    printf("Set\n");

    switch(argc)            //Change the number of iterations, particles or the block size depending on the arguments sent through the command line
    { 
        case 1:
            num_particles = NUM_PARTICLES;
            num_iterations = NUM_ITERATIONS;
            block_size = BLOCK_SIZE;
            break;

        case 2:
            num_particles = NUM_PARTICLES;
            num_iterations = NUM_ITERATIONS;
            block_size = atoi(argv[1]);
            break;
        
        case 3:
            num_particles = NUM_PARTICLES;
            num_iterations = atoi(argv[2]);
            block_size = atoi(argv[1]);
            break;

        default:
            num_particles = atoi(argv[3]);
            num_iterations = atoi(argv[2]);
            block_size = atoi(argv[1]);
            break;

        

    }
  
    particles = (Particle*)malloc(num_particles * sizeof(Particle));            //Allocate the space needed on the host
    cuda_particles = (Particle*)malloc(num_particles * sizeof(Particle));
  
    for(int i = 0; i < num_particles; ++i)
    {
        particles[i].Velocity.x = 0.7f;
        particles[i].Velocity.y = 0.6f;
        particles[i].Velocity.z = 0.5f;
        particles[i].Position.x = 0;
        particles[i].Position.y = 0;
        particles[i].Position.z = 0;
    }

   

    cudaMalloc(&d_particles, num_particles * sizeof(Particle));             //Allocate the space needed on the device
    cudaMemcpy(d_particles, particles, num_particles * sizeof(Particle), cudaMemcpyHostToDevice);   //Copy the arrays from the the host to the device
    
   

    starting_time = clock();                //Do the calculations on the CPU
    for(iteration = 0; iteration < num_iterations; ++iteration)
    {
        timestep_for_cpu(particles, num_particles, iteration);
    }
    printf("Done with the simulation on the CPU, using %d iterations and %d particles. It took %lf seconds to do!\n", num_iterations, num_particles, (double) (clock() - starting_time) / CLOCKS_PER_SEC);

    grid_size = (num_particles + block_size - 1) / block_size;
    starting_time = clock();            //Do the calculations on the GPU
    for(iteration = 0; iteration < num_iterations; ++iteration)
    {
        timestep_for_gpu<<<grid_size, block_size>>>(d_particles, num_particles, iteration);
    }
    printf("Done with the simulation on the GPU, using %d iterations and %d particles. It took %lf seconds to do!\n", num_iterations, num_particles, (double) (clock() - starting_time) / CLOCKS_PER_SEC);

    cudaMemcpy(cuda_particles, d_particles, num_particles * sizeof(Particle), cudaMemcpyDeviceToHost);

    for(int j = 0; j < num_particles; ++j)      //Check if both versions got the same results
    {
        if(particles[j].Position.x != cuda_particles[j].Position.x || particles[j].Position.y != cuda_particles[j].Position.y || particles[j].Position.z != cuda_particles[j].Position.z)
        {
            correctness = 0;
            printf("GPU and CPU does not match at index %d\n", j);
        }
    }
    
    if(correctness == 1)
    {
        printf("Comparing the output for each implementation... Correct!\n");
    }
    else
    {
        /*for(int j = 0; j < num_particles; ++j)
        {
            printf("%d:     x: %lf y: %lf z: %lf        || d_x: %lf d_y: %lf d_z: %lf\n", j, particles[j].Position.x, particles[j].Position.y, particles[j].Position.z, cuda_particles[j].Position.x, cuda_particles[j].Position.y, cuda_particles[j].Position.z);
        }*/
        printf("Comparing the output for each implementation... Incorrect!\n");
    }

    free(particles);
    free(cuda_particles);
    cudaFree(d_particles);
}