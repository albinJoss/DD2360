#include <stdio.h>
#include <math.h>
#include <windows.h>
#include <wchar.h>


#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 100
#define BLOCK_SIZE 128
#define NUM_STREAMS 1

typedef struct 
{
    float3 Position;
    float3 Velocity;
} Particle;

typedef struct
{
    int block_size;
    int num_iterations;
    int num_particles;
	int num_streams;
} Info;

typedef struct 
{
    Particle *particles;
    Particle *d_particles;
    Particle *cuda_particles;
} ParticleCollection;



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

       particles[i].Position.x += particles[i].Velocity.x * dt;        //Update the position of the particle based on the updated velocity
       particles[i].Position.y += particles[i].Velocity.y * dt;
       particles[i].Position.z += particles[i].Velocity.z * dt;
    }
    return;
}

void init(Info *info, ParticleCollection *particleCollection, cudaStream_t *streams)
{
    // particleCollection->particles = (Particle*)malloc(info->num_particles * sizeof(Particle));            //Allocate the space needed on the host
    cudaHostAlloc((void **) &(particleCollection->particles), info->num_particles * sizeof(Particle), cudaHostAllocDefault);
    // particleCollection->cuda_particles = (Particle*)malloc(info->num_particles * sizeof(Particle));

    for(int i = 0; i < info->num_particles; ++i)
    {
       particleCollection->particles[i].Velocity.x = 0.7f;
       particleCollection->particles[i].Velocity.y = 0.6f;
       particleCollection->particles[i].Velocity.z = 0.5f;
       particleCollection->particles[i].Position.x = 0;
       particleCollection->particles[i].Position.y = 0;
       particleCollection->particles[i].Position.z = 0;
    }

	for(int j = 0; j < info->num_streams; ++j)
	{
		cudaStreamCreate(&streams[j]);
	}

    cudaMalloc(&(particleCollection->d_particles), info->num_particles * sizeof(Particle));             //Allocate the space needed on the device
    //cudaMemcpy(particleCollection->d_particles, particleCollection->particles, info->num_particles * sizeof(Particle), cudaMemcpyHostToDevice);   //Copy the arrays from the the host to the device
    return;
}

void cpu_execution(Particle *particles, Info *info)
{
    clock_t starting_time = clock();
    for(int iteration = 0; iteration < info->num_iterations; ++iteration)
    {
        timestep_for_cpu(particles, info->num_particles, iteration);
    }
    //printf("Done with the simulation on the CPU, using %d iterations and %d particles. It took %lf seconds to do!\n", info->num_iterations, info->num_particles, (double) (clock() - starting_time) / CLOCKS_PER_SEC);
    return;

}

void gpu_execution(ParticleCollection *particleCollection, Info *info, cudaStream_t *streams)
{
	int batch_size = info->num_particles / info->num_streams;
	int bytes = batch_size * sizeof(Particle);
    int grid_size = (info->num_particles + info->block_size - 1) / info->block_size;
    clock_t starting_time = clock();            //Do the calculations on the GPU
    for(int iteration = 0; iteration < info->num_iterations; ++iteration)
    {
		for(int i = 0; i < info->num_streams; ++i)
		{
			int offset = batch_size * i;
			cudaMemcpyAsync(&(particleCollection->d_particles)[offset], &(particleCollection->particles)[offset], bytes, cudaMemcpyHostToDevice, streams[i]);
			timestep_for_gpu<<<grid_size, info->block_size, 0, streams[i]>>>(particleCollection->d_particles, info->num_particles, iteration);
			cudaMemcpyAsync(&(particleCollection->particles)[offset], &(particleCollection->d_particles)[offset], bytes, cudaMemcpyDeviceToHost, streams[i]);
		}
    }
	cudaDeviceSynchronize();
	
	for(int k = 0; k < info->num_streams; ++k)
	{
		cudaStreamDestroy(streams[k]);
	}
	// printf("Destroyed\n");
	printf("Done with the simulation on the GPU, using %d iterations and %d particles. It took %lf seconds to do!\n", info->num_iterations, info->num_particles, (double) (clock() - starting_time) / CLOCKS_PER_SEC);
    return;
}

int main(int argc, char *argv[])
{
    Info *info = (Info*) malloc(sizeof(Info));
    ParticleCollection *particleCollection = (ParticleCollection*) malloc(sizeof(particleCollection));
    // int correctness = 1;

    info->block_size = argc > 1 ? atoi(argv[1]) : BLOCK_SIZE;
    info->num_iterations = argc > 2 ? atoi(argv[2]) : NUM_ITERATIONS;
    info->num_particles = argc > 3 ? atoi(argv[3]) : NUM_PARTICLES;
	info->num_streams = argc > 4 ? atoi(argv[4]) : NUM_STREAMS;

    cudaStream_t *streams = (cudaStream_t *)malloc(info->num_streams * sizeof(cudaStream_t));
    init(info, particleCollection, streams);
    
    cpu_execution(particleCollection->particles, info);

    gpu_execution(particleCollection, info, streams);

    //cudaMemcpy(particleCollection->cuda_particles, particleCollection->d_particles, info->num_particles * sizeof(Particle), cudaMemcpyDeviceToHost);

    // for(int j = 0; j < info->num_particles; ++j)      //Check if both versions got the same results
    // {
    //     if(particleCollection->particles[j].Position.x != particleCollection->cuda_particles[j].Position.x || 
    //        particleCollection->particles[j].Position.y != particleCollection->cuda_particles[j].Position.y || 
    //        particleCollection->particles[j].Position.z != particleCollection->cuda_particles[j].Position.z)
    //     {
    //         correctness = 0;
    //         printf("GPU and CPU does not match at index %d\n", j);
    //     }
    // }
    
    // if(correctness == 1)
    // {
    //  //   printf("Comparing the output for each implementation... Correct!\n");
    // }
    // else
    // {
        
    //     printf("Comparing the output for each implementation... Incorrect!\n");
    // }

    // free(particleCollection->particles);
    cudaFreeHost(particleCollection->particles);
    free(info);
    free(particleCollection);
    cudaFree(particleCollection->d_particles);
    // printf("Freed\n");
    return 0;
}