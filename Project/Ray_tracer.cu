#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <io.h>
#include <stdint.h>
#include "timer.h"

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

const int width = 7680;
const int height = 7680;
const double factor = 0.703952253f * 2.38924456f;

__device__ const int width_d = width;
__device__ const int height_d = height;
__device__ const double factor_d = 0.703952253f * 2.38924456f;
__device__ const double pixel_factor_x = 2.0 / (double) width_d;
__device__ const double pixel_factor_y = 2.0 / (double) height_d; 

// Sphere properties
typedef struct Sphere
{
    double3 position;
    double radius;
    double3 color;
    double diffuse;
    double specular_c;
    double specular_k;
} Sphere;

// Light position and color
typedef struct Light
{
    double3 L;
    double3 color;
    double ambient;

} Light;

// Camera
typedef struct Camera
{
    double3 origin;    // Position
    double3 direction; // Pointing to
} Camera;


/*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------Helper functions------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------CPU---------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

//Dot product helper function for two double3
double dot(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

//Helper function to multiply a double3 (a vector in a 3d space) with a scalar
double3 scalar_mul(double3 vector, double scalar)
{
    double3 result;
    result = {vector.x * scalar, vector.y * scalar, vector.z * scalar};
    return result;
}


// Helper function with subtraction for double3
double3 subtraction(double3 a, double3 b)
{
    double3 result = {a.x - b.x, a.y - b.y, a.z - b.z};
    return result;
}

double3 addition(double3 a, double3 b)
{
    double3 result;
    result = {a.x + b.x, a.y + b.y, a.z + b.z};
    return result;
}


// Helper function for clip
int clip(double3 vector)
{
    int result = 0x0;
    vector.x = vector.x > 1 ? 255 : vector.x < 0 ? 0
                                                 : vector.x * 255;
    result = (int)vector.x << 16;
    vector.y = vector.y > 1 ? 255 : vector.y < 0 ? 0
                                                 : vector.y * 255;
    result = result | ((int)vector.y << 8);
    vector.z = vector.z > 1 ? 255 : vector.z < 0 ? 0
                                                 : vector.z * 255;
    result = result | ((int)vector.z);

    return result;
}

// Normalizes a 3 dimensional vector using the fast inverse square root algorithm
double3 normalize(double3 vector)
{
    //Based on https://stackoverflow.com/questions/11644441/fast-inverse-square-root-on-x64/11644533
    // Declare the variables used
    double inverse_sqrt = dot(vector, vector);
    double x2 = inverse_sqrt * 0.5;
    int64_t i = *(int64_t *) &inverse_sqrt;
    // The magic number is for doubles is from https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
    i = 0x5fe6eb50c7b537a9 - (i >> 1);
    inverse_sqrt = *(double *) &i;
    inverse_sqrt = inverse_sqrt * (1.5 - (x2 * inverse_sqrt * inverse_sqrt));   // 1st iteration
    //      y  = y * ( 1.5 - ( x2 * y * y ) );   // 2nd iteration, this can be removed

    double3 result;
    result.x = vector.x * inverse_sqrt;
    result.y = vector.y * inverse_sqrt;
    result.z = vector.z * inverse_sqrt;

    return result;
}


/*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------GPU---------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/


__device__ 
double3 add_gpu(double3 a, double3 b)
{
    double3 result;
    result = {a.x + b.x, a.y + b.y, a.z + b.z};
    return result;
}

__device__
double dot_gpu(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__
double3 normalize_gpu(double3 vector)
{
    //Based on https://stackoverflow.com/questions/11644441/fast-inverse-square-root-on-x64/11644533
    // Declare the variables used
    double inverse_sqrt = dot_gpu(vector, vector);
    double x2 = inverse_sqrt * 0.5;
    int64_t i = *(int64_t *) &inverse_sqrt;
    // The magic number is for doubles is from https://cs.uwaterloo.ca/~m32rober/rsqrt.pdf
    i = 0x5fe6eb50c7b537a9 - (i >> 1);
    inverse_sqrt = *(double *) &i;
    inverse_sqrt = inverse_sqrt * (1.5 - (x2 * inverse_sqrt * inverse_sqrt));   // 1st iteration
    //      y  = y * ( 1.5 - ( x2 * y * y ) );   // 2nd iteration, this can be removed

    double3 result;
    result.x = vector.x * inverse_sqrt;
    result.y = vector.y * inverse_sqrt;
    result.z = vector.z * inverse_sqrt;

    return result;
}


__device__
double3 sub_gpu(double3 a, double3 b)
{
    double3 result = {a.x - b.x, a.y - b.y, a.z - b.z};
    return result;
}

__device__
double3 scalar_mul_gpu(double3 vector, double scalar)
{
    double3 result;
    result = {vector.x * scalar, vector.y * scalar, vector.z * scalar};
    return result;
}

__device__
int clip_gpu(double3 vector)
{
    int result = 0x0;
    vector.x = vector.x > 1 ? 255 : vector.x < 0 ? 0
                                                 : vector.x * 255;
    result = (int)vector.x << 16;
    vector.y = vector.y > 1 ? 255 : vector.y < 0 ? 0
                                                 : vector.y * 255;
    result = result | ((int)vector.y << 8);
    vector.z = vector.z > 1 ? 255 : vector.z < 0 ? 0
                                                 : vector.z * 255;
    result = result | ((int)vector.z);

    return result;
}


/*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------Basic Version---------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------CPU---------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/


double intersect_sphere(Camera *camera, Sphere *sphere, double3 direction)
{
    // printf("checking for intersections\n");
    // Return the distance between the origin (camera) and the intersection of the ray with the sphere, or inf if there are no intersections.
    // camera.origin and sphere.position are points in a 3 dimensional space, direction is a normalized vector and sphere.radius is a scalar.
    double a = (double) dot(direction, direction);
    double3 OS = subtraction(camera->origin, sphere->position);
    double b = (double) 2 * dot(direction, OS);
    double c = (double) dot(OS, OS) - (sphere->radius * sphere->radius);
    double disc = (double) (b * b) - (4 * a * c);

    if(disc > 0)
    {
        double dist_sqrt = sqrt(disc);
        double q = b < 0 ? (-b - dist_sqrt) / 2.0 : (-b + dist_sqrt) / 2.0;
        double t0 = q / a;
        double t1 = c / q;

        double min = fmin(t0, t1);
        double max = fmax(t0, t1);

        if(max >= 0)
        {
            return (min < 0) ? t1 : t0;
        } 
    }

    return INFINITY;
    
}

double3 trace_ray(Camera *camera, Sphere *sphere, Light *light, double3 direction)
{
    // printf("Tracing the ray\n");
    // Find the first point of intersection in the scene
    double t = intersect_sphere(camera, sphere, direction);
    // No infinty?
    if (isinf(t))
    {
        return {NULL, NULL, NULL};
    }

    // Find the point of the intersection on the object
    double3 M = addition(camera->origin, scalar_mul(direction, t));
    double3 N = subtraction(M, sphere->position);
    N = normalize(N);
    double3 toL = subtraction(light->L, M);
    toL = normalize(toL);
    double3 toO = subtraction(camera->origin, M);
    toO = normalize(toO);

    // Ambient light
    double3 col = {light->ambient, light->ambient, light->ambient};
    
    // Lambert shading
    double saved_value = dot(N, toL);
    saved_value = fmax(saved_value, 0.0);
    saved_value *= sphere->diffuse;
    col = addition(col, scalar_mul(sphere->color, saved_value));

    // Blinn-phong shading (specular)
    double3 toO_plus_toL = normalize(addition(toL, toO));
    double saved_value2 = dot(N, toO_plus_toL);
    saved_value = sphere->specular_c * pow(fmax(saved_value2, 0), sphere->specular_k);
    col = addition(col, scalar_mul(light->color, saved_value));

    return col;
}

void run(Camera *camera, Sphere *sphere, Light *light, int *img)
{
    printf("Running\n");
    double3 direction;
    double3 color;
    double add_height = (double)2 / (double)height;
    double add_width = (double)2 / (double)width;
    double x = (double)-1;
    double y = (double)-1;
    camera->direction.x = x;
    camera->direction.y = y;
    // printf("Done with the set up\n");

    // Loop through all pixels
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            // Position of the pixel
            // Direction of the ray going through the optical center
            direction = normalize(subtraction(camera->direction, camera->origin));

            // Launch the ray and get the color of the pixel
            color = trace_ray(camera, sphere, light, direction);
            camera->direction.y += add_height;

            if (color.x == NULL)
            {
                continue;
            }
            img[i * width + j] = clip(color);
        }
       camera->direction.x += add_width;
       camera->direction.y = (double) -1;
    }
    // printf("Done with the for loop");
}



/*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------GPU---------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/


__device__
double intersect_sphere_gpu(Camera *camera, Sphere *sphere, double3 direction)
{
    // Return the distance between the origin (camera) and the intersection of the ray with the sphere, or inf if there are no intersections.
    // camera.origin and sphere.position are points in a 3 dimensional space, direction is a normalized vector and sphere.radius is a scalar.

    double a = dot_gpu(direction, direction);
    double3 OS = sub_gpu(camera->origin, sphere->position);
    double b = 2 * dot_gpu(direction, OS);
    double c = dot_gpu(OS, OS) - (sphere->radius * sphere->radius);

    double disc = (b * b) - (4 * a * c);

    if(disc > 0)
    {
        double dist_sqrt = sqrt(disc);
        double q = b < 0 ? (-b - dist_sqrt) / 2.0 : (-b + dist_sqrt) / 2.0;
        double t0 = q / a;
        double t1 = c / q;
        double tmin = min(t0, t1);
        double tmax = max(t0, t1);

        if(tmax >= 0)
        {
            return t0 < 0 ? t1 : t0;
        }
    } 
    return INFINITY;
}

__device__
double3 trace_ray_gpu(Camera *camera, Sphere *sphere, Light *light, double3 direction)
{
    //Find the firsst point of intersection with the scene.
    double t = intersect_sphere_gpu(camera, sphere, direction);
    //Check if there is an intersection
        if(t == INFINITY)
    {
        return {NULL, NULL, NULL};
    }
    //Find the intersection on the object
    double3 M = add_gpu(camera->origin, scalar_mul_gpu(direction, t));
    double3 N = normalize_gpu(sub_gpu(M, sphere->position));
    double3 toL = normalize_gpu(sub_gpu(light->L, M));
    double3 toO = normalize_gpu(sub_gpu(camera->origin, M));

    //Ambient light
    double3 col = {light->ambient, light->ambient, light->ambient};

    //Lambert shading (diffuse)
    double saved_value = dot_gpu(N, toL);
    saved_value = max(saved_value, 0.0);
    saved_value *= sphere->diffuse;
    col = add_gpu(col, scalar_mul_gpu(sphere->color, saved_value));

    //Blinn phong shading (specular)
    double3 toO_plus_toL = normalize_gpu(add_gpu(toL, toO));
    double saved_value2 = dot_gpu(N, toO_plus_toL);
    saved_value = sphere->specular_c * pow(max(saved_value2, 0.0), sphere->specular_k);
    col = add_gpu(col, scalar_mul_gpu(light->color, saved_value));

    //Check if there was an intersection, done here as to maximize performance of the function (making sure that the warps doesn't have too many different branches)

    return col;
}

__global__
void run_gpu(Camera *camera, Sphere *sphere, Light *light, int *img)
{
    int index = blockIdx.x *blockDim.x + threadIdx.x;

    if(index > height_d * width_d)
    {
        return;
    }

    int i = index % width_d;
    int j = index / width_d;

    double x = -1.0 + (double) i * pixel_factor_x;
    double y = -1.0 + (double) j * pixel_factor_y;

    double3 direction = {x - camera->origin.x, y - camera->origin.y, camera->direction.z - camera->origin.z};
    direction = normalize_gpu(direction);

    double3 col = trace_ray_gpu(camera, sphere, light, direction);

    img[index] = clip_gpu(col);

}




/*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------BMP file writing--------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

unsigned char *create_BMP_file_header(int stride)
{
    printf("info header being made\n");
    int fileSize = 54 + (stride * height);

    static unsigned char fileHeader[] = {
        0, 0,       /// signature
        0, 0, 0, 0, /// image file size in bytes
        0, 0, 0, 0, /// reserved
        0, 0, 0, 0, /// start of pixel array
    };

    fileHeader[0] = (unsigned char)('B');
    fileHeader[1] = (unsigned char)('M');
    fileHeader[2] = (unsigned char)(fileSize);
    fileHeader[3] = (unsigned char)(fileSize >> 8);
    fileHeader[4] = (unsigned char)(fileSize >> 16);
    fileHeader[5] = (unsigned char)(fileSize >> 24);
    fileHeader[10] = (unsigned char)(54);

    return fileHeader;
}

unsigned char *create_BMP_info_header()
{
    printf("info header being made\n");
    static unsigned char infoHeader[] = {
        0, 0, 0, 0, /// header size
        0, 0, 0, 0, /// image width
        0, 0, 0, 0, /// image height
        0, 0,       /// number of color planes
        0, 0,       /// bits per pixel
        0, 0, 0, 0, /// compression
        0, 0, 0, 0, /// image size
        0, 0, 0, 0, /// horizontal resolution
        0, 0, 0, 0, /// vertical resolution
        0, 0, 0, 0, /// colors in color table
        0, 0, 0, 0, /// important color count
    };

    infoHeader[0] = (unsigned char)(40);
    infoHeader[4] = (unsigned char)(width);
    infoHeader[5] = (unsigned char)(width >> 8);
    infoHeader[6] = (unsigned char)(width >> 16);
    infoHeader[7] = (unsigned char)(width >> 24);
    infoHeader[8] = (unsigned char)(height);
    infoHeader[9] = (unsigned char)(height >> 8);
    infoHeader[10] = (unsigned char)(height >> 16);
    infoHeader[11] = (unsigned char)(height >> 24);
    infoHeader[12] = (unsigned char)(1);
    infoHeader[14] = (unsigned char)(24);

    return infoHeader;
}

void generate_BMP(int *img, FILE *image_file)
{
    printf("Writing to BMP file\n");
    // Setting up all the variables
    int width_in_bytes = width * 3;
    unsigned char padding[3] = {0, 0, 0};
    int padding_size = (4 - width_in_bytes % 4) % 4;
    int stride = width_in_bytes + padding_size;

    // Opening the file
    if (image_file == NULL)
    {
        printf("Could not open the file\n");
        exit(-1);
    }

    // Creating the headers and writing them to the file
    unsigned char *file_header = create_BMP_file_header(stride);
    fwrite(file_header, 1, 14, image_file);

    unsigned char *info_header = create_BMP_info_header();
    fwrite(info_header, 1, 40, image_file);

    unsigned char *image = NULL;

    image = (unsigned char *)malloc(3 * width * height);
    int x, y, r, g, b;
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            x = i;
            y = (height - 1) - j;
            r = (img[i * width + j] >> 16) & 0xFF;
            g = (img[i * width + j] >> 8) & 0xFF;
            b = img[i * width + j] & 0xFF;

            image[(x + y * width) * 3 + 2] = (unsigned char)(r);
            image[(x + y * width) * 3 + 1] = (unsigned char)(g);
            image[(x + y * width) * 3 + 0] = (unsigned char)(b);
        }
    }

    // Write the pixel data to the BMP file
    int verification = 0;
    for (int i = 0; i < height; ++i)
    {
        // printf("For loop\n");
        verification = fwrite(image + (width * (height - i - 1) * 3), 3, width, image_file);
        if (verification != width)
        {
            printf("Was not able to write all of the values of row %d image will be incomplete.\n", i);
        }

        fwrite(padding, padding_size, 1, image_file);
    }
    free(image);
    fflush(image_file);
    fclose(image_file);
}

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------Initializing and setting up the running of program-------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/


void init(Sphere *sphere, Light *light, Camera *camera)
{
    printf("Initializing\n");
    // Set sphere properties
    sphere->position = {0.0, 0.0, 1.0};
    sphere->radius = 1.5;
    sphere->color = {0.0, 0.0, 1.0};
    sphere->diffuse = 1.0;
    sphere->specular_c = 1.0;
    sphere->specular_k = 50.0;

    // Set light properties
    light->L = {5.0, 5.0, -10.0};
    light->color = {1.0, 1.0, 1.0};
    light->ambient = 0.05;

    // Set camera properties
    camera->origin = {0.0, 0.0, -1.0};
    camera->direction = {0.0, 0.0, 0.0};
}

__device__
void init_d(Sphere *sphere, Light *light, Camera *camera)
{
    printf("Initializing\n");
    // Set sphere properties
    sphere->position = {0.0, 0.0, 1.0};
    sphere->radius = 1.0;
    sphere->color = {0.0, 0.0, 1.0};
    sphere->diffuse = 1.0;
    sphere->specular_c = 1.0;
    sphere->specular_k = 50.0;

    // Set light properties
    light->L = {5.0, 5.0, -10.0};
    light->color = {1.0, 1.0, 1.0};
    light->ambient = 0.05;

    // Set camera properties
    camera->origin = {0.0, 0.0, -1.0};
    camera->direction = {0.0, 0.0, 0.0};
}

int main(int argc, char *argv[])
{
    //General set up
    Timer_t timer = {0};
    Timer_t timer_d = {0};
    Sphere *sphere = (Sphere *)malloc(sizeof(Sphere));
    Light *light = (Light *)malloc(sizeof(Light));
    Camera *camera = (Camera *)malloc(sizeof(Camera));
    init(sphere, light, camera);
    // printf("size of double3: %d\t size of sphere: %d\t size of light: %d\t size of camera: %d\n", sizeof(double3), sizeof(Sphere), sizeof(Light), sizeof(Camera));
    // printf("Size of Sphere *: %d\t Size of Light *: %d\t Size of Camera *: %d\n", sizeof(Sphere *), sizeof(Light *), sizeof(Camera *));

    // printf("Sphere memory address: %u\tposition: %u\tradius: %u\tcolor: %u\tdiffuse: %u\tspecular_c: %u\tspecular_k: %u\n"
    //         "Light memory address: %u\tL: %u\tcolor: %u\tambient: %u\n"
    //         "Camera memory address: %u\torigin: %u\tdirection: %u\n", 
    //         sphere, &sphere->position, &sphere->radius, &sphere->color, &sphere->diffuse, &sphere->specular_c, &sphere->specular_k,
    //         light, &light->L, &light->color, &light->ambient,
    //         camera, &camera->origin, &camera->direction);
    
    //CPU set up
    FILE *CPU = fopen("cpu_img.bmp", "wb+");
    int *img;
    img = (int *)malloc(sizeof(int) * height * width);
    tstart(&timer);
    run(camera, sphere, light, img);
    tend(&timer);
    printf("The execution on the CPU took %f seconds.\n", telapsed(&timer));
    generate_BMP(img, CPU);
    fclose(CPU);
    printf("Done with CPU image\n");

    //GPU set up

    FILE *GPU = fopen("gpu_img.bmp", "wb+");

    Sphere *sphere_d;
    Light *light_d;
    Camera *camera_d;
    int *img_d;
    int *cuda_img = (int *) malloc(sizeof(int) * height * width);

    cudaMalloc(&sphere_d, sizeof(Sphere));
    cudaMalloc(&light_d, sizeof(Light));
    cudaMalloc(&camera_d, sizeof(Camera));
    init(sphere, light, camera);
    cudaMalloc(&img_d, height * width * sizeof(int *));

    cudaMemcpy(sphere_d, sphere, sizeof(Sphere), cudaMemcpyHostToDevice);
    cudaMemcpy(light_d, light, sizeof(Light), cudaMemcpyHostToDevice);
    cudaMemcpy(camera_d, camera, sizeof(Camera), cudaMemcpyHostToDevice);

    int TPB = 32;
    int BLOCKS = (height * width + TPB - 1) / TPB;

    tstart(&timer_d);
    run_gpu<<<BLOCKS, TPB>>>(camera_d, sphere_d, light_d, img_d);
    cudaDeviceSynchronize();
    tend(&timer_d);
    printf("The execution on the GPU took %f seconds.\n", telapsed(&timer_d));

    HANDLE_ERROR(cudaMemcpy(cuda_img, img_d, sizeof(int) * width_d * height_d, cudaMemcpyDeviceToHost));

    generate_BMP(cuda_img, GPU);

    cudaFree(sphere_d);
    cudaFree(light_d);
    cudaFree(camera_d);
    cudaFree(img_d);
    free(img);
    free(sphere);
    free(light);
    free(camera);
    return 0;
}