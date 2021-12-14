#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <CL/cl.h>

// This is a macro for checking the error variable.
#define CHK_ERROR(err)     \
    if (err != CL_SUCCESS) \
        fprintf(stderr, "Error: %s\n", clGetErrorString(err));

// A errorCode to string converter (forward declaration)
const char *clGetErrorString(int);

#define TPB 256

typedef struct args
{
    float *x;
    float *y;
    float *d_x;
    float *d_y;
    float a;
    int64_t array_size;
} args;

void memsetF(args *args) //Sets the arrays using memcpy to speed up the execution
{
    args->x = (float *)malloc(args->array_size * sizeof(float));
    args->y = (float *)malloc(args->array_size * sizeof(float));
    args->d_x = (float *)malloc(args->array_size * sizeof(float));
    args->d_y = (float *)malloc(args->array_size * sizeof(float));
    args->a = 4.0f;

    for (int i = 0; i < args->array_size; ++i)
    {
        args->x[i] = 2.0f;
        args->y[i] = 8.0f;

        args->d_x[i] = 2.0f;
        args->d_y[i] = 8.0f;
    }
}

void saxpy_for_cpu(args *args) //Executes saxpy on the CPU sequentially
{
    for (int i = 0; i < args->array_size; ++i)
    {
        args->y[i] += args->a * args->x[i];
    }
}

const char *execute_kernel = " \
__kernel \
void saxpy(__global float *x, __global float *y, int array_size, float a) { \
	int idx = get_global_id(0);      \
	if (idx >= array_size) return;   \
									 \
	y[idx] += a * x[idx];		     \
} \
";

void opencl(args *args)
{
    clock_t starting;
    cl_platform_id *platforms;
    cl_uint n_platform;

    // Find OpenCL Platforms
    cl_int err = clGetPlatformIDs(0, NULL, &n_platform);
    CHK_ERROR(err);
    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * n_platform);
    err = clGetPlatformIDs(n_platform, platforms, NULL);
    CHK_ERROR(err);

    // Find and sort devices
    cl_device_id *device_list;
    cl_uint n_devices;
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices);
    CHK_ERROR(err);
    device_list = (cl_device_id *)malloc(sizeof(cl_device_id) * n_devices);
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, n_devices, device_list, NULL);
    CHK_ERROR(err);

    // Create and initialize an OpenCL context
    cl_context context = clCreateContext(NULL, n_devices, device_list, NULL, NULL, &err);
    CHK_ERROR(err);

    // Create a command queue
    cl_command_queue cmd_queue = clCreateCommandQueue(context, device_list[0], 0, &err);
    CHK_ERROR(err);

    /* Insert your own code here */
    starting = clock();
    size_t test = args->array_size * sizeof(float);
    cl_mem d_x = clCreateBuffer(context, CL_MEM_READ_ONLY, test, NULL, &err);
    CHK_ERROR(err);
    cl_mem d_y = clCreateBuffer(context, CL_MEM_READ_ONLY, test, NULL, &err);
    CHK_ERROR(err);

    err = clEnqueueWriteBuffer(cmd_queue, d_x, CL_TRUE, 0, (size_t)test, args->d_x, 0, NULL, NULL);
    CHK_ERROR(err);
    err = clEnqueueWriteBuffer(cmd_queue, d_y, CL_TRUE, 0, (size_t)test, args->d_y, 0, NULL, NULL);
    CHK_ERROR(err);

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&execute_kernel, NULL, &err);
    CHK_ERROR(err);

    err = clBuildProgram(program, 1, device_list, NULL, NULL, NULL);
    CHK_ERROR(err);

    cl_kernel kernel = clCreateKernel(program, "saxpy", &err);
    CHK_ERROR(err);

    size_t workitems = (args->array_size / TPB + 1) * TPB;
    size_t workgroup = TPB;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_x);
    CHK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_y);
    CHK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), (void *)&(args->array_size));
    CHK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(args->a), (void *)&args->a);
    CHK_ERROR(err);

    err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &workitems, &workgroup, 0, NULL, NULL);
    CHK_ERROR(err);

    err = clEnqueueReadBuffer(cmd_queue, d_y, CL_TRUE, 0, test, args->d_y, 0, NULL, NULL);
    CHK_ERROR(err);
    CHK_ERROR(err);

    // printf("Computing SAXPY with %d elements with OpenCL. \nDone in %lf seconds!\n", args->array_size, (double) (clock() - starting) / CLOCKS_PER_SEC);
    printf("%f\n", (double)(clock() - starting) / CLOCKS_PER_SEC);
    err = clFlush(cmd_queue);
    CHK_ERROR(err);
    err = clFinish(cmd_queue);
    CHK_ERROR(err);

    // Finally, release all that we have allocated.
    err = clReleaseCommandQueue(cmd_queue);
    CHK_ERROR(err);
    err = clReleaseContext(context);
    CHK_ERROR(err);
    free(platforms);
    free(device_list);
}

//Driver code
int main(int argc, char *argv[])
{
    clock_t starting_time;
    double end;
    struct args *arg = (args *)malloc(sizeof(struct args));
    arg->array_size = argc == 1 ? 10000 : atoi(argv[1]);
    int correctness = 1;
    float margin = 0.01f;
    // for (arg->array_size; arg->array_size < 1000000000; arg->array_size)
    // {
        memsetF(arg);

        starting_time = clock();
        saxpy_for_cpu(arg);
        // printf("Computing SAXPY with %d elements on the CPU.\nDone in %lf seconds!\n", arg->array_size, (double)(clock() - starting_time) / CLOCKS_PER_SEC);
        end = (double)(clock() - starting_time) / CLOCKS_PER_SEC;
        printf("%I64d;%f;", arg->array_size, end);
        opencl(arg);

        for (int j = 0; j < arg->array_size; ++j)
        {

            if (abs(arg->y[j] - arg->d_y[j]) > margin)
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
        free(arg->d_x);
        free(arg->d_y);
        free(arg->x);
        free(arg->y);
        free(arg);
    // }
    return 1;
}

const char *clGetErrorString(int errorCode)
{
    switch (errorCode)
    {
    case 0:
        return "CL_SUCCESS";
    case -1:
        return "CL_DEVICE_NOT_FOUND";
    case -2:
        return "CL_DEVICE_NOT_AVAILABLE";
    case -3:
        return "CL_COMPILER_NOT_AVAILABLE";
    case -4:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5:
        return "CL_OUT_OF_RESOURCES";
    case -6:
        return "CL_OUT_OF_HOST_MEMORY";
    case -7:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8:
        return "CL_MEM_COPY_OVERLAP";
    case -9:
        return "CL_IMAGE_FORMAT_MISMATCH";
    case -10:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -12:
        return "CL_MAP_FAILURE";
    case -13:
        return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14:
        return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15:
        return "CL_COMPILE_PROGRAM_FAILURE";
    case -16:
        return "CL_LINKER_NOT_AVAILABLE";
    case -17:
        return "CL_LINK_PROGRAM_FAILURE";
    case -18:
        return "CL_DEVICE_PARTITION_FAILED";
    case -19:
        return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
    case -30:
        return "CL_INVALID_VALUE";
    case -31:
        return "CL_INVALID_DEVICE_TYPE";
    case -32:
        return "CL_INVALID_PLATFORM";
    case -33:
        return "CL_INVALID_DEVICE";
    case -34:
        return "CL_INVALID_CONTEXT";
    case -35:
        return "CL_INVALID_QUEUE_PROPERTIES";
    case -36:
        return "CL_INVALID_COMMAND_QUEUE";
    case -37:
        return "CL_INVALID_HOST_PTR";
    case -38:
        return "CL_INVALID_MEM_OBJECT";
    case -39:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40:
        return "CL_INVALID_IMAGE_SIZE";
    case -41:
        return "CL_INVALID_SAMPLER";
    case -42:
        return "CL_INVALID_BINARY";
    case -43:
        return "CL_INVALID_BUILD_OPTIONS";
    case -44:
        return "CL_INVALID_PROGRAM";
    case -45:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46:
        return "CL_INVALID_KERNEL_NAME";
    case -47:
        return "CL_INVALID_KERNEL_DEFINITION";
    case -48:
        return "CL_INVALID_KERNEL";
    case -49:
        return "CL_INVALID_ARG_INDEX";
    case -50:
        return "CL_INVALID_ARG_VALUE";
    case -51:
        return "CL_INVALID_ARG_SIZE";
    case -52:
        return "CL_INVALID_KERNEL_ARGS";
    case -53:
        return "CL_INVALID_WORK_DIMENSION";
    case -54:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case -55:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case -56:
        return "CL_INVALID_GLOBAL_OFFSET";
    case -57:
        return "CL_INVALID_EVENT_WAIT_LIST";
    case -58:
        return "CL_INVALID_EVENT";
    case -59:
        return "CL_INVALID_OPERATION";
    case -60:
        return "CL_INVALID_GL_OBJECT";
    case -61:
        return "CL_INVALID_BUFFER_SIZE";
    case -62:
        return "CL_INVALID_MIP_LEVEL";
    case -63:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64:
        return "CL_INVALID_PROPERTY";
    case -65:
        return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66:
        return "CL_INVALID_COMPILER_OPTIONS";
    case -67:
        return "CL_INVALID_LINKER_OPTIONS";
    case -68:
        return "CL_INVALID_DEVICE_PARTITION_COUNT";
    case -69:
        return "CL_INVALID_PIPE_SIZE";
    case -70:
        return "CL_INVALID_DEVICE_QUEUE";
    case -71:
        return "CL_INVALID_SPEC_ID";
    case -72:
        return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
    case -1002:
        return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003:
        return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004:
        return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005:
        return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    case -1006:
        return "CL_INVALID_D3D11_DEVICE_KHR";
    case -1007:
        return "CL_INVALID_D3D11_RESOURCE_KHR";
    case -1008:
        return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1009:
        return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
    case -1010:
        return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
    case -1011:
        return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
    case -1012:
        return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
    case -1013:
        return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
    case -1093:
        return "CL_INVALID_EGL_OBJECT_KHR";
    case -1092:
        return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
    case -1001:
        return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1057:
        return "CL_DEVICE_PARTITION_FAILED_EXT";
    case -1058:
        return "CL_INVALID_PARTITION_COUNT_EXT";
    case -1059:
        return "CL_INVALID_PARTITION_NAME_EXT";
    case -1094:
        return "CL_INVALID_ACCELERATOR_INTEL";
    case -1095:
        return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
    case -1096:
        return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
    case -1097:
        return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
    case -1000:
        return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1098:
        return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
    case -1099:
        return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
    case -1100:
        return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
    case -1101:
        return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
    default:
        return "CL_UNKNOWN_ERROR";
    }
}