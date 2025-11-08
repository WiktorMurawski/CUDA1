#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "cuda_buffer.cuh"
#include "random_fill.cuh"

#define CUDA_CHECK(call)                                        \
    do {                                                        \
        cudaError_t err = (call);                               \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",         \
                cudaGetErrorString(err), __FILE__, __LINE__);   \
            return err;                                         \
        }                                                       \
    } while (0)                                                 \

#define CUDA_CHECK_GOTO(call, label)                            \
  do {                                                          \
    cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n",         \
                cudaGetErrorString(err), __FILE__, __LINE__);   \
            goto label;                                         \
        }                                                       \
  } while (0)                                                   \

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main(int argc, char** argv)
{
    constexpr int width = 800;
    constexpr int height = 600;

    constexpr int arraySize = 64;
    int* a = new int[arraySize];
    int* b = new int[arraySize];
    fillRandom(a, arraySize, 0, 9);
    fillRandom(b, arraySize, -9, 0);

    int c[arraySize] = { 0 };

    // Choose which GPU to run on, change this on a multi-GPU system.
    CUDA_CHECK_GOTO(cudaSetDevice(0), cleanup);

    // Add vectors in parallel.
    CUDA_CHECK_GOTO(addWithCuda(c, a, b, arraySize), cleanup);

cleanup:
    printf("[");
    for (int i = 0; i < arraySize; i++)
    {
        printf("%2d, ", a[i]);
    }
    printf("] + \n");
    printf("[");
    for (int i = 0; i < arraySize; i++)
    {
        printf("%2d, ", b[i]);
    }
    printf("] = \n");
    printf("[");
    for (int i = 0; i < arraySize; i++)
    {
        printf("%2d, ", c[i]);
    }
    printf("]\n");


    delete[] a;
    delete[] b;

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    CudaBuffer<int> cd_c(size);
    CudaBuffer<int> cd_a(size);
    CudaBuffer<int> cd_b(size);

    if (!cd_c.valid()) return cd_c.error();
    if (!cd_a.valid()) return cd_a.error();
    if (!cd_b.valid()) return cd_b.error();

    int* dev_a = cd_a.get();
    int* dev_b = cd_b.get();
    int* dev_c = cd_c.get();

    // Copy input vectors from host memory to GPU buffers.
    CUDA_CHECK(cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel
    addKernel <<<1, size >>> (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    CUDA_CHECK(cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));

    // Reset device
    CUDA_CHECK(cudaDeviceReset());

    return cudaSuccess;
}
