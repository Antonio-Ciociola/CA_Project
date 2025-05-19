#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>
using std::cerr;
using std::endl;

__constant__ float d_kernel[64];
__device__ int KSIZE;
__device__ int WIDTH;
__device__ int HEIGHT;
__device__ int THRESHOLD;

__global__ void blur_horizontal(const unsigned char *input, unsigned char *output, float *d_kernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    float sum = 0.0f;
    int half = KSIZE / 2;

    for (int i = -half; i <= half; ++i)
    {
        int ix = min(max(x + i, 0), WIDTH - 1);
        sum += input[y * WIDTH + ix] * d_kernel[i + half];
    }

    output[y * WIDTH + x] = (unsigned char)(sum);
}

__global__ void blur_vertical(const unsigned char *input, unsigned char *output, float *d_kernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    float sum = 0.0f;
    int half = KSIZE / 2;

    for (int i = -half; i <= half; ++i)
    {
        int iy = min(max(y + i, 0), HEIGHT - 1);
        sum += input[iy * WIDTH + x] * d_kernel[i + half];
    }

    output[y * WIDTH + x] = (unsigned char)(sum);
}

__global__ void sumScale(const unsigned char *input1, const unsigned char *input2, unsigned char *output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    output[y * width + x] = input1[y * width + x] - input2[y * width + x];
    output[y * width + x] = 255 - 20 * output[y * width + x];
    output[y * width + x] = min(max(output[y * width + x], 0), 255);
    output[y * width + x] = output > THRESHOLD ? 255 : 0;
}

void gaussian_blur_cuda(const uint8_t *input, uint8_t *output, int width, int height, float *kernel1, float *kernel2, int ksize, int threshold)
{

    // cudaMemcpyToSymbol(d_kernel, h_kernel.data(), sizeof(float) * ksize);

    uint8_t *d_input, *d_temp, *d_output, *d_kernel1, *d_kernel2, *d_out1, *d_out2;
    size_t img_size = width * height;
    size_t kernel_size = ksize * ksize;

    KSIZE = kernelsSize;
    WIDTH = width;
    HEIGHT = height;
    THRESHOLD = threshold;

    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_temp, img_size);
    cudaMalloc(&d_out1, img_size);
    cudaMalloc(&d_out2, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMalloc(&d_kernel1, sizeof(float) * kernel_size);
    cudaMalloc(&d_kernel2, sizeof(float) * kernel_size);

    cudaMemcpy(d_input, input, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel1, kernel1, sizeof(float) * kernel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel2, kernel2, sizeof(float) * kernel_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    blur_horizontal<<<grid, block>>>(d_input, d_temp, d_kernel1);
    blur_vertical<<<grid, block>>>(d_temp, d_out1, d_kernel1);

    blur_horizontal<<<grid, block>>>(d_input, d_temp, d_kernel2);
    blur_vertical<<<grid, block>>>(d_temp, d_out2, d_kernel2);

    sumScale<<<grid, block>>>(d_out1, d_out2, d_output);

    cudaMemcpy(output, d_output, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
    cudaFree(d_kernel1);
    cudaFree(d_kernel2);

    /*
    if(cudaGetLastError() != cudaSuccess)
        cerr << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << endl;
    */
}
