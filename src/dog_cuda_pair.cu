#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>
using std::cerr;
using std::endl;

__constant__ int KSIZE;
__constant__ int WIDTH;
__constant__ int HEIGHT;
__constant__ int THRESHOLD;

uint8_t *d_input, *d_output;
float2 *d_temp, *d_out;
float *d_kernel1, *d_kernel2;

#define clamp(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

__global__ void blur_horizontal(const uint8_t *input, float2 *output, float *d_kernel1, float *d_kernel2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT)
        return;

    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    int half = KSIZE / 2;

    for (int i = -half; i <= half; ++i)
    {
        int ix = clamp(x + i, 0, WIDTH - 1);
        sum.x += input[y * WIDTH + ix] * d_kernel1[i + half];
        sum.y += input[y * WIDTH + ix] * d_kernel2[i + half];
    }

    output[y * WIDTH + x].x = (unsigned char)(sum.x);
    output[y * WIDTH + x].y = (unsigned char)(sum.y);
}

__global__ void blur_vertical(const float2 *input, float2 *output, float *d_kernel1, float *d_kernel2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT)
        return;

    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    int half = KSIZE / 2;

    for (int i = -half; i <= half; ++i)
    {
        int iy = clamp(y + i, 0, HEIGHT - 1);
        sum.x += input[iy * WIDTH + x].x * d_kernel1[i + half];
        sum.y += input[iy * WIDTH + x].y * d_kernel2[i + half];
    }

    output[y * WIDTH + x].x = (unsigned char)(sum.x);
    output[y * WIDTH + x].y = (unsigned char)(sum.y);
}

__global__ void sumScale(const float2 *input, unsigned char *output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT)
        return;

    // unsigned char val = clamp(255 - 20*(int(input2[y * WIDTH + x]) - int(input1[y * WIDTH + x])), 0, 255);
    // output[y * WIDTH + x] = THRESHOLD < 0? val : (val > THRESHOLD ? 255 : 0);
    int pos = y * WIDTH + x;
    float val = input[pos].x - input[pos].y;
    output[pos] = clamp(255 - 20 * val, 0, 255);
}

void initialize(int height, int width, float *kernel1, float *kernel2, int ksize, float threshold = -1)
{
    size_t img_size = width * height;
    int i_threshold = threshold >= 0 ? int(threshold) : -1;

    cudaMemcpyToSymbol(KSIZE, &ksize, sizeof(int));
    cudaMemcpyToSymbol(WIDTH, &width, sizeof(int));
    cudaMemcpyToSymbol(HEIGHT, &height, sizeof(int));
    cudaMemcpyToSymbol(THRESHOLD, &i_threshold, sizeof(int));

    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_temp, sizeof(float2) * img_size);
    cudaMalloc(&d_out, sizeof(float2) * img_size);
    cudaMalloc(&d_output, sizeof(uint8_t) * img_size);
    cudaMalloc(&d_kernel1, sizeof(float2) * ksize);
    cudaMalloc(&d_kernel2, sizeof(float2) * ksize);

    cudaMemcpy(d_kernel1, kernel1, sizeof(float) * ksize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel2, kernel2, sizeof(float) * ksize, cudaMemcpyHostToDevice);
}

void computeDoG(const uint8_t *input, uint8_t *output, int height, int width, int _ = -1)
{
    size_t img_size = width * height;
    cudaMemcpy(d_input, input, img_size, cudaMemcpyHostToDevice);

    const int xBlock = 32, yBlock = 2;
    dim3 block(xBlock, yBlock);
    dim3 grid((width + xBlock - 1) / xBlock, (height + yBlock - 1) / yBlock);

    blur_horizontal<<<grid, block>>>(d_input, d_temp, d_kernel1, d_kernel2);

    blur_vertical<<<grid, block>>>(d_temp, d_out, d_kernel1, d_kernel2);

    // sumScale<<<grid, block>>>(d_out, d_output);

    cudaMemcpy(output, d_output, img_size, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
}

void finalize()
{
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_out);
    cudaFree(d_output);
    cudaFree(d_kernel1);
    cudaFree(d_kernel2);
}
