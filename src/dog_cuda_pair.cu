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
float2 *d_temp;
__constant__ float2 d_kernels[16];

#define clamp(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

__global__ void blur_horizontal(const uint8_t *input, float2 *output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT)
        return;

    float2 sum;
    sum.x = 0.0f;
    sum.y = 0.0f;
    int half = KSIZE / 2;

    __shared__ uint8_t tile[4][80];
    int tx = threadIdx.x;
    int ty = threadIdx.y; // se metti più di 4 esplode fortissimo, sarà divertente

    // load itself (tile is offset by half to avoid negative indices)
    tile[ty][tx + half] = input[y * WIDTH + x];
    // if close to left edge, load left neighbor
    if(tx < half)
        tile[ty][tx] = input[y * WIDTH + (x - half < 0? 0 : x - half)];
    // if close to right edge, load right neighbor
    if(tx >= blockDim.x - half)
        tile[ty][tx + 2 * half] = input[y * WIDTH + (x + half >= WIDTH? WIDTH - 1 : x + half)];

    __syncthreads();

    #pragma unroll
    for(int i = 0; i < KSIZE; ++i){
        sum.x += tile[ty][tx + i] * d_kernels[i].x;
        sum.y += tile[ty][tx + i] * d_kernels[i].y;
    }

    output[y * WIDTH + x].x = sum.x;
    output[y * WIDTH + x].y = sum.y;
}

__global__ void blur_vertical(const float2 *input, uint8_t *output)
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
        sum.x += input[iy * WIDTH + x].x * d_kernels[i + half].x;
        sum.y += input[iy * WIDTH + x].y * d_kernels[i + half].y;
    }

    uint8_t val = clamp(255 - 20 * (sum.y - sum.x), 0, 255);
    output[y * WIDTH + x] = THRESHOLD < 0 ? val : (val > THRESHOLD ? 255 : 0);
}

void initialize(int height, int width, float *kernel1, float *kernel2, int ksize, float threshold = -1, int batchSize = 1)
{
    size_t img_size = width * height;
    int i_threshold = threshold >= 0 ? int(threshold) : -1;

    cudaMemcpyToSymbol(KSIZE, &ksize, sizeof(int));
    cudaMemcpyToSymbol(WIDTH, &width, sizeof(int));
    cudaMemcpyToSymbol(HEIGHT, &height, sizeof(int));
    cudaMemcpyToSymbol(THRESHOLD, &i_threshold, sizeof(int));

    cudaMalloc(&d_input, sizeof(uint8_t) * img_size * batchSize);
    cudaMalloc(&d_temp, sizeof(float2) * img_size);
    cudaMalloc(&d_output, sizeof(uint8_t) * img_size * batchSize);

    float2 kernels[ksize];
    for(int i = 0; i < ksize; ++i){
        kernels[i].x = kernel1[i];
        kernels[i].y = kernel2[i];
    }
    cudaMemcpyToSymbol(d_kernels, kernels, sizeof(float2) * ksize);
}

void computeDoG(const uint8_t *input, uint8_t *output, int height, int width, int _ = -1, int batchSize = 1, int xBlock = 32, int yBlock = 2)
{
    size_t img_size = width * height;

    dim3 block(xBlock, yBlock);
    dim3 grid((width + xBlock - 1) / xBlock, (height + yBlock - 1) / yBlock);

    cudaMemcpy(d_input, input, img_size * batchSize, cudaMemcpyHostToDevice);

    for (int i = 0; i < batchSize; ++i){
        const uint8_t *batch_input = d_input + i * img_size; // Access the correct batch
        uint8_t *batch_output = d_output + i * img_size;     // Access the correct batch

        blur_horizontal<<<grid, block>>>(batch_input, d_temp);
        blur_vertical<<<grid, block>>>(d_temp, batch_output);
    }
    cudaMemcpy(output, d_output, img_size * batchSize, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
}

void finalize()
{
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
}
