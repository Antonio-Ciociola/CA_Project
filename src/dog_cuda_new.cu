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

__constant__ int tile_width;
__constant__ int tile_height;

uint8_t *d_input, *d_output;
float *d_temp, *d_out1, *d_out2;
float *d_kernel1, *d_kernel2;

#define clamp(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

__global__ void blur_horizontal(const unsigned char *input, float *output, float *d_kernel)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    if (x >= WIDTH || y >= HEIGHT)
        return;

    extern __shared__ unsigned char tile[];

    int half = KSIZE / 2;
    unsigned char *tile_p = &tile[(ty + half) * tile_width + tx + half];

    int left = clamp(x - half, 0, WIDTH - 1);
    int right = clamp(x + half, 0, WIDTH - 1);

    tile_p[0] = input[y * WIDTH + x];
    tile_p[+half] = input[y * WIDTH + right];
    tile_p[-half] = input[y * WIDTH + left];
    __syncthreads();

    float sum = 0.0f;

    for (int i = -half; i <= half; ++i)
    {
        sum += tile_p[i] * d_kernel[i + half];
    }

    output[y * WIDTH + x] = sum;
}

__global__ void blur_vertical(const float *input, float *output, float *d_kernel)
{

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT)
        return;

    extern __shared__ float f_tile[];

    int half = KSIZE / 2;

    float *tile_p = &f_tile[(ty + half) * tile_width + tx + half];

    int left = clamp(y - half, 0, HEIGHT - 1);
    int right = clamp(y + half, 0, HEIGHT - 1);

    tile_p[0] = input[y * WIDTH + x];
    tile_p[+half * tile_width] = input[right * WIDTH + x];
    tile_p[-half * tile_width] = input[left * WIDTH + x];

    __syncthreads();

    float sum = 0.0f;

    for (int i = -half; i <= half; ++i)
    {
        sum += tile_p[i * tile_width] * d_kernel[i + half];
    }

    output[y * WIDTH + x] = sum;
}

__global__ void sumScale(const float *input1, const float *input2, unsigned char *output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT)
        return;

    unsigned char val = clamp(255 - 20 * (input2[y * WIDTH + x] - input1[y * WIDTH + x]), 0, 255);
    output[y * WIDTH + x] = THRESHOLD < 0 ? val : (val > THRESHOLD ? 255 : 0);
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
    cudaMalloc(&d_temp, sizeof(float) * img_size);
    cudaMalloc(&d_out1, sizeof(float) * img_size);
    cudaMalloc(&d_out2, sizeof(float) * img_size);
    cudaMalloc(&d_output, img_size);
    cudaMalloc(&d_kernel1, sizeof(float) * ksize);
    cudaMalloc(&d_kernel2, sizeof(float) * ksize);

    cudaMemcpy(d_kernel1, kernel1, sizeof(float) * ksize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel2, kernel2, sizeof(float) * ksize, cudaMemcpyHostToDevice);
}

void computeDoG(const uint8_t *input, uint8_t *output, int height, int width, int _ = -1)
{
    size_t img_size = width * height;
    cudaMemcpy(d_input, input, img_size, cudaMemcpyHostToDevice);

    const int xBlock = 32, yBlock = 32;

    dim3 block(xBlock, yBlock);
    dim3 grid((width + 32 - 1) / 32, (height + 32 - 1) / 32);

    int sharedWidth = block.x + KSIZE + 30;
    int sharedHeight = block.y + KSIZE + 30;
    int sharedMemSize = sharedWidth * sharedHeight *  (sizeof(unsigned char) + sizeof(float));

    cudaMemcpyToSymbol(tile_width, &sharedWidth, sizeof(int));
    cudaMemcpyToSymbol(tile_height, &sharedHeight, sizeof(int));

    blur_horizontal<<<grid, block, sharedMemSize>>>(d_input, d_temp, d_kernel1);
    blur_vertical<<<grid, block, sharedMemSize>>>(d_temp, d_out1, d_kernel1);

    blur_horizontal<<<grid, block, sharedMemSize>>>(d_input, d_temp, d_kernel2);
    blur_vertical<<<grid, block, sharedMemSize>>>(d_temp, d_out2, d_kernel2);

    sumScale<<<grid, block>>>(d_out1, d_out2, d_output);

    cudaMemcpy(output, d_output, img_size, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
}

void finalize()
{
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_out1);
    cudaFree(d_out2);
    cudaFree(d_output);
    cudaFree(d_kernel1);
    cudaFree(d_kernel2);
}
