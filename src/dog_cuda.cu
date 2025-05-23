#include <cuda_fp16.h>
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
__constant__ float c_kernel1[32], c_kernel2[32];

uint8_t *d_input, *d_output;
float *d_temp, *d_out1, *d_out2;

#define clamp(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

__global__ void blur_horizontal(const unsigned char *input, float *output){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    __half sum = 0.0f;
    int half = KSIZE / 2;

    for (int i = -half; i <= half; ++i){
        int ix = clamp(x + i, 0, WIDTH - 1);
        sum += input[y * WIDTH + ix] * c_kernel1[i + half];
    }

    output[y * WIDTH + x] = sum;
}
__global__ void blur_horizontal2(const unsigned char *input, float *output){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    __half sum = 0.0f;
    int half = KSIZE / 2;

    for (int i = -half; i <= half; ++i){
        int ix = clamp(x + i, 0, WIDTH - 1);
        sum += input[y * WIDTH + ix] * c_kernel2[i + half];
    }

    output[y * WIDTH + x] = sum;
}

__global__ void blur_vertical(const float *input, float *output){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    float sum = 0.0f;
    int half = KSIZE / 2;

    for (int i = -half; i <= half; ++i){
        int iy = clamp(y + i, 0, HEIGHT - 1);
        sum += input[iy * WIDTH + x] * c_kernel1[i + half];
    }

    output[y * WIDTH + x] = sum;
}

__global__ void blur_vertical2(const float *input, float *output){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    float sum = 0.0f;
    int half = KSIZE / 2;

    for (int i = -half; i <= half; ++i){
        int iy = clamp(y + i, 0, HEIGHT - 1);
        sum += input[iy * WIDTH + x] * c_kernel2[i + half];
    }

    output[y * WIDTH + x] = sum;
}

__global__ void sumScale(const float *input1, const float *input2, unsigned char *output){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    unsigned char val = clamp(255 - 20*(input2[y * WIDTH + x] - input1[y * WIDTH + x]), 0, 255);
    output[y * WIDTH + x] = THRESHOLD < 0? val : (val > THRESHOLD ? 255 : 0);
}

void initialize(int height, int width, float* kernel1, float* kernel2, int ksize, float threshold = -1){
    size_t img_size = width * height;
    int i_threshold = threshold >= 0? int(threshold) : -1;

    cudaMemcpyToSymbol(KSIZE, &ksize, sizeof(int));
    cudaMemcpyToSymbol(WIDTH, &width, sizeof(int));
    cudaMemcpyToSymbol(HEIGHT, &height, sizeof(int));
    cudaMemcpyToSymbol(THRESHOLD, &i_threshold, sizeof(int));
    cudaMemcpyToSymbol(c_kernel1, kernel1, sizeof(float) * ksize);
    cudaMemcpyToSymbol(c_kernel2, kernel2, sizeof(float) * ksize);
    
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_temp, sizeof(float) * img_size);
    cudaMalloc(&d_out1, sizeof(float) * img_size);
    cudaMalloc(&d_out2, sizeof(float) * img_size);
    cudaMalloc(&d_output, img_size);
}

void computeDoG(const uint8_t* input, uint8_t* output, int height, int width, int _ = -1){
    size_t img_size = width * height;
    cudaMemcpy(d_input, input, img_size, cudaMemcpyHostToDevice);


    int xth = 32, yth = 32;

    dim3 block(xth, yth);
    dim3 grid((width + xth - 1) / xth, (height + yth - 1) / yth);

    blur_horizontal<<<grid, block>>>(d_input, d_temp);
    blur_vertical<<<grid, block>>>(d_temp, d_out1);

    blur_horizontal2<<<grid, block>>>(d_input, d_temp);
    blur_vertical2<<<grid, block>>>(d_temp, d_out2);

    sumScale<<<grid, block>>>(d_out1, d_out2, d_output);

    cudaMemcpy(output, d_output, img_size, cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
}

void finalize(){
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_out1);
    cudaFree(d_out2);
    cudaFree(d_output);
}
