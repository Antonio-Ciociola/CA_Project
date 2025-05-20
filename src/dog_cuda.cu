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

uint8_t *d_input, *d_temp, *d_output, *d_out1, *d_out2;
float *d_kernel1, *d_kernel2;

#define clamp(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

__global__ void blur_horizontal(const unsigned char *input, unsigned char *output, float *d_kernel){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    float sum = 0.0f;
    int half = KSIZE / 2;

    for (int i = -half; i <= half; ++i){
        int ix = clamp(x + i, 0, WIDTH - 1);
        sum += input[y * WIDTH + ix] * d_kernel[i + half];
    }

    output[y * WIDTH + x] = (unsigned char)(sum);
}

__global__ void blur_vertical(const unsigned char *input, unsigned char *output, float *d_kernel){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    float sum = 0.0f;
    int half = KSIZE / 2;

    for (int i = -half; i <= half; ++i){
        int iy = clamp(y + i, 0, HEIGHT - 1);
        sum += input[iy * WIDTH + x] * d_kernel[i + half];
    }

    output[y * WIDTH + x] = (unsigned char)(sum);
}

__global__ void sumScale(const unsigned char *input1, const unsigned char *input2, unsigned char *output){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    unsigned char val = clamp(255 - 20*(int(input2[y * WIDTH + x]) - int(input1[y * WIDTH + x])), 0, 255);
    output[y * WIDTH + x] = THRESHOLD < 0? val : (val > THRESHOLD ? 255 : 0);
}

void initialize(int height, int width, float* kernel1, float* kernel2, int ksize, float threshold = -1){
    size_t img_size = width * height;
    int i_threshold = threshold >= 0? int(threshold) : -1;

    cudaMemcpyToSymbol(KSIZE, &ksize, sizeof(int));
    cudaMemcpyToSymbol(WIDTH, &width, sizeof(int));
    cudaMemcpyToSymbol(HEIGHT, &height, sizeof(int));
    cudaMemcpyToSymbol(THRESHOLD, &i_threshold, sizeof(int));
    
    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_temp, img_size);
    cudaMalloc(&d_out1, img_size);
    cudaMalloc(&d_out2, img_size);
    cudaMalloc(&d_output, img_size);
    cudaMalloc(&d_kernel1, sizeof(float) * ksize);
    cudaMalloc(&d_kernel2, sizeof(float) * ksize);

    cudaMemcpy(d_kernel1, kernel1, sizeof(float) * ksize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel2, kernel2, sizeof(float) * ksize, cudaMemcpyHostToDevice);
}

void computeDoG(const uint8_t* input, uint8_t* output, int height, int width, float* _3, float* _4, int _5, float _6 = -1, int _7 = -1){
    size_t img_size = width * height;
    cudaMemcpy(d_input, input, img_size, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((width + 32 - 1) / 32, (height + 32 - 1) / 32);

    blur_horizontal<<<grid, block>>>(d_input, d_temp, d_kernel1);
    blur_vertical<<<grid, block>>>(d_temp, d_out1, d_kernel1);

    blur_horizontal<<<grid, block>>>(d_input, d_temp, d_kernel2);
    blur_vertical<<<grid, block>>>(d_temp, d_out2, d_kernel2);

    sumScale<<<grid, block>>>(d_out1, d_out2, d_output);

    cudaMemcpy(output, d_output, img_size, cudaMemcpyDeviceToHost);

    if(cudaGetLastError() != cudaSuccess)
        cerr << "CUDA Error: " << cudaGetErrorString(cudaGetLastError()) << endl;
}

void finalize(){
    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_out1);
    cudaFree(d_out2);
    cudaFree(d_output);
    cudaFree(d_kernel1);
    cudaFree(d_kernel2);
}
