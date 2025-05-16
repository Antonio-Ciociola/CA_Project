#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdint>

__constant__ float d_kernel[64];

__global__ void blur_horizontal(const unsigned char* input, unsigned char* output, int width, int height, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int half = ksize / 2;

    for (int i = -half; i <= half; ++i) {
        int ix = min(max(x + i, 0), width - 1);
        sum += input[y * width + ix] * d_kernel[i + half];
    }

    output[y * width + x] = (unsigned char)(sum);
}

__global__ void blur_vertical(const unsigned char* input, unsigned char* output, int width, int height, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int half = ksize / 2;

    for (int i = -half; i <= half; ++i) {
        int iy = min(max(y + i, 0), height - 1);
        sum += input[iy * width + x] * d_kernel[i + half];
    }

    output[y * width + x] = (unsigned char)(sum);
}

void gaussian_blur_cuda(const std::vector<uint8_t>& input, std::vector<uint8_t>& output, int width, int height, float sigma) {
    int ksize = int(6 * sigma + 1) | 1;
    int half = ksize / 2;
    std::vector<float> h_kernel(ksize);

    float sum = 0.0f;
    for (int i = -half; i <= half; ++i) {
        float val = expf(-0.5f * i * i / (sigma * sigma));
        h_kernel[i + half] = val;
        sum += val;
    }
    for (int i = 0; i < ksize; ++i)
        h_kernel[i] /= sum;

    cudaMemcpyToSymbol(d_kernel, h_kernel.data(), sizeof(float) * ksize);

    uint8_t *d_input, *d_temp, *d_output;
    size_t img_size = width * height;

    cudaMalloc(&d_input, img_size);
    cudaMalloc(&d_temp, img_size);
    cudaMalloc(&d_output, img_size);

    cudaMemcpy(d_input, input.data(), img_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    blur_horizontal<<<grid, block>>>(d_input, d_temp, width, height, ksize);
    blur_vertical<<<grid, block>>>(d_temp, d_output, width, height, ksize);

    cudaMemcpy(output.data(), d_output, img_size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_temp);
    cudaFree(d_output);
}
