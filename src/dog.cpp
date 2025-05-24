#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::exp;

#define clamp(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

float* temp1;
float* temp2;
float* kernel1;
float* kernel2;
int kernelSize;
float threshold;

vector<float> generateGaussianKernel2D(float* kernel1, int size) {
    // multiply the 1D kernel with itself to create a 2D kernel
    vector<float> kernel(size * size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            kernel[i * size + j] = kernel1[i] * kernel1[j];
        }
    }
    return kernel;
}

void initialize(int height, int width, float* k1, float* k2, int ksize, float th = -1) {
    // Initialize global variables for kernel and threshold
    kernel1 = new float[ksize * ksize];
    kernel2 = new float[ksize * ksize];
    auto kernel1_2D = generateGaussianKernel2D(k1, ksize);
    auto kernel2_2D = generateGaussianKernel2D(k2, ksize);
    std::copy(kernel1_2D.begin(), kernel1_2D.end(), kernel1);
    std::copy(kernel2_2D.begin(), kernel2_2D.end(), kernel2);
    kernelSize = ksize;
    threshold = th;
    temp1 = new float[height * width];
    temp2 = new float[height * width];
}

// Convolve image with kernel
void convolve(
    const uint8_t* image, const float* kernel,
    int width, int height, int ksize,
    float* output) {

    int half = ksize / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0;
            for (int i = 0; i < ksize; ++i) {
                for (int j = 0; j < ksize; ++j) {
                    int iy = clamp(y + i - half, 0, height - 1);
                    int ix = clamp(x + j - half, 0, width - 1);
                    sum += image[iy*width + ix] * kernel[i*ksize + j];
                }
            }
            output[y*width + x] = clamp(sum, 0, 255);
        }
    }
}

// Compute the Difference of Gaussians
// if threshold is set to a negative value, it will not be applied
// otherwise, it shall be at most 1.0f
void computeDoG(const uint8_t* input, uint8_t* output, int h, int w, int _ = -1) {
    convolve(input, kernel1, w, h, kernelSize, temp1);
    convolve(input, kernel2, w, h, kernelSize, temp2);

    for(int i = 0; i < w * h; ++i){
        output[i] = clamp(255 - 20*(temp2[i] - temp1[i]), 0, 255);
    }

    // apply threshold
    if (threshold < 0) return;
    int z_thr = 255 * threshold;

    for(int i = 0; i < w * h; ++i)
        output[i] = (output[i] >= z_thr) ? 255 : 0;
}

void finalize() {
    delete[] temp1;
    delete[] temp2;
}
