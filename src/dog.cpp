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

// Convolve image with kernel
void convolve(
    const uint8_t* image, const float* kernel,
    int width, int height, int ksize,
    uint8_t* output) {

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
            output[y*width + x] = clamp(int(sum), 0, 255);
        }
    }
}

// Compute the Difference of Gaussians
// if threshold is set to a negative value, it will not be applied
// otherwise, it shall be at most 1.0f
void computeDoG(
    const uint8_t* input, uint8_t* output, int h, int w,
    float* kernel1, float* kernel2, int kernelSize, float threshold = -1,int numThreads = -1) {

    auto kernel1_2D = generateGaussianKernel2D(kernel1, kernelSize);
    auto kernel2_2D = generateGaussianKernel2D(kernel2, kernelSize);

    vector<uint8_t> blur1(h * w), blur2(h * w);
    convolve(input, kernel1_2D.data(), w, h, kernelSize, blur1.data());
    convolve(input, kernel2_2D.data(), w, h, kernelSize, blur2.data());

    for(int i = 0; i < w * h; ++i){
        output[i] = clamp(255 - 20*(blur2[i] - blur1[i]), 0, 255);
    }

    // apply threshold
    if (threshold < 0) return;
    int z_thr = 255 * threshold;

    for(int i = 0; i < w * h; ++i)
        output[i] = (output[i] >= z_thr) ? 255 : 0;
}
