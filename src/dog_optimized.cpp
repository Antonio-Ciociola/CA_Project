#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include "lodepng.h"
#include "png_util.h"

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::exp;

int clamp(int value, int minval, int maxval) {
	return (value < minval) ? minval : (value > maxval) ? maxval : value;
}

// Generate a Gaussian kernel
vector<float> generateGaussianKernel(int size, float sigma) {
    vector<float> kernel;
    kernel.reserve(size * size);
    float sum = 0.0;
    int half = size / 2;

    for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
            float value = exp(-(i * i + j * j) / (2 * sigma * sigma));
            kernel.push_back(value);
            sum += value;
        }
    }
    for(int i = 0; i < size * size; ++i)
        kernel[i] /= sum;
    return kernel;
}

// Convolve image with kernel
void convolve(
    const uint8_t* image, const float* kernel,
    int width, int height, int ksize,
    uint8_t* output) {

    int half = ksize / 2;

    for (int y = half; y < height - half; ++y) {
        for (int x = half; x < width - half; ++x) {
            float sum = 0.0;
            for (int i = 0; i < ksize; ++i) {
                for (int j = 0; j < ksize; ++j) {
                    int iy = y + i - half;
                    int ix = x + j - half;
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
    float sigma1, float sigma2, int kernelSize, float threshold = -1,int numThreads = -1) {

    auto kernel1 = generateGaussianKernel(kernelSize, sigma1);
    auto kernel2 = generateGaussianKernel(kernelSize, sigma2);

    for(int i = 0; i < kernelSize * kernelSize; ++i)
        kernel2[i] -= kernel1[i];

    vector<uint8_t> result(w * h);
    convolve(input, kernel2.data(), w, h, kernelSize, result.data());

    int min = 255, max = 0;
    for(int i = 0; i < w * h; ++i){
        output[i] = clamp(255 - 20 * result[i], 0, 255);
        if (output[i] < min) min = output[i];
        if (output[i] > max) max = output[i];
    }

    // apply threshold
    if (threshold < 0) return;
    int z_thr = threshold * (max - min) + min;
    cerr << min << " " << max << " " << threshold << " " << z_thr << endl;

    for(int i = 0; i < w*h; ++i)
        output[i] = (output[i] >= z_thr) ? 255 : 0;
}