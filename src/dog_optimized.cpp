#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <chrono>

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::exp;

using std::chrono::high_resolution_clock;
using std::chrono::duration;

#define clamp(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

float* temp1;
float* temp2;
float* temp;
float* kernel1;
float* kernel2;
int kernelSize;
float threshold;

void initialize(int height, int width, float* k1, float* k2, int ksize, float th = -1) {
    kernel1 = k1;
    kernel2 = k2;
    kernelSize = ksize;
    threshold = th;
    temp1 = new float[height * width];
    temp2 = new float[height * width];
    temp = new float[height * width];
}

// Perform separable convolution
void convolveSeparable(
    const uint8_t* input, const float* kernel, int ksize,
    int width, int height, float* output) {

    int half = ksize / 2;

    // Horizontal pass
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int k = -half; k <= half; ++k) {
                int idx = clamp(x + k, 0, width - 1);
                sum += input[y * width + idx] * kernel[k + half];
            }
            temp[x * height + y] = sum;
        }
    }
    std::swap(width, height);
    // "Horizontal" pass (on the transposed image)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int k = -half; k <= half; ++k) {
                int idx = clamp(x + k, 0, width - 1);
                sum += temp[y * width + idx] * kernel[k + half];
            }
            output[x * height + y] = clamp(sum, 0, 255);
        }
    }
}

// Compute the Difference of Gaussians using separable convolution
void computeDoG(const uint8_t* input, uint8_t* output, int h, int w, int _ = -1) {
    convolveSeparable(input, kernel1, kernelSize, w, h, temp1);
    convolveSeparable(input, kernel2, kernelSize, w, h, temp2);

    for (int i = 0; i < w * h; ++i) {
        output[i] = clamp(255 - 20 * (temp2[i] - temp1[i]), 0, 255);
    }

    // Apply threshold if needed
    if (threshold < 0) return;
    int i_threshold = 255 * threshold;

    for (int i = 0; i < w * h; ++i)
        output[i] = (output[i] >= i_threshold) ? 255 : 0;
}

void finalize() {
    delete[] temp1;
    delete[] temp2;
    delete[] temp;
}
