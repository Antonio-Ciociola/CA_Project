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

uint8_t* temp1;
uint8_t* temp2;
float* kernel1;
float* kernel2;
int kernelSize;
float threshold;

void initialize(int height, int width, float* k1, float* k2, int ksize, float th = -1) {
    kernel1 = k1;
    kernel2 = k2;
    kernelSize = ksize;
    threshold = th;
    temp1 = new uint8_t[height * width];
    temp2 = new uint8_t[height * width];
}

// Perform separable convolution
void convolveSeparable(
    const uint8_t* input, const float* kernel, int ksize,
    int width, int height, uint8_t* output) {

    int half = ksize / 2;
    vector<float> temp(width * height, 0.0f);

    // Horizontal pass
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int k = -half; k <= half; ++k) {
                int idx = clamp(x + k, 0, width - 1);
                sum += input[y * width + idx] * kernel[k + half];
            }
            temp[y * width + x] = sum;
        }
    }

    // Vertical pass
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int k = -half; k <= half; ++k) {
                int idy = clamp(y + k, 0, height - 1);
                sum += temp[idy * width + x] * kernel[k + half];
            }
            output[y * width + x] = clamp(int(sum), 0, 255);
        }
    }
}

// Compute the Difference of Gaussians using separable convolution
void computeDoG(
    const uint8_t* input, uint8_t* output, int h, int w,
    float* _3, float* _4, int _5, float _6 = -1, int _7 = -1) {

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
}
