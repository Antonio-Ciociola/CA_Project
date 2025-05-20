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
    float* kernel1, float* kernel2, int kernelSize, float threshold = -1, int numThreads = -1) {

    vector<uint8_t> blur1(h * w), blur2(h * w);

    convolveSeparable(input, kernel1, kernelSize, w, h, blur1.data());
    convolveSeparable(input, kernel2, kernelSize, w, h, blur2.data());

    for (int i = 0; i < w * h; ++i) {
        output[i] = clamp(255 - 20 * (blur2[i] - blur1[i]), 0, 255);
    }

    // Apply threshold if needed
    if (threshold < 0) return;
    int i_threshold = 255 * threshold;

    for (int i = 0; i < w * h; ++i)
        output[i] = (output[i] >= i_threshold) ? 255 : 0;
}
