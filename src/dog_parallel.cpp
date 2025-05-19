#include <thread>
#include <mutex>
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

// Separable convolution (horizontal then vertical)
void separableConvolution(
    const uint8_t* input, uint8_t* output,
    const float* kernel1D, int ksize,
    int width, int height, int numThreads) {

    int half = ksize / 2;
    vector<float> temp(width * height, 0.0f);

    // Horizontal pass
    auto h_worker = [&](int startY, int endY) {
        for (int y = startY; y < endY; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int k = -half; k <= half; ++k) {
                    int xk = clamp(x + k, 0, width - 1);
                    sum += input[y * width + xk] * kernel1D[k + half];
                }
                temp[y * width + x] = sum;
            }
        }
    };

    vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    int extra = height % numThreads;
    int y = 0;
    for (int i = 0; i < numThreads; ++i) {
        int startY = y;
        int endY = startY + rowsPerThread + (i < extra ? 1 : 0);
        threads.emplace_back(h_worker, startY, endY);
        y = endY;
    }
    for (auto& t : threads) t.join();
    threads.clear();

    // Vertical pass
    auto v_worker = [&](int startY, int endY) {
        for (int y = startY; y < endY; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                for (int k = -half; k <= half; ++k) {
                    int yk = clamp(y + k, 0, height - 1);
                    sum += temp[yk * width + x] * kernel1D[k + half];
                }
                output[y * width + x] = clamp(int(sum), 0, 255);
            }
        }
    };

    y = 0;
    for (int i = 0; i < numThreads; ++i) {
        int startY = y;
        int endY = startY + rowsPerThread + (i < extra ? 1 : 0);
        threads.emplace_back(v_worker, startY, endY);
        y = endY;
    }
    for (auto& t : threads) t.join();
}

// Compute the Difference of Gaussians with threshold
void computeDoG(
    const uint8_t* input, uint8_t* output, int h, int w,
    float* kernel1, float* kernel2, int kernelSize,
    float threshold = -1, int numThreads = -1) {

    if (numThreads <= 0)
        numThreads = std::thread::hardware_concurrency();

    vector<uint8_t> blur1(h * w), blur2(h * w);

    separableConvolution(input, blur1.data(), kernel1, kernelSize, w, h, numThreads);
    separableConvolution(input, blur2.data(), kernel2, kernelSize, w, h, numThreads);

    // Difference and intensity mapping
    auto dog_worker = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            int val = clamp(255 - 20 * (blur2[i] - blur1[i]), 0, 255);
            output[i] = val;
        }
    };

    vector<std::thread> threads;
    int pixelsPerThread = (w * h) / numThreads;
    int extra = (w * h) % numThreads;
    int start = 0;

    for (int i = 0; i < numThreads; ++i) {
        int end = start + pixelsPerThread + (i < extra ? 1 : 0);
        threads.emplace_back(dog_worker, start, end);
        start = end;
    }

    for (auto& t : threads) t.join();
    threads.clear();


    if (threshold < 0) return;
    int z_thr = threshold * 255;

    // Thresholding
    auto threshold_worker = [&](int start, int end) {
        for (int i = start; i < end; ++i)
            output[i] = (output[i] >= z_thr) ? 255 : 0;
    };

    start = 0;
    for (int i = 0; i < numThreads; ++i) {
        int end = start + pixelsPerThread + (i < extra ? 1 : 0);
        threads.emplace_back(threshold_worker, start, end);
        start = end;
    }

    for (auto& t : threads) t.join();
}
