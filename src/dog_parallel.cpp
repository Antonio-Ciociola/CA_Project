#include <thread>
#include <mutex>
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
    // Initialize global variables for kernel and threshold
    kernel1 = k1;
    kernel2 = k2;
    kernelSize = ksize;
    threshold = th;
    temp1 = new uint8_t[height * width];
    temp2 = new uint8_t[height * width];
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
void computeDoG(const uint8_t* input, uint8_t* output, int h, int w, int numThreads = -1) {
    if (numThreads <= 0)
        numThreads = std::thread::hardware_concurrency();

    separableConvolution(input, temp1, kernel1, kernelSize, w, h, numThreads);
    separableConvolution(input, temp2, kernel2, kernelSize, w, h, numThreads);

    // Difference and intensity mapping
    auto dog_worker = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            int val = clamp(255 - 20 * (temp2[i] - temp1[i]), 0, 255);
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

void finalize() {
    delete[] temp1;
    delete[] temp2;
}
