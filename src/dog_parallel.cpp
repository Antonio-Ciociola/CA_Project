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

    // Number of threads to use
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }
    std::cout << numThreads << std::endl;
    
    auto compute_dog_worker = [](int startRow, int endRow, int width, int height,
        const uint8_t* image, uint8_t* output, const float* kernel, int ksize) {

        int half = ksize / 2;

        for (int y = std::max(half, startRow); y < std::min(height - half, endRow); ++y) {
            for (int x = half; x < width - half; ++x) {
                float sum = 0.0;
                for (int i = 0; i < ksize; ++i) {
                    for (int j = 0; j < ksize; ++j) {
                        int iy = y + i - half;
                        int ix = x + j - half;
                        sum += image[iy*width + ix] * kernel[i * ksize + j];
                    }
                }
                output[y*width + x] = clamp(255 - int(sum)*20, 0, 255);
            }
        }
    };

    vector<std::thread> threads;
    int rowsPerThread = h / numThreads;
    int extra = h % numThreads;
    int currentRow = 0;
    
    std::cout << rowsPerThread << " " << extra << std::endl;

    for (int i = 0; i < numThreads; ++i) {
        int startRow = currentRow;
        int endRow = startRow + rowsPerThread + (i < extra ? 1 : 0);
        threads.emplace_back([=](){
            compute_dog_worker(startRow, endRow, w, h, input, output, kernel2.data(), kernelSize);
        });
        currentRow = endRow;
        std::cout << startRow << " " << endRow << std::endl;
    }

    for (auto& t : threads) t.join();

    if (threshold < 0) return;

    float th = 256 * threshold;

    // Thresholding
    auto threshold_worker = [&](int start, int end) {
        for(int i = start; i < end; ++i)
            output[i] = (output[i] >= th) ? 255 : 0;
    };

    threads.clear();
    currentRow = 0;
    for (int i = 0; i < numThreads; ++i) {
        int startRow = currentRow;
        int endRow = startRow + rowsPerThread + (i < extra ? 1 : 0);
        threads.emplace_back(threshold_worker, startRow * w, endRow * w);
        currentRow = endRow;
    }

    for (auto& t : threads) t.join();
}

