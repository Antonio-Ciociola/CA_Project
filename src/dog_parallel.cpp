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
vector<vector<float>> generateGaussianKernel(int size, float sigma) {
    vector<vector<float>> kernel(size, vector<float>(size));
    float sum = 0.0;
    int half = size / 2;

    for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
            float value = exp(-(i * i + j * j) / (2 * sigma * sigma));
            kernel[i + half][j + half] = value;
            sum += value;
        }
    }

    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            kernel[i][j] /= sum;

    return kernel;
}

// Convolve image with kernel
vector<vector<uint8_t>> convolve(
    const vector<vector<uint8_t>> &image,
    const vector<vector<float>> &kernel) {

    int height = image.size();
    int width = image[0].size();
    int ksize = kernel.size();
    int half = ksize / 2;

    vector<vector<uint8_t>> output(height, vector<uint8_t>(width, 0));

    for (int y = half; y < height - half; ++y) {
        for (int x = half; x < width - half; ++x) {
            float sum = 0.0;
            for (int i = 0; i < ksize; ++i) {
                for (int j = 0; j < ksize; ++j) {
                    int iy = y + i - half;
                    int ix = x + j - half;
                    sum += image[iy][ix] * kernel[i][j];
                }
            }
            output[y][x] = clamp(int(sum), 0, 255);
        }
    }
    return output;
}
vector<vector<uint8_t>> computeDoG(
    const vector<vector<uint8_t>> &image,
    float sigma1, float sigma2, int kernelSize, float threshold = -1, int numThreads = -1) { 

    auto kernel1 = generateGaussianKernel(kernelSize, sigma1);
    auto kernel2 = generateGaussianKernel(kernelSize, sigma2);

    // make the difference of kernels
    for (int i = 0; i < kernelSize; ++i) {
        for (int j = 0; j < kernelSize; ++j) {
            kernel1[i][j] -= kernel2[i][j];
        }
    }

    //auto blur1 = convolve(image, kernel1);
    //auto blur2 = convolve(image, kernel2);

    int height = image.size();
    int width = image[0].size();
    vector<vector<uint8_t>> dog(height, vector<uint8_t>(width));

    // Number of threads to use
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }
    std::cout << numThreads << std::endl;
    int min = 255, max = 0;


    auto compute_dog_worker = [&](int startRow, int endRow,
        const vector<vector<uint8_t>> &image,
        const vector<vector<float>> &kernel) {

        int height = image.size();
        int width = image[0].size();
        int ksize = kernel.size();
        int half = ksize / 2;


        for (int y = std::max(half, startRow); y < std::min(height - half, endRow); ++y) {
            for (int x = half; x < width - half; ++x) {
                float sum = 0.0;
                for (int i = 0; i < ksize; ++i) {
                    for (int j = 0; j < ksize; ++j) {
                        int iy = y + i - half;
                        int ix = x + j - half;
                        sum += image[iy][ix] * kernel[i][j];
                    }
                }
                dog[y][x] = clamp(255 - int(sum)*20, 0, 255);

                if (dog[y][x] < min) min = dog[y][x];
                if (dog[y][x] > max) max = dog[y][x];
            }
        }
/*
        for (int y = startRow; y < endRow; ++y) {
            for (int x = 0; x < width; ++x) {
                dog[y][x] = clamp(255 - 20 * (blur1[y][x] - blur2[y][x]), 0, 255);
            }
        } */
    };

    vector<std::thread> threads;
    int rowsPerThread = height / numThreads;
    int extra = height % numThreads;
    int currentRow = 0;
    
    std::cout << rowsPerThread << " " << extra << std::endl;

    for (int i = 0; i < numThreads; ++i) {
        int startRow = currentRow;
        int endRow = startRow + rowsPerThread + (i < extra ? 1 : 0);
        threads.emplace_back(compute_dog_worker, startRow, endRow, 
            image, kernel1);
        currentRow = endRow;
        std::cout << startRow << " " << endRow << std::endl;
    }

    for (auto& t : threads) t.join();

    //return dog;

    if (threshold < 0) return dog;

    std::cout << "min: " << min << " max: " << max << std::endl;
    float range = max - min;
    float th = min + range * threshold;

    // Thresholding
    auto threshold_worker = [&](int startRow, int endRow) {
        for (int y = startRow; y < endRow; ++y) {
            for (int x = 0; x < width; ++x) {
                dog[y][x] = (dog[y][x] > th) ? 255 : 0;
            }
        }
    };

    threads.clear();
    currentRow = 0;
    for (int i = 0; i < numThreads; ++i) {
        int startRow = currentRow;
        int endRow = startRow + rowsPerThread + (i < extra ? 1 : 0);
        threads.emplace_back(threshold_worker, startRow, endRow);
        currentRow = endRow;
    }

    for (auto& t : threads) t.join();

    return dog;
}

