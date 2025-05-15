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

// Compute the Difference of Gaussians
vector<vector<uint8_t>> computeDoG(
    const vector<vector<uint8_t>> &image,
    float sigma1, float sigma2, int kernelSize) {

    auto kernel1 = generateGaussianKernel(kernelSize, sigma1);
    auto kernel2 = generateGaussianKernel(kernelSize, sigma2);

    auto blur1 = convolve(image, kernel1);
    auto blur2 = convolve(image, kernel2);
    
    savePNGGrayscale("blur1.png", blur1);
    savePNGGrayscale("blur2.png", blur2);

    int height = image.size();
    int width = image[0].size();
    vector<vector<uint8_t>> dog(height, vector<uint8_t>(width));

    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            dog[y][x] = clamp(255 - 20*(blur1[y][x] - blur2[y][x]), 0, 255);


    // save the DoG image
    savePNGGrayscale("dog.png", dog);


    // apply threshold
    // estimate a good threshold
    // for the DoG image

    // 0.5 * (max - min) + min
    int min = 255;
    int max = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (dog[y][x] < min) {
                min = dog[y][x];
            }
            if (dog[y][x] > max) {
                max = dog[y][x];
            }
        }
    }
    int threshold = 0.1 * (max - min) + min;


    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (dog[y][x] < threshold) {
                dog[y][x] = 0;
            } else {
                dog[y][x] = 255;
            }
        }
    }


    return dog;
}
