#ifndef DOG_H
#define DOG_H

#include <vector>
#include <cstdint>

using std::vector;

// Generate a Gaussian kernel
vector<vector<float>> generateGaussianKernel(int size, float sigma);

// Convolve image with kernel
vector<vector<uint8_t>> convolve(
    const vector<vector<uint8_t>> &image,
    const vector<vector<float>> &kernel);

// Compute the Difference of Gaussians
vector<vector<uint8_t>> computeDoG(
    const vector<vector<uint8_t>> &image,
    float sigma1, float sigma2, int kernelSize, float threshold = -1, int numThreads = -1);

#endif // DOG_H