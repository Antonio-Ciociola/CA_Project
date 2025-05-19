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

int clamp(int value, int minval, int maxval) {
	return (value < minval) ? minval : (value > maxval) ? maxval : value;
}

extern void gaussian_blur_cuda(
    const uint8_t* input, uint8_t* output,
    int width, int height, float sigma, int kernelSize);


extern void gaussian_blur_cuda(const uint8_t *input, uint8_t *output, int width, int height, float *kernel1, float *kernel2, int ksize, int threshold);


// Generate 1D Gaussian kernel
vector<float> generateGaussianKernel1D(int size, float sigma) {
    vector<float> kernel(size);
    int half = size / 2;
    float sum = 0.0f;

    for (int i = -half; i <= half; ++i) {
        float value = exp(-(i * i) / (2 * sigma * sigma));
        kernel[i + half] = value;
        sum += value;
    }

    // Normalize
    for (float &val : kernel)
        val /= sum;

    return kernel;
}

void computeDoG(
    const uint8_t* input, uint8_t* output, int h, int w,
    float sigma1, float sigma2, int kernelSize, float threshold = -1,int numThreads = -1) {

    // Generate Gaussian kernels
    vector<float> kernel1 = generateGaussianKernel1D(kernelSize, sigma1);
    vector<float> kernel2 = generateGaussianKernel1D(kernelSize, sigma2);


    auto time1 = std::chrono::high_resolution_clock::now();
    gaussian_blur_cuda(input, output, w, h, kernel1.data(), kernel2.data(), kernelSize, int(threshold * 255));
    auto time2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = time2 - time1;
    cout << "Elapsed time for Gaussian blur: " << elapsed.count() << " seconds" << endl;
    cout << w << " " << h << " " << kernelSize << " " << sigma1 << " " << sigma2 << endl;
}