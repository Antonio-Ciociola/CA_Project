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
    const std::vector<uint8_t>& input, std::vector<uint8_t>& output,
    int width, int height, float sigma, int kernelSize);


vector<vector<uint8_t>> computeDoG(
    const vector<vector<uint8_t>> &image,
    float sigma1, float sigma2, int kernelSize, float threshold = -1,int numThreads = -1) {

    int h = image.size(), w = image[0].size();
    vector<uint8_t> input(w * h), blur1(w * h), blur2(w * h);
    vector<vector<uint8_t>> dog(h, vector<uint8_t>(w));

    for(int i = 0; i < h; ++i)
        for(int j = 0; j < w; ++j)
            input[i*w + j] = image[i][j];

    gaussian_blur_cuda(input, blur1, w, h, sigma1, kernelSize);
    gaussian_blur_cuda(input, blur2, w, h, sigma2, kernelSize);

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            dog[y][x] = clamp(255 - 20*(blur1[y*w + x] - blur2[y*w + x]), 0, 255);


    // save the DoG image before thresholding
    // savePNGGrayscale("dog.png", dog);
    
    // apply threshold
    if (threshold < 0) return dog;
    
    int min = 255, max = 0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (dog[y][x] < min) {
                min = dog[y][x];
            }
            if (dog[y][x] > max) {
                max = dog[y][x];
            }
        }
    }
    int z_thr = threshold * (max - min) + min;
    cerr << min << " " << max << " " << threshold << " " << z_thr << endl;

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            dog[y][x] = (dog[y][x] >= z_thr) ? 255 : 0;

    return dog;
}

// nvcc dog_cuda_project/gaussian_blur.cu dog_cuda_project/dog.cpp dog_cuda_project/lodepng.cpp src/main.cpp src/png_util.cpp -o dog_cuda_project/dog_gpu