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


void computeDoG(
    const uint8_t* input, uint8_t* output, int h, int w,
    float sigma1, float sigma2, int kernelSize, float threshold = -1,int numThreads = -1) {

    vector<uint8_t> blur1(w * h), blur2(w * h);

    gaussian_blur_cuda(input, blur1.data(), w, h, sigma1, kernelSize);
    gaussian_blur_cuda(input, blur2.data(), w, h, sigma2, kernelSize);

    int min = 255, max = 0;
    for(int i = 0; i < w * h; ++i){
        output[i] = clamp(255 - 20*(blur1[i] - blur2[i]), 0, 255);
        if (output[i] < min) min = output[i];
        if (output[i] > max) max = output[i];
    }

    // apply threshold
    if (threshold < 0) return;
    int z_thr = threshold * (max - min) + min;
    cerr << min << " " << max << " " << threshold << " " << z_thr << endl;

    for(int i = 0; i < w * h; ++i)
        output[i] = (output[i] >= z_thr) ? 255 : 0;
}