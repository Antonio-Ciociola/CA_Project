#ifndef DOG_H
#define DOG_H

#include <vector>
#include <cstdint>
#include <opencv2/opencv.hpp>


using std::vector;

// Compute the Difference of Gaussians
void computeDoG(
    const uint8_t* image, uint8_t* output, int height, int width,
    float sigma1, float sigma2, int kernelSize, float threshold = -1, int numThreads = -1);

#endif // DOG_H