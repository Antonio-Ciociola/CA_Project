#ifndef DOG_H
#define DOG_H

#include <cstdint>

// Compute the Difference of Gaussians
void computeDoG(
    const uint8_t* image, uint8_t* output, int height, int width,
    float* kernel1, float* kernel2, int kernelSize, float threshold = -1, int numThreads = -1);

#endif // DOG_H