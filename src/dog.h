#ifndef DOG_H
#define DOG_H

#include <cstdint>

// initialize memory and constant (across frames) parameters
void initialize(int height, int width, int batchSize, float *kernel1, float *kernel2, int ksize, float threshold = -1);

// Compute the Difference of Gaussians
void computeDoG(const uint8_t *image, uint8_t *output, int batchSize, int h, int w, int numThreads = -1);

// un-initialize
void finalize();

#endif // DOG_H