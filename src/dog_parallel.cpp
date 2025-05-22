#include <thread>
#include <mutex>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <queue>
#include <condition_variable>
#include <functional>
#include <chrono>

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::exp;

using std::chrono::high_resolution_clock;
using std::chrono::duration;

#define clamp(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

uint8_t* temp1;
uint8_t* temp2;
float* temp;
float* kernel1;
float* kernel2;
int kernelSize;
float threshold;

void initialize(int height, int width, float* k1, float* k2, int ksize, float th = -1) {
    // Initialize global variables for kernel and threshold
    kernel1 = k1;
    kernel2 = k2;
    kernelSize = ksize;
    threshold = th;
    temp1 = new uint8_t[height * width];
    temp2 = new uint8_t[height * width];
    temp = new float[height * width];
}

void h_worker(const uint8_t* input, float* temp, const float* kernel1D, int half, int width, int startY, int endY) {
    for (int y = startY; y < endY; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int k = -half; k <= half; ++k) {
                int xk = clamp(x + k, 0, width - 1);
                sum += input[y * width + xk] * kernel1D[k + half];
            }
            temp[y * width + x] = sum;
        }
    }
}


void v_worker(const float* temp, uint8_t* output, const float* kernel1D, int half, int width, int height, int startY, int endY) {
    for (int y = startY; y < endY; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int k = -half; k <= half; ++k) {
                int yk = clamp(y + k, 0, height - 1);
                sum += temp[yk * width + x] * kernel1D[k + half];
            }
            output[y * width + x] = clamp(int(sum), 0, 255);
        }
    }
}


void dog_worker(const uint8_t* temp1, const uint8_t* temp2, uint8_t* output, int start, int end) {
    for (int i = start; i < end; ++i) {
        int val = clamp(255 - 20 * (temp2[i] - temp1[i]), 0, 255);
        output[i] = val;
    }
}

void threshold_worker(uint8_t* output, int z_thr, int start, int end) {
    for (int i = start; i < end; ++i) {
        output[i] = (output[i] >= z_thr) ? 255 : 0;
    }
}

class ThreadPool {
public:
    ThreadPool(size_t numThreads) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this]() { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
    }

    void enqueue(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.push(std::move(task));
        }
        condition.notify_one();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop = false;
};

// Separable convolution (horizontal then vertical)
void separableConvolution(
    const uint8_t* input, uint8_t* output,
    const float* kernel1D, int ksize,
    int width, int height, int numThreads) {

    int half = ksize / 2;
    ThreadPool threadPool(numThreads);

    int rowsPerThread = height / numThreads;
    int extra = height % numThreads;
    int y = 0;

    for (int i = 0; i < numThreads; ++i) {
        int startY = y;
        int endY = startY + rowsPerThread + (i < extra ? 1 : 0);
        threadPool.enqueue([=]() { h_worker(input, temp, kernel1D, half, width, startY, endY); });
        y = endY;
    }

    y = 0;
    for (int i = 0; i < numThreads; ++i) {
        int startY = y;
        int endY = startY + rowsPerThread + (i < extra ? 1 : 0);
        threadPool.enqueue([=]() { v_worker(temp, output, kernel1D, half, width, height, startY, endY); });
        y = endY;
    }
}

// Compute the Difference of Gaussians with threshold
void computeDoG(const uint8_t* input, uint8_t* output, int h, int w, int numThreads = -1) {
    if (numThreads <= 0)
        numThreads = std::thread::hardware_concurrency();

    ThreadPool threadPool(numThreads);

    separableConvolution(input, temp1, kernel1, kernelSize, w, h, numThreads);
    separableConvolution(input, temp2, kernel2, kernelSize, w, h, numThreads);

    int pixelsPerThread = (w * h) / numThreads;
    int extra = (w * h) % numThreads;
    int start = 0;

    for (int i = 0; i < numThreads; ++i) {
        int end = start + pixelsPerThread + (i < extra ? 1 : 0);
        threadPool.enqueue([=]() { dog_worker(temp1, temp2, output, start, end); });
        start = end;
    }

    if (threshold < 0) return;
    int z_thr = threshold * 255;

    start = 0;
    for (int i = 0; i < numThreads; ++i) {
        int end = start + pixelsPerThread + (i < extra ? 1 : 0);
        threadPool.enqueue([=]() { threshold_worker(output, z_thr, start, end); });
        start = end;
    }
}

void finalize() {
    delete[] temp1;
    delete[] temp2;
}
