#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <chrono>
#include "png_util.h"
#include "dog.h"

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::exp;
using std::min;
using std::max;
using std::stof;
using std::stoi;
using std::swap;

using cv::VideoCapture;
using cv::VideoWriter;
using cv::Mat;
using cv::Size;

using std::chrono::high_resolution_clock;
using std::chrono::duration;

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

int main(int argc, char** argv) {
    unsigned width, height;
    cout << std::fixed << std::setprecision(6);
    cerr << std::fixed << std::setprecision(6);

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_video.mp4> [output_video.mp4] [sigma1] [sigma2] [threshold] [numThreads]" << endl;
        return 1;
    }

    string inputVideo = argv[1];
    string outputVideo = (argc > 2) ? argv[2] : "output.mp4";

    float sigma1 = 1.0f, sigma2 = 2.0f, threshold = -1, numThreads = -1;
    if (argc > 3) {
        sigma1 = stof(argv[3]);
        sigma2 = (argc > 4) ? stof(argv[4]) : 2 * sigma1;
        if (sigma1 > sigma2) swap(sigma1, sigma2);
    }

    // Ensure kernel size is odd and proportional to sigma
    int kernelSize = int(2 * sigma2) | 1;

    if (argc > 5) {
        threshold = min(stof(argv[5]), 1.0f);
    }

    if (argc > 6) {
        numThreads = stoi(argv[6]);
    }

    // Open the input video
    VideoCapture cap(inputVideo);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open input video " << inputVideo << endl;
        return 1;
    }

    // Get video properties
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    // Specify a compatible codec for MP4 (avc1)
    int codec = VideoWriter::fourcc('a', 'v', 'c', '1');

    // Create the output video writer
    Size frameSize(frameWidth, frameHeight);
    VideoWriter writer(outputVideo, codec, fps, frameSize, false); // false = grayscale

    if (!writer.isOpened()) {
        cerr << "Error: Could not open the output video file." << endl;
        return -1;
    }

    // Allocate memory for the frames and DoG output
    Mat frame, grayFrame;
    uint8_t* dog = new uint8_t[frameWidth * frameHeight];

    // Initialize timing variables
    duration<double> total_read_elapsed(0);
    duration<double> total_gray_elapsed(0);
    duration<double> total_dog_elapsed(0);
    duration<double> total_writer_elapsed(0);
    duration<double> total_total_elapsed(0);

    // Generate Gaussian kernels
    vector<float> kernel1_vec = generateGaussianKernel1D(kernelSize, sigma1);
    float* kernel1 = kernel1_vec.data();
    vector<float> kernel2_vec = generateGaussianKernel1D(kernelSize, sigma2);
    float* kernel2 = kernel2_vec.data();

    initialize(frameHeight, frameWidth, kernel1, kernel2, kernelSize, threshold);

    // Process each frame and measure time for each step
    auto read_start = high_resolution_clock::now();
    while (cap.read(frame)) {
        auto read_end = high_resolution_clock::now();
        
        // Convert to grayscale
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
        auto grayscale_end = high_resolution_clock::now();

        // Apply computeDoG
        computeDoG(grayFrame.data, dog, frameHeight, frameWidth, numThreads);
        auto computeDoG_end = high_resolution_clock::now();

        // Write each frame
        cv::Mat frame = cv::Mat(frameHeight, frameWidth, CV_8UC1, dog);
        writer.write(frame);

        auto frame_end = high_resolution_clock::now();

        // Calculate elapsed time for each step and accumulate
        duration<double> read_elapsed = read_end - read_start;
        duration<double> gray_elapsed = grayscale_end - read_end;
        duration<double> dog_elapsed = computeDoG_end - grayscale_end;
        duration<double> writer_elapsed = frame_end - computeDoG_end;
        duration<double> total_elapsed = frame_end - read_start;

        total_read_elapsed += read_elapsed;
        total_gray_elapsed += gray_elapsed;
        total_dog_elapsed += dog_elapsed;
        total_writer_elapsed += writer_elapsed;
        total_total_elapsed += total_elapsed;

        cout << read_elapsed.count() << "\t" << gray_elapsed.count() << "\t" << dog_elapsed.count() << "\t" << writer_elapsed.count() << "\t" << total_elapsed.count() << endl;

        // Reset read start for the next frame
        read_start = high_resolution_clock::now(); 
    }
    cout << "Read\tGrayscale\tDoG\tWriter\tTotal" << endl;
    cout << total_read_elapsed.count() << "\t" << total_gray_elapsed.count() << "\t" << total_dog_elapsed.count() << "\t" << total_writer_elapsed.count() << "\t" << total_total_elapsed.count() << endl;

    // Clean up and release resources
    finalize();
    delete[] dog;
    cap.release();
    writer.release();

    return 0;
}