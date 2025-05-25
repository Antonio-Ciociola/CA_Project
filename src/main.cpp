#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include <chrono>
#include "png_util.h"
#include "dog.h"

using std::cerr;
using std::cout;
using std::endl;
using std::exp;
using std::max;
using std::min;
using std::stof;
using std::stoi;
using std::string;
using std::swap;
using std::vector;

using cv::Mat;
using cv::Size;
using cv::VideoCapture;
using cv::VideoWriter;

using std::chrono::duration;
using std::chrono::high_resolution_clock;

void conditionalPrint(bool flag, const std::string &message)
{
    if (flag)
    {
        cout << message << endl;
    }
}

// Generate 1D Gaussian kernel
vector<float> generateGaussianKernel1D(int size, float sigma)
{
    vector<float> kernel(size);
    int half = size / 2;
    float sum = 0.0f;

    for (int i = -half; i <= half; ++i)
    {
        float value = exp(-(i * i) / (2 * sigma * sigma));
        kernel[i + half] = value;
        sum += value;
    }

    // Normalize
    for (float &val : kernel)
        val /= sum;

    return kernel;
}

int main(int argc, char **argv)
{
    unsigned width, height;
    cout << std::fixed << std::setprecision(6);
    cerr << std::fixed << std::setprecision(6);

    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <input_file> [output_file] [sigma1] [sigma2] [threshold] [numThreads]" << endl;
        return 1;
    }

    string inputFile = argv[1];
    string outputFile = (argc > 2) ? argv[2] : "output.png";

    float sigma1 = 1.0f, sigma2 = 2.0f, threshold = -1, numThreads = -1, printDebug = 1;
    if (argc > 3)
    {
        sigma1 = stof(argv[3]);
        sigma2 = (argc > 4) ? stof(argv[4]) : 2 * sigma1;
        if (sigma1 > sigma2)
            swap(sigma1, sigma2);
    }

    // Ensure kernel size is odd and proportional to sigma
    int kernelSize = int(2 * sigma2) | 1;
    int batchSize = 1; // Default batch size for image processing

    if (argc > 5)
    {
        threshold = min(stof(argv[5]), 1.0f);
    }

    if (argc > 6)
    {
        numThreads = stoi(argv[6]);
    }

    if (argc > 7)
    {
        printDebug = stoi(argv[7]);
    }

    if (argc > 8)
    {
        batchSize = stoi(argv[8]);
    }

    // Check if input is an image or video
    bool isImage = inputFile.substr(inputFile.find_last_of(".") + 1) != "mp4";

    int frameHeight = 0;
    int frameWidth = 0;
    int frameSize = 0;

    if (isImage)
    {
        // Process image

        // Generate Gaussian kernels
        vector<float> kernel1_vec = generateGaussianKernel1D(kernelSize, sigma1);
        float *kernel1 = kernel1_vec.data();
        vector<float> kernel2_vec = generateGaussianKernel1D(kernelSize, sigma2);
        float *kernel2 = kernel2_vec.data();

        auto read_start_img = high_resolution_clock::now();

        uint8_t *image_data = nullptr;

        if (inputFile.find(".raw") != string::npos)
        {
            // Read raw image
            FILE *file = fopen(inputFile.c_str(), "rb");
            if (!file)
            {
                cerr << "Error: Could not open input file " << inputFile << endl;
                return 1;
            }

            fread(&frameHeight, sizeof(int), 1, file);
            fread(&frameWidth, sizeof(int), 1, file);

            frameSize = frameHeight * frameWidth;

            image_data = new uint8_t[frameSize];
            fread(image_data, sizeof(uint8_t), frameSize, file);
            fclose(file);
        }
        else
        {
            Mat image = cv::imread(inputFile, cv::IMREAD_GRAYSCALE);
            if (image.empty())
            {
                cerr << "Error: Could not open input image " << inputFile << endl;
                return 1;
            }

            frameHeight = image.rows;
            frameWidth = image.cols;
            frameSize = frameHeight * frameWidth;
            image_data = image.data;
        }

        // Allocate memory for DoG output
        uint8_t *dog = new uint8_t[frameSize];

        initialize(frameHeight, frameWidth, batchSize, kernel1, kernel2, kernelSize, threshold);

        auto read_end_img = high_resolution_clock::now();

        // // fwrite the image
        // FILE* file = fopen("raw_image.raw", "wb");
        // if (file) {
        //     fwrite(&frameHeight, sizeof(int), 1, file);
        //     fwrite(&frameWidth, sizeof(int), 1, file);
        //     fwrite(image.data, sizeof(uint8_t), frameWidth * frameHeight, file);
        //     fclose(file);
        // }

        // Apply computeDoG
        computeDoG(image_data, dog, batchSize, frameHeight, frameWidth, numThreads);

        auto computeDoG_end_img = high_resolution_clock::now();

        if (inputFile.find(".raw") != string::npos)
        {
            // Save the result as a raw image
            FILE *file = fopen(outputFile.c_str(), "wb");
            if (!file)
            {
                cerr << "Error: Could not write output file " << outputFile << endl;
                delete[] dog;
                return 1;
            }

            fwrite(&frameHeight, sizeof(int), 1, file);
            fwrite(&frameWidth, sizeof(int), 1, file);
            fwrite(dog, sizeof(uint8_t), frameWidth * frameHeight, file);
            fclose(file);
            delete[] image_data;
        }
        else
        {
            // Save the result
            Mat outputImage(frameHeight, frameWidth, CV_8UC1, dog);
            if (!cv::imwrite(outputFile, outputImage))
            {
                cerr << "Error: Could not write output image " << outputFile << endl;
                delete[] dog;
                finalize();
                return 1;
            }
        }

        auto write_end_img = high_resolution_clock::now();

        // Clean up
        delete[] dog;
        finalize();

        if (printDebug)
        {
            duration<double> read_elapsed = read_end_img - read_start_img;
            duration<double> dog_elapsed = computeDoG_end_img - read_end_img;
            duration<double> write_elapsed = write_end_img - computeDoG_end_img;

            cout << "Read\tDoG\tWrite" << endl;
            cout << read_elapsed.count() << "\t" << dog_elapsed.count() << "\t" << write_elapsed.count() << endl;
        }

        return 0;
    }

    string inputVideo = argv[1];
    string outputVideo = (argc > 2) ? argv[2] : "output.mp4";

    // Open the input video
    VideoCapture cap(inputVideo);
    if (!cap.isOpened())
    {
        cerr << "Error: Could not open input video " << inputVideo << endl;
        return 1;
    }

    // Get video properties
    frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    frameSize = frameHeight * frameWidth;
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    // Specify a compatible codec for MP4 (avc1)
    int codec = VideoWriter::fourcc('a', 'v', 'c', '1');

    // Create the output video writer
    Size frameSiz(frameWidth, frameHeight);
    VideoWriter writer(outputVideo, codec, fps, frameSiz, false); // false = grayscale

    if (!writer.isOpened())
    {
        cerr << "Error: Could not open the output video file." << endl;
        return -1;
    }

    // Allocate memory for the frames and DoG output
    Mat frame, grayFrame;
    uint8_t *dog = new uint8_t[frameSize * batchSize];

    // Initialize timing variables
    duration<double> total_read_elapsed(0);
    duration<double> total_gray_elapsed(0);
    duration<double> total_dog_elapsed(0);
    duration<double> total_writer_elapsed(0);
    duration<double> total_total_elapsed(0);

    // Generate Gaussian kernels
    vector<float> kernel1_vec = generateGaussianKernel1D(kernelSize, sigma1);
    float *kernel1 = kernel1_vec.data();
    vector<float> kernel2_vec = generateGaussianKernel1D(kernelSize, sigma2);
    float *kernel2 = kernel2_vec.data();

    initialize(frameHeight, frameWidth, batchSize, kernel1, kernel2, kernelSize, threshold);

    // Process each frame and measure time for each step
    auto read_start = high_resolution_clock::now();

    uint8_t *frame_test = new uint8_t[frameWidth * frameHeight];
    while (cap.read(frame))
    {

        uint8_t *data = static_cast<uint8_t *>(malloc(frameHeight * frameWidth * batchSize * sizeof(uint8_t)));

        int numFrames = 0;
        do
        {
            // Convert to grayscale
            cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
            memcpy(data + numFrames * frameSize, grayFrame.data, frameSize * sizeof(uint8_t));
            numFrames++;
        } while (numFrames < batchSize && cap.read(frame));

        auto read_end = high_resolution_clock::now();
        auto grayscale_end = high_resolution_clock::now();

        // Apply computeDoG
        // cout<< "Processing " << numFrames << " frames..." << endl;
        computeDoG(data, dog, numFrames, frameHeight, frameWidth, numThreads);
        auto computeDoG_end = high_resolution_clock::now();

        for (int i = 0; i < numFrames; ++i)
        {
            // Write each frame
            uint8_t *frame_data = dog + i * frameSize;
            cv::Mat frame = cv::Mat(frameHeight, frameWidth, CV_8UC1, frame_data);
            writer.write(frame);
        }

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

        if (printDebug)
            cout << read_elapsed.count() << "\t" << gray_elapsed.count() << "\t" << dog_elapsed.count() << "\t" << writer_elapsed.count() << "\t" << total_elapsed.count() << endl;

        // Reset read start for the next frame
        read_start = high_resolution_clock::now();
    }

    if (printDebug)
    {
        cout << "Read\tGrayscale\tDoG\tWriter\tTotal" << endl;
        cout << total_read_elapsed.count() << "\t" << total_gray_elapsed.count() << "\t" << total_dog_elapsed.count() << "\t" << total_writer_elapsed.count() << "\t" << total_total_elapsed.count() << endl;
    }

    // Clean up and release resources
    finalize();
    delete[] dog;
    cap.release();
    writer.release();

    return 0;
}