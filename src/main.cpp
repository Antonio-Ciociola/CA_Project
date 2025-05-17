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


// Function to convert 2D uint8_t vector to OpenCV grayscale Mat
cv::Mat vectorToMat(const vector<vector<uint8_t>>& frameData) {
    int rows = frameData.size();
    int cols = frameData[0].size();
    cv::Mat mat(rows, cols, CV_8UC1); // 8-bit single channel

    for (int i = 0; i < rows; ++i) {
        memcpy(mat.ptr(i), frameData[i].data(), cols);
    }

    return mat;
}


int main(int argc, char** argv) {
    unsigned width, height;

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_video.mp4> [output_video.mp4] [sigma1] [sigma2] [threshold] [numThreads]" << endl;
        return 1;
    }

    string inputVideo = argv[1];
    string outputVideo = (argc > 2) ? argv[2] : "output.mp4";

    float sigma1, sigma2, threshold = -1, numThreads = -1;
    if (argc > 3) {
        sigma1 = std::stof(argv[3]);
        sigma2 = (argc > 4) ? std::stof(argv[4]) : 2 * sigma1;
        if (sigma1 > sigma2) std::swap(sigma1, sigma2);
    } else {
        sigma1 = 1.0f;
        sigma2 = 2.0f;
    }
    int kernelSize = int(2 * sigma2) | 1; // Ensure kernel size is odd

    if (argc > 5)
        threshold = std::min(std::stof(argv[5]), 1.0f);

    if (argc > 6) {
        numThreads = std::stoi(argv[6]);
    }

    // Open the input video
    cv::VideoCapture cap(inputVideo);
    if (!cap.isOpened()) {
        cerr << "Error: Could not open input video " << inputVideo << endl;
        return 1;
    }

    // Get video properties
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));

    // Specify a compatible codec for MP4
    int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1'); // Use 'avc1' codec

/*
    // Open the output video
    cv::VideoWriter writer(outputVideo, fourcc, fps, cv::Size(frameWidth, frameHeight), false);
    if (!writer.isOpened()) {
        cerr << "Error: Could not open output video " << outputVideo << endl;
        return 1;
    }
*/

    //string outputFilename = "output.avi";
    // int codec = VideoWriter::fourcc('M', 'J', 'P', 'G'); // Or 'X','V','I','D'
    //double fps = 30.0;
    cv::Size frameSize(frameWidth, frameHeight);

    cv::VideoWriter writer(outputVideo, codec, fps, frameSize, false); // false = grayscale

    if (!writer.isOpened()) {
        cerr << "Error: Could not open the output video file." << endl;
        return -1;
    }

    


    // Process each frame
    cv::Mat frame, grayFrame, dogFrame;
    while (cap.read(frame)) {
        // Convert to grayscale
        std::cout << "A" << std::endl;
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        std::cout << "B" << std::endl;

        // Convert OpenCV Mat to 2D vector<uint8_t> for computeDoG
        std::vector<std::vector<uint8_t>> image(grayFrame.rows, std::vector<uint8_t>(grayFrame.cols));
        for (int i = 0; i < grayFrame.rows; ++i) {
            for (int j = 0; j < grayFrame.cols; ++j) {
                image[i][j] = grayFrame.at<uint8_t>(i, j);
            }
        }
        std::cout << "C" << std::endl;

        // Apply computeDoG
        auto start = std::chrono::high_resolution_clock::now();
        auto dog = computeDoG(image, sigma1, sigma2, kernelSize, threshold, numThreads);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        cout << "Processed frame in: " << elapsed.count() << " seconds" << endl;

        // Convert vector<float> back to OpenCV Mat
        // get size of dog matrix
        int dogRows = dog.size();
        int dogCols = dogRows > 0 ? dog[0].size() : 0;

        // Save the DoG frame as an image
        static int frameCounter = 0;
        std::string outputFile = "dog_frame.png";
        savePNGGrayscale(outputFile, dog);

        // Write each fram
        cv::Mat frame = vectorToMat(dog);
        writer.write(frame);

/*
        dogFrame = cv::Mat(dogRows, dogCols, CV_32F, dog.data());

        std::cout << "D" << std::endl;

        // Normalize and convert to 8-bit for video output
        cv::normalize(dogFrame, dogFrame, 0, 255, cv::NORM_MINMAX);
        std::cout << "E" << std::endl;
        dogFrame.convertTo(dogFrame, CV_8U);

        std::cout << "F" << std::endl;

        // Write the frame to the output video
        writer.write(dogFrame);
*/
        std::cout << "G" << std::endl;
    }

    cap.release();
    writer.release();
    cout << "Video processing complete. Output saved to " << outputVideo << endl;

    return 0;
}