#include <opencv2/opencv.hpp>
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

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_file> [output_file] [sigma1] [sigma2] [threshold] [numThreads]" << endl;
        return 1;
    }

    string inputFile = argv[1];
    string outputFile = (argc > 2) ? argv[2] : "output.png";

    uint8_t* image_data = nullptr;
    int frameHeight = 0;
    int frameWidth = 0;

    // Read raw image
    FILE* file = fopen(inputFile.c_str(), "rb");
    if (!file) {
        cerr << "Error: Could not open input file " << inputFile << endl;
        return 1;
    }

    fread(&frameHeight, sizeof(int), 1, file);
    fread(&frameWidth, sizeof(int), 1, file);

    image_data = new uint8_t[frameHeight * frameWidth];
    fread(image_data, sizeof(uint8_t), frameHeight * frameWidth, file);
    fclose(file);

    // Write encoded image
    Mat outputImage(frameHeight, frameWidth, CV_8UC1, image_data);
    if (!cv::imwrite(outputFile, outputImage)) {
        cerr << "Error: Could not write output image " << outputFile << endl;
        return 1;
    }
    return 0;
}