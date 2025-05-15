#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include "png_util.h"
#include "dog.h"

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::exp;

int main(int argc, char** argv){
    unsigned width, height;

    if(argc < 2){
        cerr << "Usage: " << argv[0] << " <input_image.png> [output_image.png] [sigma1] [sigma2]" << endl;
        return 1;
    }

    string inputFile = argv[1];
    string outputFile;
    if(argc > 2){
        outputFile = argv[2];
    } else {
        outputFile = inputFile.substr(0, inputFile.find_last_of('.')) + "_edges.png";
    }

    auto image = loadPNGGrayscale(inputFile, width, height);

    if(image.empty()){
        cerr << "Error: Could not load image " << inputFile << endl;
        return 1;
    }

    float sigma1, sigma2;
    if(argc > 3){
        sigma1 = std::stof(argv[3]);
        sigma2 = (argc > 4)? std::stof(argv[4]) : 2 * sigma1;
        if(sigma1 > sigma2) std::swap(sigma1, sigma2);
    }
    else {
        sigma1 = 1.0f;
        sigma2 = 2.0f;
    }
    int kernelSize = int(2 * sigma2) | 1; // Ensure kernel size is odd

    auto dog = computeDoG(image, sigma1, sigma2, kernelSize);

    savePNGGrayscale(outputFile, dog);
    return 0;
}