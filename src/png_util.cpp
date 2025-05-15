#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#include "lodepng.h"
#include "png_util.h"

using std::vector;
using std::string;
using std::cout;
using std::cerr;
using std::endl;


// Load PNG and convert to grayscale
vector<vector<uint8_t>> loadPNGGrayscale(const string &filename, unsigned &width, unsigned &height) {
    vector<unsigned char> png, image;
    lodepng::load_file(png, filename);
    unsigned error = lodepng::decode(image, width, height, png);
    if (error) {
        cerr << "Decoder error: " << lodepng_error_text(error) << endl;
        exit(1);
    }

    vector<vector<uint8_t>> gray(height, vector<uint8_t>(width));
    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            int idx = 4 * (y * width + x);
            uint8_t r = image[idx];
            uint8_t g = image[idx + 1];
            uint8_t b = image[idx + 2];
            gray[y][x] = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
    return gray;
}

// Save grayscale image to PNG
void savePNGGrayscale(const string &filename, const vector<vector<uint8_t>> &gray) {
    unsigned height = gray.size();
    unsigned width = gray[0].size();
    vector<unsigned char> image(width * height * 4);

    for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
            int idx = 4 * (y * width + x);
            uint8_t val = gray[y][x];
            image[idx + 0] = val;
            image[idx + 1] = val;
            image[idx + 2] = val;
            image[idx + 3] = 255;
        }
    }

    vector<unsigned char> png;
    unsigned error = lodepng::encode(png, image, width, height);
    if (!error) {
        lodepng::save_file(png, filename);
    } else {
        cerr << "Encoder error: " << lodepng_error_text(error) << endl;
    }
}
