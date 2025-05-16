#include "lodepng.h"
#include <iostream>
#include <vector>
#include <cmath>

int clamp(int value, int minval, int maxval) {
	return (value < minval) ? minval : (value > maxval) ? maxval : value;
}

extern void gaussian_blur_cuda(
    const std::vector<uint8_t>& input, std::vector<uint8_t>& output,
    int width, int height, float sigma);

// Grayscale loading from PNG
std::vector<uint8_t> loadGrayscale(const std::string& filename, unsigned& w, unsigned& h) {
    std::vector<unsigned char> png, image;
    lodepng::load_file(png, filename);
    unsigned error = lodepng::decode(image, w, h, png);
    if (error) {
        std::cerr << "Decode error: " << lodepng_error_text(error) << "\n";
        exit(1);
    }

    std::vector<uint8_t> gray(w * h);
    for (unsigned y = 0; y < h; ++y)
        for (unsigned x = 0; x < w; ++x) {
            int idx = 4 * (y * w + x);
            uint8_t r = image[idx];
            uint8_t g = image[idx + 1];
            uint8_t b = image[idx + 2];
            gray[y * w + x] = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
        }
    return gray;
}

void saveGrayscale(const std::string& filename, const std::vector<uint8_t>& img, unsigned w, unsigned h) {
    std::vector<unsigned char> image(w * h * 4);
    for (unsigned i = 0; i < w * h; ++i) {
        image[4 * i + 0] = img[i];
        image[4 * i + 1] = img[i];
        image[4 * i + 2] = img[i];
        image[4 * i + 3] = 255;
    }
    std::vector<unsigned char> png;
    unsigned error = lodepng::encode(png, image, w, h);
    if (!error)
        lodepng::save_file(png, filename);
    else
        std::cerr << "Encode error: " << lodepng_error_text(error) << "\n";
}

int main() {
    unsigned w, h;
    auto gray = loadGrayscale("../images/camilla.png", w, h);

    std::vector<uint8_t> blur1(w * h), blur2(w * h), dog(w * h);

    gaussian_blur_cuda(gray, blur1, w, h, 3.0f);
    gaussian_blur_cuda(gray, blur2, w, h, 6.0f);

    for (unsigned i = 0; i < w * h; ++i) {
        int diff = int(blur1[i]) - int(blur2[i]);
        dog[i] = clamp(128 + diff, 0, 255);
    }

    saveGrayscale("output_dog.png", dog, w, h);
    std::cout << "Done. Saved to output_dog.png\n";
}
