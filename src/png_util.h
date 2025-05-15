#ifndef PNG_UTIL_H
#define PNG_UTIL_H

#include <string>
#include <vector>
#include <cstdint>

using std::vector;
using std::string;

// Load PNG and convert to grayscale
vector<vector<uint8_t>> loadPNGGrayscale(const string &filename, unsigned &width, unsigned &height);

// Save grayscale image to PNG
void savePNGGrayscale(const string &filename, const vector<vector<uint8_t>> &gray);

#endif // PNG_UTIL_H