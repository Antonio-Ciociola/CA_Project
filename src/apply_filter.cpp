#include <vector>
#include <iostream>
using std::vector;
using std::cout;
using std::endl;

void apply_filter(vector<short> &samples, size_t frames, int channels, const vector<double> &filter){
	int filter_size = filter.size();
	vector<short> output(samples.size(), 0);
	for (size_t i = 0; i < frames; ++i){
		for (size_t j = 0; j < channels; ++j){
			double sum = 0.0;
			for (int k = -filter_size/2; k <= filter_size/2; ++k){
				if (i - k >= 0 && i - k < frames){
					sum += samples[(i - k) * channels + j] * filter[k];
				}
			}
			output[i * channels + j] = static_cast<short>(sum);
		}
	}
	samples = output;
}