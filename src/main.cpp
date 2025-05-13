#include <iostream>
#include <fstream>
#include <vector>
#include <sndfile.h>

using std::vector, std::ifstream, std::cout, std::cerr, std::endl;

void apply_filter(vector<short> &samples, size_t frames, int channels, const vector<double> &filter);

int main(int argc, char *argv[]){
	if (argc < 4){
		cerr << "Usage: " << argv[0] << " <input file> <output file> <filter file>\n";
		return 1;
	}
	const char *input_file = argv[1];
	const char *output_file = argv[2];
	const char *filter_file = argv[3];

	SF_INFO sf_info;
	SNDFILE *in_file = sf_open(input_file, SFM_READ, &sf_info);
	if (!in_file){
		cerr << "Failed to open input file " << argv[1] << "\n";
		return 1;
	}
	size_t frames = sf_info.frames;
	int channels = sf_info.channels;
	vector<short> samples(channels * frames);
	sf_count_t read_frames = sf_readf_short(in_file, samples.data(), frames);
	sf_close(in_file);
	if (read_frames != frames){
		cerr << "Warning: failed to read all frames from input file\n";
	}

	vector<double> filter;
	ifstream filter_stream(filter_file);
	if (!filter_stream){
		cerr << "Failed to open filter file " << argv[3] << "\n";
		return 1;
	}
	double value;
	while (filter_stream >> value){
		filter.push_back(value);
	}
	filter_stream.close();
	
	apply_filter(samples, frames, channels, filter);

	SNDFILE *out_file = sf_open(output_file, SFM_WRITE, &sf_info);
	if (!out_file){
		cerr << "Failed to open output file " << argv[2] << "\n";
		return 1;
	}
	sf_count_t written_frames = sf_writef_short(out_file, samples.data(), read_frames);
	sf_close(out_file);
	if (written_frames != frames)
		cerr << "Warning: failed to write all frames to output file\n";
	cerr << "Filtered audio written to " << output_file << "\n";
	return 0;
}