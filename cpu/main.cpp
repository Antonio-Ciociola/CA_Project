#include <iostream>
#include <vector>
#include <sndfile.h>

void applyMovingAverage(std::vector<short> &samples, int channels, int windowSize)
{
    std::vector<short> filtered(samples.size());

    for (int ch = 0; ch < channels; ++ch)
    {
        for (size_t i = 0; i < samples.size() / channels; ++i)
        {
            int sum = 0;
            int count = 0;

            for (int j = -windowSize / 2; j <= windowSize / 2; ++j)
            {
                int idx = i + j;
                if (idx >= 0 && idx < samples.size() / channels)
                {
                    sum += samples[idx * channels + ch];
                    count++;
                }
            }
            if (i % 10000000 == 0)
            {

                printf("i: %d, ch: %d, sum: %d, count: %d, %d\n", i, ch, sum, count, samples.size());
            }

            filtered[i * channels + ch] = static_cast<short>(sum / count);
        }
    }

    samples = filtered;
}

int main(int argc, char *argv[])
{

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input file> <output file>\n";
        return 1;
    }

    const char *inputFile = argv[1];
    const char *outputFile = argv[2];

    int windowSize = 64;

    SF_INFO sfInfo;
    SNDFILE *inFile = sf_open(inputFile, SFM_READ, &sfInfo);
    if (!inFile)
    {
        std::cerr << "Failed to open input.wav\n";
        return 1;
    }

    sf_count_t numFrames = sfInfo.frames;
    int numChannels = sfInfo.channels;

    std::vector<short> samples(numFrames * numChannels);
    sf_count_t framesRead = sf_readf_short(inFile, samples.data(), numFrames);
    sf_close(inFile);

    if (framesRead != numFrames)
    {
        std::cerr << "Warning: Could not read all frames.\n";
    }

    applyMovingAverage(samples, numChannels, windowSize);

    SNDFILE *outFile = sf_open(outputFile, SFM_WRITE, &sfInfo);
    if (!outFile)
    {
        std::cerr << "Failed to open output.wav\n";
        return 1;
    }

    sf_count_t framesWritten = sf_writef_short(outFile, samples.data(), framesRead);
    if (framesWritten != framesRead)
    {
        std::cerr << "Warning: Could not write all frames.\n";
    }

    sf_close(outFile);
    std::cout << "Filtered audio written to " << outputFile << "\n";

    return 0;
}
