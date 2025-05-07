#include <iostream>
#include <vector>
#include <thread>
#include <sndfile.h>

void applyMovingAverage(std::vector<short> &samples, int channels, int windowSize)
{
    size_t totalFrames = samples.size() / channels;
    std::vector<short> filtered(samples.size());

    unsigned int numThreads = 8;
    if (numThreads == 0) numThreads = 2; // fallback
    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    auto worker = [&](size_t startFrame, size_t endFrame) {
        for (size_t i = startFrame; i < endFrame; ++i) {
            for (int ch = 0; ch < channels; ++ch) {
                int sum = 0;
                int count = 0;
                for (int j = -windowSize/2; j <= windowSize/2; ++j) {
                    int idx = static_cast<int>(i) + j;
                    if (idx >= 0 && idx < static_cast<int>(totalFrames)) {
                        sum += samples[idx * channels + ch];
                        ++count;
                    }
                }
                filtered[i * channels + ch] = static_cast<short>(sum / count);
            }
        }
    };

    // Launch threads to process chunks of frames
    for (unsigned int t = 0; t < numThreads; ++t) {
        size_t start = t * totalFrames / numThreads;
        size_t end = (t + 1) * totalFrames / numThreads;
        threads.emplace_back(worker, start, end);
    }

    // Join threads
    for (auto &th : threads) {
        th.join();
    }

    samples.swap(filtered);
}

int main()
{
    const char *inputFile = "input.wav";
    const char *outputFile = "output.wav";
    int windowSize = 64;

    SF_INFO sfInfo;
    SNDFILE *inFile = sf_open(inputFile, SFM_READ, &sfInfo);
    if (!inFile) {
        std::cerr << "Failed to open " << inputFile << std::endl;
        return 1;
    }

    sf_count_t numFrames = sfInfo.frames;
    int numChannels = sfInfo.channels;

    std::vector<short> samples(numFrames * numChannels);
    sf_count_t framesRead = sf_readf_short(inFile, samples.data(), numFrames);
    sf_close(inFile);

    if (framesRead != numFrames) {
        std::cerr << "Warning: Could not read all frames.\n";
    }

    applyMovingAverage(samples, numChannels, windowSize);

    SNDFILE *outFile = sf_open(outputFile, SFM_WRITE, &sfInfo);
    if (!outFile) {
        std::cerr << "Failed to open " << outputFile << std::endl;
        return 1;
    }

    sf_count_t framesWritten = sf_writef_short(outFile, samples.data(), framesRead);
    if (framesWritten != framesRead) {
        std::cerr << "Warning: Could not write all frames.\n";
    }

    sf_close(outFile);
    std::cout << "Filtered audio written to " << outputFile << std::endl;

    return 0;
}
