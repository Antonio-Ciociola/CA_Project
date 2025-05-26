PKG_CFLAGS := $(shell pkg-config --cflags opencv4)
PKG_LIBS := $(shell pkg-config --libs opencv4)
CXXFLAGS := -std=c++11 $(PKG_CFLAGS)
OPTFLAGS := -O3 -flto -ffast-math

all: dog dog_flag dog_optimized dog_parallel_flag dog_parallel rawtojpg dog_gpu dog_gpu_new dog_gpu_pair

dog: src/main.cpp src/dog.cpp src/dog.h src/png_util.cpp src/png_util.h src/lodepng.cpp src/lodepng.h
	g++ $(CXXFLAGS) src/main.cpp src/dog.cpp src/png_util.cpp src/lodepng.cpp -o dog $(PKG_LIBS)

dog_flag: src/main.cpp src/dog.cpp src/dog.h src/png_util.cpp src/png_util.h src/lodepng.cpp src/lodepng.h
	g++ $(CXXFLAGS) $(OPTFLAGS) src/main.cpp src/dog.cpp src/png_util.cpp src/lodepng.cpp -o dog_flag $(PKG_LIBS)

dog_optimized: src/main.cpp src/dog_optimized.cpp src/dog.h src/png_util.cpp src/png_util.h src/lodepng.cpp src/lodepng.h
	g++ $(CXXFLAGS) $(OPTFLAGS) src/main.cpp src/dog_optimized.cpp src/png_util.cpp src/lodepng.cpp -o dog_optimized $(PKG_LIBS)

dog_parallel: src/main.cpp src/dog_parallel.cpp src/dog.h src/png_util.cpp src/png_util.h src/lodepng.cpp src/lodepng.h
	g++ $(CXXFLAGS) $(OPTFLAGS) src/main.cpp src/dog_parallel.cpp src/png_util.cpp src/lodepng.cpp -o dog_parallel $(PKG_LIBS)

rawtojpg: src/rawtojpg.cpp
	g++ $(CXXFLAGS) src/rawtojpg.cpp -o rawtojpg $(PKG_LIBS)

dog_gpu: src/main.cpp src/dog_cuda.cu src/dog.h src/png_util.cpp src/png_util.h src/lodepng.cpp src/lodepng.h
	nvcc $(CXXFLAGS) -O3 src/main.cpp src/dog_cuda.cu src/png_util.cpp src/lodepng.cpp -o dog_gpu $(PKG_LIBS)

dog_gpu_new: src/main.cpp src/dog_cuda_new.cu src/dog.h src/png_util.cpp src/png_util.h src/lodepng.cpp src/lodepng.h
	nvcc $(CXXFLAGS) src/main.cpp src/dog_cuda_new.cu src/png_util.cpp src/lodepng.cpp -o dog_gpu_new $(PKG_LIBS)

dog_gpu_pair: src/main.cpp src/dog_cuda_pair.cu src/dog.h src/png_util.cpp src/png_util.h src/lodepng.cpp src/lodepng.h
	nvcc $(CXXFLAGS) -O3 src/main.cpp src/dog_cuda_pair.cu src/png_util.cpp src/lodepng.cpp -o dog_gpu_pair $(PKG_LIBS)

clean:
	rm -f dog dog_optimized dog_parallel dog_gpu dog_gpu_new* dog_gpu_pair rawtojpg
	rm -f outputs/*
	rm -f *.mp4 *.png
