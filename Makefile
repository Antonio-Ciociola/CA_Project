PKG_CFLAGS := $(shell pkg-config --cflags opencv4)
PKG_LIBS := $(shell pkg-config --libs opencv4)
CXXFLAGS := -std=c++11 $(PKG_CFLAGS)

all: dog dog_optimized dog_parallel dog_gpu

dog: src/main.cpp src/dog.cpp src/png_util.cpp src/lodepng.cpp
	g++ $(CXXFLAGS) src/main.cpp src/dog.cpp src/png_util.cpp src/lodepng.cpp -o dog $(PKG_LIBS)

dog_optimized: src/main.cpp src/dog_optimized.cpp src/png_util.cpp src/lodepng.cpp
	g++ $(CXXFLAGS) src/main.cpp src/dog_optimized.cpp src/png_util.cpp src/lodepng.cpp -o dog_optimized $(PKG_LIBS)

dog_parallel: src/main.cpp src/dog_parallel.cpp src/png_util.cpp src/lodepng.cpp
	g++ $(CXXFLAGS) src/main.cpp src/dog_parallel.cpp src/png_util.cpp src/lodepng.cpp -o dog_parallel $(PKG_LIBS)

dog_gpu: dog_cuda_project/gaussian_blur.cu dog_cuda_project/dog.cpp dog_cuda_project/lodepng.cpp src/main.cpp src/png_util.cpp
	nvcc $(CXXFLAGS) dog_cuda_project/gaussian_blur.cu dog_cuda_project/dog.cpp dog_cuda_project/lodepng.cpp src/main.cpp src/png_util.cpp -o dog_gpu $(PKG_LIBS)

clean:
	rm -f dog dog_optimized dog_parallel
