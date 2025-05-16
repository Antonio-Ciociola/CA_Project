all:
	g++ -std=c++11 src/main.cpp src/dog.cpp src/png_util.cpp src/lodepng.cpp -o dog
	g++ -std=c++11 src/main.cpp src/dog_optimized.cpp src/png_util.cpp src/lodepng.cpp -o dog_optimized
	g++ -std=c++11 src/main.cpp src/dog_parallel.cpp src/png_util.cpp src/lodepng.cpp -o dog_parallel
