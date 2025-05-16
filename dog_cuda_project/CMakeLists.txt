cmake_minimum_required(VERSION 3.10)
project(DOG_CUDA LANGUAGES C CXX CUDA)

add_executable(dog_cuda main.cpp gaussian_blur.cu lodepng.cpp)
set_target_properties(dog_cuda PROPERTIES CUDA_STANDARD 11 CXX_STANDARD 11)
