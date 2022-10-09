cmake_minimum_required(VERSION 3.23)
project(cudaBasicInfo CUDA)


set(CMAKE_CUDA_STANDARD 14)

add_executable(cudaBasicInfo main.cu cuda_basic_info.cuh)

set_target_properties(cudaBasicInfo PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
