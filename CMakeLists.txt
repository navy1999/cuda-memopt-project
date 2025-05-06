cmake_minimum_required(VERSION 3.10)
project(cuda_memopt LANGUAGES CXX CUDA)

add_executable(matmul_naive src/matmul_naive.cu)
add_executable(matmul_tiled src/matmul_tiled.cu)
