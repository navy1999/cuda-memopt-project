# CUDA Matrix Multiplication Optimizations

## Compile All Versions
nvcc -o build/matmul_naive.exe src/matmul_naive.cu
nvcc -o build/matmul_tiled.exe src/matmul_tiled.cu
nvcc -o build/matmul_tiled_vec.exe src/matmul_tiled_vec.cu
nvcc -o build/matmul_tiled_vec_unroll.exe src/matmul_tiled_vec_unroll.cu
nvcc -o build/matmul_tuned.exe src/matmul_tuned.cu

text

## Run Benchmark
python benchmark.py

text

## View Results
- CSV file: `results/benchmark.csv`
- Plot: `results/comparison.png`