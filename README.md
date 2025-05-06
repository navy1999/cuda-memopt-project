CUDA Memory Access Pattern Optimization Project
Overview
This project explores the impact of memory access pattern optimization on GPU performance using CUDA. We implement and compare two versions of matrix multiplication: a naive approach and an optimized version using tiling and shared memory. The goal is to demonstrate how optimizing memory accesses (coalescing, shared memory usage) can significantly improve computation throughput on NVIDIA GPUs.

Folder Structure
text
cuda-memopt-project/
├── src/                  # Source code (.cu files)
│   ├── matmul_naive.cu
│   └── matmul_tiled.cu
├── benchmarks/           # Benchmarking and profiling scripts
│   └── run_benchmarks.bat
├── results/              # Output and profiling results
│   ├── matmul_naive_ncu.txt
│   └── matmul_tiled_ncu.txt
├── report/               # Project report and presentation
│   ├── project_report.pdf
│   └── presentation.pptx
├── README.md             # This file
├── CMakeLists.txt        # Build configuration (if using CMake)
└── .vscode/              # VS Code configuration files (optional)
Prerequisites
NVIDIA GPU with CUDA support

CUDA Toolkit 11.x or newer (nvcc should be in your PATH)

Nsight Compute (for profiling, optional but recommended)

CMake (if using CMake for builds)

(Optional) Visual Studio Code or another C++ IDE

Building the Project
Using CMake
bash
mkdir build
cd build
cmake ..
cmake --build .
Using nvcc Directly
bash
nvcc src/matmul_naive.cu -o build/matmul_naive.exe
nvcc src/matmul_tiled.cu -o build/matmul_tiled.exe
Running the Kernels
From the build directory (or project root if not using CMake):

bash
./build/matmul_naive.exe
./build/matmul_tiled.exe
Benchmarking and Profiling
To benchmark and profile the kernels, use the provided script:

text
benchmarks\run_benchmarks.bat
Or run Nsight Compute manually:

bash
ncu ./build/matmul_naive.exe
ncu ./build/matmul_tiled.exe
Profiling reports will be saved in the results/ folder.

Project Report
See report/project_report.pdf for:

Problem statement

Review of existing solutions

Solution approach

Implementation details

Evaluation (benchmarks, experiments, metrics, baselines, results, lessons learned)

Conclusions

Contact
For questions or contributions, please contact Navneet Shankar or Ayush Gupta.

End of README