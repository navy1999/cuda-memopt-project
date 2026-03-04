@echo off
REM Build the project
cmake -S .. -B ../build
cmake --build ../build

REM Run naive kernel and profile with Nsight Compute
ncu --set full --target-processes all -o ../results/matmul_naive_ncu.txt ../build/matmul_naive.exe

REM Run optimized kernel and profile
ncu --set full --target-processes all -o ../results/matmul_tiled_ncu.txt ../build/matmul_tiled.exe

echo Benchmarking and profiling complete. See results folder.
pause
