#!/usr/bin/env python3
"""
LLVM loop-tiling autotuner: explore tile sizes for matmul_naive.cu and write results/autotune.csv.
Requires: clang++ (CUDA), opt (LLVM), nvcc on PATH; LoopTilingPass built in llvm-pass/build.
"""
import os
import sys
import subprocess
import tempfile
import csv
import argparse
import shutil
import numpy as np

# Tile sizes to explore
DEFAULT_TILE_SIZES = [8, 16, 32, 64]
DEFAULT_N = 1024
DEFAULT_TRIALS = 5
SRC_CU = "src/matmul_naive.cu"
CSV_OUT = "results/autotune.csv"


def _pass_lib_default():
    """Default path to LoopTilingPass plugin: llvm-pass/build/Release or build, .dll on Windows else .so."""
    ext = ".dll" if sys.platform == "win32" else ".so"
    for base in ["llvm-pass/build/Release", "llvm-pass/build", "build"]:
        path = os.path.join(base, f"LoopTilingPass{ext}")
        if os.path.isfile(path):
            return path
    return os.path.join("llvm-pass", "build", "Release", f"LoopTilingPass{ext}")


def _check_tools():
    """Ensure clang++, opt, nvcc are on PATH; exit with clear message if not."""
    missing = []
    for cmd in ["clang++", "opt", "nvcc"]:
        if shutil.which(cmd) is None:
            missing.append(cmd)
    if missing:
        print("Error: required tools not found on PATH:", ", ".join(missing), file=sys.stderr)
        print("Install LLVM (clang++, opt) and CUDA Toolkit (nvcc), then re-run.", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Autotune LLVM loop-tiling pass tile sizes for matmul_naive.cu")
    parser.add_argument("--pass-plugin", default=None, help="Path to LoopTilingPass plugin (.so or .dll). Default: auto-detect")
    parser.add_argument("--sizes", type=str, default=None, help="Comma-separated tile sizes (default: 8,16,32,64)")
    parser.add_argument("-N", "--matrix-size", type=int, default=DEFAULT_N, help=f"Matrix dimension (default: {DEFAULT_N})")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help=f"Trials per tile size (default: {DEFAULT_TRIALS})")
    parser.add_argument("--cuda-arch", type=str, default="sm_61", help="CUDA GPU architecture for clang (default: sm_61)")
    args = parser.parse_args()

    _check_tools()

    pass_lib = args.pass_plugin or os.environ.get("LOOP_TILING_PASS", _pass_lib_default())
    if not os.path.isfile(pass_lib):
        print("Error: LoopTilingPass plugin not found:", pass_lib, file=sys.stderr)
        print("Build it first: cd llvm-pass && mkdir -p build && cd build && cmake .. -DLLVM_DIR=... && cmake --build . --config Release", file=sys.stderr)
        sys.exit(1)

    tile_sizes = [int(x) for x in args.sizes.split(",")] if args.sizes else DEFAULT_TILE_SIZES
    N = args.matrix_size
    trials = args.trials

    if not os.path.isfile(SRC_CU):
        print("Error: source not found:", SRC_CU, file=sys.stderr)
        sys.exit(1)

    os.makedirs("results", exist_ok=True)
    temp_files = []

    try:
        with open(CSV_OUT, "w", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["TileSize", "AvgTime_ms"])

            for T in tile_sizes:
                # 1. Compile CUDA -> LLVM bitcode
                fd_bc, bc = tempfile.mkstemp(suffix=".bc")
                os.close(fd_bc)
                temp_files.append(bc)
                subprocess.run([
                    "clang++", "-x", "cuda", f"--cuda-gpu-arch={args.cuda_arch}",
                    "-O3", "-emit-llvm", "-c", SRC_CU, "-o", bc
                ], check=True)

                # 2. Apply tiling pass
                fd_tiled, tiled_bc = tempfile.mkstemp(suffix=".bc")
                os.close(fd_tiled)
                temp_files.append(tiled_bc)
                subprocess.run([
                    "opt", "-load-pass-plugin=" + pass_lib,
                    "--passes=loop-tiling",
                    f"-loop-tiling-tilesize={T}",
                    bc, "-o", tiled_bc
                ], check=True)

                # 3. Compile back to executable (platform-agnostic name)
                exe_suffix = ".exe" if sys.platform == "win32" else ""
                fd_exe, exe = tempfile.mkstemp(suffix=exe_suffix)
                os.close(fd_exe)
                temp_files.append(exe)
                subprocess.run(["nvcc", "-O3", tiled_bc, "-o", exe], check=True)

                # 4. Benchmark
                times = []
                for _ in range(trials):
                    out = subprocess.check_output([exe, str(N)], text=True)
                    for line in out.splitlines():
                        if "AvgTime=" in line:
                            times.append(float(line.split("AvgTime=")[1].split()[0]))
                            break
                avg_t = np.mean(times)
                print(f"T={T} avg={avg_t:.3f} ms")
                writer.writerow([T, f"{avg_t:.3f}"])
    finally:
        for p in temp_files:
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass

    print("Wrote", CSV_OUT)


if __name__ == "__main__":
    main()
