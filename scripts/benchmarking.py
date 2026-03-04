#!/usr/bin/env python3
"""
Compile (optional), run, and plot CUDA matrix multiplication benchmarks.
Outputs: results/benchmark.csv, results/comparison.png
"""
import os
import sys
import subprocess
import csv
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (label, source-file-prefix)
VERSIONS = [
    ("naive", "matmul_naive"),
    ("tiled", "matmul_tiled"),
    ("vec", "matmul_tiled_vec"),
    ("unroll", "matmul_tiled_vec_unroll"),
    ("tuned", "matmul_tuned"),
]

DEFAULT_SIZES = [256, 512, 1024, 2048]
DEFAULT_TRIALS = 5


def _exe_name(label):
    """Platform-agnostic executable name: matmul_<label> or matmul_<label>.exe on Windows."""
    base = os.path.join("build", f"matmul_{label}")
    return base + ".exe" if sys.platform == "win32" else base


def compile_all(versions=None):
    """Compile each CUDA source into build/matmul_<label> (or .exe on Windows)."""
    versions = versions or VERSIONS
    os.makedirs("build", exist_ok=True)
    for label, src in versions:
        exe = _exe_name(label)
        cu = os.path.join("src", f"{src}.cu")
        if not os.path.isfile(cu):
            print("Warning: source not found:", cu, file=sys.stderr)
            continue
        cmd = ["nvcc", "-O3", "-o", exe, cu]
        print("Compiling", cu, "->", exe)
        subprocess.run(cmd, check=True)


def run_benchmarks(sizes=None, trials=None, versions=None):
    """Run each executable over all sizes, collect and write results."""
    sizes = sizes or DEFAULT_SIZES
    trials = trials or DEFAULT_TRIALS
    versions = versions or VERSIONS

    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "benchmark.csv")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Size", "Version", "AvgTime_ms", "StdDev_ms"])

        for N in sizes:
            print("\nBenchmarking N=%d" % N)
            for label, _ in versions:
                exe = _exe_name(label)
                if not os.path.isfile(exe):
                    print("  [SKIP] %s: executable not found: %s (run with compile or: cmake --build build)" % (label, exe))
                    continue
                times = []
                for _ in range(trials):
                    try:
                        output = subprocess.check_output(
                            [exe, str(N)], stderr=subprocess.STDOUT, text=True
                        )
                    except subprocess.CalledProcessError as e:
                        print("  [ERROR] %s on N=%d:\n%s" % (label, N, (e.output or "").strip()))
                        break
                    for line in output.splitlines():
                        if "AvgTime=" in line:
                            part = line.split("AvgTime=")[1]
                            ms = float(part.split()[0])
                            times.append(ms)
                            break

                if times:
                    avg = np.mean(times)
                    std = np.std(times)
                    print("  %-8s Avg=%.3f ms  Std=%.3f" % (label, avg, std))
                    writer.writerow([N, label, "%.3f" % avg, "%.3f" % std])
                else:
                    print("  %-8s No valid runs for N=%d" % (label, N))

    return csv_path


def plot_results(csv_path=None):
    """Read CSV and plot execution time curves."""
    csv_path = csv_path or os.path.join("results", "benchmark.csv")
    if not os.path.isfile(csv_path):
        print("Error: CSV not found:", csv_path, "(run benchmarks first)", file=sys.stderr)
        return
    data = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            size, label, avg, std = int(row[0]), row[1], float(row[2]), float(row[3])
            rec = data.setdefault(label, {"sizes": [], "avg": [], "std": []})
            rec["sizes"].append(size)
            rec["avg"].append(avg)
            rec["std"].append(std)

    plt.figure(figsize=(10, 6))
    for label, rec in data.items():
        plt.errorbar(rec["sizes"], rec["avg"], yerr=rec["std"], marker="o", capsize=4, label=label)
    plt.title("CUDA Matrix Multiplication Performance")
    plt.xlabel("Matrix Dimension N")
    plt.ylabel("Average Kernel Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_png = os.path.join("results", "comparison.png")
    plt.savefig(out_png, dpi=300)
    print("Plot saved to", out_png)


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA matmul kernels and optionally plot.")
    parser.add_argument("--no-compile", action="store_true", help="Skip compile; use existing build/ executables")
    parser.add_argument("--sizes", type=str, default=None, help="Comma-separated matrix sizes (default: 256,512,1024,2048)")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Trials per size (default: %d)" % DEFAULT_TRIALS)
    parser.add_argument("--versions", type=str, default=None, help="Comma-separated version labels (default: all)")
    parser.add_argument("--plot-only", action="store_true", help="Only plot from existing results/benchmark.csv")
    args = parser.parse_args()

    if args.plot_only:
        plot_results()
        return

    sizes = [int(x) for x in args.sizes.split(",")] if args.sizes else DEFAULT_SIZES
    versions = None
    if args.versions:
        want = set(v.strip() for v in args.versions.split(","))
        versions = [(l, s) for l, s in VERSIONS if l in want]
        if not versions:
            print("Error: no matching versions; use e.g. --versions naive,tiled", file=sys.stderr)
            sys.exit(1)

    if not args.no_compile:
        compile_all(versions)
    csv_path = run_benchmarks(sizes=sizes, trials=args.trials, versions=versions)
    plot_results(csv_path)
    print("\nDone. See results/benchmark.csv and results/comparison.png")


if __name__ == "__main__":
    main()
