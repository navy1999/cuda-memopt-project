import os
import subprocess
import csv
import matplotlib.pyplot as plt
import numpy as np

# (label, source‐file‐prefix)
versions = [
    ("naive",  "matmul_naive"),
    ("tiled",  "matmul_tiled"),
    ("vec",    "matmul_tiled_vec"),
    ("unroll","matmul_tiled_vec_unroll"),
    ("tuned",  "matmul_tuned")
]

sizes  = [256, 512, 1024, 2048]
trials = 5

def compile_all():
    """Compile each CUDA source into build\matmul_<label>.exe"""
    os.makedirs("build", exist_ok=True)
    for label, src in versions:
        exe = os.path.join("build", f"matmul_{label}.exe")
        cu  = os.path.join("src",   f"{src}.cu")
        cmd = ["nvcc", "-O3", "-o", exe, cu]
        print(f"Compiling {cu} -> {exe}")
        subprocess.run(cmd, check=True)

def run_benchmarks():
    """Run each executable over all sizes, collect and write results."""
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "benchmark.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Size","Version","AvgTime_ms","StdDev_ms"])
        
        for N in sizes:
            print(f"\nBenchmarking N={N}")
            for label, _ in versions:
                exe = os.path.join("build", f"matmul_{label}.exe")
                times = []
                
                for _ in range(trials):
                    try:
                        # call without shell, so Windows finds the .exe directly
                        output = subprocess.check_output([exe, str(N)],
                                                         stderr=subprocess.STDOUT,
                                                         text=True)
                    except subprocess.CalledProcessError as e:
                        print(f"  [ERROR] {label} on N={N}:\n{e.output.strip()}")
                        break
                    
                    # parse line like "[Naive]  N=256  AvgTime=XX.XXX ms"
                    for line in output.splitlines():
                        if "AvgTime=" in line:
                            part = line.split("AvgTime=")[1]
                            ms   = float(part.split()[0])
                            times.append(ms)
                            break

                if times:
                    avg = np.mean(times)
                    std = np.std(times)
                    print(f"  {label:8s} Avg={avg:.3f} ms  Std={std:.3f}")
                    writer.writerow([N, label, f"{avg:.3f}", f"{std:.3f}"])
                else:
                    print(f"  {label:8s} No valid runs for N={N}")

def plot_results():
    """Read CSV and plot execution time curves."""
    csv_path = os.path.join("results", "benchmark.csv")
    data = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for size, label, avg, std in reader:
            size = int(size)
            avg   = float(avg)
            std   = float(std)
            rec = data.setdefault(label, {"sizes":[], "avg":[], "std":[]})
            rec["sizes"].append(size)
            rec["avg"].append(avg)
            rec["std"].append(std)
    
    plt.figure(figsize=(10,6))
    for label, rec in data.items():
        plt.errorbar(rec["sizes"], rec["avg"], yerr=rec["std"],
                     marker='o', capsize=4, label=label)
    plt.title("CUDA Matrix Multiplication Performance")
    plt.xlabel("Matrix Dimension N")
    plt.ylabel("Average Kernel Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_png = os.path.join("results","comparison.png")
    plt.savefig(out_png, dpi=300)
    print(f"\nPlot saved to {out_png}")

if __name__ == "__main__":
    compile_all()
    run_benchmarks()
    plot_results()
    print("\nDone. See results/benchmark.csv and results/comparison.png")
