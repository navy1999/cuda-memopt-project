import os
import subprocess
import csv
import matplotlib.pyplot as plt
import numpy as np
import math

# Ensure results directory exists
os.makedirs('results', exist_ok=True)

# Define matrix sizes and number of trials
matrix_sizes = [256, 512, 1024]
trials = 5

# Prepare data storage
results = []

def run_program(exe, size):
    try:
        output = subprocess.check_output([exe, str(size)], text=True)
        for line in output.splitlines():
            if 'Avg time' in line:
                # Extract the float value from the line
                time_ms = float(line.split(':')[-1].strip().split()[0])
                return time_ms
    except Exception as e:
        print(f"Error running {exe} with size {size}: {e}")
        return None

for size in matrix_sizes:
    naive_times = []
    tiled_times = []
    for _ in range(trials):
        naive_time = run_program('./build/matmul_naive.exe', size)
        tiled_time = run_program('./build/matmul_tiled.exe', size)
        if naive_time is not None:
            naive_times.append(naive_time)
        if tiled_time is not None:
            tiled_times.append(tiled_time)
    # Calculate average and std dev, handle missing data
    naive_avg = np.mean(naive_times) if naive_times else np.nan
    naive_std = np.std(naive_times) if naive_times else np.nan
    tiled_avg = np.mean(tiled_times) if tiled_times else np.nan
    tiled_std = np.std(tiled_times) if tiled_times else np.nan
    speedup = naive_avg / tiled_avg if (naive_avg and tiled_avg and tiled_avg != 0) else np.nan
    results.append({
        'size': size,
        'naive_avg': naive_avg,
        'naive_std': naive_std,
        'tiled_avg': tiled_avg,
        'tiled_std': tiled_std,
        'speedup': speedup
    })

# Save results to CSV
csv_file = 'results/benchmark_results.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['size', 'naive_avg', 'naive_std', 'tiled_avg', 'tiled_std', 'speedup'])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# Prepare data for plotting, replacing nan with 0 for plotting
for r in results:
    if r['naive_std'] is None or math.isnan(r['naive_std']):
        r['naive_std'] = 0
    if r['tiled_std'] is None or math.isnan(r['tiled_std']):
        r['tiled_std'] = 0
    if r['naive_avg'] is None or math.isnan(r['naive_avg']):
        r['naive_avg'] = 0
    if r['tiled_avg'] is None or math.isnan(r['tiled_avg']):
        r['tiled_avg'] = 0
    if r['speedup'] is None or math.isnan(r['speedup']):
        r['speedup'] = 0

sizes = [r['size'] for r in results]
naive_avg = [r['naive_avg'] for r in results]
tiled_avg = [r['tiled_avg'] for r in results]
naive_std = [r['naive_std'] for r in results]
tiled_std = [r['tiled_std'] for r in results]
speedup = [r['speedup'] for r in results]

plt.figure(figsize=(12, 6))
plt.errorbar(sizes, naive_avg, yerr=naive_std, label='Naive Kernel', fmt='-o', capsize=5)
plt.errorbar(sizes, tiled_avg, yerr=tiled_std, label='Tiled Kernel', fmt='-o', capsize=5)
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Average Kernel Execution Time (ms)')
plt.title('CUDA Matrix Multiplication Kernel Execution Time')
plt.legend()
plt.grid(True)
plt.savefig('results/kernel_execution_time.png')

plt.figure(figsize=(12, 6))
plt.plot(sizes, speedup, '-o', color='green')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('Speedup (Naive / Tiled)')
plt.title('Speedup of Tiled Kernel over Naive Kernel')
plt.grid(True)
plt.savefig('results/speedup.png')

plt.show()
