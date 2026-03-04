#!/usr/bin/env python3
"""Read results/autotune.csv and plot tile size vs. runtime to results/autotune.png."""
import os
import sys
import csv
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot autotune results (tile size vs. time)")
    parser.add_argument("--csv", default="results/autotune.csv", help="Input CSV path")
    parser.add_argument("--out", default="results/autotune.png", help="Output PNG path")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print("Error: CSV not found:", args.csv, "(run scripts/autotune.py first)", file=sys.stderr)
        sys.exit(1)

    sizes = []
    times = []
    with open(args.csv) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                sizes.append(int(row[0]))
                times.append(float(row[1]))

    if not sizes:
        print("Error: no data in", args.csv, file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.bar(sizes, times, color="steelblue", edgecolor="navy")
    best_idx = min(range(len(times)), key=lambda i: times[i])
    plt.axvline(x=sizes[best_idx], color="red", linestyle="--", alpha=0.8, label="best T=%d" % sizes[best_idx])
    plt.xlabel("Tile Size T")
    plt.ylabel("Average Time (ms)")
    plt.title("LLVM Loop-Tiling Autotune: Tile Size vs. Runtime (N=1024)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
