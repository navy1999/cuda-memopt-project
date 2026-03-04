import Link from "next/link";
import { getBenchmarkRows } from "@/lib/loadData";
import BenchmarkChart from "@/components/BenchmarkChart";

export default async function Home() {
  const rows = getBenchmarkRows();

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
      <main className="max-w-5xl mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50 mb-2">
          CUDA Matrix Multiplication Optimizations
        </h1>
        <p className="text-zinc-600 dark:text-zinc-400 mb-6">
          This project compares hand-tuned CUDA kernels: <strong>naive</strong> (global memory),
          <strong> tiled</strong> (shared memory), <strong>vec</strong> (vectorized loads),{" "}
          <strong>unroll</strong> (loop unrolling), and <strong>tuned</strong> (32×32 tiles with padding).
          Results are produced by <code className="bg-zinc-200 dark:bg-zinc-800 px-1 rounded">scripts/benchmarking.py</code>.
        </p>

        {rows.length > 0 ? (
          <section className="mb-8">
            <h2 className="text-lg font-semibold text-zinc-800 dark:text-zinc-200 mb-4">
              Kernel time vs matrix size N
            </h2>
            <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800 p-4">
              <BenchmarkChart rows={rows} />
            </div>
            <p className="mt-2 text-sm text-zinc-500">
              <Link href="/benchmarks" className="underline">Explore benchmarks</Link> for toggles and speedup table.
            </p>
          </section>
        ) : (
          <p className="text-zinc-500">
            No benchmark data yet. Run <code className="bg-zinc-200 dark:bg-zinc-800 px-1 rounded">python scripts/benchmarking.py</code> in the
            project root and ensure <code className="bg-zinc-200 dark:bg-zinc-800 px-1 rounded">results/benchmark.csv</code> exists, then rebuild the frontend.
          </p>
        )}

        <div className="flex flex-wrap gap-4 mt-8">
          <Link
            href="/benchmarks"
            className="rounded-lg bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 px-4 py-2 font-medium"
          >
            Benchmarks
          </Link>
          <Link
            href="/autotune"
            className="rounded-lg border border-zinc-300 dark:border-zinc-600 px-4 py-2 font-medium"
          >
            Autotune
          </Link>
          <Link
            href="/report"
            className="rounded-lg border border-zinc-300 dark:border-zinc-600 px-4 py-2 font-medium"
          >
            Report
          </Link>
          <Link
            href="/demo"
            className="rounded-lg border border-zinc-300 dark:border-zinc-600 px-4 py-2 font-medium"
          >
            GPU Demo
          </Link>
        </div>
      </main>
    </div>
  );
}
