import { getBenchmarkRows } from "@/lib/loadData";
import BenchmarkChart from "@/components/BenchmarkChart";
import BenchmarkExplorer from "@/components/BenchmarkExplorer";

export const metadata = {
  title: "Benchmarks | CUDA Memopt",
  description: "Interactive benchmark explorer and speedup table",
};

export default async function BenchmarksPage() {
  const rows = getBenchmarkRows();

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
      <main className="max-w-5xl mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50 mb-2">Benchmark Explorer</h1>
        <p className="text-zinc-600 dark:text-zinc-400 mb-6">
          Compare kernel runtimes and speedups (tiled/vec/unroll/tuned vs naive).
        </p>
        {rows.length > 0 ? (
          <>
            <BenchmarkExplorer initialRows={rows} />
            <div className="mt-8 bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800 p-4">
              <BenchmarkChart rows={rows} />
            </div>
          </>
        ) : (
          <p className="text-zinc-500">No data. Run benchmarking.py and copy results into public/results (or rebuild).</p>
        )}
      </main>
    </div>
  );
}
