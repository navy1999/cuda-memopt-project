import { getAutotuneRows } from "@/lib/loadData";
import AutotuneChart from "@/components/AutotuneChart";

export const metadata = {
  title: "Autotune | CUDA Memopt",
  description: "LLVM loop-tiling tile size vs. runtime",
};

export default async function AutotunePage() {
  const rows = getAutotuneRows();

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
      <main className="max-w-5xl mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50 mb-2">Autotune Results</h1>
        <p className="text-zinc-600 dark:text-zinc-400 mb-6">
          Tile size vs. average kernel time (N=1024) from <code className="bg-zinc-200 dark:bg-zinc-800 px-1 rounded">scripts/autotune.py</code>.
          Best tile size is highlighted.
        </p>
        {rows.length > 0 ? (
          <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800 p-4">
            <AutotuneChart rows={rows} />
          </div>
        ) : (
          <p className="text-zinc-500">No autotune data. Run scripts/autotune.py and ensure results/autotune.csv exists, then rebuild.</p>
        )}
      </main>
    </div>
  );
}
