export const metadata = {
  title: "Report | CUDA Memopt",
  description: "Project report and summary",
};

export default function ReportPage() {
  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
      <main className="max-w-5xl mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50 mb-2">Project Report</h1>
        <div className="prose dark:prose-invert max-w-none mb-6">
          <ul className="list-disc pl-6 text-zinc-700 dark:text-zinc-300">
            <li>Naive kernel is memory-bound with non-coalesced global accesses; tiling and shared memory yield the largest gains (2–3×).</li>
            <li>Vectorization (float4) adds ~10% improvement; unrolling adds marginal benefit (~1–2%).</li>
            <li>Larger 32×32 tiles can trade off occupancy; the tuned kernel uses padding to reduce bank conflicts.</li>
            <li>Compiler-driven tiling (LLVM pass + autotuner) can match hand-tuned performance when tile size is chosen well.</li>
          </ul>
        </div>
        <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800 p-4">
          <p className="mb-2 text-sm text-zinc-600 dark:text-zinc-400">Download the full report (PDF):</p>
          <a
            href="/report/report_compiler.pdf"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 rounded-lg bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 px-4 py-2 font-medium"
          >
            Open report_compiler.pdf
          </a>
        </div>
      </main>
    </div>
  );
}
