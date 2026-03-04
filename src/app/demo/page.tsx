import WebGPUDemo from "@/components/WebGPUDemo";

export const metadata = {
  title: "GPU Demo | CUDA Memopt",
  description: "Run a WebGPU matrix multiplication demo in your browser",
};

export default function DemoPage() {
  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950">
      <main className="max-w-5xl mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50 mb-2">GPU Demo (WebGPU)</h1>
        <p className="text-zinc-600 dark:text-zinc-400 mb-6">
          Run a small matrix multiplication on your GPU in the browser. This uses WebGPU (not CUDA), so results are for
          illustration only. For real CUDA runs, use the local helper (see instructions below).
        </p>
        <WebGPUDemo />
      </main>
    </div>
  );
}
