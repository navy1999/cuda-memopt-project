import Link from "next/link";

export default function Nav() {
  return (
    <nav className="border-b border-zinc-200 dark:border-zinc-800 bg-white/80 dark:bg-zinc-900/80 backdrop-blur">
      <div className="max-w-5xl mx-auto px-4 flex items-center gap-6 h-12">
        <Link href="/" className="font-semibold text-zinc-900 dark:text-zinc-50">
          CUDA Memopt
        </Link>
        <Link href="/benchmarks" className="text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-50">
          Benchmarks
        </Link>
        <Link href="/autotune" className="text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-50">
          Autotune
        </Link>
        <Link href="/report" className="text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-50">
          Report
        </Link>
        <Link href="/demo" className="text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-50">
          Demo
        </Link>
      </div>
    </nav>
  );
}
