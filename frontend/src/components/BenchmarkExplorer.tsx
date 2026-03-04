"use client";

import { useMemo, useState } from "react";
import type { BenchmarkRow } from "@/lib/types";
import BenchmarkChart from "./BenchmarkChart";

const VERSIONS = ["naive", "tiled", "vec", "unroll", "tuned"];

export default function BenchmarkExplorer({ initialRows }: { initialRows: BenchmarkRow[] }) {
  const [selected, setSelected] = useState<string[]>(VERSIONS);
  const [logScale, setLogScale] = useState(false);

  const speedups = useMemo(() => {
    const bySize: Record<number, Record<string, number>> = {};
    for (const r of initialRows) {
      if (!bySize[r.size]) bySize[r.size] = {};
      bySize[r.size][r.version] = r.avgTimeMs;
    }
    const naiveBySize: Record<number, number> = {};
    for (const r of initialRows) {
      if (r.version === "naive") naiveBySize[r.size] = r.avgTimeMs;
    }
    const out: { size: number; version: string; speedup: number }[] = [];
    for (const [size, vers] of Object.entries(bySize)) {
      const naive = naiveBySize[Number(size)];
      if (!naive || naive <= 0) continue;
      for (const [ver, t] of Object.entries(vers)) {
        if (ver !== "naive") out.push({ size: Number(size), version: ver, speedup: naive / t });
      }
    }
    return out;
  }, [initialRows]);

  const filteredRows = useMemo(
    () => initialRows.filter((r) => selected.includes(r.version)),
    [initialRows, selected]
  );

  const toggle = (v: string) => {
    setSelected((prev) => (prev.includes(v) ? prev.filter((x) => x !== v) : [...prev, v]));
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-4">
        <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">Versions:</span>
        {VERSIONS.map((v) => (
          <label key={v} className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={selected.includes(v)}
              onChange={() => toggle(v)}
              className="rounded border-zinc-300"
            />
            <span>{v}</span>
          </label>
        ))}
        <label className="flex items-center gap-2 cursor-pointer ml-4">
          <input
            type="checkbox"
            checked={logScale}
            onChange={(e) => setLogScale(e.target.checked)}
            className="rounded border-zinc-300"
          />
          <span>Log scale Y</span>
        </label>
      </div>
      <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-200 dark:border-zinc-700">
              <th className="text-left p-2">N</th>
              <th className="text-left p-2">Version</th>
              <th className="text-right p-2">Speedup vs naive</th>
            </tr>
          </thead>
          <tbody>
            {speedups
              .sort((a, b) => a.size - b.size || a.version.localeCompare(b.version))
              .map((s, i) => (
                <tr key={i} className="border-b border-zinc-100 dark:border-zinc-800">
                  <td className="p-2">{s.size}</td>
                  <td className="p-2">{s.version}</td>
                  <td className="text-right p-2 font-mono">{s.speedup.toFixed(2)}×</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-800 p-4">
        <BenchmarkChart rows={filteredRows} versions={selected} logScale={logScale} />
      </div>
    </div>
  );
}
