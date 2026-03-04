import type { BenchmarkRow, AutotuneRow } from "./types";

function parseBenchmarkLine(row: string[]): BenchmarkRow | null {
  if (row.length < 4) return null;
  const size = parseInt(row[0], 10);
  const avg = parseFloat(row[2]);
  const std = parseFloat(row[3]);
  if (Number.isNaN(size) || Number.isNaN(avg) || Number.isNaN(std)) return null;
  return { size, version: row[1], avgTimeMs: avg, stdDevMs: std };
}

function parseAutotuneLine(row: string[]): AutotuneRow | null {
  if (row.length < 2) return null;
  const tileSize = parseInt(row[0], 10);
  const avg = parseFloat(row[1]);
  if (Number.isNaN(tileSize) || Number.isNaN(avg)) return null;
  return { tileSize, avgTimeMs: avg };
}

export async function fetchBenchmarkCsv(): Promise<BenchmarkRow[]> {
  const base = typeof window !== "undefined" ? "" : "";
  const res = await fetch(`${base}/results/benchmark.csv`, { cache: "no-store" });
  if (!res.ok) return [];
  const text = await res.text();
  const lines = text.trim().split("\n");
  const out: BenchmarkRow[] = [];
  for (let i = 1; i < lines.length; i++) {
    const row = lines[i].split(",").map((c) => c.trim());
    const r = parseBenchmarkLine(row);
    if (r) out.push(r);
  }
  return out;
}

export async function fetchAutotuneCsv(): Promise<AutotuneRow[]> {
  const base = typeof window !== "undefined" ? "" : "";
  const res = await fetch(`${base}/results/autotune.csv`, { cache: "no-store" });
  if (!res.ok) return [];
  const text = await res.text();
  const lines = text.trim().split("\n");
  const out: AutotuneRow[] = [];
  for (let i = 1; i < lines.length; i++) {
    const row = lines[i].split(",").map((c) => c.trim());
    const r = parseAutotuneLine(row);
    if (r) out.push(r);
  }
  return out;
}
