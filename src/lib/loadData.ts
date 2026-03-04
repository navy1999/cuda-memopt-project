/**
 * Server-side only: read CSV from public/ (after copy-data.js).
 */
import { readFileSync } from "fs";
import path from "path";
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

export function getBenchmarkRows(): BenchmarkRow[] {
  try {
    const p = path.join(process.cwd(), "public", "results", "benchmark.csv");
    const text = readFileSync(p, "utf-8");
    const lines = text.trim().split("\n");
    const out: BenchmarkRow[] = [];
    for (let i = 1; i < lines.length; i++) {
      const row = lines[i].split(",").map((c) => c.trim());
      const r = parseBenchmarkLine(row);
      if (r) out.push(r);
    }
    return out;
  } catch {
    return [];
  }
}

export function getAutotuneRows(): AutotuneRow[] {
  try {
    const p = path.join(process.cwd(), "public", "results", "autotune.csv");
    const text = readFileSync(p, "utf-8");
    const lines = text.trim().split("\n");
    const out: AutotuneRow[] = [];
    for (let i = 1; i < lines.length; i++) {
      const row = lines[i].split(",").map((c) => c.trim());
      const r = parseAutotuneLine(row);
      if (r) out.push(r);
    }
    return out;
  } catch {
    return [];
  }
}
