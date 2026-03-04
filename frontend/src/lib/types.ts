export interface BenchmarkRow {
  size: number;
  version: string;
  avgTimeMs: number;
  stdDevMs: number;
}

export interface AutotuneRow {
  tileSize: number;
  avgTimeMs: number;
}
