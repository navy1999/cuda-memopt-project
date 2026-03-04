"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import type { BenchmarkRow } from "@/lib/types";

const COLORS: Record<string, string> = {
  naive: "#64748b",
  tiled: "#0ea5e9",
  vec: "#8b5cf6",
  unroll: "#ec4899",
  tuned: "#22c55e",
};

function groupByVersion(rows: BenchmarkRow[]) {
  const byVersion: Record<string, { size: number; avg: number; std: number }[]> = {};
  for (const r of rows) {
    if (!byVersion[r.version]) byVersion[r.version] = [];
    byVersion[r.version].push({ size: r.size, avg: r.avgTimeMs, std: r.stdDevMs });
  }
  const sizes = [...new Set(rows.map((r) => r.size))].sort((a, b) => a - b);
  const data = sizes.map((size) => {
    const point: Record<string, number | string> = { size };
    for (const [ver, pts] of Object.entries(byVersion)) {
      const p = pts.find((x) => x.size === size);
      if (p) {
        point[ver] = p.avg;
        point[`${ver}_std`] = p.std;
      }
    }
    return point;
  });
  return { data, versions: Object.keys(byVersion) };
}

export default function BenchmarkChart({
  rows,
  versions,
  logScale,
}: {
  rows: BenchmarkRow[];
  versions?: string[];
  logScale?: boolean;
}) {
  const { data, versions: allVersions } = groupByVersion(rows);
  const show = versions && versions.length > 0 ? versions : allVersions;

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="size" type="number" name="N" />
        <YAxis scale={logScale ? "log" : "linear"} domain={logScale ? ["auto", "auto"] : undefined} />
        <Tooltip />
        <Legend />
        {show.map((ver) => (
          <Line
            key={ver}
            type="monotone"
            dataKey={ver}
            stroke={COLORS[ver] ?? "#888"}
            strokeWidth={2}
            dot={{ r: 4 }}
            name={ver}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
