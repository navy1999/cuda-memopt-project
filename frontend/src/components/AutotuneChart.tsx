"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { AutotuneRow } from "@/lib/types";

export default function AutotuneChart({ rows }: { rows: AutotuneRow[] }) {
  const best = rows.length ? rows.reduce((a, b) => (a.avgTimeMs <= b.avgTimeMs ? a : b)) : null;

  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={rows} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="tileSize" type="number" name="Tile size" />
        <YAxis />
        <Tooltip />
        <Bar dataKey="avgTimeMs" fill="#0ea5e9" name="Avg time (ms)" radius={[4, 4, 0, 0]} />
        {best && (
          <ReferenceLine
            x={best.tileSize}
            stroke="#22c55e"
            strokeDasharray="4 4"
            label={{ value: `Best T=${best.tileSize}`, position: "top" }}
          />
        )}
      </BarChart>
    </ResponsiveContainer>
  );
}
