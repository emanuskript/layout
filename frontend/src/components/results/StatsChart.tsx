"use client";

import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface StatsChartProps {
  stats: Record<string, number>;
  colorMap?: Map<string, string>;
}

export function StatsChart({ stats, colorMap }: StatsChartProps) {
  const data = Object.entries(stats)
    .filter(([, v]) => v > 0)
    .sort(([, a], [, b]) => b - a)
    .map(([name, count]) => ({ name, count }));

  if (data.length === 0) return null;

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" angle={-45} textAnchor="end" interval={0} fontSize={11} />
        <YAxis allowDecimals={false} />
        <Tooltip />
        <Bar dataKey="count">
          {data.map((entry) => (
            <Cell key={entry.name} fill={colorMap?.get(entry.name) || "#6366f1"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
