"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { DailyVisits } from "@/lib/types/analytics";

interface DailyChartProps {
  data: DailyVisits[];
}

export function DailyChart({ data }: DailyChartProps) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" fontSize={11} />
        <YAxis allowDecimals={false} />
        <Tooltip />
        <Line type="monotone" dataKey="visits" stroke="#6366f1" strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}
