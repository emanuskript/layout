"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { CountryVisits } from "@/lib/types/analytics";

interface CountryChartProps {
  data: CountryVisits[];
}

export function CountryChart({ data }: CountryChartProps) {
  const sorted = [...data].sort((a, b) => b.visits - a.visits).slice(0, 20);

  return (
    <ResponsiveContainer width="100%" height={350}>
      <BarChart data={sorted} margin={{ top: 5, right: 10, left: 10, bottom: 60 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="country" angle={-45} textAnchor="end" interval={0} fontSize={11} />
        <YAxis allowDecimals={false} />
        <Tooltip />
        <Bar dataKey="visits" fill="#6366f1" />
      </BarChart>
    </ResponsiveContainer>
  );
}
