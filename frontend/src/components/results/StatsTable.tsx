"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface StatsTableProps {
  stats: Record<string, number>;
  colorMap?: Map<string, string>;
}

export function StatsTable({ stats, colorMap }: StatsTableProps) {
  const entries = Object.entries(stats)
    .filter(([, v]) => v > 0)
    .sort(([, a], [, b]) => b - a);

  if (entries.length === 0) {
    return <p className="text-sm text-muted-foreground">No elements detected.</p>;
  }

  const total = entries.reduce((sum, [, v]) => sum + v, 0);

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Class</TableHead>
          <TableHead className="text-right">Count</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {entries.map(([name, count]) => (
          <TableRow key={name}>
            <TableCell className="flex items-center gap-2">
              {colorMap && (
                <span
                  className="inline-block h-3 w-3 rounded-full"
                  style={{ backgroundColor: colorMap.get(name) || "#888" }}
                />
              )}
              {name}
            </TableCell>
            <TableCell className="text-right">{count}</TableCell>
          </TableRow>
        ))}
        <TableRow className="font-medium">
          <TableCell>Total</TableCell>
          <TableCell className="text-right">{total}</TableCell>
        </TableRow>
      </TableBody>
    </Table>
  );
}
