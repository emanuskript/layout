"use client";

interface MiniBarChartProps {
  stats: Record<string, number>;
  colorMap: Map<string, string>;
}

export function MiniBarChart({ stats, colorMap }: MiniBarChartProps) {
  const entries = Object.entries(stats)
    .filter(([, v]) => v > 0)
    .sort(([, a], [, b]) => b - a);

  if (entries.length === 0) return null;

  const max = entries[0][1];

  return (
    <div className="space-y-1.5">
      {entries.map(([name, count]) => (
        <div key={name} className="flex items-center gap-2">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 mb-0.5">
              <span
                className="inline-block h-2 w-2 rounded-full flex-shrink-0"
                style={{ backgroundColor: colorMap.get(name) || "#6366f1" }}
              />
              <span className="truncate text-xs text-muted-foreground">{name}</span>
            </div>
            <div className="h-2 w-full rounded-full bg-muted">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${(count / max) * 100}%`,
                  backgroundColor: colorMap.get(name) || "#6366f1",
                }}
              />
            </div>
          </div>
          <span className="text-xs font-medium tabular-nums w-8 text-right">{count}</span>
        </div>
      ))}
    </div>
  );
}
