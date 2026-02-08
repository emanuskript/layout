"use client";

import { useAtomValue, useSetAtom } from "jotai";

import { StatsChart } from "@/components/results/StatsChart";
import { chartModalOpenAtom, selectedFileAtom, colorMapAtom } from "@/lib/atoms";

export function ChartModal() {
  const open = useAtomValue(chartModalOpenAtom);
  const setOpen = useSetAtom(chartModalOpenAtom);
  const selectedFile = useAtomValue(selectedFileAtom);
  const colorMap = useAtomValue(colorMapAtom);

  if (!open) return null;

  const stats =
    selectedFile?.singleResult?.stats ??
    selectedFile?.batchResults?.stats_summary ??
    null;

  if (!stats) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setOpen(false)}>
      <div
        className="relative w-full max-w-2xl rounded-lg bg-popover p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-center justify-between">
          <h3 className="text-sm font-medium">Detection Distribution</h3>
          <button
            onClick={() => setOpen(false)}
            className="rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <StatsChart stats={stats} colorMap={colorMap} />
      </div>
    </div>
  );
}
