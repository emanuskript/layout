"use client";

import { Progress } from "@/components/ui/progress";
import { BatchProgress } from "@/lib/types/predict";

interface ProgressBarProps {
  progress: BatchProgress;
}

export function ProgressBar({ progress: data }: ProgressBarProps) {
  const pct = data.total > 0 ? (data.progress / data.total) * 100 : 0;

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-muted-foreground">
          {data.status === "processing"
            ? `Processing: ${data.current_image}`
            : data.status === "completed"
              ? "Completed"
              : data.status}
        </span>
        <span className="font-medium">
          {data.progress} / {data.total}
        </span>
      </div>
      <Progress value={pct} />
    </div>
  );
}
