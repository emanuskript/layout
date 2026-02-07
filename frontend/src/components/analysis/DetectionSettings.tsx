"use client";

import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";

interface DetectionSettingsProps {
  confidence: number;
  iou: number;
  onConfidenceChange: (v: number) => void;
  onIouChange: (v: number) => void;
}

export function DetectionSettings({
  confidence,
  iou,
  onConfidenceChange,
  onIouChange,
}: DetectionSettingsProps) {
  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>Confidence Threshold</Label>
          <span className="text-sm text-muted-foreground">{confidence.toFixed(2)}</span>
        </div>
        <Slider
          min={0.05}
          max={0.95}
          step={0.05}
          value={[confidence]}
          onValueChange={([v]) => onConfidenceChange(v)}
        />
      </div>
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label>IoU Threshold</Label>
          <span className="text-sm text-muted-foreground">{iou.toFixed(2)}</span>
        </div>
        <Slider
          min={0.05}
          max={0.95}
          step={0.05}
          value={[iou]}
          onValueChange={([v]) => onIouChange(v)}
        />
      </div>
    </div>
  );
}
