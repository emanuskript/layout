"use client";

import { useAtomValue, useSetAtom } from "jotai";
import { tourStepIndexAtom, tourTotalStepsAtom, goToTourStepAtom } from "@/lib/atoms";
import { cn } from "@/lib/utils";

export function TourProgress() {
  const currentIndex = useAtomValue(tourStepIndexAtom);
  const totalSteps = useAtomValue(tourTotalStepsAtom);
  const goToStep = useSetAtom(goToTourStepAtom);

  return (
    <div className="flex justify-center gap-1.5 px-4 py-2" role="tablist" aria-label="Tour progress">
      {Array.from({ length: totalSteps }, (_, i) => (
        <button
          key={i}
          role="tab"
          aria-selected={i === currentIndex}
          aria-label={`Go to step ${i + 1}`}
          onClick={() => goToStep(i)}
          className={cn(
            "h-2 rounded-full border-none transition-all duration-200",
            i === currentIndex
              ? "w-6 bg-primary"
              : "w-2 cursor-pointer bg-muted hover:bg-muted-foreground",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
          )}
        />
      ))}
    </div>
  );
}
