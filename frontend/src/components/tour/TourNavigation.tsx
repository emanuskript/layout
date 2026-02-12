"use client";

import { useAtomValue, useSetAtom } from "jotai";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight, Check } from "lucide-react";
import {
  tourIsFirstStepAtom,
  tourIsLastStepAtom,
  nextTourStepAtom,
  prevTourStepAtom,
  skipTourAtom,
} from "@/lib/atoms";

export function TourNavigation() {
  const isFirst = useAtomValue(tourIsFirstStepAtom);
  const isLast = useAtomValue(tourIsLastStepAtom);
  const next = useSetAtom(nextTourStepAtom);
  const prev = useSetAtom(prevTourStepAtom);
  const skip = useSetAtom(skipTourAtom);

  return (
    <div className="flex items-center justify-between border-t px-4 py-3">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => skip()}
        className="text-muted-foreground hover:text-foreground"
      >
        Skip tour
      </Button>

      <div className="flex gap-2">
        {!isFirst && (
          <Button variant="outline" size="sm" onClick={() => prev()}>
            <ChevronLeft className="h-4 w-4" />
            Back
          </Button>
        )}

        <Button size="sm" onClick={() => next()}>
          {isLast ? "Finish" : "Next"}
          {isLast ? (
            <Check className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </Button>
      </div>
    </div>
  );
}
