"use client";

import { useState } from "react";
import { useSetAtom } from "jotai";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  GraduationCap,
  Upload,
  Grid3X3,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";
import { skipTourAtom } from "@/lib/atoms";

interface TourWelcomeProps {
  onStart: () => void;
  onSkip: () => void;
}

const features = [
  { icon: Upload, text: "Upload images or ZIP archives for analysis" },
  { icon: Grid3X3, text: "Detect manuscript layout elements with YOLO" },
  { icon: SlidersHorizontal, text: "Fine-tune confidence and IoU thresholds" },
  { icon: Sparkles, text: "Interactive canvas with pan, zoom, and inspect" },
];

export function TourWelcome({ onStart, onSkip }: TourWelcomeProps) {
  const skipTour = useSetAtom(skipTourAtom);
  const [dontShow, setDontShow] = useState(false);

  function handleSkip() {
    if (dontShow) {
      skipTour();
    }
    onSkip();
  }

  function handleOpenChange(open: boolean) {
    if (!open) handleSkip();
  }

  return (
    <Dialog open onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[440px]" showCloseButton={false}>
        <DialogHeader className="text-center sm:text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 text-primary">
            <GraduationCap className="h-8 w-8" />
          </div>
          <DialogTitle className="text-center text-xl">
            Welcome to Layout Analysis!
          </DialogTitle>
          <DialogDescription className="text-center">
            Detect and annotate layout elements in medieval manuscripts with
            powerful YOLO models. Would you like a quick tour?
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3 rounded-lg bg-muted p-4">
          {features.map((f) => (
            <div key={f.text} className="flex items-center gap-3 text-sm text-foreground">
              <f.icon className="h-[18px] w-[18px] shrink-0 text-primary" />
              <span>{f.text}</span>
            </div>
          ))}
        </div>

        <DialogFooter className="flex-col gap-4 sm:flex-col">
          <div className="flex items-center gap-2">
            <Checkbox
              id="dont-show"
              checked={dontShow}
              onCheckedChange={(v) => setDontShow(v === true)}
            />
            <label
              htmlFor="dont-show"
              className="text-[13px] text-muted-foreground cursor-pointer"
            >
              Don&apos;t show this again
            </label>
          </div>

          <div className="flex w-full gap-2">
            <Button variant="ghost" className="flex-1" onClick={handleSkip}>
              Skip for now
            </Button>
            <Button className="flex-1" onClick={onStart}>
              Start Tour
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
