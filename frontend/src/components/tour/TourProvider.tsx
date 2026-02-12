"use client";

import { useEffect, useState } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import {
  tourActiveAtom,
  tourHasSeenAtom,
  currentTourStepAtom,
  startTourAtom,
  loadTourState,
} from "@/lib/atoms";
import { TourSpotlight } from "./TourSpotlight";
import { TourWelcome } from "./TourWelcome";

export function TourProvider({ children }: { children: React.ReactNode }) {
  const tourActive = useAtomValue(tourActiveAtom);
  const hasSeenTour = useAtomValue(tourHasSeenAtom);
  const currentStep = useAtomValue(currentTourStepAtom);
  const setHasSeen = useSetAtom(tourHasSeenAtom);
  const startTour = useSetAtom(startTourAtom);

  const [showWelcome, setShowWelcome] = useState(false);
  const [isReady, setIsReady] = useState(false);

  // Load persisted tour state on mount
  useEffect(() => {
    const state = loadTourState();
    if (state.hasSeenTour) {
      setHasSeen(true);
    }
    // Delay readiness to let the workspace render
    const timer = setTimeout(() => setIsReady(true), 1500);
    return () => clearTimeout(timer);
  }, [setHasSeen]);

  // Show welcome dialog for first-time visitors once ready
  useEffect(() => {
    if (isReady && !hasSeenTour) {
      setShowWelcome(true);
    }
  }, [isReady, hasSeenTour]);

  function handleStartTour() {
    setShowWelcome(false);
    // Small delay to let the dialog close before starting spotlight
    setTimeout(() => startTour(), 150);
  }

  function handleSkipWelcome() {
    setShowWelcome(false);
  }

  return (
    <>
      {children}
      {showWelcome && isReady && (
        <TourWelcome onStart={handleStartTour} onSkip={handleSkipWelcome} />
      )}
      {tourActive && currentStep && isReady && <TourSpotlight />}
    </>
  );
}
