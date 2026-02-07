"use client";

import { useCallback, useState } from "react";

import { predictSingle } from "../api/predict";
import { PredictSingleResponse } from "../types/predict";

export function usePrediction() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictSingleResponse | null>(null);

  const run = useCallback(
    async (image: File, confidence: number, iou: number, classes?: string[]) => {
      setLoading(true);
      setError(null);
      setResult(null);
      try {
        const data = await predictSingle(image, confidence, iou, classes);
        setResult(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Prediction failed");
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { run, loading, error, result, reset };
}
