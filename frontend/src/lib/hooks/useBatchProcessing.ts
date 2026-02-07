"use client";

import { useCallback, useRef, useState } from "react";

import { getBatchResults, predictBatch } from "../api/predict";
import { subscribeBatchProgress } from "../api/sse";
import { BatchProgress, BatchResults } from "../types/predict";

export function useBatchProcessing() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<BatchProgress | null>(null);
  const [results, setResults] = useState<BatchResults | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const unsubRef = useRef<(() => void) | null>(null);

  const run = useCallback(
    async (zipFile: File, confidence: number, iou: number, classes?: string[]) => {
      setLoading(true);
      setError(null);
      setProgress(null);
      setResults(null);

      try {
        const { task_id } = await predictBatch(zipFile, confidence, iou, classes);
        setTaskId(task_id);

        unsubRef.current = subscribeBatchProgress(
          task_id,
          async (data) => {
            setProgress(data);
            if (data.status === "completed") {
              try {
                const res = await getBatchResults(task_id);
                setResults(res);
              } catch (err) {
                setError(err instanceof Error ? err.message : "Failed to fetch results");
              } finally {
                setLoading(false);
              }
            } else if (data.status === "error") {
              setError(data.message || "Batch processing failed");
              setLoading(false);
            }
          },
          () => {
            setError("Lost connection to server");
            setLoading(false);
          },
        );
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to start batch");
        setLoading(false);
      }
    },
    [],
  );

  const reset = useCallback(() => {
    unsubRef.current?.();
    setResults(null);
    setProgress(null);
    setError(null);
  }, []);

  return { run, loading, error, progress, results, taskId, reset };
}
