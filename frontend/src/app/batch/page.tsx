"use client";

import { useMemo, useState } from "react";

import { ClassSelector } from "@/components/analysis/ClassSelector";
import { DetectionSettings } from "@/components/analysis/DetectionSettings";
import { ZipUploader } from "@/components/analysis/ZipUploader";
import { DownloadButtons } from "@/components/results/DownloadButtons";
import { ImageGallery } from "@/components/results/ImageGallery";
import { StatsChart } from "@/components/results/StatsChart";
import { StatsTable } from "@/components/results/StatsTable";
import { ProgressBar } from "@/components/shared/ProgressBar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useClassSelection } from "@/lib/hooks/useClassSelection";
import { useBatchProcessing } from "@/lib/hooks/useBatchProcessing";

export default function BatchPage() {
  const [file, setFile] = useState<File | null>(null);
  const [confidence, setConfidence] = useState(0.25);
  const [iou, setIou] = useState(0.3);
  const { classes, selected, selectedNames, toggle, selectAll, unselectAll, loading: classesLoading } =
    useClassSelection();
  const { run, loading, error, progress, results, taskId, reset } = useBatchProcessing();

  const colorMap = useMemo(
    () => new Map(classes.map((c) => [c.name, c.color])),
    [classes],
  );

  const handleRun = () => {
    if (!file) return;
    run(file, confidence, iou, selectedNames.length < classes.length ? selectedNames : undefined);
  };

  const handleReset = () => {
    reset();
    setFile(null);
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Batch Processing</h1>

      {!results && !loading ? (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Upload ZIP Archive</CardTitle>
            </CardHeader>
            <CardContent>
              <ZipUploader file={file} onFileChange={setFile} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Detection Settings</CardTitle>
            </CardHeader>
            <CardContent>
              <DetectionSettings
                confidence={confidence}
                iou={iou}
                onConfidenceChange={setConfidence}
                onIouChange={setIou}
              />
            </CardContent>
          </Card>

          {!classesLoading && (
            <Card>
              <CardHeader>
                <CardTitle>Class Filter</CardTitle>
              </CardHeader>
              <CardContent>
                <ClassSelector
                  classes={classes}
                  selected={selected}
                  onToggle={toggle}
                  onSelectAll={selectAll}
                  onUnselectAll={unselectAll}
                />
              </CardContent>
            </Card>
          )}

          {error && (
            <div className="rounded-lg border border-destructive bg-destructive/5 p-4 text-sm text-destructive">
              {error}
            </div>
          )}

          <Button onClick={handleRun} disabled={!file || loading || selected.size === 0} size="lg">
            Start Batch Processing
          </Button>
        </>
      ) : loading && progress ? (
        <Card>
          <CardHeader>
            <CardTitle>Processing...</CardTitle>
          </CardHeader>
          <CardContent>
            <ProgressBar progress={progress} />
          </CardContent>
        </Card>
      ) : results ? (
        <>
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">
              Results — {results.total_processed} images processed
            </h2>
            <Button variant="outline" onClick={handleReset}>
              New Batch
            </Button>
          </div>

          {results.errors.length > 0 && (
            <div className="rounded-lg border border-destructive bg-destructive/5 p-4 text-sm">
              <p className="font-medium text-destructive">Errors ({results.errors.length}):</p>
              <ul className="mt-1 list-inside list-disc text-destructive/80">
                {results.errors.map((e, i) => (
                  <li key={i}>{e}</li>
                ))}
              </ul>
            </div>
          )}

          {taskId && <DownloadButtons taskId={taskId} isBatch />}

          <Card>
            <CardHeader>
              <CardTitle>Gallery</CardTitle>
            </CardHeader>
            <CardContent>
              <ImageGallery gallery={results.gallery} />
            </CardContent>
          </Card>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Summary Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <StatsTable stats={results.stats_summary} colorMap={colorMap} />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <StatsChart stats={results.stats_summary} colorMap={colorMap} />
              </CardContent>
            </Card>
          </div>

          {results.stats_per_image.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Per-Image Statistics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="max-h-96 overflow-auto">
                  {results.stats_per_image.map((item, idx) => (
                    <details key={idx} className="mb-2">
                      <summary className="cursor-pointer rounded px-2 py-1 text-sm font-medium hover:bg-muted">
                        {item.image}
                      </summary>
                      <div className="ml-4 mt-1">
                        <StatsTable stats={item.stats} colorMap={colorMap} />
                      </div>
                    </details>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </>
      ) : null}
    </div>
  );
}
