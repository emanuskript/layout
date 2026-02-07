"use client";

import { useMemo, useState } from "react";

import { ClassSelector } from "@/components/analysis/ClassSelector";
import { DetectionSettings } from "@/components/analysis/DetectionSettings";
import { ImageUploader } from "@/components/analysis/ImageUploader";
import { AnnotatedImage } from "@/components/results/AnnotatedImage";
import { DownloadButtons } from "@/components/results/DownloadButtons";
import { StatsChart } from "@/components/results/StatsChart";
import { StatsTable } from "@/components/results/StatsTable";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useClassSelection } from "@/lib/hooks/useClassSelection";
import { usePrediction } from "@/lib/hooks/usePrediction";

export default function SinglePage() {
  const [file, setFile] = useState<File | null>(null);
  const [confidence, setConfidence] = useState(0.25);
  const [iou, setIou] = useState(0.3);
  const { classes, selected, selectedNames, toggle, selectAll, unselectAll, loading: classesLoading } =
    useClassSelection();
  const { run, loading, error, result, reset } = usePrediction();

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

  // For canvas rendering, we use the original uploaded image as src
  const imageSrc = useMemo(() => {
    if (file) return URL.createObjectURL(file);
    return "";
  }, [file]);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Single Image Analysis</h1>

      {!result ? (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Upload Image</CardTitle>
            </CardHeader>
            <CardContent>
              <ImageUploader file={file} onFileChange={setFile} />
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
            {loading ? "Analyzing..." : "Run Analysis"}
          </Button>
        </>
      ) : (
        <>
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Results</h2>
            <Button variant="outline" onClick={handleReset}>
              New Analysis
            </Button>
          </div>

          <DownloadButtons taskId={result.task_id} />

          <div className="grid gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>Annotated Image</CardTitle>
                </CardHeader>
                <CardContent>
                  <AnnotatedImage
                    imageSrc={imageSrc}
                    cocoJson={result.coco_json}
                    colorMap={colorMap}
                    selectedClasses={selected}
                  />
                </CardContent>
              </Card>
            </div>

            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Statistics</CardTitle>
                </CardHeader>
                <CardContent>
                  <StatsTable stats={result.stats} colorMap={colorMap} />
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Filter Classes</CardTitle>
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
            </div>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <StatsChart stats={result.stats} colorMap={colorMap} />
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
