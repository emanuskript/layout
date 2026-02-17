"use client";

import { useAtomValue } from "jotai";

import { Button } from "@/components/ui/button";
import { apiUrl } from "@/lib/api/client";
import {
  selectedFileAtom,
  effectiveCocoJsonAtom,
  colorMapAtom,
  selectedClassNamesAtom,
} from "@/lib/atoms";
import { downloadAnnotatedImage } from "@/lib/utils/downloadAnnotatedImage";

interface DownloadButtonsProps {
  taskId: string;
  isBatch?: boolean;
}

export function DownloadButtons({ taskId, isBatch = false }: DownloadButtonsProps) {
  const selectedFile = useAtomValue(selectedFileAtom);
  const effectiveCocoJson = useAtomValue(effectiveCocoJsonAtom);
  const colorMap = useAtomValue(colorMapAtom);
  const selectedClasses = useAtomValue(selectedClassNamesAtom);

  const handleDownloadImage = () => {
    if (!selectedFile || !effectiveCocoJson) return;
    const file = selectedFile.file;
    const url = URL.createObjectURL(file);
    downloadAnnotatedImage(
      url,
      effectiveCocoJson,
      colorMap,
      selectedClasses,
      `annotated_${file.name.replace(/\.[^.]+$/, "")}.jpg`,
    );
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  };

  return (
    <div className="flex flex-wrap gap-2">
      <Button variant="outline" size="sm" asChild>
        <a href={apiUrl(`/download/${taskId}/coco_json`)} download>
          Download COCO JSON
        </a>
      </Button>
      {!isBatch && effectiveCocoJson && (
        <Button variant="outline" size="sm" onClick={handleDownloadImage}>
          Download Annotated Image
        </Button>
      )}
      {isBatch && (
        <Button variant="outline" size="sm" asChild>
          <a href={apiUrl(`/download/${taskId}/results_zip`)} download>
            Download All (ZIP)
          </a>
        </Button>
      )}
    </div>
  );
}
