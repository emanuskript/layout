"use client";

import { Button } from "@/components/ui/button";
import { apiUrl } from "@/lib/api/client";

interface DownloadButtonsProps {
  taskId: string;
  isBatch?: boolean;
}

export function DownloadButtons({ taskId, isBatch = false }: DownloadButtonsProps) {
  return (
    <div className="flex flex-wrap gap-2">
      <Button variant="outline" size="sm" asChild>
        <a href={apiUrl(`/download/${taskId}/coco_json`)} download>
          Download COCO JSON
        </a>
      </Button>
      {!isBatch && (
        <Button variant="outline" size="sm" asChild>
          <a href={apiUrl(`/download/${taskId}/annotated_image`)} download>
            Download Annotated Image
          </a>
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
