"use client";

import { Pencil, Check, RotateCcw } from "lucide-react";
import { COCOAnnotation, COCOCategory, COCOImage } from "@/lib/types/coco";

interface AnnotationInspectorProps {
  annotation: COCOAnnotation;
  categories: COCOCategory[];
  colorMap: Map<string, string>;
  imageInfo?: COCOImage;
  onClose: () => void;
  isEditing?: boolean;
  onEdit?: () => void;
  onDoneEditing?: () => void;
  onReset?: () => void;
  hasEdits?: boolean;
}

export function AnnotationInspector({
  annotation,
  categories,
  colorMap,
  imageInfo,
  onClose,
  isEditing,
  onEdit,
  onDoneEditing,
  onReset,
  hasEdits,
}: AnnotationInspectorProps) {
  const catById = new Map(categories.map((c) => [c.id, c.name]));
  const className = catById.get(annotation.category_id) || "Unknown";
  const color = colorMap.get(className) || "#888888";

  const imgArea = imageInfo ? imageInfo.width * imageInfo.height : 0;
  const coverage = imgArea > 0 ? ((annotation.area / imgArea) * 100).toFixed(1) : "\u2014";

  const segPolygons = annotation.segmentation?.length ?? 0;
  const totalVertices = annotation.segmentation
    ? annotation.segmentation.reduce((sum, seg) => sum + seg.length / 2, 0)
    : 0;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Annotation Details</h3>
        <div className="flex items-center gap-1">
          {/* Edit / Done button */}
          {isEditing ? (
            <button
              onClick={onDoneEditing}
              className="rounded-md p-1 text-primary hover:bg-primary/10"
              aria-label="Done editing"
              title="Done editing"
            >
              <Check className="h-4 w-4" />
            </button>
          ) : (
            onEdit && (
              <button
                onClick={onEdit}
                className="rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
                aria-label="Edit bounding box"
                title="Edit bounding box"
              >
                <Pencil className="h-4 w-4" />
              </button>
            )
          )}
          {/* Reset button (only when editing and has edits) */}
          {isEditing && hasEdits && (
            <button
              onClick={onReset}
              className="rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              aria-label="Reset to original"
              title="Reset to original"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          )}
          {/* Close button */}
          <button
            onClick={onClose}
            className="rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
            aria-label="Close inspector"
          >
            <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      {isEditing && (
        <div className="rounded-md bg-primary/10 px-2.5 py-1.5 text-xs text-primary">
          Drag handles to resize &middot; Drag inside to move
        </div>
      )}

      <div className="space-y-4">
        {/* Class */}
        <div>
          <p className="mb-1 text-xs font-medium text-muted-foreground">Class</p>
          <div className="flex items-center gap-2">
            <span className="inline-block h-3 w-3 rounded-full" style={{ backgroundColor: color }} />
            <span className="text-sm font-medium">{className}</span>
          </div>
        </div>

        {/* Annotation ID */}
        <div>
          <p className="mb-1 text-xs font-medium text-muted-foreground">Annotation ID</p>
          <p className="text-sm">{annotation.id}</p>
        </div>

        {/* Area */}
        <div>
          <p className="mb-1 text-xs font-medium text-muted-foreground">Area</p>
          <p className="text-sm">{Math.round(annotation.area).toLocaleString()} px&sup2;</p>
        </div>

        {/* Coverage */}
        <div>
          <p className="mb-1 text-xs font-medium text-muted-foreground">Image Coverage</p>
          <p className="text-sm">{coverage}%</p>
        </div>

        {/* BBox */}
        {annotation.bbox && (
          <div>
            <p className="mb-1 text-xs font-medium text-muted-foreground">
              Bounding Box
              {hasEdits && <span className="ml-1.5 text-primary">(edited)</span>}
            </p>
            <div className="grid grid-cols-2 gap-1 text-sm">
              <span className="text-muted-foreground">x:</span>
              <span>{Math.round(annotation.bbox[0])}</span>
              <span className="text-muted-foreground">y:</span>
              <span>{Math.round(annotation.bbox[1])}</span>
              <span className="text-muted-foreground">w:</span>
              <span>{Math.round(annotation.bbox[2])}</span>
              <span className="text-muted-foreground">h:</span>
              <span>{Math.round(annotation.bbox[3])}</span>
            </div>
          </div>
        )}

        {/* Segmentation info */}
        {segPolygons > 0 && (
          <div>
            <p className="mb-1 text-xs font-medium text-muted-foreground">Segmentation</p>
            <p className="text-sm">
              {segPolygons} polygon{segPolygons !== 1 ? "s" : ""}, {totalVertices} vertices
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
