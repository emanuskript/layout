"use client";

import { DetectionSettings } from "@/components/analysis/DetectionSettings";
import { ProgressBar } from "@/components/shared/ProgressBar";
import { FileEntry } from "@/lib/atoms/types";

interface FileEntryItemProps {
  entry: FileEntry;
  isSelected: boolean;
  isExpanded: boolean;
  onSelect: () => void;
  onToggleExpand: () => void;
  onRemove: () => void;
  onRun: () => void;
  onUpdateSettings: (patch: { confidence?: number; iou?: number }) => void;
  /** Currently selected batch image index (ZIP only) */
  selectedBatchImageIndex: number | null;
  /** Callback when a child batch image is clicked */
  onSelectBatchImage: (index: number) => void;
}

const statusConfig: Record<
  FileEntry["status"],
  { label: string; dotClass: string; textClass: string }
> = {
  idle: { label: "Idle", dotClass: "bg-gray-400", textClass: "text-muted-foreground" },
  analyzing: { label: "Analyzing", dotClass: "bg-blue-500 animate-pulse", textClass: "text-blue-600" },
  done: { label: "Done", dotClass: "bg-green-500", textClass: "text-green-600" },
  error: { label: "Error", dotClass: "bg-red-500", textClass: "text-red-600" },
};

export function FileEntryItem({
  entry,
  isSelected,
  isExpanded,
  onSelect,
  onToggleExpand,
  onRemove,
  onRun,
  onUpdateSettings,
  selectedBatchImageIndex,
  onSelectBatchImage,
}: FileEntryItemProps) {
  const { label, dotClass, textClass } = statusConfig[entry.status];

  const runLabel =
    entry.status === "done"
      ? "Re-run"
      : entry.status === "error"
        ? "Retry"
        : "Run";

  return (
    <div
      className={`group border-b transition-colors ${
        isSelected ? "border-l-2 border-l-primary bg-primary/5" : "border-l-2 border-l-transparent hover:bg-muted/50"
      }`}
    >
      {/* Compact row */}
      <div className="flex items-center gap-2 px-2 py-2 cursor-pointer" onClick={onSelect}>
        {/* Thumbnail or ZIP icon */}
        <div className="h-12 w-10 flex-shrink-0 overflow-hidden rounded bg-muted">
          {entry.kind === "image" && entry.thumbnailUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={entry.thumbnailUrl}
              alt={entry.file.name}
              className="h-full w-full object-cover"
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center text-muted-foreground">
              <svg className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
              </svg>
            </div>
          )}
        </div>

        {/* File info */}
        <div className="flex-1 min-w-0">
          <p className="truncate text-sm font-medium" title={entry.file.name}>
            {entry.file.name}
          </p>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">
              {formatSize(entry.file.size)}
            </span>
            <span className={`flex items-center gap-1 text-xs ${textClass}`}>
              <span className={`inline-block h-1.5 w-1.5 rounded-full ${dotClass}`} />
              {label}
            </span>
          </div>
        </div>

        {/* Remove button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          className="flex-shrink-0 rounded p-1 text-muted-foreground/0 transition-colors group-hover:text-muted-foreground hover:!text-destructive hover:bg-destructive/10"
          title="Remove file"
        >
          <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Settings toggle + run button row */}
      <div className="flex items-center justify-between px-2 pb-1">
        <button
          onClick={(e) => {
            e.stopPropagation();
            onToggleExpand();
          }}
          className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        >
          <svg
            className={`h-3 w-3 transition-transform ${isExpanded ? "rotate-90" : ""}`}
            fill="none"
            stroke="currentColor"
            strokeWidth={2}
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          Settings
        </button>
        {entry.status !== "analyzing" && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onRun();
            }}
            className="flex items-center gap-1 text-xs text-primary hover:text-primary/80"
          >
            <svg className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
            </svg>
            {runLabel}
          </button>
        )}
      </div>

      {/* Batch progress (inline) */}
      {entry.status === "analyzing" && entry.kind === "zip" && entry.batchProgress && (
        <div className="px-3 pb-2">
          <ProgressBar progress={entry.batchProgress} />
        </div>
      )}

      {/* Error message */}
      {entry.status === "error" && entry.error && (
        <div className="mx-2 mb-2 rounded bg-destructive/5 px-2 py-1 text-xs text-destructive">
          {entry.error}
        </div>
      )}

      {/* Expanded settings */}
      {isExpanded && (
        <div className="border-t bg-muted/20 px-3 py-2">
          <DetectionSettings
            confidence={entry.confidence}
            iou={entry.iou}
            onConfidenceChange={(v) => onUpdateSettings({ confidence: v })}
            onIouChange={(v) => onUpdateSettings({ iou: v })}
          />
        </div>
      )}

      {/* Batch image children (ZIP only, after results) */}
      {entry.kind === "zip" && entry.batchResults && (
        <div className="border-t">
          {entry.batchResults.gallery.map((g, idx) => {
            const thumbSrc = entry.extractedImages?.get(g.filename) ?? g.annotated_url;
            const isActive = isSelected && selectedBatchImageIndex === idx;
            return (
              <button
                key={idx}
                onClick={() => onSelectBatchImage(idx)}
                className={`flex w-full items-center gap-2 py-1 pl-6 pr-2 text-left transition-colors ${
                  isActive
                    ? "bg-primary/10 text-foreground"
                    : "text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                }`}
              >
                <div className="h-8 w-7 flex-shrink-0 overflow-hidden rounded bg-muted">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={thumbSrc}
                    alt={g.filename}
                    className="h-full w-full object-cover"
                  />
                </div>
                <span className="truncate text-xs">{g.filename}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
