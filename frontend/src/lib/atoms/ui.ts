import { atom } from "jotai";
import { COCOAnnotation } from "../types/coco";
import { selectedFileIdAtom } from "./files";

// ── Inspected annotation (single mode, clicked annotation) ──
export const inspectedAnnotationAtom = atom<COCOAnnotation | null>(null);

// ── Per-file batch image index (Map: fileId → index | null) ──
const batchImageIndexMapAtom = atom<Map<string, number | null>>(new Map());

// ── Read/write atom: selected batch image index for current file ──
export const selectedBatchImageIndexAtom = atom(
  (get) => {
    const fileId = get(selectedFileIdAtom);
    if (!fileId) return null;
    return get(batchImageIndexMapAtom).get(fileId) ?? null;
  },
  (get, set, value: number | null) => {
    const fileId = get(selectedFileIdAtom);
    if (!fileId) return;
    const map = new Map(get(batchImageIndexMapAtom));
    map.set(fileId, value);
    set(batchImageIndexMapAtom, map);
  },
);

// ── Chart modal open state ──
export const chartModalOpenAtom = atom(false);

// ── Which file entry has its settings expanded ──
export const expandedFileIdAtom = atom<string | null>(null);
