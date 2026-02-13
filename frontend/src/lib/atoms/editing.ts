import { atom } from "jotai";
import { COCOJson } from "../types/coco";
import { selectedFileAtom } from "./files";

/** Which annotation is currently being edited (null = not editing) */
export const editingAnnotationIdAtom = atom<number | null>(null);

/**
 * Overlay edits: fileId → (annotationId → edited bbox).
 * Original annotations are never mutated.
 */
export const annotationEditsAtom = atom<
  Map<string, Map<number, [number, number, number, number]>>
>(new Map());

/** Apply (or update) a bbox edit for the current file */
export const applyBboxEditAtom = atom(
  null,
  (
    get,
    set,
    {
      fileId,
      annotationId,
      bbox,
    }: {
      fileId: string;
      annotationId: number;
      bbox: [number, number, number, number];
    },
  ) => {
    const edits = new Map(get(annotationEditsAtom));
    const fileEdits = new Map(edits.get(fileId) ?? []);
    fileEdits.set(annotationId, bbox);
    edits.set(fileId, fileEdits);
    set(annotationEditsAtom, edits);
  },
);

/** Reset a single annotation's edit back to original */
export const resetAnnotationEditAtom = atom(
  null,
  (
    get,
    set,
    { fileId, annotationId }: { fileId: string; annotationId: number },
  ) => {
    const edits = new Map(get(annotationEditsAtom));
    const fileEdits = edits.get(fileId);
    if (!fileEdits) return;
    const next = new Map(fileEdits);
    next.delete(annotationId);
    edits.set(fileId, next);
    set(annotationEditsAtom, edits);
  },
);

/** Derived: effective COCO JSON for the selected file (originals merged with edits) */
export const effectiveCocoJsonAtom = atom<COCOJson | null>((get) => {
  const file = get(selectedFileAtom);
  const coco = file?.singleResult?.coco_json ?? null;
  if (!coco || !file) return coco;

  const fileEdits = get(annotationEditsAtom).get(file.id);
  if (!fileEdits || fileEdits.size === 0) return coco;

  return {
    ...coco,
    annotations: coco.annotations.map((ann) => {
      const editedBbox = fileEdits.get(ann.id);
      if (!editedBbox) return ann;

      const [ox, oy, ow, oh] = ann.bbox;
      const [nx, ny, nw, nh] = editedBbox;

      // Transform segmentation vertices to match the new bbox
      let newSeg = ann.segmentation;
      if (ann.segmentation?.length && ow > 0 && oh > 0) {
        newSeg = ann.segmentation.map((seg) => {
          const t = new Array(seg.length);
          for (let i = 0; i < seg.length; i += 2) {
            t[i] = nx + (seg[i] - ox) * (nw / ow);
            t[i + 1] = ny + (seg[i + 1] - oy) * (nh / oh);
          }
          return t;
        });
      }

      return {
        ...ann,
        bbox: editedBbox,
        area: editedBbox[2] * editedBbox[3],
        segmentation: newSeg,
      };
    }),
  };
});

/** Does the current file have any bbox edits? */
export const hasEditsAtom = atom<boolean>((get) => {
  const file = get(selectedFileAtom);
  if (!file) return false;
  const fileEdits = get(annotationEditsAtom).get(file.id);
  return !!fileEdits && fileEdits.size > 0;
});

/** Does a specific annotation in the current file have an edit? */
export const hasAnnotationEditAtom = atom<(annotationId: number) => boolean>(
  (get) => {
    const file = get(selectedFileAtom);
    if (!file) return () => false;
    const fileEdits = get(annotationEditsAtom).get(file.id);
    if (!fileEdits) return () => false;
    return (annotationId: number) => fileEdits.has(annotationId);
  },
);
