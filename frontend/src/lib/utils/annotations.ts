import { COCOAnnotation, COCOCategory, COCOImage } from "../types/coco";

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export interface DrawOptions {
  fillOpacity?: number;
  strokeOpacity?: number;
  highlightedId?: number | null;
  selectedId?: number | null;
  editingId?: number | null;
  showLabels?: boolean;
}

export function drawAnnotations(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  annotations: COCOAnnotation[],
  categories: COCOCategory[],
  colorMap: Map<string, string>,
  selectedClasses: Set<string>,
  imageInfo?: COCOImage,
  options?: DrawOptions,
) {
  const imgW = imageInfo?.width ?? image.naturalWidth;
  const imgH = imageInfo?.height ?? image.naturalHeight;
  const fillOpacity = options?.fillOpacity ?? 0.25;
  const strokeOpacity = options?.strokeOpacity ?? 0.8;
  const highlightedId = options?.highlightedId ?? null;
  const showLabels = options?.showLabels ?? true;

  ctx.canvas.width = imgW;
  ctx.canvas.height = imgH;
  ctx.drawImage(image, 0, 0, imgW, imgH);

  const catById = new Map(categories.map((c) => [c.id, c.name]));

  for (const ann of annotations) {
    const className = catById.get(ann.category_id);
    if (!className || !selectedClasses.has(className)) continue;

    const color = colorMap.get(className) || "#888888";
    const isHighlighted = highlightedId === ann.id;
    const currentFill = isHighlighted ? Math.min(fillOpacity * 2, 0.5) : fillOpacity;
    const currentStroke = isHighlighted ? 1.0 : strokeOpacity;
    const lineWidth = isHighlighted ? 3 : 2;

    if (ann.segmentation && ann.segmentation.length > 0) {
      for (const seg of ann.segmentation) {
        if (seg.length < 6) continue;
        ctx.beginPath();
        ctx.moveTo(seg[0], seg[1]);
        for (let i = 2; i < seg.length; i += 2) {
          ctx.lineTo(seg[i], seg[i + 1]);
        }
        ctx.closePath();
        ctx.fillStyle = hexToRgba(color, currentFill);
        ctx.fill();
        ctx.strokeStyle = hexToRgba(color, currentStroke);
        ctx.lineWidth = lineWidth;
        ctx.stroke();
      }
    } else if (ann.bbox) {
      const [x, y, w, h] = ann.bbox;
      ctx.fillStyle = hexToRgba(color, currentFill);
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = hexToRgba(color, currentStroke);
      ctx.lineWidth = lineWidth;
      ctx.strokeRect(x, y, w, h);
    }

    if (showLabels && className) {
      const labelX = ann.bbox ? ann.bbox[0] : ann.segmentation?.[0]?.[0] ?? 0;
      const labelY = ann.bbox ? ann.bbox[1] : ann.segmentation?.[0]?.[1] ?? 0;
      const fontSize = Math.max(12, Math.min(16, imgW / 80));
      ctx.font = `bold ${fontSize}px sans-serif`;
      const metrics = ctx.measureText(className);
      const pad = 3;
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.fillRect(labelX, labelY - fontSize - pad, metrics.width + pad * 2, fontSize + pad * 2);
      ctx.fillStyle = color;
      ctx.fillText(className, labelX + pad, labelY - pad);
    }
  }
}

/**
 * Draw image + annotations with a pan/zoom transform applied.
 * All coordinates are in image-space; the ctx transform handles the viewport.
 */
export function drawAnnotationsTransformed(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  annotations: COCOAnnotation[],
  categories: COCOCategory[],
  colorMap: Map<string, string>,
  selectedClasses: Set<string>,
  scale: number,
  offsetX: number,
  offsetY: number,
  imageInfo?: COCOImage,
  options?: DrawOptions,
) {
  const imgW = imageInfo?.width ?? image.naturalWidth;
  const imgH = imageInfo?.height ?? image.naturalHeight;
  const fillOpacity = options?.fillOpacity ?? 0.25;
  const strokeOpacity = options?.strokeOpacity ?? 0.8;
  const highlightedId = options?.highlightedId ?? null;
  const selectedId = options?.selectedId ?? null;
  const editingId = options?.editingId ?? null;
  const showLabels = options?.showLabels ?? true;

  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);

  ctx.drawImage(image, 0, 0, imgW, imgH);

  const catById = new Map(categories.map((c) => [c.id, c.name]));

  for (const ann of annotations) {
    const className = catById.get(ann.category_id);
    if (!className || !selectedClasses.has(className)) continue;

    const isEditing = editingId === ann.id;
    const isSelected = selectedId === ann.id;
    const isHighlighted = highlightedId === ann.id;

    // When editing an annotation, hide all other annotations
    if (editingId != null && !isEditing) continue;

    // Editing annotation gets blue, selected gets a boosted class color
    const color = isEditing ? "#3b82f6" : (colorMap.get(className) || "#888888");
    const currentFill = isEditing
      ? Math.min(fillOpacity * 2.5, 0.4)
      : isSelected
        ? Math.min(fillOpacity * 2.5, 0.45)
        : isHighlighted
          ? Math.min(fillOpacity * 2, 0.5)
          : fillOpacity;
    const currentStroke = (isEditing || isSelected) ? 1.0 : isHighlighted ? 1.0 : strokeOpacity;
    const lineWidth = (isEditing ? 2.5 : (isSelected || isHighlighted) ? 3 : 2) / scale; // constant screen-space width

    if (ann.segmentation && ann.segmentation.length > 0) {
      for (const seg of ann.segmentation) {
        if (seg.length < 6) continue;
        ctx.beginPath();
        ctx.moveTo(seg[0], seg[1]);
        for (let i = 2; i < seg.length; i += 2) {
          ctx.lineTo(seg[i], seg[i + 1]);
        }
        ctx.closePath();
        ctx.fillStyle = hexToRgba(color, currentFill);
        ctx.fill();
        ctx.strokeStyle = hexToRgba(color, currentStroke);
        ctx.lineWidth = lineWidth;
        ctx.stroke();
      }
    } else if (ann.bbox) {
      const [x, y, w, h] = ann.bbox;
      ctx.fillStyle = hexToRgba(color, currentFill);
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = hexToRgba(color, currentStroke);
      ctx.lineWidth = lineWidth;
      ctx.strokeRect(x, y, w, h);
    }

    if (showLabels && className) {
      const labelX = ann.bbox ? ann.bbox[0] : ann.segmentation?.[0]?.[0] ?? 0;
      const labelY = ann.bbox ? ann.bbox[1] : ann.segmentation?.[0]?.[1] ?? 0;
      const fontSize = Math.max(12, Math.min(16, imgW / 80)) / scale;
      ctx.font = `bold ${fontSize}px sans-serif`;
      const metrics = ctx.measureText(className);
      const pad = 3 / scale;
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.fillRect(labelX, labelY - fontSize - pad, metrics.width + pad * 2, fontSize + pad * 2);
      ctx.fillStyle = color;
      ctx.fillText(className, labelX + pad, labelY - pad);
    }
  }

  ctx.restore();
}

/**
 * Hit-test: find which annotation is at (x, y) in image coordinates.
 */
export function hitTestAnnotation(
  x: number,
  y: number,
  annotations: COCOAnnotation[],
  categories: COCOCategory[],
  selectedClasses: Set<string>,
): COCOAnnotation | null {
  const catById = new Map(categories.map((c) => [c.id, c.name]));

  // Iterate in reverse so topmost (last-drawn) annotation is hit first
  for (let i = annotations.length - 1; i >= 0; i--) {
    const ann = annotations[i];
    const className = catById.get(ann.category_id);
    if (!className || !selectedClasses.has(className)) continue;

    if (ann.segmentation && ann.segmentation.length > 0) {
      for (const seg of ann.segmentation) {
        if (pointInPolygon(x, y, seg)) return ann;
      }
    }

    if (ann.bbox) {
      const [bx, by, bw, bh] = ann.bbox;
      if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) return ann;
    }
  }
  return null;
}

// ── BBox editing helpers ──

export type HandleType =
  | "tl"
  | "t"
  | "tr"
  | "r"
  | "br"
  | "b"
  | "bl"
  | "l"
  | "move"
  | `vertex-${number}-${number}`; // vertex-segIndex-pointIndex

interface HandlePosition {
  type: HandleType;
  x: number;
  y: number;
}

/** Return the 8 handle positions for a bbox [x, y, w, h]. */
export function getHandlePositions(
  bbox: [number, number, number, number],
): HandlePosition[] {
  const [x, y, w, h] = bbox;
  return [
    { type: "tl", x, y },
    { type: "t", x: x + w / 2, y },
    { type: "tr", x: x + w, y },
    { type: "r", x: x + w, y: y + h / 2 },
    { type: "br", x: x + w, y: y + h },
    { type: "b", x: x + w / 2, y: y + h },
    { type: "bl", x, y: y + h },
    { type: "l", x, y: y + h / 2 },
  ];
}

const HANDLE_SCREEN_RADIUS = 5; // px in screen space

/**
 * Test if (imageX, imageY) is over a resize handle.
 * Returns the handle type or null.
 */
export function hitTestHandle(
  imageX: number,
  imageY: number,
  bbox: [number, number, number, number],
  scale: number,
): HandleType | null {
  const radius = HANDLE_SCREEN_RADIUS / scale;
  for (const h of getHandlePositions(bbox)) {
    if (
      Math.abs(imageX - h.x) <= radius &&
      Math.abs(imageY - h.y) <= radius
    ) {
      return h.type;
    }
  }
  return null;
}

/**
 * Draw 8 resize handles around a bbox.
 * Called inside the canvas transform (image-space), so size is divided by scale
 * to appear constant on screen.
 */
export function drawEditHandles(
  ctx: CanvasRenderingContext2D,
  bbox: [number, number, number, number],
  scale: number,
) {
  const size = HANDLE_SCREEN_RADIUS / scale;
  const handles = getHandlePositions(bbox);

  // Draw dashed bbox outline
  const [x, y, w, h] = bbox;
  ctx.save();
  ctx.setLineDash([6 / scale, 4 / scale]);
  ctx.strokeStyle = "#3b82f6";
  ctx.lineWidth = 2 / scale;
  ctx.strokeRect(x, y, w, h);
  ctx.restore();

  // Draw handles
  for (const handle of handles) {
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 1.5 / scale;
    ctx.fillRect(handle.x - size, handle.y - size, size * 2, size * 2);
    ctx.strokeRect(handle.x - size, handle.y - size, size * 2, size * 2);
  }
}

const VERTEX_SCREEN_RADIUS = 7; // px in screen space
const TARGET_VERTEX_COUNT = 8; // Number of control handles to display

/** Compute the decimation step for a given segment */
function getDecimation(segLength: number): number {
  const totalPoints = segLength / 2;
  return Math.max(1, Math.floor(totalPoints / TARGET_VERTEX_COUNT));
}

/**
 * Hit-test segmentation polygon vertices.
 * Returns a handle type like 'vertex-0-2' (segIndex-pointIndex) or null.
 */
export function hitTestVertex(
  imageX: number,
  imageY: number,
  segmentation: number[][],
  scale: number,
): HandleType | null {
  const radius = VERTEX_SCREEN_RADIUS / scale;
  for (let si = 0; si < segmentation.length; si++) {
    const seg = segmentation[si];
    const decimation = getDecimation(seg.length);
    for (let pi = 0; pi < seg.length; pi += 2 * decimation) {
      if (
        Math.abs(imageX - seg[pi]) <= radius &&
        Math.abs(imageY - seg[pi + 1]) <= radius
      ) {
        return `vertex-${si}-${pi}` as HandleType;
      }
    }
  }
  return null;
}

/**
 * Draw draggable vertex handles on polygon segmentation points.
 */
export function drawVertexHandles(
  ctx: CanvasRenderingContext2D,
  segmentation: number[][],
  scale: number,
) {
  const r = VERTEX_SCREEN_RADIUS / scale;
  for (const seg of segmentation) {
    const decimation = getDecimation(seg.length);
    for (let i = 0; i < seg.length; i += 2 * decimation) {
      ctx.beginPath();
      ctx.arc(seg[i], seg[i + 1], r, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = 1.5 / scale;
      ctx.stroke();
    }
  }
}

/**
 * Apply a drag to a segmentation vertex with smooth falloff.
 * Nearby vertices move proportionally so the shape deforms naturally
 * instead of creating a spike at the dragged point.
 * Returns a new segmentation array (immutable).
 */
export function applyVertexDrag(
  segmentation: number[][],
  handle: HandleType,
  deltaX: number,
  deltaY: number,
): number[][] {
  const match = (handle as string).match(/^vertex-(\d+)-(\d+)$/);
  if (!match) return segmentation;
  const segIdx = parseInt(match[1], 10);
  const ptIdx = parseInt(match[2], 10);

  return segmentation.map((seg, si) => {
    if (si !== segIdx) return seg;

    const totalPoints = seg.length / 2;
    const decimation = getDecimation(seg.length);
    // Influence radius: half the distance between two displayed handles (in point indices)
    const influenceRadius = decimation;
    const dragPointIdx = ptIdx / 2; // Convert flat index to point index

    const newSeg = [...seg];
    for (let p = 0; p < totalPoints; p++) {
      // Compute shortest distance on the polygon ring
      let dist = Math.abs(p - dragPointIdx);
      dist = Math.min(dist, totalPoints - dist); // wrap around

      if (dist <= influenceRadius) {
        // Smooth cosine falloff: 1 at center, 0 at edge
        const t = 1 - dist / influenceRadius;
        const influence = t * t * (3 - 2 * t); // smoothstep
        newSeg[p * 2] = seg[p * 2] + deltaX * influence;
        newSeg[p * 2 + 1] = seg[p * 2 + 1] + deltaY * influence;
      }
    }
    return newSeg;
  });
}

/** Recompute bbox from segmentation polygons */
export function bboxFromSegmentation(
  segmentation: number[][],
): [number, number, number, number] {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const seg of segmentation) {
    for (let i = 0; i < seg.length; i += 2) {
      minX = Math.min(minX, seg[i]);
      minY = Math.min(minY, seg[i + 1]);
      maxX = Math.max(maxX, seg[i]);
      maxY = Math.max(maxY, seg[i + 1]);
    }
  }
  return [minX, minY, maxX - minX, maxY - minY];
}

/**
 * Compute new bbox after dragging a handle by (deltaX, deltaY) in image coords.
 * Ensures width/height stay positive (minimum 1px).
 */
export function applyHandleDrag(
  bbox: [number, number, number, number],
  handle: HandleType,
  deltaX: number,
  deltaY: number,
): [number, number, number, number] {
  let [x, y, w, h] = bbox;

  switch (handle) {
    case "tl":
      x += deltaX;
      y += deltaY;
      w -= deltaX;
      h -= deltaY;
      break;
    case "t":
      y += deltaY;
      h -= deltaY;
      break;
    case "tr":
      y += deltaY;
      w += deltaX;
      h -= deltaY;
      break;
    case "r":
      w += deltaX;
      break;
    case "br":
      w += deltaX;
      h += deltaY;
      break;
    case "b":
      h += deltaY;
      break;
    case "bl":
      x += deltaX;
      w -= deltaX;
      h += deltaY;
      break;
    case "l":
      x += deltaX;
      w -= deltaX;
      break;
    case "move":
      x += deltaX;
      y += deltaY;
      break;
  }

  // Enforce minimum dimensions
  if (w < 1) {
    w = 1;
  }
  if (h < 1) {
    h = 1;
  }

  return [x, y, w, h];
}

/** Check if a handle type refers to a polygon vertex */
export function isVertexHandle(handle: HandleType | null): boolean {
  return typeof handle === "string" && handle.startsWith("vertex-");
}

/** Map handle type to CSS cursor */
export function handleCursor(handle: HandleType | null): string {
  if (isVertexHandle(handle)) return "crosshair";
  switch (handle) {
    case "tl":
    case "br":
      return "nwse-resize";
    case "tr":
    case "bl":
      return "nesw-resize";
    case "t":
    case "b":
      return "ns-resize";
    case "l":
    case "r":
      return "ew-resize";
    case "move":
      return "move";
    default:
      return "";
  }
}

function pointInPolygon(px: number, py: number, polygon: number[]): boolean {
  let inside = false;
  const n = polygon.length / 2;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i * 2];
    const yi = polygon[i * 2 + 1];
    const xj = polygon[j * 2];
    const yj = polygon[j * 2 + 1];
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}
