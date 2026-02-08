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
  const showLabels = options?.showLabels ?? true;

  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);

  ctx.drawImage(image, 0, 0, imgW, imgH);

  const catById = new Map(categories.map((c) => [c.id, c.name]));

  for (const ann of annotations) {
    const className = catById.get(ann.category_id);
    if (!className || !selectedClasses.has(className)) continue;

    const color = colorMap.get(className) || "#888888";
    const isHighlighted = highlightedId === ann.id;
    const currentFill = isHighlighted ? Math.min(fillOpacity * 2, 0.5) : fillOpacity;
    const currentStroke = isHighlighted ? 1.0 : strokeOpacity;
    const lineWidth = (isHighlighted ? 3 : 2) / scale; // constant screen-space width

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
