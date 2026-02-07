import { COCOAnnotation, COCOCategory, COCOImage } from "../types/coco";

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function drawAnnotations(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  annotations: COCOAnnotation[],
  categories: COCOCategory[],
  colorMap: Map<string, string>,
  selectedClasses: Set<string>,
  imageInfo?: COCOImage,
) {
  const imgW = imageInfo?.width ?? image.naturalWidth;
  const imgH = imageInfo?.height ?? image.naturalHeight;

  ctx.canvas.width = imgW;
  ctx.canvas.height = imgH;
  ctx.drawImage(image, 0, 0, imgW, imgH);

  const catById = new Map(categories.map((c) => [c.id, c.name]));

  for (const ann of annotations) {
    const className = catById.get(ann.category_id);
    if (!className || !selectedClasses.has(className)) continue;

    const color = colorMap.get(className) || "#888888";

    // Draw segmentation polygons
    if (ann.segmentation && ann.segmentation.length > 0) {
      for (const seg of ann.segmentation) {
        if (seg.length < 6) continue;
        ctx.beginPath();
        ctx.moveTo(seg[0], seg[1]);
        for (let i = 2; i < seg.length; i += 2) {
          ctx.lineTo(seg[i], seg[i + 1]);
        }
        ctx.closePath();
        ctx.fillStyle = hexToRgba(color, 0.25);
        ctx.fill();
        ctx.strokeStyle = hexToRgba(color, 0.8);
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    } else if (ann.bbox) {
      // Fallback to bbox
      const [x, y, w, h] = ann.bbox;
      ctx.fillStyle = hexToRgba(color, 0.25);
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = hexToRgba(color, 0.8);
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
    }

    // Draw label
    if (className) {
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
