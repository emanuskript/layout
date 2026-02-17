import { COCOJson } from "../types/coco";
import { drawAnnotations } from "./annotations";

/**
 * Render the image with all visible annotations at full resolution
 * and trigger a high-quality JPEG download.
 */
export function downloadAnnotatedImage(
  imageSrc: string,
  cocoJson: COCOJson,
  colorMap: Map<string, string>,
  selectedClasses: Set<string>,
  filename = "annotated_image.jpg",
): void {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Draw at full image resolution with annotations
    drawAnnotations(ctx, img, cocoJson.annotations, cocoJson.categories, colorMap, selectedClasses, cocoJson.images[0], {
      fillOpacity: 0.25,
      strokeOpacity: 0.8,
      showLabels: true,
    });

    // Export as highest quality JPEG
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
      },
      "image/jpeg",
      1.0, // maximum quality
    );
  };
  img.src = imageSrc;
}
