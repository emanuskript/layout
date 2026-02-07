"use client";

import { useEffect, useRef } from "react";

import { COCOJson } from "@/lib/types/coco";
import { drawAnnotations } from "@/lib/utils/annotations";

interface AnnotatedImageProps {
  imageSrc: string;
  cocoJson: COCOJson;
  colorMap: Map<string, string>;
  selectedClasses: Set<string>;
}

export function AnnotatedImage({
  imageSrc,
  cocoJson,
  colorMap,
  selectedClasses,
}: AnnotatedImageProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imgRef.current = img;
      const ctx = canvasRef.current?.getContext("2d");
      if (ctx) {
        drawAnnotations(
          ctx,
          img,
          cocoJson.annotations,
          cocoJson.categories,
          colorMap,
          selectedClasses,
          cocoJson.images[0],
        );
      }
    };
    img.src = imageSrc;
  }, [imageSrc, cocoJson, colorMap, selectedClasses]);

  // Re-render on class filter changes without reloading image
  useEffect(() => {
    if (!imgRef.current) return;
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) {
      drawAnnotations(
        ctx,
        imgRef.current,
        cocoJson.annotations,
        cocoJson.categories,
        colorMap,
        selectedClasses,
        cocoJson.images[0],
      );
    }
  }, [selectedClasses, cocoJson, colorMap]);

  return (
    <canvas
      ref={canvasRef}
      className="max-h-[70vh] w-full rounded-lg border object-contain"
      style={{ imageRendering: "auto" }}
    />
  );
}
