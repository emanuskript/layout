"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Slider } from "@/components/ui/slider";
import { COCOAnnotation, COCOJson } from "@/lib/types/coco";
import { drawAnnotationsTransformed, hitTestAnnotation } from "@/lib/utils/annotations";

interface InteractiveCanvasProps {
  imageSrc: string;
  cocoJson: COCOJson;
  colorMap: Map<string, string>;
  selectedClasses: Set<string>;
  onAnnotationClick?: (ann: COCOAnnotation | null) => void;
}

interface Transform {
  scale: number;
  offsetX: number;
  offsetY: number;
}

function computeFit(cw: number, ch: number, imgW: number, imgH: number): Transform {
  const scale = Math.min(cw / imgW, ch / imgH, 1);
  return {
    scale,
    offsetX: (cw - imgW * scale) / 2,
    offsetY: (ch - imgH * scale) / 2,
  };
}

export function InteractiveCanvas({
  imageSrc,
  cocoJson,
  colorMap,
  selectedClasses,
  onAnnotationClick,
}: InteractiveCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const tRef = useRef<Transform>({ scale: 1, offsetX: 0, offsetY: 0 });

  const [opacity, setOpacity] = useState(0.25);
  const hoveredIdRef = useRef<number | null>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    className: string;
    area: number;
    bbox: [number, number, number, number];
  } | null>(null);
  const [zoomDisplay, setZoomDisplay] = useState(100);

  // Pan tracking (dragged distinguishes click from drag)
  const panRef = useRef({ active: false, dragged: false, startX: 0, startY: 0, origOX: 0, origOY: 0 });

  const catById = useRef(new Map<number, string>());

  const getImgSize = useCallback(() => {
    const imgW = cocoJson.images[0]?.width ?? imgRef.current?.naturalWidth ?? 1;
    const imgH = cocoJson.images[0]?.height ?? imgRef.current?.naturalHeight ?? 1;
    return { imgW, imgH };
  }, [cocoJson]);

  // ── Draw ──
  const redraw = useCallback(
    (highlightId: number | null = null) => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      const img = imgRef.current;
      if (!canvas || !container || !img) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      const { width: cw, height: ch } = container.getBoundingClientRect();

      canvas.width = cw * dpr;
      canvas.height = ch * dpr;
      canvas.style.width = `${cw}px`;
      canvas.style.height = `${ch}px`;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cw, ch);

      const t = tRef.current;
      drawAnnotationsTransformed(
        ctx,
        img,
        cocoJson.annotations,
        cocoJson.categories,
        colorMap,
        selectedClasses,
        t.scale,
        t.offsetX,
        t.offsetY,
        cocoJson.images[0],
        { fillOpacity: opacity, highlightedId: highlightId },
      );
    },
    [cocoJson, colorMap, selectedClasses, opacity],
  );

  // ── Fit to view ──
  const fitToView = useCallback(() => {
    const container = containerRef.current;
    if (!container || !imgRef.current) return;
    const { width: cw, height: ch } = container.getBoundingClientRect();
    const { imgW, imgH } = getImgSize();
    tRef.current = computeFit(cw, ch, imgW, imgH);
    setZoomDisplay(Math.round(tRef.current.scale * 100));
    redraw(hoveredIdRef.current);
  }, [getImgSize, redraw]);

  // ── Load image ──
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      imgRef.current = img;
      catById.current = new Map(cocoJson.categories.map((c) => [c.id, c.name]));
      // Compute initial fit
      const container = containerRef.current;
      if (container) {
        const { width: cw, height: ch } = container.getBoundingClientRect();
        const imgW = cocoJson.images[0]?.width ?? img.naturalWidth;
        const imgH = cocoJson.images[0]?.height ?? img.naturalHeight;
        tRef.current = computeFit(cw, ch, imgW, imgH);
        setZoomDisplay(Math.round(tRef.current.scale * 100));
      }
      redraw();
    };
    img.src = imageSrc;
  }, [imageSrc, cocoJson, redraw]);

  // Redraw on filter / opacity changes
  useEffect(() => {
    redraw(hoveredIdRef.current);
  }, [selectedClasses, opacity, redraw]);

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(() => fitToView());
    ro.observe(container);
    return () => ro.disconnect();
  }, [fitToView]);

  // ── Screen → image coords ──
  const screenToImage = useCallback(
    (clientX: number, clientY: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const sx = clientX - rect.left;
      const sy = clientY - rect.top;
      const t = tRef.current;
      return {
        x: (sx - t.offsetX) / t.scale,
        y: (sy - t.offsetY) / t.scale,
        clientX,
        clientY,
      };
    },
    [],
  );

  // ── Wheel zoom (centered on cursor) ──
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = container.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      const t = tRef.current;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const newScale = Math.min(Math.max(t.scale * factor, 0.05), 30);

      tRef.current = {
        scale: newScale,
        offsetX: mx - (mx - t.offsetX) * (newScale / t.scale),
        offsetY: my - (my - t.offsetY) * (newScale / t.scale),
      };
      setZoomDisplay(Math.round(newScale * 100));
      redraw(hoveredIdRef.current);
    };

    container.addEventListener("wheel", onWheel, { passive: false });
    return () => container.removeEventListener("wheel", onWheel);
  }, [redraw]);

  // ── Mouse handlers ──
  const DRAG_THRESHOLD = 4; // px — below this it's a click, above it's a drag

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0 || e.button === 1) {
      e.preventDefault();
      panRef.current = {
        active: true,
        dragged: false,
        startX: e.clientX,
        startY: e.clientY,
        origOX: tRef.current.offsetX,
        origOY: tRef.current.offsetY,
      };
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const p = panRef.current;
    if (p.active) {
      const dx = e.clientX - p.startX;
      const dy = e.clientY - p.startY;
      if (!p.dragged && Math.abs(dx) + Math.abs(dy) > DRAG_THRESHOLD) {
        p.dragged = true;
      }
      if (p.dragged) {
        tRef.current = {
          ...tRef.current,
          offsetX: p.origOX + dx,
          offsetY: p.origOY + dy,
        };
        redraw(hoveredIdRef.current);
        setTooltip(null);
        return;
      }
    }

    const coords = screenToImage(e.clientX, e.clientY);
    if (!coords) return;

    const ann = hitTestAnnotation(
      coords.x,
      coords.y,
      cocoJson.annotations,
      cocoJson.categories,
      selectedClasses,
    );

    if (ann) {
      if (hoveredIdRef.current !== ann.id) {
        hoveredIdRef.current = ann.id;
        redraw(ann.id);
      }
      setTooltip({
        x: coords.clientX,
        y: coords.clientY,
        className: catById.current.get(ann.category_id) || "Unknown",
        area: Math.round(ann.area),
        bbox: ann.bbox,
      });
    } else {
      if (hoveredIdRef.current !== null) {
        hoveredIdRef.current = null;
        redraw(null);
      }
      setTooltip(null);
    }
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    const p = panRef.current;
    const wasDrag = p.dragged;
    panRef.current = { ...p, active: false, dragged: false };

    // If it wasn't a drag, treat as a click for annotation selection
    if (!wasDrag && e.button === 0) {
      const coords = screenToImage(e.clientX, e.clientY);
      if (!coords) return;
      const ann = hitTestAnnotation(
        coords.x,
        coords.y,
        cocoJson.annotations,
        cocoJson.categories,
        selectedClasses,
      );
      onAnnotationClick?.(ann);
    }
  };

  const handleMouseLeave = () => {
    panRef.current = { ...panRef.current, active: false, dragged: false };
    if (hoveredIdRef.current !== null) {
      hoveredIdRef.current = null;
      redraw(null);
    }
    setTooltip(null);
  };

  const handleDoubleClick = () => fitToView();

  return (
    <div ref={containerRef} className="relative h-full w-full overflow-hidden">
      <canvas
        ref={canvasRef}
        className="absolute inset-0 cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onDoubleClick={handleDoubleClick}
      />

      {/* Floating controls */}
      <div className="absolute bottom-3 left-3 flex items-center gap-3 rounded-lg border bg-white/90 px-3 py-1.5 shadow-sm backdrop-blur-sm">
        <span className="text-xs font-medium text-muted-foreground">Opacity</span>
        <Slider
          className="w-24"
          min={0}
          max={1}
          step={0.05}
          value={[opacity]}
          onValueChange={([v]) => setOpacity(v)}
        />
        <span className="w-8 text-xs text-muted-foreground">{Math.round(opacity * 100)}%</span>
        <span className="border-l pl-3 text-xs text-muted-foreground">{zoomDisplay}%</span>
        <button
          onClick={fitToView}
          className="rounded px-1.5 py-0.5 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
          title="Fit to view (or double-click)"
        >
          Fit
        </button>
      </div>

      {/* Pan hint */}
      <div className="absolute right-3 top-3 rounded-md bg-black/40 px-2 py-1 text-[10px] text-white/70 backdrop-blur-sm">
        Scroll to zoom &middot; Drag to pan &middot; Click annotation to inspect &middot; Double-click to fit
      </div>

      {/* Hover tooltip */}
      {tooltip && (
        <div
          className="pointer-events-none fixed z-50 rounded-md border bg-white px-3 py-2 text-xs shadow-lg"
          style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}
        >
          <p className="font-semibold">{tooltip.className}</p>
          <p className="text-muted-foreground">
            Area: {tooltip.area.toLocaleString()} px&sup2;
          </p>
          <p className="text-muted-foreground">
            BBox: {Math.round(tooltip.bbox[2])}&times;{Math.round(tooltip.bbox[3])}
          </p>
        </div>
      )}
    </div>
  );
}
