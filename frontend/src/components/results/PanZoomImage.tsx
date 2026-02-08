"use client";

import { useCallback, useEffect, useRef, useState } from "react";

interface PanZoomImageProps {
  src: string;
  alt: string;
}

interface Transform {
  scale: number;
  offsetX: number;
  offsetY: number;
}

export function PanZoomImage({ src, alt }: PanZoomImageProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const tRef = useRef<Transform>({ scale: 1, offsetX: 0, offsetY: 0 });
  const panRef = useRef({ active: false, dragged: false, startX: 0, startY: 0, origOX: 0, origOY: 0 });
  const [, forceRender] = useState(0);
  const DRAG_THRESHOLD = 4;

  const fitToView = useCallback(() => {
    const container = containerRef.current;
    const img = imgRef.current;
    if (!container || !img || !img.naturalWidth) return;
    const { width: cw, height: ch } = container.getBoundingClientRect();
    const scale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight, 1);
    tRef.current = {
      scale,
      offsetX: (cw - img.naturalWidth * scale) / 2,
      offsetY: (ch - img.naturalHeight * scale) / 2,
    };
    forceRender((n) => n + 1);
  }, []);

  // Fit on load and resize
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(() => fitToView());
    ro.observe(container);
    return () => ro.disconnect();
  }, [fitToView]);

  // Reset fit when src changes
  useEffect(() => {
    tRef.current = { scale: 1, offsetX: 0, offsetY: 0 };
    // fitToView will fire via onLoad
  }, [src]);

  // Wheel zoom
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
      forceRender((n) => n + 1);
    };
    container.addEventListener("wheel", onWheel, { passive: false });
    return () => container.removeEventListener("wheel", onWheel);
  }, []);

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
    if (!p.active) return;
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
      forceRender((n) => n + 1);
    }
  };

  const handleMouseUp = () => {
    panRef.current = { ...panRef.current, active: false, dragged: false };
  };

  const handleMouseLeave = () => {
    panRef.current = { ...panRef.current, active: false, dragged: false };
  };

  const t = tRef.current;

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full cursor-grab overflow-hidden active:cursor-grabbing"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
      onDoubleClick={fitToView}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        ref={(el) => { imgRef.current = el; }}
        src={src}
        alt={alt}
        onLoad={fitToView}
        className="absolute left-0 top-0 max-w-none"
        style={{
          transformOrigin: "0 0",
          transform: `translate(${t.offsetX}px, ${t.offsetY}px) scale(${t.scale})`,
        }}
        draggable={false}
      />
    </div>
  );
}
