"use client";

import { GalleryItem } from "@/lib/types/predict";

interface BatchGalleryProps {
  gallery: GalleryItem[];
  statsPerImage?: { image: string; stats: Record<string, number> }[];
  selected: number | null;
  onSelect: (idx: number | null) => void;
}

export function BatchGallery({ gallery, statsPerImage, selected, onSelect }: BatchGalleryProps) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
      {gallery.map((item, idx) => {
        const stats = statsPerImage?.find((s) => s.image === item.filename)?.stats;
        const annCount = stats ? Object.values(stats).reduce((a, b) => a + b, 0) : 0;
        const classCount = stats ? Object.keys(stats).filter((k) => stats[k] > 0).length : 0;

        return (
          <button
            key={idx}
            onClick={() => onSelect(selected === idx ? null : idx)}
            className={`group relative overflow-hidden rounded-lg border transition-all hover:ring-2 hover:ring-primary ${
              selected === idx ? "ring-2 ring-primary" : ""
            }`}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={item.annotated_url}
              alt={item.filename}
              className="aspect-[3/4] w-full object-cover"
            />
            {/* Gradient overlay with metadata */}
            <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 via-black/30 to-transparent px-2 pb-2 pt-6">
              <p className="truncate text-xs font-medium text-white">{item.filename}</p>
              <div className="mt-0.5 flex gap-2 text-[10px] text-white/80">
                <span>{annCount} ann</span>
                <span>{classCount} classes</span>
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
