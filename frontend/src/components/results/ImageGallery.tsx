"use client";

import { useState } from "react";

import { GalleryItem } from "@/lib/types/predict";

interface ImageGalleryProps {
  gallery: GalleryItem[];
}

export function ImageGallery({ gallery }: ImageGalleryProps) {
  const [selected, setSelected] = useState<number | null>(null);

  return (
    <div>
      {selected !== null && (
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-medium">{gallery[selected].filename}</p>
            <button
              className="text-sm text-muted-foreground hover:text-foreground"
              onClick={() => setSelected(null)}
            >
              Close
            </button>
          </div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={gallery[selected].annotated_url}
            alt={gallery[selected].filename}
            className="max-h-[60vh] w-full rounded-lg border object-contain"
          />
        </div>
      )}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
        {gallery.map((item, idx) => (
          <button
            key={idx}
            onClick={() => setSelected(idx)}
            className={`overflow-hidden rounded-lg border transition-all hover:ring-2 hover:ring-primary ${
              selected === idx ? "ring-2 ring-primary" : ""
            }`}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={item.annotated_url}
              alt={item.filename}
              className="aspect-square w-full object-cover"
            />
            <p className="truncate px-2 py-1 text-xs">{item.filename}</p>
          </button>
        ))}
      </div>
    </div>
  );
}
