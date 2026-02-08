"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { ManuscriptClass } from "@/lib/types/classes";

interface ClassFilterSidebarProps {
  classes: ManuscriptClass[];
  selected: Set<string>;
  onToggle: (name: string) => void;
  onSelectAll: () => void;
  onUnselectAll: () => void;
}

export function ClassFilterSidebar({
  classes,
  selected,
  onToggle,
  onSelectAll,
  onUnselectAll,
}: ClassFilterSidebarProps) {
  const [search, setSearch] = useState("");

  const filtered = search
    ? classes.filter((c) => c.name.toLowerCase().includes(search.toLowerCase()))
    : classes;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between pb-2">
        <span className="text-sm font-medium">
          Classes ({selected.size}/{classes.length})
        </span>
        <div className="flex gap-1">
          <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={onSelectAll}>
            All
          </Button>
          <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={onUnselectAll}>
            None
          </Button>
        </div>
      </div>

      <Input
        placeholder="Search classes..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="mb-2 h-8 text-sm"
      />

      <div className="flex-1 space-y-0.5 overflow-y-auto">
        {filtered.map((cls) => (
          <label
            key={cls.id}
            className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors hover:bg-muted"
          >
            <Checkbox
              checked={selected.has(cls.name)}
              onCheckedChange={() => onToggle(cls.name)}
            />
            <span
              className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
              style={{ backgroundColor: cls.color }}
            />
            <span className="truncate">{cls.name}</span>
          </label>
        ))}
        {filtered.length === 0 && (
          <p className="px-2 py-4 text-center text-xs text-muted-foreground">No classes match.</p>
        )}
      </div>
    </div>
  );
}
