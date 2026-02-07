"use client";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { ManuscriptClass } from "@/lib/types/classes";

interface ClassSelectorProps {
  classes: ManuscriptClass[];
  selected: Set<string>;
  onToggle: (name: string) => void;
  onSelectAll: () => void;
  onUnselectAll: () => void;
}

export function ClassSelector({
  classes,
  selected,
  onToggle,
  onSelectAll,
  onUnselectAll,
}: ClassSelectorProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">
          Classes ({selected.size}/{classes.length})
        </span>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={onSelectAll}>
            Select All
          </Button>
          <Button variant="outline" size="sm" onClick={onUnselectAll}>
            Unselect All
          </Button>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
        {classes.map((cls) => (
          <label
            key={cls.id}
            className="flex cursor-pointer items-center gap-2 rounded-md border px-3 py-2 text-sm transition-colors hover:bg-muted"
          >
            <Checkbox
              checked={selected.has(cls.name)}
              onCheckedChange={() => onToggle(cls.name)}
            />
            <span
              className="inline-block h-3 w-3 rounded-full"
              style={{ backgroundColor: cls.color }}
            />
            <span className="truncate">{cls.name}</span>
          </label>
        ))}
      </div>
    </div>
  );
}
