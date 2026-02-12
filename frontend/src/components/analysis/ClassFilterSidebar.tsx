"use client";

import { useMemo, useState } from "react";
import { ChevronRight } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { ManuscriptClass } from "@/lib/types/classes";
import { classCategories, getDisplayName } from "@/data/classCategories";
import { cn } from "@/lib/utils";

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
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  // Build a lookup map: backend class name → ManuscriptClass
  const classMap = useMemo(
    () => new Map(classes.map((c) => [c.name, c])),
    [classes],
  );

  const searchLower = search.toLowerCase();

  // Build groups with their resolved classes, filtered by search
  const groups = useMemo(() => {
    return classCategories
      .map((cat) => {
        const items = cat.classes
          .map((name) => classMap.get(name))
          .filter((c): c is ManuscriptClass => c != null)
          .filter((c) =>
            search
              ? getDisplayName(c.name).toLowerCase().includes(searchLower) ||
                c.name.toLowerCase().includes(searchLower)
              : true,
          );

        return { name: cat.name, items };
      })
      .filter((g) => g.items.length > 0);
  }, [classMap, search, searchLower]);

  function toggleGroup(groupName: string) {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(groupName)) next.delete(groupName);
      else next.add(groupName);
      return next;
    });
  }

  // Count how many classes in a group are selected
  function selectedInGroup(items: ManuscriptClass[]): number {
    return items.filter((c) => selected.has(c.name)).length;
  }

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

      <div className="flex-1 overflow-y-auto">
        {groups.map((group) => {
          const isCollapsed = collapsed.has(group.name);
          const selCount = selectedInGroup(group.items);

          return (
            <div key={group.name} className="mb-1">
              {/* Group header */}
              <button
                onClick={() => toggleGroup(group.name)}
                className="flex w-full items-center gap-1 rounded-md px-1.5 py-1.5 text-xs font-semibold text-foreground transition-colors hover:bg-muted"
              >
                <ChevronRight
                  className={cn(
                    "h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform duration-200",
                    !isCollapsed && "rotate-90",
                  )}
                />
                <span className="flex-1 text-left">{group.name}</span>
                <span className="text-[10px] font-normal text-muted-foreground">
                  {selCount}/{group.items.length}
                </span>
              </button>

              {/* Class items */}
              {!isCollapsed && (
                <div className="space-y-0.5 pl-2">
                  {group.items.map((cls) => (
                    <label
                      key={cls.id}
                      className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-1 text-sm transition-colors hover:bg-muted"
                    >
                      <Checkbox
                        checked={selected.has(cls.name)}
                        onCheckedChange={() => onToggle(cls.name)}
                      />
                      <span
                        className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                        style={{ backgroundColor: cls.color }}
                      />
                      <span className="truncate">{getDisplayName(cls.name)}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>
          );
        })}

        {groups.length === 0 && (
          <p className="px-2 py-4 text-center text-xs text-muted-foreground">No classes match.</p>
        )}
      </div>
    </div>
  );
}
