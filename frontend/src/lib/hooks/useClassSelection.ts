"use client";

import { useCallback, useEffect, useState } from "react";

import { getClasses } from "../api/classes";
import { ManuscriptClass } from "../types/classes";

export function useClassSelection() {
  const [classes, setClasses] = useState<ManuscriptClass[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getClasses()
      .then((data) => {
        setClasses(data);
        setSelected(new Set(data.map((c) => c.name)));
      })
      .finally(() => setLoading(false));
  }, []);

  const toggle = useCallback((name: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }, []);

  const selectAll = useCallback(() => {
    setSelected(new Set(classes.map((c) => c.name)));
  }, [classes]);

  const unselectAll = useCallback(() => {
    setSelected(new Set());
  }, []);

  const selectedNames = Array.from(selected);

  return { classes, selected, selectedNames, toggle, selectAll, unselectAll, loading };
}
