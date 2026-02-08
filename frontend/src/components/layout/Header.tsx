"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useAtomValue, useSetAtom } from "jotai";

import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { apiUrl } from "@/lib/api/client";
import {
  fileCountAtom,
  anyLoadingAtom,
  anyHasResultsAtom,
  selectedFileAtom,
  selectedClassNamesAtom,
  runAllIdleFilesAtom,
  resetWorkspaceAtom,
} from "@/lib/atoms";

export function Header() {
  const fileCount = useAtomValue(fileCountAtom);
  const anyLoading = useAtomValue(anyLoadingAtom);
  const anyHasResults = useAtomValue(anyHasResultsAtom);
  const selectedFile = useAtomValue(selectedFileAtom);
  const selectedClassNames = useAtomValue(selectedClassNamesAtom);
  const runAll = useSetAtom(runAllIdleFilesAtom);
  const resetWorkspace = useSetAtom(resetWorkspaceAtom);

  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const hasFiles = fileCount > 0;
  const runDisabled = !hasFiles || anyLoading || selectedClassNames.size === 0;
  const runLabel = anyLoading ? "Processing..." : "Run Analysis";

  // Export info from selected file
  const taskId =
    selectedFile?.singleResult?.task_id ??
    selectedFile?.batchTaskId ??
    null;
  const isBatch = selectedFile?.kind === "zip";

  return (
    <header className="sticky top-0 z-40 h-14 border-b bg-background/95 backdrop-blur-sm">
      <div className="flex h-full items-center justify-between px-4">
        {/* Left: title */}
        <Link href="/analyze" className="text-lg font-bold tracking-tight">
          Manuscript Layout Analysis
        </Link>

        {/* Right: actions */}
        <div className="flex items-center gap-2">
          {/* Run button */}
          {hasFiles && !anyHasResults && (
            <button
              onClick={() => runAll()}
              disabled={runDisabled}
              className="rounded-md bg-primary px-4 py-1.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50 disabled:pointer-events-none"
            >
              {runLabel}
            </button>
          )}
          {anyHasResults && (
            <button
              onClick={() => resetWorkspace()}
              className="rounded-md border px-4 py-1.5 text-sm font-medium transition-colors hover:bg-muted"
            >
              New Analysis
            </button>
          )}

          {/* Theme toggle */}
          <ThemeToggle />

          {/* Export dropdown */}
          {taskId && (
            <div ref={menuRef} className="relative">
              <button
                onClick={() => setOpen((o) => !o)}
                className="flex items-center gap-1.5 rounded-md border px-3 py-1.5 text-sm font-medium transition-colors hover:bg-muted"
              >
                <svg className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                </svg>
                Export
                <svg className={`h-3 w-3 transition-transform ${open ? "rotate-180" : ""}`} fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {open && (
                <div className="absolute right-0 top-full mt-1 w-56 rounded-md border bg-popover py-1 shadow-lg">
                  <a
                    href={apiUrl(`/download/${taskId}/coco_json`)}
                    download
                    onClick={() => setOpen(false)}
                    className="flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted"
                  >
                    <svg className="h-4 w-4 text-muted-foreground" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m.75 12l3 3m0 0l3-3m-3 3v-6m-1.5-9H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                    </svg>
                    COCO JSON
                  </a>

                  {!isBatch && (
                    <a
                      href={apiUrl(`/download/${taskId}/annotated_image`)}
                      download
                      onClick={() => setOpen(false)}
                      className="flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted"
                    >
                      <svg className="h-4 w-4 text-muted-foreground" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3.75 21h16.5A2.25 2.25 0 0022.5 18.75V5.25A2.25 2.25 0 0020.25 3H3.75A2.25 2.25 0 001.5 5.25v13.5A2.25 2.25 0 003.75 21z" />
                      </svg>
                      Annotated Image
                    </a>
                  )}

                  {isBatch && (
                    <a
                      href={apiUrl(`/download/${taskId}/results_zip`)}
                      download
                      onClick={() => setOpen(false)}
                      className="flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted"
                    >
                      <svg className="h-4 w-4 text-muted-foreground" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 7.5l-.625 10.632a2.25 2.25 0 01-2.247 2.118H6.622a2.25 2.25 0 01-2.247-2.118L3.75 7.5m8.25 3v6.75m0 0l-3-3m3 3l3-3M3.375 7.5h17.25c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125H3.375c-.621 0-1.125.504-1.125 1.125v1.5c0 .621.504 1.125 1.125 1.125z" />
                      </svg>
                      Download All (ZIP)
                    </a>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
