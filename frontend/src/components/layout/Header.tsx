"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const tabs = [
  { href: "/single", label: "Single Image" },
  { href: "/batch", label: "Batch Processing" },
  { href: "/analytics", label: "Analytics" },
];

export function Header() {
  const pathname = usePathname();

  return (
    <header className="border-b bg-white">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <Link href="/single" className="text-xl font-bold tracking-tight">
          Manuscript Layout Analysis
        </Link>
        <nav className="flex gap-1">
          {tabs.map((tab) => (
            <Link
              key={tab.href}
              href={tab.href}
              className={cn(
                "rounded-md px-4 py-2 text-sm font-medium transition-colors",
                pathname === tab.href
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground",
              )}
            >
              {tab.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
