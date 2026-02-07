"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { RecentVisit } from "@/lib/types/analytics";

interface DataTableProps {
  data: RecentVisit[];
}

export function DataTable({ data }: DataTableProps) {
  return (
    <div className="max-h-96 overflow-auto rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Time</TableHead>
            <TableHead>Country</TableHead>
            <TableHead>City</TableHead>
            <TableHead className="hidden md:table-cell">IP</TableHead>
            <TableHead className="hidden lg:table-cell">User Agent</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.map((visit, idx) => (
            <TableRow key={idx}>
              <TableCell className="whitespace-nowrap text-xs">
                {new Date(visit.timestamp).toLocaleString()}
              </TableCell>
              <TableCell>{visit.country}</TableCell>
              <TableCell>{visit.city}</TableCell>
              <TableCell className="hidden md:table-cell text-xs font-mono">
                {visit.ip}
              </TableCell>
              <TableCell className="hidden lg:table-cell max-w-xs truncate text-xs">
                {visit.user_agent}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
