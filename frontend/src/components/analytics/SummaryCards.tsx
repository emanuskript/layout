"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface SummaryCardsProps {
  totalVisits: number;
  uniqueCountries: number;
  periodDays: number;
}

export function SummaryCards({ totalVisits, uniqueCountries, periodDays }: SummaryCardsProps) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">Total Visits</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-3xl font-bold">{totalVisits.toLocaleString()}</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">Countries</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-3xl font-bold">{uniqueCountries}</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">Period</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-3xl font-bold">{periodDays} days</p>
        </CardContent>
      </Card>
    </div>
  );
}
