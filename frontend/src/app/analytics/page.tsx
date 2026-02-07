"use client";

import { useEffect, useState } from "react";

import { CountryChart } from "@/components/analytics/CountryChart";
import { DailyChart } from "@/components/analytics/DailyChart";
import { DataTable } from "@/components/analytics/DataTable";
import { LoginForm } from "@/components/analytics/LoginForm";
import { SummaryCards } from "@/components/analytics/SummaryCards";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAnalytics } from "@/lib/hooks/useAnalytics";

export default function AnalyticsPage() {
  const { token, data, loading, error, login, fetchData, logout } = useAnalytics();
  const [days, setDays] = useState(30);

  useEffect(() => {
    if (token) fetchData(days);
  }, [token, days, fetchData]);

  if (!token) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold">Analytics</h1>
        <LoginForm onLogin={login} loading={loading} error={error} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Analytics Dashboard</h1>
        <Button variant="outline" onClick={logout}>
          Logout
        </Button>
      </div>

      <div className="flex items-center gap-4">
        <Label>Period:</Label>
        <Slider
          className="w-48"
          min={7}
          max={365}
          step={1}
          value={[days]}
          onValueChange={([v]) => setDays(v)}
        />
        <span className="text-sm text-muted-foreground">{days} days</span>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive bg-destructive/5 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      {loading && !data && <p className="text-muted-foreground">Loading analytics data...</p>}

      {data && (
        <>
          <SummaryCards
            totalVisits={data.summary.total_visits}
            uniqueCountries={data.summary.unique_countries}
            periodDays={data.summary.period_days}
          />

          <Tabs defaultValue="countries">
            <TabsList>
              <TabsTrigger value="countries">By Country</TabsTrigger>
              <TabsTrigger value="daily">Daily</TabsTrigger>
              <TabsTrigger value="recent">Recent Visits</TabsTrigger>
            </TabsList>
            <TabsContent value="countries">
              <Card>
                <CardHeader>
                  <CardTitle>Visits by Country</CardTitle>
                </CardHeader>
                <CardContent>
                  <CountryChart data={data.by_country} />
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="daily">
              <Card>
                <CardHeader>
                  <CardTitle>Daily Visits</CardTitle>
                </CardHeader>
                <CardContent>
                  <DailyChart data={data.by_day} />
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="recent">
              <Card>
                <CardHeader>
                  <CardTitle>Recent Visits</CardTitle>
                </CardHeader>
                <CardContent>
                  <DataTable data={data.recent_visits} />
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </>
      )}
    </div>
  );
}
