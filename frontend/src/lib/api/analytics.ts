import { AnalyticsData } from "../types/analytics";
import { apiFetch } from "./client";

export async function analyticsLogin(
  username: string,
  password: string,
): Promise<string> {
  const data = await apiFetch<{ token: string }>("/analytics/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  return data.token;
}

export async function getAnalyticsData(
  token: string,
  days: number = 30,
): Promise<AnalyticsData> {
  return apiFetch<AnalyticsData>(`/analytics/data?days=${days}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
}
