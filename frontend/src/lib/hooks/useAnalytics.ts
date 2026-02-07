"use client";

import { useCallback, useState } from "react";

import { analyticsLogin, getAnalyticsData } from "../api/analytics";
import { AnalyticsData } from "../types/analytics";

export function useAnalytics() {
  const [token, setToken] = useState<string | null>(() => {
    if (typeof window !== "undefined") {
      return sessionStorage.getItem("analytics_token");
    }
    return null;
  });
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const login = useCallback(async (username: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      const t = await analyticsLogin(username, password);
      setToken(t);
      sessionStorage.setItem("analytics_token", t);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchData = useCallback(
    async (days: number = 30) => {
      if (!token) return;
      setLoading(true);
      setError(null);
      try {
        const d = await getAnalyticsData(token, days);
        setData(d);
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Failed to fetch";
        if (msg.includes("expired") || msg.includes("Invalid token")) {
          setToken(null);
          sessionStorage.removeItem("analytics_token");
        }
        setError(msg);
      } finally {
        setLoading(false);
      }
    },
    [token],
  );

  const logout = useCallback(() => {
    setToken(null);
    setData(null);
    sessionStorage.removeItem("analytics_token");
  }, []);

  return { token, data, loading, error, login, fetchData, logout };
}
