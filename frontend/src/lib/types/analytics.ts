export interface CountryVisits {
  country: string;
  visits: number;
}

export interface DailyVisits {
  date: string;
  visits: number;
}

export interface RecentVisit {
  timestamp: string;
  ip: string;
  country: string;
  city: string;
  user_agent: string;
}

export interface AnalyticsData {
  summary: {
    total_visits: number;
    unique_countries: number;
    period_days: number;
  };
  by_country: CountryVisits[];
  by_day: DailyVisits[];
  recent_visits: RecentVisit[];
}
