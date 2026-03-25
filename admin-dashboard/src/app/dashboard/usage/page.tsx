"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertTriangle, AlertCircle, RefreshCw, TrendingUp, CheckCircle2 } from "lucide-react";

type DayPoint = { date: string; count: number };

type TenantUsage = {
  tenant_id: string;
  slug: string;
  name: string;
  daily_review_cap: number | null;
  daily_download_cap: number | null;
  today_count: number;
  trend: DayPoint[];
  overCap: boolean;
  nearCap: boolean;
};

type UsageData = {
  days: string[];
  tenants: TenantUsage[];
};

function shortDate(d: string) {
  const date = new Date(d + "T12:00:00Z");
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function BarChart({ trend, cap }: { trend: DayPoint[]; cap: number | null }) {
  const maxVal = Math.max(...trend.map((d) => d.count), cap ?? 0, 1);

  return (
    <div className="flex items-end gap-1 h-16">
      {trend.map((d) => {
        const pct = (d.count / maxVal) * 100;
        const isToday = d === trend[trend.length - 1];
        const overCap = cap !== null && cap > 0 && d.count >= cap;
        const nearCap = cap !== null && cap > 0 && !overCap && d.count >= cap * 0.8;
        const barColor = overCap
          ? "bg-red-500"
          : nearCap
          ? "bg-amber-400"
          : isToday
          ? "bg-primary"
          : "bg-primary/40";

        return (
          <div key={d.date} className="flex-1 flex flex-col items-center gap-1 group relative">
            {/* Tooltip */}
            <div className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 hidden group-hover:block z-10 bg-foreground text-background text-xs rounded px-2 py-1 whitespace-nowrap">
              {shortDate(d.date)}: {d.count}{cap ? ` / ${cap}` : ""}
            </div>
            <div className="w-full rounded-t-sm" style={{ height: `${Math.max(pct, 2)}%` }}>
              <div className={`w-full h-full rounded-t-sm ${barColor} transition-all`} />
            </div>
          </div>
        );
      })}
      {/* Cap line */}
      {cap !== null && cap > 0 && (
        <div
          className="absolute w-full border-t-2 border-dashed border-red-400 pointer-events-none"
          style={{ bottom: `${(cap / maxVal) * 64}px` }}
        />
      )}
    </div>
  );
}

function SparklineChart({ trend, cap }: { trend: DayPoint[]; cap: number | null }) {
  const maxVal = Math.max(...trend.map((d) => d.count), cap ?? 0, 1);
  const w = 200;
  const h = 64;
  const pts = trend.map((d, i) => {
    const x = (i / (trend.length - 1)) * w;
    const y = h - (d.count / maxVal) * h;
    return `${x},${y}`;
  });
  const polyline = pts.join(" ");
  const capY = cap !== null && cap > 0 ? h - (cap / maxVal) * h : null;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-16" preserveAspectRatio="none">
      {/* Cap line */}
      {capY !== null && (
        <line x1={0} y1={capY} x2={w} y2={capY} stroke="#f87171" strokeWidth={1.5} strokeDasharray="4 3" />
      )}
      {/* Area fill */}
      <polyline
        points={`0,${h} ${polyline} ${w},${h}`}
        fill="hsl(var(--primary) / 0.15)"
        stroke="none"
      />
      {/* Line */}
      <polyline points={polyline} fill="none" stroke="hsl(var(--primary))" strokeWidth={2} strokeLinejoin="round" />
      {/* Today dot */}
      {trend.length > 0 && (() => {
        const last = trend[trend.length - 1];
        const x = w;
        const y = h - (last.count / maxVal) * h;
        return <circle cx={x} cy={y} r={3} fill="hsl(var(--primary))" />;
      })()}
    </svg>
  );
}

export default function UsagePage() {
  const [data, setData] = useState<UsageData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [view, setView] = useState<"bar" | "line">("bar");

  async function load() {
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/usage");
      const json = await res.json();
      if (!res.ok) { setError(json.error || "Failed"); return; }
      setData(json);
    } catch {
      setError("Network error");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  const alerts = data?.tenants.filter((t) => t.overCap || t.nearCap) ?? [];

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Usage & Caps</h1>
          <p className="text-muted-foreground mt-1">7-day usage trend per tenant. Red dashed line = daily cap.</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant={view === "bar" ? "default" : "outline"} size="sm" onClick={() => setView("bar")}>Bar</Button>
          <Button variant={view === "line" ? "default" : "outline"} size="sm" onClick={() => setView("line")}>Line</Button>
          <Button variant="outline" size="sm" onClick={load} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </div>

      {/* Alert banner */}
      {alerts.length > 0 && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="py-3 px-4">
            <div className="flex items-start gap-2">
              <AlertTriangle className="h-5 w-5 text-red-500 shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-red-700">
                  {alerts.filter((t) => t.overCap).length > 0
                    ? `${alerts.filter((t) => t.overCap).length} tenant(s) have exceeded their daily cap today`
                    : `${alerts.length} tenant(s) are near their daily cap (≥80%)`}
                </p>
                <p className="text-xs text-red-600 mt-0.5">
                  {alerts.map((t) => t.name).join(", ")}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {error && (
        <div className="flex items-center gap-2 text-destructive text-sm">
          <AlertCircle className="h-4 w-4" />{error}
        </div>
      )}

      {!data && !loading && !error && null}

      {/* Summary row */}
      {data && (
        <div className="grid gap-4 sm:grid-cols-3">
          <Card>
            <CardContent className="pt-4 pb-4">
              <p className="text-xs text-muted-foreground">Active Tenants</p>
              <p className="text-2xl font-bold">{data.tenants.length}</p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-4 pb-4">
              <p className="text-xs text-muted-foreground">Total Usage Today</p>
              <p className="text-2xl font-bold">
                {data.tenants.reduce((s, t) => s + t.today_count, 0)}
              </p>
            </CardContent>
          </Card>
          <Card className={alerts.filter((t) => t.overCap).length > 0 ? "border-red-300" : alerts.length > 0 ? "border-amber-300" : ""}>
            <CardContent className="pt-4 pb-4 flex items-center gap-3">
              <div>
                <p className="text-xs text-muted-foreground">Cap Alerts</p>
                <p className="text-2xl font-bold">{alerts.length}</p>
              </div>
              {alerts.length === 0
                ? <CheckCircle2 className="h-6 w-6 text-green-500 ml-auto" />
                : <AlertTriangle className="h-6 w-6 text-red-500 ml-auto" />}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Per-tenant charts */}
      {data && (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {data.tenants.map((t) => (
            <Card
              key={t.tenant_id}
              className={
                t.overCap
                  ? "border-red-300"
                  : t.nearCap
                  ? "border-amber-300"
                  : ""
              }
            >
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <CardTitle className="text-base">{t.name}</CardTitle>
                    <CardDescription className="text-xs">{t.slug}</CardDescription>
                  </div>
                  {t.overCap && (
                    <span className="text-xs font-medium text-red-600 bg-red-50 border border-red-200 rounded-full px-2 py-0.5 flex items-center gap-1 whitespace-nowrap">
                      <AlertTriangle className="h-3 w-3" /> Over Cap
                    </span>
                  )}
                  {t.nearCap && !t.overCap && (
                    <span className="text-xs font-medium text-amber-600 bg-amber-50 border border-amber-200 rounded-full px-2 py-0.5 flex items-center gap-1 whitespace-nowrap">
                      <TrendingUp className="h-3 w-3" /> Near Cap
                    </span>
                  )}
                </div>
              </CardHeader>
              <CardContent className="pb-4 space-y-3">
                {/* Chart */}
                <div className="relative">
                  {view === "bar"
                    ? <BarChart trend={t.trend} cap={t.daily_review_cap} />
                    : <SparklineChart trend={t.trend} cap={t.daily_review_cap} />}
                </div>

                {/* Day labels */}
                <div className="flex justify-between">
                  {t.trend.map((d, i) => (
                    <span
                      key={d.date}
                      className={`text-xs ${i === t.trend.length - 1 ? "font-semibold text-foreground" : "text-muted-foreground"}`}
                    >
                      {i === t.trend.length - 1 ? "Today" : shortDate(d.date)}
                    </span>
                  ))}
                </div>

                {/* Stats row */}
                <div className="flex items-center justify-between text-xs pt-1 border-t">
                  <span className="text-muted-foreground">
                    Today: <span className="font-semibold text-foreground">{t.today_count}</span>
                    {t.daily_review_cap !== null && (
                      <> / <span className="text-muted-foreground">{t.daily_review_cap} cap</span></>
                    )}
                  </span>
                  {t.daily_review_cap !== null && t.daily_review_cap > 0 && (
                    <div className="flex items-center gap-1">
                      <div className="h-1.5 w-20 bg-slate-200 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${t.overCap ? "bg-red-500" : t.nearCap ? "bg-amber-400" : "bg-primary"}`}
                          style={{ width: `${Math.min((t.today_count / t.daily_review_cap) * 100, 100)}%` }}
                        />
                      </div>
                      <span className={t.overCap ? "text-red-600 font-medium" : t.nearCap ? "text-amber-600" : "text-muted-foreground"}>
                        {Math.round((t.today_count / t.daily_review_cap) * 100)}%
                      </span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {data && data.tenants.length === 0 && (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            No active tenants found.
          </CardContent>
        </Card>
      )}
    </div>
  );
}
