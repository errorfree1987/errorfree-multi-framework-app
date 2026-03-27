"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  AlertTriangle, AlertCircle, RefreshCw, TrendingUp, CheckCircle2,
  Users, BarChart3, Table2, ChevronDown, ChevronUp,
} from "lucide-react";

type DayPoint = { date: string; count: number };
type TopUser = { email: string; count: number };
type TenantUsage = {
  tenant_id: string; slug: string; name: string;
  daily_review_cap: number | null; daily_download_cap: number | null;
  today_count: number; period_total: number;
  peak_day: { date: string; count: number } | null;
  cap_hit_days: number;
  trend: DayPoint[]; overCap: boolean; nearCap: boolean;
  top_users: TopUser[];
};
type UsageData = { days: string[]; period: number; tenants: TenantUsage[] };

type SortKey = "name" | "today" | "total" | "utilization";

function shortDate(d: string) {
  return new Date(d + "T12:00:00Z").toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

// ── Bar chart ─────────────────────────────────────────────────────────────────
function BarChart({ trend, cap }: { trend: DayPoint[]; cap: number | null }) {
  const maxVal = Math.max(...trend.map((d) => d.count), cap ?? 0, 1);
  return (
    <div className="flex items-end gap-0.5" style={{ height: 64 }}>
      {trend.map((d, i) => {
        const pct = (d.count / maxVal) * 100;
        const isToday = i === trend.length - 1;
        const over = cap && cap > 0 && d.count >= cap;
        const near = cap && cap > 0 && !over && d.count >= cap * 0.8;
        const color = over ? "bg-red-500" : near ? "bg-amber-400" : isToday ? "bg-primary" : "bg-primary/35";
        return (
          <div key={d.date} className="flex-1 flex flex-col items-center group relative">
            <div className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 hidden group-hover:block z-10 bg-foreground text-background text-xs rounded px-2 py-1 whitespace-nowrap">
              {shortDate(d.date)}: {d.count}{cap ? ` / ${cap}` : ""}
            </div>
            <div className="w-full rounded-t-sm" style={{ height: `${Math.max(pct, 2)}%` }}>
              <div className={`w-full h-full rounded-t-sm ${color}`} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Utilization ring ──────────────────────────────────────────────────────────
function UtilRing({ pct, over, near }: { pct: number; over: boolean; near: boolean }) {
  const r = 20; const c = 2 * Math.PI * r;
  const dash = Math.min((pct / 100) * c, c);
  const color = over ? "#ef4444" : near ? "#f59e0b" : "#3b82f6";
  return (
    <svg width="48" height="48" viewBox="0 0 48 48">
      <circle cx="24" cy="24" r={r} fill="none" stroke="#e2e8f0" strokeWidth="4" />
      <circle cx="24" cy="24" r={r} fill="none" stroke={color} strokeWidth="4"
        strokeDasharray={`${dash} ${c - dash}`} strokeLinecap="round"
        transform="rotate(-90 24 24)" />
      <text x="24" y="28" textAnchor="middle" fontSize="10" fontWeight="bold" fill={color}>
        {pct > 999 ? "∞" : `${pct}%`}
      </text>
    </svg>
  );
}

// ── Tenant card ───────────────────────────────────────────────────────────────
function TenantCard({ t }: { t: TenantUsage }) {
  const [expanded, setExpanded] = useState(false);
  const utilPct = t.daily_review_cap && t.daily_review_cap > 0
    ? Math.round((t.today_count / t.daily_review_cap) * 100) : 0;

  return (
    <Card className={t.overCap ? "border-red-300" : t.nearCap ? "border-amber-300" : ""}>
      <CardHeader className="pb-2 pt-4">
        <div className="flex items-start gap-3">
          {t.daily_review_cap && t.daily_review_cap > 0 ? (
            <UtilRing pct={utilPct} over={t.overCap} near={t.nearCap} />
          ) : (
            <div className="h-12 w-12 rounded-full bg-slate-100 flex items-center justify-center text-lg font-bold text-slate-400">
              {t.name.charAt(0).toUpperCase()}
            </div>
          )}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 justify-between">
              <CardTitle className="text-sm truncate">{t.name}</CardTitle>
              {t.overCap && <span className="text-xs font-medium text-red-600 bg-red-50 border border-red-200 rounded-full px-2 py-0.5 flex items-center gap-1 whitespace-nowrap shrink-0"><AlertTriangle className="h-3 w-3" />Over Cap</span>}
              {t.nearCap && !t.overCap && <span className="text-xs font-medium text-amber-600 bg-amber-50 border border-amber-200 rounded-full px-2 py-0.5 flex items-center gap-1 whitespace-nowrap shrink-0"><TrendingUp className="h-3 w-3" />Near Cap</span>}
            </div>
            <CardDescription className="text-xs mt-0.5">{t.slug}</CardDescription>
          </div>
        </div>
      </CardHeader>

      <CardContent className="pb-4 space-y-3">
        {/* Bar chart */}
        <BarChart trend={t.trend} cap={t.daily_review_cap} />
        <div className="flex justify-between">
          {t.trend.map((d, i) => (
            <span key={d.date} className={`text-xs ${i === t.trend.length - 1 ? "font-semibold" : "text-muted-foreground"}`}>
              {i === t.trend.length - 1 ? "Today" : t.trend.length <= 10 ? shortDate(d.date) : i % 3 === 0 ? shortDate(d.date) : ""}
            </span>
          ))}
        </div>

        {/* Stats grid */}
        <div className="grid grid-cols-3 gap-2 pt-1 border-t text-center">
          <div>
            <p className="text-lg font-bold">{t.today_count}</p>
            <p className="text-xs text-muted-foreground">Today</p>
          </div>
          <div>
            <p className="text-lg font-bold">{t.period_total}</p>
            <p className="text-xs text-muted-foreground">Period Total</p>
          </div>
          <div>
            <p className="text-lg font-bold">{t.peak_day?.count ?? 0}</p>
            <p className="text-xs text-muted-foreground">Peak Day</p>
          </div>
        </div>

        {/* Cap info + hit days */}
        <div className="flex flex-wrap gap-3 text-xs text-muted-foreground border-t pt-2">
          <span>Review cap: <strong className="text-foreground">{t.daily_review_cap ?? "Unlimited"}</strong>/day</span>
          {t.daily_download_cap != null && <span>Download cap: <strong className="text-foreground">{t.daily_download_cap}</strong>/day</span>}
          {t.cap_hit_days > 0 && <span className="text-red-600 font-medium">Cap hit: {t.cap_hit_days}d in period</span>}
          {t.peak_day && <span>Peak: {shortDate(t.peak_day.date)}</span>}
        </div>

        {/* Expandable: top users */}
        {t.top_users.length > 0 && (
          <>
            <button
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
              onClick={() => setExpanded(!expanded)}
            >
              <Users className="h-3.5 w-3.5" />Top Users
              {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
            </button>
            {expanded && (
              <div className="space-y-1.5 border-t pt-2">
                {t.top_users.map((u) => {
                  const pct = t.period_total > 0 ? Math.round((u.count / t.period_total) * 100) : 0;
                  return (
                    <div key={u.email} className="flex items-center gap-2">
                      <span className="text-xs truncate flex-1 max-w-[140px]">{u.email}</span>
                      <div className="h-1.5 w-16 bg-slate-100 rounded-full overflow-hidden">
                        <div className="h-full bg-primary/60 rounded-full" style={{ width: `${pct}%` }} />
                      </div>
                      <span className="text-xs font-semibold w-8 text-right">{u.count}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

// ── Table view ────────────────────────────────────────────────────────────────
function TableView({ tenants, sortKey, setSortKey }: {
  tenants: TenantUsage[];
  sortKey: SortKey;
  setSortKey: (k: SortKey) => void;
}) {
  function SortBtn({ k, label }: { k: SortKey; label: string }) {
    return (
      <button className={`flex items-center gap-1 text-xs font-medium ${sortKey === k ? "text-primary" : "text-muted-foreground hover:text-foreground"}`} onClick={() => setSortKey(k)}>
        {label}{sortKey === k && <ChevronDown className="h-3 w-3" />}
      </button>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b bg-slate-50">
            <th className="text-left py-3 px-4"><SortBtn k="name" label="Tenant" /></th>
            <th className="text-right py-3 px-3"><SortBtn k="today" label="Today" /></th>
            <th className="text-right py-3 px-3"><SortBtn k="total" label="Period Total" /></th>
            <th className="text-center py-3 px-3">Peak Day</th>
            <th className="text-center py-3 px-3"><SortBtn k="utilization" label="Today's Util" /></th>
            <th className="text-center py-3 px-3">Cap Hits</th>
            <th className="text-left py-3 px-3">Caps</th>
            <th className="text-left py-3 px-3">Top User</th>
          </tr>
        </thead>
        <tbody>
          {tenants.map((t) => {
            const utilPct = t.daily_review_cap && t.daily_review_cap > 0
              ? Math.round((t.today_count / t.daily_review_cap) * 100) : null;
            return (
              <tr key={t.tenant_id} className={`border-b hover:bg-slate-50 ${t.overCap ? "bg-red-50/50" : t.nearCap ? "bg-amber-50/30" : ""}`}>
                <td className="py-3 px-4">
                  <div className="flex items-center gap-2">
                    <div className={`h-7 w-7 rounded-md flex items-center justify-center text-xs font-bold ${t.overCap ? "bg-red-100 text-red-600" : "bg-primary/10 text-primary"}`}>
                      {t.name.charAt(0).toUpperCase()}
                    </div>
                    <div>
                      <p className="font-medium">{t.name}</p>
                      <p className="text-xs text-muted-foreground">{t.slug}</p>
                    </div>
                  </div>
                </td>
                <td className="py-3 px-3 text-right font-mono font-semibold">{t.today_count}</td>
                <td className="py-3 px-3 text-right font-mono">{t.period_total}</td>
                <td className="py-3 px-3 text-center text-xs">
                  {t.peak_day ? <><span className="font-semibold">{t.peak_day.count}</span><span className="text-muted-foreground ml-1">({shortDate(t.peak_day.date)})</span></> : "—"}
                </td>
                <td className="py-3 px-3">
                  {utilPct !== null ? (
                    <div className="flex items-center gap-2 justify-center">
                      <div className="h-1.5 w-16 bg-slate-200 rounded-full overflow-hidden">
                        <div className={`h-full rounded-full ${t.overCap ? "bg-red-500" : t.nearCap ? "bg-amber-400" : "bg-primary"}`} style={{ width: `${Math.min(utilPct, 100)}%` }} />
                      </div>
                      <span className={`text-xs font-medium ${t.overCap ? "text-red-600" : t.nearCap ? "text-amber-600" : "text-muted-foreground"}`}>{utilPct}%</span>
                    </div>
                  ) : <span className="text-xs text-muted-foreground text-center block">Unlimited</span>}
                </td>
                <td className="py-3 px-3 text-center">
                  {t.cap_hit_days > 0 ? <span className="text-xs text-red-600 font-semibold">{t.cap_hit_days}d</span> : <span className="text-xs text-muted-foreground">—</span>}
                </td>
                <td className="py-3 px-3 text-xs text-muted-foreground">
                  <span>Review: {t.daily_review_cap ?? "∞"}</span>
                  {t.daily_download_cap != null && <><br /><span>DL: {t.daily_download_cap}</span></>}
                </td>
                <td className="py-3 px-3 text-xs text-muted-foreground">
                  {t.top_users[0] ? <><span className="truncate max-w-[120px] block">{t.top_users[0].email}</span><span className="text-muted-foreground">{t.top_users[0].count} events</span></> : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function UsagePage() {
  const [data, setData] = useState<UsageData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [period, setPeriod] = useState(7);
  const [view, setView] = useState<"card" | "table">("card");
  const [sortKey, setSortKey] = useState<SortKey>("total");

  async function load(p = period) {
    setLoading(true); setError("");
    try {
      const res = await fetch(`/api/usage?period=${p}`);
      const json = await res.json();
      if (!res.ok) { setError(json.error || "Failed"); return; }
      setData(json);
    } catch { setError("Network error"); }
    finally { setLoading(false); }
  }

  useEffect(() => { load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  function handlePeriod(p: number) { setPeriod(p); load(p); }

  const sorted = (data?.tenants ?? []).slice().sort((a, b) => {
    if (sortKey === "name") return a.name.localeCompare(b.name);
    if (sortKey === "today") return b.today_count - a.today_count;
    if (sortKey === "total") return b.period_total - a.period_total;
    if (sortKey === "utilization") {
      const ua = a.daily_review_cap ? a.today_count / a.daily_review_cap : 0;
      const ub = b.daily_review_cap ? b.today_count / b.daily_review_cap : 0;
      return ub - ua;
    }
    return 0;
  });

  const alerts = sorted.filter((t) => t.overCap || t.nearCap);
  const totalToday = sorted.reduce((s, t) => s + t.today_count, 0);
  const totalPeriod = sorted.reduce((s, t) => s + t.period_total, 0);
  const totalCapHits = sorted.reduce((s, t) => s + t.cap_hit_days, 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Usage & Caps</h1>
          <p className="text-muted-foreground mt-1">Monitor usage trends, cap utilization, and top active users per tenant.</p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {/* Period selector */}
          <div className="flex rounded-md border overflow-hidden">
            {[7, 14, 30].map((p) => (
              <button key={p} onClick={() => handlePeriod(p)} className={`px-3 py-1.5 text-sm font-medium transition-colors ${period === p ? "bg-primary text-primary-foreground" : "bg-white text-muted-foreground hover:bg-slate-50"}`}>
                {p}d
              </button>
            ))}
          </div>
          {/* View toggle */}
          <div className="flex rounded-md border overflow-hidden">
            <button onClick={() => setView("card")} className={`p-1.5 transition-colors ${view === "card" ? "bg-primary text-primary-foreground" : "bg-white text-muted-foreground hover:bg-slate-50"}`}><BarChart3 className="h-4 w-4" /></button>
            <button onClick={() => setView("table")} className={`p-1.5 transition-colors ${view === "table" ? "bg-primary text-primary-foreground" : "bg-white text-muted-foreground hover:bg-slate-50"}`}><Table2 className="h-4 w-4" /></button>
          </div>
          <Button variant="outline" size="sm" onClick={() => load()} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </div>
      </div>

      {/* Alert banner */}
      {alerts.length > 0 && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="py-3 px-4 flex items-start gap-2">
            <AlertTriangle className="h-5 w-5 text-red-500 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-red-700">{alerts.filter((t) => t.overCap).length > 0 ? `${alerts.filter((t) => t.overCap).length} tenant(s) exceeded daily cap` : `${alerts.length} tenant(s) near daily cap (≥80%)`}</p>
              <p className="text-xs text-red-600 mt-0.5">{alerts.map((t) => t.name).join(", ")}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Summary KPIs */}
      {data && (
        <div className="grid gap-3 sm:grid-cols-4">
          {[
            { label: "Active Tenants", value: sorted.length, icon: BarChart3, color: "text-blue-600", bg: "bg-blue-50" },
            { label: `Today's Total Usage`, value: totalToday, icon: TrendingUp, color: "text-green-600", bg: "bg-green-50" },
            { label: `${period}-day Total`, value: totalPeriod, icon: TrendingUp, color: "text-purple-600", bg: "bg-purple-50" },
            { label: "Cap Hit Days", value: totalCapHits, icon: alerts.length > 0 ? AlertTriangle : CheckCircle2, color: totalCapHits > 0 ? "text-red-600" : "text-green-600", bg: totalCapHits > 0 ? "bg-red-50" : "bg-green-50" },
          ].map((k) => {
            const Icon = k.icon;
            return (
              <Card key={k.label} className="shadow-none">
                <CardContent className="pt-3 pb-3 flex items-center gap-3">
                  <div className={`p-2 rounded-lg ${k.bg}`}><Icon className={`h-4 w-4 ${k.color}`} /></div>
                  <div>
                    <p className="text-xs text-muted-foreground">{k.label}</p>
                    <p className="text-2xl font-bold">{k.value}</p>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {error && <div className="flex items-center gap-2 text-destructive text-sm"><AlertCircle className="h-4 w-4" />{error}</div>}

      {/* Content */}
      {data && sorted.length === 0 && <Card><CardContent className="py-12 text-center text-muted-foreground">No active tenants found.</CardContent></Card>}

      {data && view === "card" && (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {sorted.map((t) => <TenantCard key={t.tenant_id} t={t} />)}
        </div>
      )}

      {data && view === "table" && (
        <Card>
          <CardHeader className="pb-0">
            <CardTitle className="text-base flex items-center gap-2"><Table2 className="h-4 w-4" />All Tenants — {period}-day Overview</CardTitle>
            <CardDescription>Click column headers to sort. Red dashed line on charts = daily cap.</CardDescription>
          </CardHeader>
          <CardContent className="pt-3">
            <TableView tenants={sorted} sortKey={sortKey} setSortKey={setSortKey} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
