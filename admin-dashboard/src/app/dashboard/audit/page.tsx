"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Download, RefreshCw, ShieldCheck, Users, Settings,
  AlertCircle, LogIn, Trash2, Clock, ChevronDown, ChevronRight,
  TrendingUp, CheckCircle2, XCircle, Activity, Search,
} from "lucide-react";

type AuditEvent = {
  id: string;
  action: string;
  tenant_slug?: string;
  result?: string;
  actor_email?: string;
  email?: string;
  deny_reason?: string;
  context?: Record<string, unknown>;
  created_at: string;
};

type Tenant = { id: string; slug: string; name?: string };

// ── Action metadata ──────────────────────────────────────────────────────────
const ACTION_META: Record<string, { label: string; icon: React.ReactNode; color: string; category: string }> = {
  members_batch_added: { label: "Batch Members Added", icon: <Users className="h-4 w-4" />, color: "bg-blue-100 text-blue-700 border-blue-200", category: "members" },
  member_updated:      { label: "Member Updated",       icon: <Users className="h-4 w-4" />, color: "bg-sky-100 text-sky-700 border-sky-200", category: "members" },
  member_deleted:      { label: "Member Deleted",       icon: <Trash2 className="h-4 w-4" />, color: "bg-orange-100 text-orange-700 border-orange-200", category: "members" },
  tenant_updated:      { label: "Tenant Updated",       icon: <Settings className="h-4 w-4" />, color: "bg-purple-100 text-purple-700 border-purple-200", category: "tenants" },
  tenant_created:      { label: "Tenant Created",       icon: <Settings className="h-4 w-4" />, color: "bg-emerald-100 text-emerald-700 border-emerald-200", category: "tenants" },
  epoch_revoke:        { label: "Sessions Revoked",     icon: <ShieldCheck className="h-4 w-4" />, color: "bg-red-100 text-red-700 border-red-200", category: "security" },
  member_revoked:      { label: "Access Revoked",       icon: <ShieldCheck className="h-4 w-4" />, color: "bg-red-100 text-red-700 border-red-200", category: "security" },
  admin_login:         { label: "Admin Login",          icon: <LogIn className="h-4 w-4" />, color: "bg-green-100 text-green-700 border-green-200", category: "auth" },
  sso_verify:          { label: "SSO Verify",           icon: <LogIn className="h-4 w-4" />, color: "bg-green-100 text-green-700 border-green-200", category: "auth" },
  access_denied:       { label: "Access Denied",        icon: <XCircle className="h-4 w-4" />, color: "bg-red-100 text-red-700 border-red-200", category: "security" },
  analyzer_launch:     { label: "Analyzer Launch",      icon: <Activity className="h-4 w-4" />, color: "bg-indigo-100 text-indigo-700 border-indigo-200", category: "usage" },
};
function getMeta(action: string) {
  return ACTION_META[action] ?? { label: action, icon: <Clock className="h-4 w-4" />, color: "bg-slate-100 text-slate-700 border-slate-200", category: "other" };
}

const QUICK_FILTERS = [
  { label: "All", value: "", category: "" },
  { label: "Security", value: "", category: "security" },
  { label: "Members", value: "", category: "members" },
  { label: "Tenants", value: "", category: "tenants" },
  { label: "Auth", value: "", category: "auth" },
  { label: "Failed", value: "failed", category: "" },
];

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmtDate(s: string) {
  return new Date(s).toLocaleString("en-US", { month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit" });
}
function fmtDateGroup(s: string) {
  return new Date(s).toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" });
}
function toDateKey(s: string) { return new Date(s).toDateString(); }

function eventsToCSV(events: AuditEvent[]) {
  const hdr = ["Time", "Action", "Tenant", "Actor", "Email", "Result", "Deny Reason", "Context"];
  const rows = events.map((e) => [
    e.created_at, e.action, e.tenant_slug ?? "", e.actor_email ?? "",
    e.email ?? "", e.result ?? "", e.deny_reason ?? "",
    e.context ? JSON.stringify(e.context) : "",
  ]);
  return [hdr, ...rows].map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(",")).join("\n");
}

// ── Mini sparkline ─────────────────────────────────────────────────────────────
function MiniBar({ data }: { data: { date: string; count: number }[] }) {
  const max = Math.max(...data.map((d) => d.count), 1);
  return (
    <div className="flex items-end gap-0.5 h-8">
      {data.map((d) => (
        <div key={d.date} className="flex-1 flex flex-col items-center group relative">
          <div className="absolute bottom-full mb-0.5 left-1/2 -translate-x-1/2 hidden group-hover:block z-10 bg-foreground text-background text-xs rounded px-1.5 py-0.5 whitespace-nowrap">
            {new Date(d.date + "T12:00:00Z").toLocaleDateString("en-US", { month: "short", day: "numeric" })}: {d.count}
          </div>
          <div className="w-full rounded-t-sm bg-primary/70" style={{ height: `${Math.max((d.count / max) * 100, 4)}%` }} />
        </div>
      ))}
    </div>
  );
}

// ── Event row ─────────────────────────────────────────────────────────────────
function EventRow({ e }: { e: AuditEvent }) {
  const [open, setOpen] = useState(false);
  const meta = getMeta(e.action);
  const isSuccess = !e.result || e.result === "success";
  const hasDot = e.action === "access_denied" || e.result === "denied" || e.result === "error";

  return (
    <div className={`relative flex gap-4 items-start ${open ? "mb-1" : ""}`}>
      {/* Timeline dot */}
      <div className={`absolute -left-4 mt-1.5 flex h-4 w-4 items-center justify-center rounded-full border-2 border-background ${hasDot ? "bg-red-500" : isSuccess ? "bg-primary" : "bg-amber-400"}`}>
        <div className="h-1.5 w-1.5 rounded-full bg-white" />
      </div>

      <div className="flex-1 min-w-0">
        <button
          className="w-full text-left"
          onClick={() => setOpen(!open)}
        >
          <Card className={`shadow-none border transition-colors ${open ? "border-primary/40 bg-slate-50" : "hover:border-slate-300"}`}>
            <CardContent className="py-2.5 px-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium border ${meta.color}`}>
                    {meta.icon}{meta.label}
                  </span>
                  {e.tenant_slug && <span className="text-xs bg-slate-100 rounded px-2 py-0.5 font-mono">{e.tenant_slug}</span>}
                  {e.email && <span className="text-xs text-muted-foreground">{e.email}</span>}
                  {e.deny_reason && (
                    <span className="text-xs bg-red-50 text-red-600 border border-red-200 rounded px-2 py-0.5">{e.deny_reason}</span>
                  )}
                  {!isSuccess && !e.deny_reason && <span className="text-xs text-destructive font-medium">{e.result}</span>}
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-muted-foreground whitespace-nowrap">{fmtDate(e.created_at)}</span>
                  {(e.context && Object.keys(e.context).length > 0) && (
                    open ? <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" /> : <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                  )}
                </div>
              </div>
              {e.actor_email && <p className="text-xs text-muted-foreground mt-0.5">by {e.actor_email}</p>}
            </CardContent>
          </Card>
        </button>

        {/* Expanded context */}
        {open && e.context && Object.keys(e.context).length > 0 && (
          <div className="ml-0 mt-1 rounded-md border border-slate-200 bg-white p-3 text-xs font-mono text-slate-700 space-y-1">
            <p className="text-xs font-sans font-semibold text-slate-500 uppercase tracking-wide mb-2">Event Context</p>
            {Object.entries(e.context).map(([k, v]) => (
              <div key={k} className="flex gap-2">
                <span className="text-muted-foreground min-w-[120px] shrink-0">{k}:</span>
                <span className="break-all">{Array.isArray(v) ? `[${(v as unknown[]).join(", ")}]` : String(v)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function AuditPage() {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [tenants, setTenants] = useState<Tenant[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 50;

  // Filters
  const [tenantFilter, setTenantFilter] = useState("");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");
  const [actorSearch, setActorSearch] = useState("");
  const [quickFilter, setQuickFilter] = useState<{ category: string; result: string }>({ category: "", result: "" });

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const loadEvents = useCallback(async () => {
    setLoading(true);
    setError("");
    setPage(1);
    try {
      const params = new URLSearchParams({ limit: "500" });
      if (tenantFilter) params.set("tenant_slug", tenantFilter);
      if (fromDate) params.set("from", fromDate);
      if (toDate) params.set("to", toDate);
      const res = await fetch(`/api/audit?${params}`);
      const data = await res.json();
      if (!res.ok) { setError(data.error || "Failed to load"); return; }
      setEvents(Array.isArray(data) ? data : []);
    } catch { setError("Network error"); }
    finally { setLoading(false); }
  }, [tenantFilter, fromDate, toDate]);

  useEffect(() => { loadEvents(); }, [loadEvents]);
  useEffect(() => {
    fetch("/api/tenants").then((r) => r.json()).then((d) => { if (Array.isArray(d)) setTenants(d); });
  }, []);

  // Client-side filter
  const filtered = events.filter((e) => {
    if (quickFilter.category && getMeta(e.action).category !== quickFilter.category) return false;
    if (quickFilter.result === "failed" && e.result !== "denied" && e.result !== "error") return false;
    if (actorSearch) {
      const q = actorSearch.toLowerCase();
      if (!(e.actor_email ?? "").toLowerCase().includes(q) && !(e.email ?? "").toLowerCase().includes(q)) return false;
    }
    return true;
  });

  const paginated = filtered.slice(0, page * PAGE_SIZE);
  const hasMore = filtered.length > paginated.length;

  // Stats
  const successCount = events.filter((e) => e.result === "success" || !e.result).length;
  const failedCount = events.filter((e) => e.result === "denied" || e.result === "error").length;
  const successRate = events.length > 0 ? Math.round((successCount / events.length) * 100) : 100;
  const uniqueTenants = new Set(events.map((e) => e.tenant_slug).filter(Boolean)).size;
  const uniqueActors = new Set(events.map((e) => e.actor_email).filter(Boolean)).size;

  // Per-day counts (last 7 days for sparkline)
  const last7 = Array.from({ length: 7 }, (_, i) => {
    const d = new Date(); d.setDate(d.getDate() - (6 - i));
    return d.toISOString().slice(0, 10);
  });
  const dayMap = new Map<string, number>();
  for (const e of events) {
    const k = e.created_at.slice(0, 10);
    dayMap.set(k, (dayMap.get(k) ?? 0) + 1);
  }
  const sparkData = last7.map((d) => ({ date: d, count: dayMap.get(d) ?? 0 }));

  // Deny reason breakdown
  const denyMap = new Map<string, number>();
  for (const e of events) {
    if (e.deny_reason) denyMap.set(e.deny_reason, (denyMap.get(e.deny_reason) ?? 0) + 1);
  }
  const denyBreakdown = [...denyMap.entries()].sort((a, b) => b[1] - a[1]);

  // Group paginated by date
  const grouped: { dateKey: string; dateLabel: string; items: AuditEvent[] }[] = [];
  for (const e of paginated) {
    const dk = toDateKey(e.created_at);
    const g = grouped.find((x) => x.dateKey === dk);
    if (g) g.items.push(e);
    else grouped.push({ dateKey: dk, dateLabel: fmtDateGroup(e.created_at), items: [e] });
  }

  function handleExportCSV() {
    const blob = new Blob([eventsToCSV(filtered)], { type: "text/csv;charset=utf-8;" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `audit-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click(); URL.revokeObjectURL(a.href);
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Audit Logs</h1>
          <p className="text-muted-foreground mt-1">Full operation history with context drill-down. Click any event to expand details.</p>
        </div>
        <Button variant="outline" size="sm" onClick={handleExportCSV} disabled={filtered.length === 0}>
          <Download className="h-4 w-4 mr-1" />Export CSV ({filtered.length})
        </Button>
      </div>

      {/* Stats row */}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        {[
          { label: "Total Events", value: events.length, icon: Activity, color: "text-blue-600", bg: "bg-blue-50" },
          { label: "Success Rate", value: `${successRate}%`, icon: CheckCircle2, color: successRate >= 90 ? "text-green-600" : "text-amber-600", bg: successRate >= 90 ? "bg-green-50" : "bg-amber-50" },
          { label: "Failed / Denied", value: failedCount, icon: XCircle, color: failedCount > 0 ? "text-red-600" : "text-slate-400", bg: failedCount > 0 ? "bg-red-50" : "bg-slate-50" },
          { label: "Unique Tenants", value: uniqueTenants, icon: Settings, color: "text-purple-600", bg: "bg-purple-50" },
          { label: "Unique Actors", value: uniqueActors, icon: Users, color: "text-indigo-600", bg: "bg-indigo-50" },
        ].map((k) => {
          const Icon = k.icon;
          return (
            <Card key={k.label} className="shadow-none">
              <CardContent className="pt-3 pb-3 flex items-center gap-3">
                <div className={`p-2 rounded-lg ${k.bg}`}><Icon className={`h-4 w-4 ${k.color}`} /></div>
                <div>
                  <p className="text-xs text-muted-foreground">{k.label}</p>
                  <p className="text-xl font-bold">{loading ? "—" : k.value}</p>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Sparkline + deny breakdown */}
      {events.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card className="lg:col-span-2">
            <CardHeader className="pb-2 pt-4">
              <CardTitle className="text-sm flex items-center gap-2"><TrendingUp className="h-4 w-4" />Events per Day (last 7 days)</CardTitle>
            </CardHeader>
            <CardContent className="pb-4">
              <MiniBar data={sparkData} />
              <div className="flex justify-between mt-1">
                {sparkData.map((d, i) => (
                  <span key={d.date} className={`text-xs ${i === 6 ? "font-semibold" : "text-muted-foreground"}`}>
                    {i === 6 ? "Today" : new Date(d.date + "T12:00:00Z").toLocaleDateString("en-US", { weekday: "short" })}
                  </span>
                ))}
              </div>
            </CardContent>
          </Card>

          {denyBreakdown.length > 0 && (
            <Card>
              <CardHeader className="pb-2 pt-4">
                <CardTitle className="text-sm flex items-center gap-2"><XCircle className="h-4 w-4 text-red-500" />Denial Reasons</CardTitle>
              </CardHeader>
              <CardContent className="pb-4 space-y-2">
                {denyBreakdown.map(([reason, count]) => (
                  <div key={reason} className="flex items-center justify-between text-sm">
                    <span className="text-xs text-slate-600 font-mono">{reason}</span>
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 w-16 bg-slate-100 rounded-full overflow-hidden">
                        <div className="h-full bg-red-400 rounded-full" style={{ width: `${(count / events.length) * 100}%` }} />
                      </div>
                      <span className="text-xs font-semibold text-red-600 w-6 text-right">{count}</span>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Filters */}
      <Card>
        <CardContent className="pt-4 pb-4 space-y-3">
          {/* Quick filter pills */}
          <div className="flex flex-wrap gap-2">
            {QUICK_FILTERS.map((f) => {
              const active = quickFilter.category === f.category && quickFilter.result === f.value;
              return (
                <button
                  key={f.label}
                  onClick={() => setQuickFilter({ category: f.category, result: f.value })}
                  className={`px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
                    active ? "bg-primary text-primary-foreground border-primary" : "bg-white text-muted-foreground border-slate-200 hover:border-slate-400"
                  }`}
                >
                  {f.label}
                  {f.label === "Failed" && failedCount > 0 && (
                    <span className={`ml-1.5 rounded-full px-1.5 py-0.5 text-xs ${active ? "bg-white/20" : "bg-red-100 text-red-600"}`}>{failedCount}</span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Detailed filters */}
          <div className="flex flex-wrap gap-3 items-end">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <Input
                placeholder="Search actor or user email..."
                value={actorSearch}
                onChange={(e) => {
                  setActorSearch(e.target.value);
                  if (debounceRef.current) clearTimeout(debounceRef.current);
                }}
                className="pl-8 h-9 w-56 text-sm"
              />
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">Tenant</p>
              <select value={tenantFilter} onChange={(e) => setTenantFilter(e.target.value)} className="h-9 rounded-md border border-input bg-background px-2 text-sm">
                <option value="">All Tenants</option>
                {tenants.map((t) => <option key={t.id} value={t.slug}>{t.name || t.slug}</option>)}
              </select>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">From</p>
              <Input type="date" value={fromDate} onChange={(e) => setFromDate(e.target.value)} className="h-9 w-36 text-sm" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">To</p>
              <Input type="date" value={toDate} onChange={(e) => setToDate(e.target.value)} className="h-9 w-36 text-sm" />
            </div>
            <Button variant="outline" size="sm" onClick={loadEvents} disabled={loading}>
              <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            </Button>
          </div>

          {filtered.length !== events.length && (
            <p className="text-xs text-muted-foreground">Showing {filtered.length} of {events.length} events</p>
          )}
        </CardContent>
      </Card>

      {error && <div className="flex items-center gap-2 text-destructive text-sm"><AlertCircle className="h-4 w-4" />{error}</div>}

      {/* Timeline */}
      {filtered.length === 0 && !loading ? (
        <Card><CardContent className="py-12 text-center text-muted-foreground">No events match the current filters.</CardContent></Card>
      ) : (
        <div className="space-y-8">
          {grouped.map((group) => (
            <div key={group.dateKey}>
              <div className="flex items-center gap-3 mb-4">
                <div className="h-px flex-1 bg-border" />
                <span className="text-xs font-medium text-muted-foreground whitespace-nowrap">{group.dateLabel} — {group.items.length} event{group.items.length !== 1 ? "s" : ""}</span>
                <div className="h-px flex-1 bg-border" />
              </div>
              <div className="relative space-y-2 pl-6">
                <div className="absolute left-2 top-0 bottom-0 w-px bg-border" />
                {group.items.map((e) => <EventRow key={e.id} e={e} />)}
              </div>
            </div>
          ))}

          {hasMore && (
            <div className="text-center">
              <Button variant="outline" onClick={() => setPage((p) => p + 1)}>
                Load more ({filtered.length - paginated.length} remaining)
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
