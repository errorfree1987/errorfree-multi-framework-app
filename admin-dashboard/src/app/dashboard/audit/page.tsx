"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Card, CardContent, CardHeader, CardTitle, CardDescription,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Download, RefreshCw, ShieldCheck, Users, Settings,
  AlertCircle, LogIn, Trash2, Clock,
} from "lucide-react";

type AuditEvent = {
  id: string;
  action: string;
  tenant_slug?: string;
  result?: string;
  actor_email?: string;
  context?: Record<string, unknown>;
  created_at: string;
};

type Tenant = { id: string; slug: string; name?: string };

const ACTION_META: Record<string, { label: string; icon: React.ReactNode; color: string }> = {
  members_batch_added: { label: "Batch Add Members", icon: <Users className="h-4 w-4" />, color: "bg-blue-100 text-blue-700" },
  tenant_updated:      { label: "Tenant Updated",    icon: <Settings className="h-4 w-4" />, color: "bg-purple-100 text-purple-700" },
  member_revoked:      { label: "Access Revoked",    icon: <ShieldCheck className="h-4 w-4" />, color: "bg-red-100 text-red-700" },
  admin_login:         { label: "Admin Login",        icon: <LogIn className="h-4 w-4" />, color: "bg-green-100 text-green-700" },
  tenant_created:      { label: "Tenant Created",    icon: <Settings className="h-4 w-4" />, color: "bg-emerald-100 text-emerald-700" },
  member_deleted:      { label: "Member Deleted",    icon: <Trash2 className="h-4 w-4" />, color: "bg-orange-100 text-orange-700" },
};

function getActionMeta(action: string) {
  return ACTION_META[action] ?? {
    label: action,
    icon: <Clock className="h-4 w-4" />,
    color: "bg-slate-100 text-slate-700",
  };
}

function formatDate(s: string) {
  const d = new Date(s);
  return d.toLocaleString("en-US", {
    month: "short", day: "numeric", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

function formatDateGroup(s: string) {
  const d = new Date(s);
  return d.toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" });
}

function toDateKey(s: string) {
  return new Date(s).toDateString();
}

function eventsToCSV(events: AuditEvent[]): string {
  const header = ["Time", "Action", "Tenant", "Actor", "Result", "Context"];
  const rows = events.map((e) => [
    e.created_at,
    e.action,
    e.tenant_slug ?? "",
    e.actor_email ?? "",
    e.result ?? "",
    e.context ? JSON.stringify(e.context) : "",
  ]);
  return [header, ...rows].map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(",")).join("\n");
}

export default function AuditPage() {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [tenants, setTenants] = useState<Tenant[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [tenantFilter, setTenantFilter] = useState("");
  const [actionFilter, setActionFilter] = useState("");
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");

  const loadEvents = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams();
      if (tenantFilter) params.set("tenant_slug", tenantFilter);
      if (actionFilter) params.set("action", actionFilter);
      if (fromDate) params.set("from", fromDate);
      if (toDate) params.set("to", toDate);
      const res = await fetch(`/api/audit?${params.toString()}`);
      const data = await res.json();
      if (!res.ok) { setError(data.error || "Failed to load"); return; }
      setEvents(Array.isArray(data) ? data : []);
    } catch {
      setError("Network error");
    } finally {
      setLoading(false);
    }
  }, [tenantFilter, actionFilter, fromDate, toDate]);

  useEffect(() => { loadEvents(); }, [loadEvents]);

  useEffect(() => {
    fetch("/api/tenants").then((r) => r.json()).then((d) => {
      if (Array.isArray(d)) setTenants(d);
    });
  }, []);

  function handleExportCSV() {
    const csv = eventsToCSV(events);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `audit-log-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Group events by date
  const grouped: { dateKey: string; dateLabel: string; items: AuditEvent[] }[] = [];
  for (const e of events) {
    const dk = toDateKey(e.created_at);
    const existing = grouped.find((g) => g.dateKey === dk);
    if (existing) {
      existing.items.push(e);
    } else {
      grouped.push({ dateKey: dk, dateLabel: formatDateGroup(e.created_at), items: [e] });
    }
  }

  const uniqueActions = Array.from(new Set(events.map((e) => e.action))).sort();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Audit Logs</h1>
        <p className="text-muted-foreground mt-1">
          Full operation history. Filter by tenant, action, or date range.
        </p>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-wrap gap-3 items-end justify-between">
            <div className="flex flex-wrap gap-3 items-end">
              {/* Tenant filter */}
              <div>
                <p className="text-xs text-muted-foreground mb-1">Tenant</p>
                <select
                  value={tenantFilter}
                  onChange={(e) => setTenantFilter(e.target.value)}
                  className="h-9 rounded-md border border-input bg-background px-3 text-sm"
                >
                  <option value="">All Tenants</option>
                  {tenants.map((t) => (
                    <option key={t.id} value={t.slug}>{t.name || t.slug}</option>
                  ))}
                </select>
              </div>

              {/* Action filter */}
              <div>
                <p className="text-xs text-muted-foreground mb-1">Action</p>
                <select
                  value={actionFilter}
                  onChange={(e) => setActionFilter(e.target.value)}
                  className="h-9 rounded-md border border-input bg-background px-3 text-sm"
                >
                  <option value="">All Actions</option>
                  {uniqueActions.map((a) => (
                    <option key={a} value={a}>{getActionMeta(a).label}</option>
                  ))}
                </select>
              </div>

              {/* Date range */}
              <div>
                <p className="text-xs text-muted-foreground mb-1">From</p>
                <Input type="date" value={fromDate} onChange={(e) => setFromDate(e.target.value)} className="h-9 w-36" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">To</p>
                <Input type="date" value={toDate} onChange={(e) => setToDate(e.target.value)} className="h-9 w-36" />
              </div>

              <Button variant="outline" size="sm" onClick={loadEvents} disabled={loading}>
                <RefreshCw className={`h-4 w-4 mr-1 ${loading ? "animate-spin" : ""}`} />
                Refresh
              </Button>
            </div>

            {/* Export CSV */}
            <Button
              variant="outline"
              size="sm"
              onClick={handleExportCSV}
              disabled={events.length === 0}
            >
              <Download className="h-4 w-4 mr-1" />
              Export CSV ({events.length})
            </Button>
          </div>
        </CardHeader>
      </Card>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 text-destructive text-sm">
          <AlertCircle className="h-4 w-4" />
          {error}
        </div>
      )}

      {/* Timeline */}
      {events.length === 0 && !loading ? (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            No audit events found for the selected filters.
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-8">
          {grouped.map((group) => (
            <div key={group.dateKey}>
              {/* Date separator */}
              <div className="flex items-center gap-3 mb-4">
                <div className="h-px flex-1 bg-border" />
                <span className="text-xs font-medium text-muted-foreground whitespace-nowrap">
                  {group.dateLabel}
                </span>
                <div className="h-px flex-1 bg-border" />
              </div>

              {/* Events for this date */}
              <div className="relative space-y-3 pl-6">
                {/* Vertical line */}
                <div className="absolute left-2 top-0 bottom-0 w-px bg-border" />

                {group.items.map((e) => {
                  const meta = getActionMeta(e.action);
                  const isSuccess = !e.result || e.result === "success";
                  return (
                    <div key={e.id} className="relative flex gap-4 items-start">
                      {/* Dot on timeline */}
                      <div className={`absolute -left-4 mt-1 flex h-5 w-5 items-center justify-center rounded-full border-2 border-background ${isSuccess ? "bg-primary" : "bg-destructive"}`}>
                        <div className="h-1.5 w-1.5 rounded-full bg-white" />
                      </div>

                      <Card className="flex-1 shadow-none border">
                        <CardContent className="py-3 px-4">
                          <div className="flex flex-wrap items-start justify-between gap-2">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${meta.color}`}>
                                {meta.icon}
                                {meta.label}
                              </span>
                              {e.tenant_slug && (
                                <span className="text-xs text-muted-foreground bg-slate-100 rounded px-2 py-0.5">
                                  {e.tenant_slug}
                                </span>
                              )}
                              {!isSuccess && (
                                <span className="text-xs text-destructive font-medium">Failed</span>
                              )}
                            </div>
                            <span className="text-xs text-muted-foreground whitespace-nowrap">
                              {formatDate(e.created_at)}
                            </span>
                          </div>

                          {/* Context details */}
                          {e.context && Object.keys(e.context).length > 0 && (
                            <div className="mt-2 text-xs text-muted-foreground space-y-0.5">
                              {Object.entries(e.context).map(([k, v]) => (
                                <div key={k}>
                                  <span className="font-medium text-foreground">{k}:</span>{" "}
                                  {Array.isArray(v)
                                    ? `[${(v as unknown[]).slice(0, 3).join(", ")}${(v as unknown[]).length > 3 ? ` +${(v as unknown[]).length - 3} more` : ""}]`
                                    : String(v)}
                                </div>
                              ))}
                            </div>
                          )}

                          {e.actor_email && (
                            <p className="mt-1 text-xs text-muted-foreground">
                              by {e.actor_email}
                            </p>
                          )}
                        </CardContent>
                      </Card>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
