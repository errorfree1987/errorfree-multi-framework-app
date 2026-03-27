"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Building2, Search, Plus, Users, Activity, Hash,
  ChevronDown, ChevronUp, Pencil, Loader2, AlertTriangle,
  CheckCircle2, Clock, XCircle, Zap,
} from "lucide-react";

type Tenant = {
  id: string; slug: string; name: string; display_name?: string;
  status: string; trial_start: string; trial_end: string;
  is_active: boolean; created_at: string;
};
type TenantStats = {
  memberCount: number; todayUsage: number; epoch: number;
  daily_review_cap?: number | null; daily_download_cap?: number | null;
};

type FilterTab = "all" | "active" | "expiring" | "inactive";

function daysUntil(s?: string): number | null {
  if (!s) return null;
  return Math.ceil((new Date(s).getTime() - Date.now()) / 86400000);
}

function formatDate(s?: string) {
  return s ? new Date(s).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }) : "—";
}

function TrialBadge({ trialEnd, isActive }: { trialEnd?: string; isActive: boolean }) {
  if (!isActive) return <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium bg-slate-100 text-slate-500"><XCircle className="h-3 w-3" />Inactive</span>;
  const d = daysUntil(trialEnd);
  if (d === null) return <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-700"><CheckCircle2 className="h-3 w-3" />No Expiry</span>;
  if (d < 0) return <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium bg-red-100 text-red-700"><AlertTriangle className="h-3 w-3" />Expired {Math.abs(d)}d ago</span>;
  if (d === 0) return <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium bg-red-100 text-red-700"><AlertTriangle className="h-3 w-3" />Expires today</span>;
  if (d <= 7) return <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium bg-amber-100 text-amber-700"><Clock className="h-3 w-3" />{d}d left</span>;
  return <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium bg-green-100 text-green-700"><CheckCircle2 className="h-3 w-3" />{d}d left</span>;
}

function TrialBar({ trialStart, trialEnd }: { trialStart?: string; trialEnd?: string }) {
  if (!trialStart || !trialEnd) return null;
  const total = new Date(trialEnd).getTime() - new Date(trialStart).getTime();
  const elapsed = Date.now() - new Date(trialStart).getTime();
  const pct = Math.min(Math.max((elapsed / total) * 100, 0), 100);
  const d = daysUntil(trialEnd);
  const color = d !== null && d < 0 ? "bg-red-400" : d !== null && d <= 7 ? "bg-amber-400" : "bg-primary";
  return (
    <div className="mt-1.5">
      <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function TenantsPage() {
  const [tenants, setTenants] = useState<Tenant[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [tab, setTab] = useState<FilterTab>("all");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [statsCache, setStatsCache] = useState<Record<string, TenantStats>>({});
  const [extendingId, setExtendingId] = useState<string | null>(null);

  const [createOpen, setCreateOpen] = useState(false);
  const [createLoading, setCreateLoading] = useState(false);
  const [createError, setCreateError] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editLoading, setEditLoading] = useState(false);
  const [editError, setEditError] = useState("");
  const [editForm, setEditForm] = useState({
    trial_end: "", is_active: true, daily_review_cap: 50, daily_download_cap: 20,
  });
  const [form, setForm] = useState({
    slug: "", name: "", display_name: "", trial_days: 30,
    daily_review_cap: 50, daily_download_cap: 20,
  });

  const loadTenants = useCallback(() => {
    fetch("/api/tenants")
      .then((r) => r.json())
      .then((d) => { if (Array.isArray(d)) setTenants(d); })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { loadTenants(); }, [loadTenants]);

  useEffect(() => {
    if (!expandedId) return;
    const t = tenants.find((x) => x.id === expandedId);
    if (!t || statsCache[t.id]) return;
    fetch(`/api/tenants/stats?tenant_id=${t.id}&tenant_slug=${encodeURIComponent(t.slug)}`)
      .then((r) => r.json())
      .then((s) => setStatsCache((c) => ({ ...c, [t.id]: s })))
      .catch(() => {});
  }, [expandedId, tenants, statsCache]);

  // Summary counts
  const active = tenants.filter((t) => t.is_active);
  const expiring = tenants.filter((t) => { const d = daysUntil(t.trial_end); return d !== null && d >= 0 && d <= 7 && t.is_active; });
  const inactive = tenants.filter((t) => !t.is_active);
  const expired = tenants.filter((t) => { const d = daysUntil(t.trial_end); return d !== null && d < 0 && t.is_active; });

  const filtered = tenants.filter((t) => {
    const q = search.toLowerCase();
    const matchSearch = !q || t.slug.toLowerCase().includes(q) || (t.name || "").toLowerCase().includes(q);
    if (!matchSearch) return false;
    if (tab === "active") return t.is_active;
    if (tab === "expiring") { const d = daysUntil(t.trial_end); return d !== null && d >= 0 && d <= 7 && t.is_active; }
    if (tab === "inactive") return !t.is_active;
    return true;
  });

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    setCreateError("");
    setCreateLoading(true);
    try {
      const res = await fetch("/api/tenants/create", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (!res.ok) { setCreateError(data.error || "Failed"); return; }
      setForm({ slug: "", name: "", display_name: "", trial_days: 30, daily_review_cap: 50, daily_download_cap: 20 });
      setCreateOpen(false);
      setTenants((prev) => [data.tenant, ...prev]);
    } catch { setCreateError("Network error"); }
    finally { setCreateLoading(false); }
  }

  async function handleQuickExtend(t: Tenant, days: number) {
    setExtendingId(t.id);
    try {
      const currentEnd = t.trial_end ? new Date(t.trial_end) : new Date();
      currentEnd.setDate(currentEnd.getDate() + days);
      const newEnd = currentEnd.toISOString().slice(0, 10) + "T23:59:59.000Z";
      await fetch("/api/tenants/update", {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tenant_id: t.id, trial_end: newEnd, is_active: t.is_active }),
      });
      loadTenants();
      setStatsCache((c) => { const n = { ...c }; delete n[t.id]; return n; });
    } catch { /* silent */ }
    finally { setExtendingId(null); }
  }

  function openEdit(t: Tenant, stats?: TenantStats) {
    setEditForm({
      trial_end: t.trial_end ? t.trial_end.slice(0, 10) : "",
      is_active: t.is_active,
      daily_review_cap: stats?.daily_review_cap ?? 50,
      daily_download_cap: stats?.daily_download_cap ?? 20,
    });
    setEditingId(t.id);
    setEditError("");
  }

  async function handleSaveEdit(e: React.FormEvent) {
    e.preventDefault();
    if (!editingId) return;
    setEditLoading(true);
    setEditError("");
    try {
      const body: Record<string, unknown> = { tenant_id: editingId, is_active: editForm.is_active };
      if (editForm.trial_end) body.trial_end = editForm.trial_end + "T23:59:59.000Z";
      body.daily_review_cap = editForm.daily_review_cap === 0 ? null : editForm.daily_review_cap;
      body.daily_download_cap = editForm.daily_download_cap === 0 ? null : editForm.daily_download_cap;
      const res = await fetch("/api/tenants/update", {
        method: "PATCH", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) { setEditError(data.error || "Failed"); return; }
      setEditingId(null);
      setStatsCache((c) => { const n = { ...c }; delete n[editingId]; return n; });
      loadTenants();
    } catch { setEditError("Network error"); }
    finally { setEditLoading(false); }
  }

  const tabs: { key: FilterTab; label: string; count: number; color?: string }[] = [
    { key: "all", label: "All", count: tenants.length },
    { key: "active", label: "Active", count: active.length, color: "text-green-700" },
    { key: "expiring", label: "Expiring Soon", count: expiring.length, color: "text-amber-700" },
    { key: "inactive", label: "Inactive", count: inactive.length, color: "text-slate-500" },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Tenant Management</h1>
          <p className="text-muted-foreground mt-1">Create, configure, and monitor all tenants.</p>
        </div>
        <Button onClick={() => setCreateOpen(!createOpen)}>
          <Plus className="h-4 w-4 mr-2" />{createOpen ? "Cancel" : "Create Tenant"}
        </Button>
      </div>

      {/* Summary KPI cards */}
      <div className="grid gap-3 sm:grid-cols-4">
        {[
          { label: "Total Tenants", value: tenants.length, icon: Building2, bg: "bg-slate-50", color: "text-slate-600" },
          { label: "Active", value: active.length, icon: CheckCircle2, bg: "bg-green-50", color: "text-green-600" },
          { label: "Expiring ≤ 7d", value: expiring.length + expired.length, icon: Clock, bg: expiring.length + expired.length > 0 ? "bg-amber-50" : "bg-slate-50", color: expiring.length + expired.length > 0 ? "text-amber-600" : "text-slate-400" },
          { label: "Inactive", value: inactive.length, icon: XCircle, bg: "bg-slate-50", color: "text-slate-400" },
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

      {/* Create form */}
      {createOpen && (
        <Card>
          <CardHeader><CardTitle>Create New Tenant</CardTitle><CardDescription>All fields marked * are required.</CardDescription></CardHeader>
          <CardContent>
            <form onSubmit={handleCreate} className="space-y-4">
              <div className="grid gap-4 sm:grid-cols-2">
                <div><Label htmlFor="slug">Slug *</Label><Input id="slug" value={form.slug} onChange={(e) => setForm((f) => ({ ...f, slug: e.target.value }))} placeholder="acme-corp" required className="mt-1" /></div>
                <div><Label htmlFor="name">Name *</Label><Input id="name" value={form.name} onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))} placeholder="Acme Corporation" required className="mt-1" /></div>
                <div><Label htmlFor="display_name">Display Name</Label><Input id="display_name" value={form.display_name} onChange={(e) => setForm((f) => ({ ...f, display_name: e.target.value }))} placeholder="ACME Corp (optional)" className="mt-1" /></div>
                <div><Label htmlFor="trial_days">Trial Days</Label><Input id="trial_days" type="number" min={1} value={form.trial_days} onChange={(e) => setForm((f) => ({ ...f, trial_days: Number(e.target.value) }))} className="mt-1" /></div>
                <div><Label htmlFor="review_cap">Daily Review Cap <span className="text-muted-foreground text-xs">(0 = unlimited)</span></Label><Input id="review_cap" type="number" min={0} value={form.daily_review_cap} onChange={(e) => setForm((f) => ({ ...f, daily_review_cap: Number(e.target.value) }))} className="mt-1" /></div>
                <div><Label htmlFor="download_cap">Daily Download Cap <span className="text-muted-foreground text-xs">(0 = unlimited)</span></Label><Input id="download_cap" type="number" min={0} value={form.daily_download_cap} onChange={(e) => setForm((f) => ({ ...f, daily_download_cap: Number(e.target.value) }))} className="mt-1" /></div>
              </div>
              {createError && <p className="text-sm text-destructive">{createError}</p>}
              <div className="flex gap-2">
                <Button type="submit" disabled={createLoading}>{createLoading ? <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Creating...</> : "Create Tenant"}</Button>
                <Button type="button" variant="outline" onClick={() => setCreateOpen(false)}>Cancel</Button>
              </div>
            </form>
          </CardContent>
        </Card>
      )}

      {/* Filter + Search */}
      <Card>
        <CardHeader className="pb-0 pt-4">
          <div className="flex flex-col sm:flex-row gap-3 justify-between">
            <div className="flex gap-1 flex-wrap">
              {tabs.map((t) => (
                <button
                  key={t.key}
                  onClick={() => setTab(t.key)}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    tab === t.key ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-slate-100"
                  }`}
                >
                  {t.label}
                  <span className={`text-xs rounded-full px-1.5 py-0.5 min-w-[1.25rem] text-center ${
                    tab === t.key ? "bg-primary-foreground/20 text-primary-foreground" : "bg-slate-200 text-slate-600"
                  }`}>{t.count}</span>
                </button>
              ))}
            </div>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input placeholder="Search slug or name..." value={search} onChange={(e) => setSearch(e.target.value)} className="pl-9 h-9 max-w-xs" />
            </div>
          </div>
        </CardHeader>

        <CardContent className="pt-4">
          {loading ? (
            <div className="flex items-center justify-center py-12"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
          ) : filtered.length === 0 ? (
            <p className="text-muted-foreground py-8 text-center">No tenants found. {tab !== "all" && <button onClick={() => setTab("all")} className="text-primary underline">Clear filter</button>}</p>
          ) : (
            <div className="space-y-2">
              {filtered.map((t) => {
                const expanded = expandedId === t.id;
                const stats = statsCache[t.id];
                const d = daysUntil(t.trial_end);
                const isExpiring = d !== null && d >= 0 && d <= 7;
                const isExpired = d !== null && d < 0;

                return (
                  <div key={t.id} className={`border rounded-lg overflow-visible ${isExpired ? "border-red-200" : isExpiring ? "border-amber-200" : ""}`}>
                    <button
                      className="w-full flex items-center gap-3 p-4 text-left hover:bg-slate-50 transition-colors"
                      onClick={() => setExpandedId(expanded ? null : t.id)}
                    >
                      {/* Icon */}
                      <div className={`h-9 w-9 rounded-lg flex items-center justify-center text-sm font-bold shrink-0 ${t.is_active ? "bg-primary/10 text-primary" : "bg-slate-100 text-slate-400"}`}>
                        {(t.name || t.slug).charAt(0).toUpperCase()}
                      </div>

                      {/* Name + slug */}
                      <div className="flex-1 min-w-0">
                        <p className="font-semibold truncate">{t.name || t.slug}</p>
                        <div className="flex items-center gap-2 mt-0.5">
                          <code className="text-xs text-muted-foreground">{t.slug}</code>
                          <TrialBar trialStart={t.trial_start} trialEnd={t.trial_end} />
                        </div>
                      </div>

                      {/* Right side */}
                      <div className="flex items-center gap-3 shrink-0">
                        <TrialBadge trialEnd={t.trial_end} isActive={t.is_active} />
                        {expanded ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
                      </div>
                    </button>

                    {expanded && (
                      <div className="border-t bg-slate-50 p-4 space-y-4">
                        {/* Stats row */}
                        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
                          {[
                            { label: "Members", value: stats ? `${stats.memberCount}` : "—", icon: Users },
                            { label: "Today's Usage", value: stats ? `${stats.todayUsage}` : "—", icon: Activity },
                            { label: "Session Epoch", value: stats ? `${stats.epoch}` : "—", icon: Hash },
                            { label: "Review Cap / day", value: stats?.daily_review_cap != null ? `${stats.daily_review_cap}` : "Unlimited", icon: Zap },
                            { label: "Download Cap / day", value: stats?.daily_download_cap != null ? `${stats.daily_download_cap}` : "Unlimited", icon: Zap },
                          ].map((s) => {
                            const Icon = s.icon;
                            return (
                              <div key={s.label} className="bg-white rounded-md border px-3 py-2">
                                <p className="text-xs text-muted-foreground flex items-center gap-1"><Icon className="h-3 w-3" />{s.label}</p>
                                <p className="font-semibold text-sm mt-0.5">{s.value}</p>
                              </div>
                            );
                          })}
                        </div>

                        {/* Trial dates */}
                        <div className="flex flex-wrap gap-4 text-sm">
                          <div><span className="text-xs text-muted-foreground">Trial Start: </span><span className="font-medium">{formatDate(t.trial_start)}</span></div>
                          <div><span className="text-xs text-muted-foreground">Trial End: </span><span className="font-medium">{formatDate(t.trial_end)}</span></div>
                          <div><span className="text-xs text-muted-foreground">Created: </span><span className="font-medium">{formatDate(t.created_at)}</span></div>
                        </div>

                        {/* Action bar */}
                        <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-slate-200">
                          {editingId !== t.id && (
                            <>
                              <Button size="sm" onClick={(e) => { e.stopPropagation(); openEdit(t, stats); }}>
                                <Pencil className="h-3.5 w-3.5 mr-1" />Edit
                              </Button>
                              <Button size="sm" variant="outline" disabled={extendingId === t.id} onClick={(e) => { e.stopPropagation(); handleQuickExtend(t, 7); }}>
                                {extendingId === t.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "+7 days"}
                              </Button>
                              <Button size="sm" variant="outline" disabled={extendingId === t.id} onClick={(e) => { e.stopPropagation(); handleQuickExtend(t, 30); }}>
                                {extendingId === t.id ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "+30 days"}
                              </Button>
                            </>
                          )}
                        </div>

                        {/* Edit form */}
                        {editingId === t.id && (
                          <form onSubmit={handleSaveEdit} className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 items-end p-3 rounded-md border border-slate-200 bg-white">
                            <div><Label className="text-xs">Trial End Date</Label><Input type="date" value={editForm.trial_end} onChange={(e) => setEditForm((f) => ({ ...f, trial_end: e.target.value }))} className="h-9 mt-1" /></div>
                            <div><Label className="text-xs">Status</Label>
                              <select value={editForm.is_active ? "1" : "0"} onChange={(e) => setEditForm((f) => ({ ...f, is_active: e.target.value === "1" }))} className="h-9 mt-1 w-full rounded-md border px-2 text-sm">
                                <option value="1">Active</option><option value="0">Inactive</option>
                              </select>
                            </div>
                            <div><Label className="text-xs">Daily Review Cap <span className="text-muted-foreground">(0=∞)</span></Label><Input type="number" min={0} value={editForm.daily_review_cap} onChange={(e) => setEditForm((f) => ({ ...f, daily_review_cap: Number(e.target.value) }))} className="h-9 mt-1" /></div>
                            <div><Label className="text-xs">Daily Download Cap <span className="text-muted-foreground">(0=∞)</span></Label><Input type="number" min={0} value={editForm.daily_download_cap} onChange={(e) => setEditForm((f) => ({ ...f, daily_download_cap: Number(e.target.value) }))} className="h-9 mt-1" /></div>
                            {editError && <p className="text-sm text-destructive col-span-full">{editError}</p>}
                            <div className="flex gap-2 col-span-full">
                              <Button type="submit" size="sm" disabled={editLoading}>{editLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Save Changes"}</Button>
                              <Button type="button" variant="outline" size="sm" onClick={() => setEditingId(null)}>Cancel</Button>
                            </div>
                          </form>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
