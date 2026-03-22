"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Building2,
  Search,
  Plus,
  Users,
  Activity,
  Hash,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

type Tenant = {
  id: string;
  slug: string;
  name: string;
  display_name?: string;
  status: string;
  trial_start: string;
  trial_end: string;
  is_active: boolean;
  created_at: string;
};

type TenantStats = {
  memberCount: number;
  todayUsage: number;
  epoch: number;
};

export default function TenantsPage() {
  const [tenants, setTenants] = useState<Tenant[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [statsCache, setStatsCache] = useState<Record<string, TenantStats>>({});

  const [createOpen, setCreateOpen] = useState(false);
  const [createLoading, setCreateLoading] = useState(false);
  const [createError, setCreateError] = useState("");
  const [form, setForm] = useState({
    slug: "",
    name: "",
    display_name: "",
    trial_days: 30,
    daily_review_cap: 50,
    daily_download_cap: 20,
  });

  useEffect(() => {
    fetch("/api/tenants")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setTenants(data);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!expandedId) return;
    const t = tenants.find((x) => x.id === expandedId);
    if (!t || statsCache[t.id]) return;
    fetch(
      `/api/tenants/stats?tenant_id=${t.id}&tenant_slug=${encodeURIComponent(t.slug)}`
    )
      .then((r) => r.json())
      .then((s) => setStatsCache((c) => ({ ...c, [t.id]: s })))
      .catch(() => {});
  }, [expandedId, tenants, statsCache]);

  const filtered = tenants.filter(
    (t) =>
      !search ||
      t.slug.toLowerCase().includes(search.toLowerCase()) ||
      (t.name || "").toLowerCase().includes(search.toLowerCase())
  );

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    setCreateError("");
    setCreateLoading(true);
    try {
      const res = await fetch("/api/tenants/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (!res.ok) {
        setCreateError(data.error || data.details || "Failed");
        return;
      }
      setForm({
        slug: "",
        name: "",
        display_name: "",
        trial_days: 30,
        daily_review_cap: 50,
        daily_download_cap: 20,
      });
      setCreateOpen(false);
      setTenants((prev) => [data.tenant, ...prev]);
    } catch {
      setCreateError("Network error");
    } finally {
      setCreateLoading(false);
    }
  }

  function formatDate(s: string) {
    return s ? new Date(s).toLocaleDateString() : "—";
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Tenant Management</h1>
          <p className="text-muted-foreground mt-1">
            Create and manage tenants. Search by slug or name.
          </p>
        </div>
        <Button onClick={() => setCreateOpen(!createOpen)}>
          <Plus className="h-4 w-4 mr-2" />
          Create Tenant
        </Button>
      </div>

      {createOpen && (
        <Card>
          <CardHeader>
            <CardTitle>Create New Tenant</CardTitle>
            <CardDescription>Fill in the required fields</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleCreate} className="space-y-4">
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="slug">Slug *</Label>
                  <Input
                    id="slug"
                    value={form.slug}
                    onChange={(e) => setForm((f) => ({ ...f, slug: e.target.value }))}
                    placeholder="acme-corp"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="name">Name *</Label>
                  <Input
                    id="name"
                    value={form.name}
                    onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
                    placeholder="Acme Corporation"
                    required
                  />
                </div>
              </div>
              <div className="grid gap-4 sm:grid-cols-3">
                <div className="space-y-2">
                  <Label htmlFor="trial_days">Trial Days</Label>
                  <Input
                    id="trial_days"
                    type="number"
                    min={1}
                    value={form.trial_days}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, trial_days: Number(e.target.value) }))
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="review_cap">Daily Review Cap</Label>
                  <Input
                    id="review_cap"
                    type="number"
                    min={0}
                    value={form.daily_review_cap}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, daily_review_cap: Number(e.target.value) }))
                    }
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="download_cap">Daily Download Cap</Label>
                  <Input
                    id="download_cap"
                    type="number"
                    min={0}
                    value={form.daily_download_cap}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, daily_download_cap: Number(e.target.value) }))
                    }
                  />
                </div>
              </div>
              {createError && (
                <p className="text-sm text-destructive">{createError}</p>
              )}
              <div className="flex gap-2">
                <Button type="submit" disabled={createLoading}>
                  {createLoading ? "Creating..." : "Create Tenant"}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setCreateOpen(false)}
                >
                  Cancel
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Search className="h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search by slug or name..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="max-w-sm"
            />
          </div>
        </CardHeader>
        <CardContent>
          {loading ? (
            <p className="text-muted-foreground py-8 text-center">Loading tenants...</p>
          ) : filtered.length === 0 ? (
            <p className="text-muted-foreground py-8 text-center">
              No tenants found. Create your first tenant above.
            </p>
          ) : (
            <div className="space-y-2">
              {filtered.map((t) => {
                const expanded = expandedId === t.id;
                const stats = statsCache[t.id];
                return (
                  <div
                    key={t.id}
                    className="border rounded-lg overflow-hidden"
                  >
                    <button
                      className="w-full flex items-center justify-between p-4 text-left hover:bg-slate-50 transition-colors"
                      onClick={() =>
                        setExpandedId(expanded ? null : t.id)
                      }
                    >
                      <div className="flex items-center gap-3">
                        <Building2 className="h-5 w-5 text-slate-400" />
                        <div>
                          <p className="font-medium">{t.slug}</p>
                          <p className="text-sm text-muted-foreground">
                            {t.name || t.display_name || "—"}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <span
                          className={`text-xs px-2 py-1 rounded ${
                            t.is_active
                              ? "bg-green-100 text-green-800"
                              : "bg-red-100 text-red-800"
                          }`}
                        >
                          {t.is_active ? "Active" : "Inactive"}
                        </span>
                        {expanded ? (
                          <ChevronUp className="h-4 w-4" />
                        ) : (
                          <ChevronDown className="h-4 w-4" />
                        )}
                      </div>
                    </button>
                    {expanded && (
                      <div className="border-t bg-slate-50 p-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                        <div className="flex items-center gap-2">
                          <Users className="h-4 w-4 text-muted-foreground" />
                          <div>
                            <p className="text-xs text-muted-foreground">Members</p>
                            <p className="font-medium">
                              {stats ? stats.memberCount : "—"}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Activity className="h-4 w-4 text-muted-foreground" />
                          <div>
                            <p className="text-xs text-muted-foreground">Today&apos;s Usage</p>
                            <p className="font-medium">
                              {stats ? stats.todayUsage : "—"}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Hash className="h-4 w-4 text-muted-foreground" />
                          <div>
                            <p className="text-xs text-muted-foreground">Epoch</p>
                            <p className="font-medium">
                              {stats ? stats.epoch : "—"}
                            </p>
                          </div>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Trial</p>
                          <p className="font-medium">
                            {formatDate(t.trial_start)} → {formatDate(t.trial_end)}
                          </p>
                        </div>
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
