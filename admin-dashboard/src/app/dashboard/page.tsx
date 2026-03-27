"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Building2, Users, BarChart3, FileText, Shield,
  AlertTriangle, TrendingUp, Clock, CheckCircle2,
  ArrowRight, Activity,
} from "lucide-react";

type Tenant = {
  id: string; slug: string; name?: string; is_active: boolean;
  trial_end?: string; trial_start?: string; status?: string;
};
type AuditEvent = {
  id: string; action: string; tenant_slug?: string;
  result?: string; actor_email?: string; created_at: string;
};

function daysUntil(dateStr?: string): number | null {
  if (!dateStr) return null;
  const diff = new Date(dateStr).getTime() - Date.now();
  return Math.ceil(diff / (1000 * 60 * 60 * 24));
}

function timeAgo(dateStr: string) {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

const ACTION_LABELS: Record<string, string> = {
  members_batch_added: "Batch members added",
  tenant_updated: "Tenant updated",
  member_revoked: "Access revoked",
  epoch_revoke: "Sessions revoked",
  admin_login: "Admin login",
  tenant_created: "Tenant created",
  member_updated: "Member updated",
  member_deleted: "Member deleted",
};

const ACTION_COLORS: Record<string, string> = {
  members_batch_added: "bg-blue-500",
  tenant_updated: "bg-purple-500",
  epoch_revoke: "bg-red-500",
  member_revoked: "bg-red-500",
  admin_login: "bg-green-500",
  tenant_created: "bg-emerald-500",
  member_updated: "bg-blue-400",
};

export default function DashboardPage() {
  const [tenants, setTenants] = useState<Tenant[]>([]);
  const [recentEvents, setRecentEvents] = useState<AuditEvent[]>([]);
  const [todayUsage, setTodayUsage] = useState<number | null>(null);
  const [totalMembers, setTotalMembers] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("/api/tenants").then((r) => r.json()),
      fetch("/api/audit?limit=8").then((r) => r.json()),
      fetch("/api/members").then((r) => r.json()),
      fetch("/api/usage").then((r) => r.json()),
    ])
      .then(([t, a, m, u]) => {
        if (Array.isArray(t)) setTenants(t);
        if (Array.isArray(a)) setRecentEvents(a);
        if (Array.isArray(m)) setTotalMembers(m.length);
        if (u?.tenants) {
          const total = (u.tenants as { today_count: number }[]).reduce(
            (s: number, x) => s + (x.today_count ?? 0), 0
          );
          setTodayUsage(total);
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const activeTenants = tenants.filter((t) => t.is_active);
  const expiringSoon = tenants.filter((t) => {
    const d = daysUntil(t.trial_end);
    return d !== null && d >= 0 && d <= 7 && t.is_active;
  });
  const expiredTrials = tenants.filter((t) => {
    const d = daysUntil(t.trial_end);
    return d !== null && d < 0 && t.is_active;
  });
  const activeMembers = totalMembers;

  const kpis = [
    {
      title: "Total Tenants",
      value: tenants.length,
      sub: `${activeTenants.length} active`,
      icon: Building2,
      color: "text-blue-600",
      bg: "bg-blue-50",
      href: "/dashboard/tenants",
    },
    {
      title: "Total Members",
      value: activeMembers ?? "—",
      sub: "across all tenants",
      icon: Users,
      color: "text-purple-600",
      bg: "bg-purple-50",
      href: "/dashboard/members",
    },
    {
      title: "Today's Usage",
      value: todayUsage ?? "—",
      sub: "total actions today",
      icon: Activity,
      color: "text-green-600",
      bg: "bg-green-50",
      href: "/dashboard/usage",
    },
    {
      title: "Trials Expiring",
      value: expiringSoon.length,
      sub: expiringSoon.length > 0 ? "within 7 days" : "all trials healthy",
      icon: Clock,
      color: expiringSoon.length > 0 ? "text-amber-600" : "text-slate-500",
      bg: expiringSoon.length > 0 ? "bg-amber-50" : "bg-slate-50",
      href: "/dashboard/tenants",
    },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground mt-1">
          Error-Free® Admin — system overview and quick actions.
        </p>
      </div>

      {/* KPI cards */}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {kpis.map((k) => {
          const Icon = k.icon;
          return (
            <Link key={k.title} href={k.href}>
              <Card className="hover:shadow-md transition-shadow cursor-pointer group">
                <CardContent className="pt-5 pb-5">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                        {k.title}
                      </p>
                      <p className="text-3xl font-bold mt-1">
                        {loading ? <span className="text-muted-foreground text-2xl">—</span> : k.value}
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">{k.sub}</p>
                    </div>
                    <div className={`p-2.5 rounded-lg ${k.bg}`}>
                      <Icon className={`h-5 w-5 ${k.color}`} />
                    </div>
                  </div>
                </CardContent>
              </Card>
            </Link>
          );
        })}
      </div>

      {/* Alert banners */}
      {!loading && expiringSoon.length > 0 && (
        <Card className="border-amber-200 bg-amber-50">
          <CardContent className="py-3 px-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-amber-800">
                  {expiringSoon.length} tenant{expiringSoon.length > 1 ? "s" : ""} expiring within 7 days
                </p>
                <div className="flex flex-wrap gap-2 mt-1">
                  {expiringSoon.map((t) => {
                    const d = daysUntil(t.trial_end);
                    return (
                      <span key={t.id} className="text-xs bg-amber-100 text-amber-700 rounded-full px-2 py-0.5 border border-amber-200">
                        {t.name || t.slug} — {d === 0 ? "expires today" : `${d}d left`}
                      </span>
                    );
                  })}
                </div>
              </div>
              <Link href="/dashboard/tenants">
                <Button size="sm" variant="outline" className="border-amber-300 text-amber-700 hover:bg-amber-100 whitespace-nowrap">
                  Manage <ArrowRight className="h-3 w-3 ml-1" />
                </Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      )}

      {!loading && expiredTrials.length > 0 && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="py-3 px-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className="h-5 w-5 text-red-500 shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-semibold text-red-800">
                  {expiredTrials.length} active tenant{expiredTrials.length > 1 ? "s" : ""} with expired trials
                </p>
                <p className="text-xs text-red-600 mt-0.5">
                  {expiredTrials.map((t) => t.name || t.slug).join(", ")}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 lg:grid-cols-5">
        {/* Recent activity feed */}
        <Card className="lg:col-span-3">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base flex items-center gap-2">
                <FileText className="h-4 w-4" />
                Recent Activity
              </CardTitle>
              <Link href="/dashboard/audit">
                <Button variant="ghost" size="sm" className="text-xs text-muted-foreground">
                  View all <ArrowRight className="h-3 w-3 ml-1" />
                </Button>
              </Link>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {loading ? (
              <p className="text-sm text-muted-foreground text-center py-4">Loading...</p>
            ) : recentEvents.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">No recent activity.</p>
            ) : (
              recentEvents.map((e) => {
                const isSuccess = !e.result || e.result === "success";
                const dotColor = ACTION_COLORS[e.action] ?? "bg-slate-400";
                return (
                  <div key={e.id} className="flex items-start gap-3">
                    <div className={`mt-1.5 h-2 w-2 rounded-full shrink-0 ${isSuccess ? dotColor : "bg-red-400"}`} />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm">
                        <span className="font-medium">
                          {ACTION_LABELS[e.action] ?? e.action}
                        </span>
                        {e.tenant_slug && (
                          <span className="text-muted-foreground"> — {e.tenant_slug}</span>
                        )}
                      </p>
                      <p className="text-xs text-muted-foreground">{timeAgo(e.created_at)}</p>
                    </div>
                    {!isSuccess && (
                      <span className="text-xs text-red-500 font-medium shrink-0">failed</span>
                    )}
                  </div>
                );
              })
            )}
          </CardContent>
        </Card>

        {/* Right column: quick links + tenant health */}
        <div className="lg:col-span-2 space-y-4">
          {/* Quick navigation */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <TrendingUp className="h-4 w-4" />
                Quick Navigation
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {[
                { href: "/dashboard/tenants", label: "Tenant Management", icon: Building2, desc: `${activeTenants.length} active` },
                { href: "/dashboard/members", label: "Members", icon: Users, desc: `${totalMembers ?? "—"} total` },
                { href: "/dashboard/usage", label: "Usage & Caps", icon: BarChart3, desc: `${todayUsage ?? "—"} today` },
                { href: "/dashboard/audit", label: "Audit Logs", icon: FileText, desc: "Full history" },
                { href: "/dashboard/revoke", label: "Revoke Access", icon: Shield, desc: "Session control" },
              ].map((l) => {
                const Icon = l.icon;
                return (
                  <Link key={l.href} href={l.href}>
                    <div className="flex items-center justify-between p-2.5 rounded-md hover:bg-slate-50 transition-colors cursor-pointer group">
                      <div className="flex items-center gap-2.5">
                        <Icon className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm font-medium">{l.label}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">{loading ? "—" : l.desc}</span>
                        <ArrowRight className="h-3.5 w-3.5 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                      </div>
                    </div>
                  </Link>
                );
              })}
            </CardContent>
          </Card>

          {/* Tenant health */}
          {!loading && tenants.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4" />
                  Tenant Health
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {tenants.slice(0, 5).map((t) => {
                  const d = daysUntil(t.trial_end);
                  const isExpired = d !== null && d < 0;
                  const isExpiring = d !== null && d >= 0 && d <= 7;
                  const totalDays = t.trial_start && t.trial_end
                    ? Math.ceil((new Date(t.trial_end).getTime() - new Date(t.trial_start).getTime()) / 86400000)
                    : null;
                  const usedDays = t.trial_start
                    ? Math.ceil((Date.now() - new Date(t.trial_start).getTime()) / 86400000)
                    : null;
                  const pct = totalDays && usedDays ? Math.min((usedDays / totalDays) * 100, 100) : null;

                  return (
                    <div key={t.id} className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium truncate max-w-[140px]">{t.name || t.slug}</span>
                        <span className={`text-xs ${isExpired ? "text-red-500 font-medium" : isExpiring ? "text-amber-500 font-medium" : "text-muted-foreground"}`}>
                          {!t.is_active ? "Inactive" : d === null ? "No expiry" : isExpired ? `Expired ${Math.abs(d)}d ago` : d === 0 ? "Expires today" : `${d}d left`}
                        </span>
                      </div>
                      {pct !== null && (
                        <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all ${isExpired ? "bg-red-400" : isExpiring ? "bg-amber-400" : "bg-primary"}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      )}
                    </div>
                  );
                })}
                {tenants.length > 5 && (
                  <Link href="/dashboard/tenants">
                    <p className="text-xs text-primary hover:underline">+{tenants.length - 5} more tenants →</p>
                  </Link>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
