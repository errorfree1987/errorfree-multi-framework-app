"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  ShieldOff, AlertTriangle, RefreshCw, AlertCircle, CheckCircle2,
  Users, Hash, Clock, Shield, ChevronDown, ChevronUp, Zap,
} from "lucide-react";

type TenantRevoke = {
  id: string; slug: string; name: string; is_active: boolean;
  epoch: number; estimated_active_sessions: number;
};
type RevokeHistoryItem = {
  id: string; created_at: string; context?: { old_epoch?: number; new_epoch?: number };
};

function timeAgo(s: string) {
  const diff = Date.now() - new Date(s).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function riskBadge(sessions: number) {
  if (sessions >= 10) return { label: "High Risk", color: "bg-red-100 text-red-700 border-red-200" };
  if (sessions >= 3)  return { label: "Medium", color: "bg-amber-100 text-amber-700 border-amber-200" };
  return { label: "Low", color: "bg-green-100 text-green-700 border-green-200" };
}

// ── Per-tenant revoke card ────────────────────────────────────────────────────
function TenantRevokeCard({
  t,
  onRevoked,
}: {
  t: TenantRevoke;
  onRevoked: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [history, setHistory] = useState<RevokeHistoryItem[]>([]);
  const [histLoading, setHistLoading] = useState(false);
  const [confirmSlug, setConfirmSlug] = useState(false);
  const [revoking, setRevoking] = useState(false);
  const [success, setSuccess] = useState("");
  const [error, setError] = useState("");

  async function loadHistory() {
    if (history.length > 0) return;
    setHistLoading(true);
    try {
      const res = await fetch(`/api/audit?tenant_slug=${t.slug}&action=epoch_revoke&limit=5`);
      const data = await res.json();
      if (Array.isArray(data)) setHistory(data);
    } catch { /* silent */ }
    finally { setHistLoading(false); }
  }

  async function handleRevoke() {
    setConfirmSlug(false);
    setRevoking(true);
    setError(""); setSuccess("");
    try {
      const res = await fetch("/api/revoke", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tenant_slug: t.slug }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.error || "Failed"); return; }
      setSuccess(`Sessions revoked. New epoch: ${data.new_epoch}`);
      setHistory([]); // reset to reload
      onRevoked();
    } catch { setError("Network error"); }
    finally { setRevoking(false); }
  }

  const risk = riskBadge(t.estimated_active_sessions);

  return (
    <Card className={!t.is_active ? "opacity-60" : t.estimated_active_sessions >= 10 ? "border-red-200" : ""}>
      <CardHeader className="pb-2 pt-4">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <div className={`h-9 w-9 rounded-lg flex items-center justify-center text-sm font-bold shrink-0 ${t.is_active ? "bg-primary/10 text-primary" : "bg-slate-100 text-slate-400"}`}>
              {t.name.charAt(0).toUpperCase()}
            </div>
            <div className="min-w-0">
              <CardTitle className="text-base flex items-center gap-2 flex-wrap">
                {t.name}
                {!t.is_active && <span className="text-xs font-normal text-muted-foreground bg-slate-100 rounded px-2 py-0.5">Inactive</span>}
                <span className={`text-xs font-medium rounded-full px-2 py-0.5 border ${risk.color}`}>{risk.label}</span>
              </CardTitle>
              <CardDescription className="text-xs">{t.slug}</CardDescription>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {confirmSlug ? (
              <>
                <span className="text-sm text-destructive font-medium">Confirm?</span>
                <Button size="sm" variant="destructive" onClick={handleRevoke} disabled={revoking}>
                  {revoking ? "Revoking..." : "Yes, Revoke"}
                </Button>
                <Button size="sm" variant="outline" onClick={() => setConfirmSlug(false)}>Cancel</Button>
              </>
            ) : (
              <Button size="sm" variant="destructive" onClick={() => setConfirmSlug(true)} disabled={revoking || !t.is_active}>
                <ShieldOff className="h-4 w-4 mr-1" />Revoke All Sessions
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="pb-4 space-y-3">
        {/* Stats row */}
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="bg-slate-50 rounded-md p-2">
            <p className="text-lg font-bold font-mono">{t.epoch}</p>
            <p className="text-xs text-muted-foreground flex items-center justify-center gap-1"><Hash className="h-3 w-3" />Current Epoch</p>
          </div>
          <div className={`rounded-md p-2 ${t.estimated_active_sessions > 0 ? "bg-amber-50" : "bg-slate-50"}`}>
            <p className={`text-lg font-bold ${t.estimated_active_sessions > 0 ? "text-amber-700" : ""}`}>{t.estimated_active_sessions}</p>
            <p className="text-xs text-muted-foreground flex items-center justify-center gap-1"><Users className="h-3 w-3" />Est. Sessions</p>
          </div>
          <div className="bg-slate-50 rounded-md p-2">
            <p className="text-lg font-bold">{t.estimated_active_sessions > 0 ? "Active" : "Clear"}</p>
            <p className="text-xs text-muted-foreground flex items-center justify-center gap-1"><Shield className="h-3 w-3" />State</p>
          </div>
        </div>

        {/* Success / error */}
        {success && <div className="flex items-center gap-2 text-green-600 text-sm"><CheckCircle2 className="h-4 w-4" />{success}</div>}
        {error && <div className="flex items-center gap-2 text-destructive text-sm"><AlertTriangle className="h-4 w-4" />{error}</div>}

        {/* Revocation history toggle */}
        <button
          className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
          onClick={() => { setExpanded(!expanded); if (!expanded) loadHistory(); }}
        >
          <Clock className="h-3.5 w-3.5" />Revocation History
          {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        </button>

        {expanded && (
          <div className="border rounded-md overflow-hidden">
            {histLoading ? (
              <p className="text-xs text-muted-foreground text-center py-3">Loading...</p>
            ) : history.length === 0 ? (
              <p className="text-xs text-muted-foreground text-center py-3">No revocation history found.</p>
            ) : (
              <div className="divide-y">
                {history.map((h) => (
                  <div key={h.id} className="flex items-center justify-between px-3 py-2 text-xs hover:bg-slate-50">
                    <div className="flex items-center gap-2">
                      <ShieldOff className="h-3.5 w-3.5 text-red-500" />
                      <span>
                        Epoch {h.context?.old_epoch ?? "?"} → {h.context?.new_epoch ?? "?"}
                      </span>
                    </div>
                    <span className="text-muted-foreground">{timeAgo(h.created_at)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function RevokePage() {
  const [tenants, setTenants] = useState<TenantRevoke[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [emergencyConfirm, setEmergencyConfirm] = useState(false);
  const [emergencyConfirm2, setEmergencyConfirm2] = useState(false);
  const [emergencyRunning, setEmergencyRunning] = useState(false);
  const [emergencyDone, setEmergencyDone] = useState(0);

  const load = useCallback(async () => {
    setLoading(true); setError("");
    try {
      const res = await fetch("/api/revoke");
      const data = await res.json();
      if (!res.ok) { setError(data.error || "Failed"); return; }
      setTenants(Array.isArray(data) ? data : []);
    } catch { setError("Network error"); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  async function handleEmergencyRevokeAll() {
    setEmergencyRunning(true);
    setEmergencyDone(0);
    const active = tenants.filter((t) => t.is_active);
    for (const t of active) {
      await fetch("/api/revoke", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tenant_slug: t.slug }),
      }).catch(() => {});
      setEmergencyDone((n) => n + 1);
      await new Promise((r) => setTimeout(r, 400)); // pace requests
    }
    setEmergencyRunning(false);
    setEmergencyConfirm(false);
    setEmergencyConfirm2(false);
    await load();
  }

  const totalSessions = tenants.reduce((s, t) => s + t.estimated_active_sessions, 0);
  const highRisk = tenants.filter((t) => t.estimated_active_sessions >= 10);
  const activeTenants = tenants.filter((t) => t.is_active);

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Revoke Access</h1>
          <p className="text-muted-foreground mt-1">
            Increment session epoch to immediately invalidate all active tokens. Includes revocation history and risk assessment.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={load} disabled={loading}>
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
        </Button>
      </div>

      {/* How it works */}
      <Card className="border-slate-200 bg-slate-50">
        <CardContent className="py-3 px-4">
          <p className="text-sm text-slate-600">
            <span className="font-semibold">How it works:</span> Each tenant has a session epoch. Revoking increments it by 1 — any token with the old epoch is immediately rejected on next request, without requiring a password reset. Rate limited to 1 revoke per tenant per 30 seconds. Session estimates are based on unique users active in the last 24h.
          </p>
        </CardContent>
      </Card>

      {/* Summary */}
      <div className="grid gap-3 sm:grid-cols-4">
        {[
          { label: "Active Tenants", value: activeTenants.length, icon: Shield, color: "text-blue-600", bg: "bg-blue-50" },
          { label: "Est. Active Sessions", value: totalSessions, icon: Users, color: totalSessions > 0 ? "text-amber-600" : "text-green-600", bg: totalSessions > 0 ? "bg-amber-50" : "bg-green-50" },
          { label: "High Risk Tenants", value: highRisk.length, icon: AlertTriangle, color: highRisk.length > 0 ? "text-red-600" : "text-slate-400", bg: highRisk.length > 0 ? "bg-red-50" : "bg-slate-50" },
          { label: "Rate Limit", value: "30s", icon: Clock, color: "text-slate-500", bg: "bg-slate-50" },
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

      {/* Emergency revoke all */}
      <Card className={`border-2 ${emergencyConfirm ? "border-red-400 bg-red-50" : "border-dashed border-slate-300"}`}>
        <CardContent className="py-4 px-4">
          {!emergencyConfirm ? (
            <div className="flex items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-red-100 rounded-lg"><Zap className="h-5 w-5 text-red-600" /></div>
                <div>
                  <p className="text-sm font-semibold">Emergency: Revoke All Tenants</p>
                  <p className="text-xs text-muted-foreground">Immediately invalidates ALL sessions across ALL active tenants. Use only in security emergencies.</p>
                </div>
              </div>
              <Button variant="destructive" size="sm" onClick={() => setEmergencyConfirm(true)} disabled={emergencyRunning}>
                <Zap className="h-4 w-4 mr-1" />Emergency Revoke All
              </Button>
            </div>
          ) : !emergencyConfirm2 ? (
            <div className="space-y-3">
              <p className="text-sm font-semibold text-red-700 flex items-center gap-2"><AlertTriangle className="h-4 w-4" />This will revoke sessions for ALL {activeTenants.length} active tenants ({totalSessions} estimated active sessions). Are you sure?</p>
              <div className="flex gap-2">
                <Button variant="destructive" size="sm" onClick={() => setEmergencyConfirm2(true)}>Yes, I understand — continue</Button>
                <Button variant="outline" size="sm" onClick={() => setEmergencyConfirm(false)}>Cancel</Button>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <p className="text-sm font-semibold text-red-700 flex items-center gap-2"><AlertTriangle className="h-4 w-4" />FINAL CONFIRMATION: Revoke all {activeTenants.length} tenant sessions now?</p>
              {emergencyRunning && (
                <div className="space-y-1">
                  <div className="h-2 w-full bg-red-200 rounded-full overflow-hidden">
                    <div className="h-full bg-red-500 transition-all" style={{ width: `${(emergencyDone / activeTenants.length) * 100}%` }} />
                  </div>
                  <p className="text-xs text-red-600">Revoking {emergencyDone} / {activeTenants.length} tenants...</p>
                </div>
              )}
              <div className="flex gap-2">
                <Button variant="destructive" size="sm" onClick={handleEmergencyRevokeAll} disabled={emergencyRunning}>
                  {emergencyRunning ? "Revoking..." : "CONFIRM — Revoke Everything"}
                </Button>
                <Button variant="outline" size="sm" onClick={() => { setEmergencyConfirm(false); setEmergencyConfirm2(false); }} disabled={emergencyRunning}>Cancel</Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {error && <div className="flex items-center gap-2 text-destructive text-sm"><AlertCircle className="h-4 w-4" />{error}</div>}

      {/* Tenant cards — high risk first */}
      <div className="space-y-3">
        {[...tenants]
          .sort((a, b) => b.estimated_active_sessions - a.estimated_active_sessions)
          .map((t) => (
            <TenantRevokeCard key={t.id} t={t} onRevoked={load} />
          ))}
        {tenants.length === 0 && !loading && !error && (
          <Card><CardContent className="py-12 text-center text-muted-foreground">No tenants found.</CardContent></Card>
        )}
      </div>
    </div>
  );
}
