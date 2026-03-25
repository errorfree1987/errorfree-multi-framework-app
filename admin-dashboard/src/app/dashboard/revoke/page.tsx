"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ShieldOff, AlertTriangle, RefreshCw, AlertCircle, CheckCircle2, Users, Hash } from "lucide-react";

type TenantRevoke = {
  id: string;
  slug: string;
  name: string;
  is_active: boolean;
  epoch: number;
  estimated_active_sessions: number;
};

export default function RevokePage() {
  const [tenants, setTenants] = useState<TenantRevoke[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [confirmSlug, setConfirmSlug] = useState<string | null>(null);
  const [revoking, setRevoking] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<Record<string, string>>({});
  const [errorMsg, setErrorMsg] = useState<Record<string, string>>({});

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/revoke");
      const data = await res.json();
      if (!res.ok) { setError(data.error || "Failed to load"); return; }
      setTenants(Array.isArray(data) ? data : []);
    } catch {
      setError("Network error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  async function handleRevoke(slug: string) {
    setConfirmSlug(null);
    setRevoking(slug);
    setErrorMsg((m) => ({ ...m, [slug]: "" }));
    setSuccessMsg((m) => ({ ...m, [slug]: "" }));
    try {
      const res = await fetch("/api/revoke", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tenant_slug: slug }),
      });
      const data = await res.json();
      if (!res.ok) {
        setErrorMsg((m) => ({ ...m, [slug]: data.error || "Failed to revoke" }));
      } else {
        setSuccessMsg((m) => ({ ...m, [slug]: `All sessions revoked. New epoch: ${data.new_epoch}` }));
        // Refresh tenant list to show new epoch
        await load();
      }
    } catch {
      setErrorMsg((m) => ({ ...m, [slug]: "Network error" }));
    } finally {
      setRevoking(null);
    }
  }

  const totalActiveSessions = tenants.reduce((s, t) => s + t.estimated_active_sessions, 0);

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Revoke Access</h1>
          <p className="text-muted-foreground mt-1">
            Increment session epoch to immediately invalidate all active tokens for a tenant.
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
            <span className="font-semibold">How it works:</span> Each tenant has a session epoch number. When you revoke, the epoch increments by 1. Any user whose token carries the old epoch will be immediately rejected on their next request — no password reset required. Rate limited to 1 revoke per tenant per 30 seconds.
          </p>
        </CardContent>
      </Card>

      {/* Summary */}
      <div className="grid gap-4 sm:grid-cols-2">
        <Card>
          <CardContent className="pt-4 pb-4 flex items-center gap-3">
            <Users className="h-6 w-6 text-muted-foreground" />
            <div>
              <p className="text-xs text-muted-foreground">Est. Active Sessions (24h)</p>
              <p className="text-2xl font-bold">{totalActiveSessions}</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 pb-4 flex items-center gap-3">
            <Hash className="h-6 w-6 text-muted-foreground" />
            <div>
              <p className="text-xs text-muted-foreground">Tenants with Active Sessions</p>
              <p className="text-2xl font-bold">
                {tenants.filter((t) => t.estimated_active_sessions > 0).length}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-destructive text-sm">
          <AlertCircle className="h-4 w-4" />{error}
        </div>
      )}

      {/* Tenant list */}
      <div className="space-y-3">
        {tenants.map((t) => (
          <Card key={t.id} className={!t.is_active ? "opacity-60" : ""}>
            <CardHeader className="pb-2 pt-4">
              <div className="flex items-start justify-between gap-4">
                <div className="space-y-0.5">
                  <CardTitle className="text-base flex items-center gap-2">
                    {t.name}
                    {!t.is_active && (
                      <span className="text-xs font-normal text-muted-foreground bg-slate-100 rounded px-2 py-0.5">
                        Inactive
                      </span>
                    )}
                  </CardTitle>
                  <CardDescription>{t.slug}</CardDescription>
                </div>

                {/* Revoke button or confirm dialog */}
                {confirmSlug === t.slug ? (
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span className="text-sm text-destructive font-medium">Confirm revoke?</span>
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => handleRevoke(t.slug)}
                      disabled={revoking === t.slug}
                    >
                      Yes, Revoke
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setConfirmSlug(null)}
                    >
                      Cancel
                    </Button>
                  </div>
                ) : (
                  <Button
                    size="sm"
                    variant="destructive"
                    onClick={() => setConfirmSlug(t.slug)}
                    disabled={revoking === t.slug || !t.is_active}
                  >
                    <ShieldOff className="h-4 w-4 mr-1" />
                    Revoke All Sessions
                  </Button>
                )}
              </div>
            </CardHeader>

            <CardContent className="pb-4">
              <div className="flex flex-wrap gap-4 text-sm">
                <div className="flex items-center gap-1.5">
                  <Hash className="h-4 w-4 text-muted-foreground" />
                  <span className="text-muted-foreground">Current Epoch:</span>
                  <span className="font-mono font-semibold">{t.epoch}</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <Users className="h-4 w-4 text-muted-foreground" />
                  <span className="text-muted-foreground">Est. Active Sessions (24h):</span>
                  <span className={`font-semibold ${t.estimated_active_sessions > 0 ? "text-amber-600" : "text-muted-foreground"}`}>
                    {t.estimated_active_sessions}
                  </span>
                  {t.estimated_active_sessions > 0 && (
                    <span className="text-xs text-muted-foreground">(will be revoked)</span>
                  )}
                </div>
              </div>

              {/* Success / error messages */}
              {successMsg[t.slug] && (
                <div className="mt-3 flex items-center gap-2 text-green-600 text-sm">
                  <CheckCircle2 className="h-4 w-4" />
                  {successMsg[t.slug]}
                </div>
              )}
              {errorMsg[t.slug] && (
                <div className="mt-3 flex items-center gap-2 text-destructive text-sm">
                  <AlertTriangle className="h-4 w-4" />
                  {errorMsg[t.slug]}
                </div>
              )}
            </CardContent>
          </Card>
        ))}

        {tenants.length === 0 && !loading && !error && (
          <Card>
            <CardContent className="py-12 text-center text-muted-foreground">
              No tenants found.
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
