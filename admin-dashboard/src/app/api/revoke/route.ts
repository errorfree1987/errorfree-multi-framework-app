import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { supabaseFetch } from "@/lib/supabase";

async function requireAuth() {
  const cookieStore = await cookies();
  const session = cookieStore.get("admin_session");
  if (!session?.value || !session.value.startsWith("admin_")) return null;
  return true;
}

// In-memory rate limit: one revoke per tenant per 30 seconds
const rateLimit = new Map<string, number>();
const RATE_LIMIT_MS = 30_000;

export async function GET() {
  const auth = await requireAuth();
  if (!auth) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  try {
    const [tenantsRes, epochRes] = await Promise.all([
      supabaseFetch("/tenants?select=id,slug,name,is_active&order=name.asc"),
      supabaseFetch("/tenant_session_epoch?select=tenant,epoch"),
    ]);

    const tenants: { id: string; slug: string; name?: string; is_active: boolean }[] =
      tenantsRes.ok ? await tenantsRes.json() : [];
    const epochs: { tenant: string; epoch: number }[] =
      epochRes.ok ? await epochRes.json() : [];

    const epochMap = new Map(epochs.map((e) => [e.tenant, e.epoch]));

    // Estimate affected sessions: unique emails with sso_verify success in last 24h per tenant
    const since = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
    const recentRes = await supabaseFetch(
      `/audit_events?action=eq.sso_verify&result=eq.success&created_at=gte.${since}&select=tenant_slug,email`
    );
    const recentEvents: { tenant_slug: string; email: string }[] =
      recentRes.ok ? await recentRes.json() : [];

    // Count unique emails per tenant
    const sessionMap = new Map<string, Set<string>>();
    for (const e of recentEvents) {
      if (!sessionMap.has(e.tenant_slug)) sessionMap.set(e.tenant_slug, new Set());
      if (e.email) sessionMap.get(e.tenant_slug)!.add(e.email);
    }

    const result = tenants.map((t) => ({
      id: t.id,
      slug: t.slug,
      name: t.name ?? t.slug,
      is_active: t.is_active,
      epoch: epochMap.get(t.slug) ?? 0,
      estimated_active_sessions: sessionMap.get(t.slug)?.size ?? 0,
    }));

    return NextResponse.json(result);
  } catch (e) {
    return NextResponse.json({ error: "Failed", details: String(e) }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  const auth = await requireAuth();
  if (!auth) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { tenant_slug } = await req.json();
  if (!tenant_slug) return NextResponse.json({ error: "tenant_slug required" }, { status: 400 });

  // Rate limit check
  const lastRevoke = rateLimit.get(tenant_slug) ?? 0;
  const elapsed = Date.now() - lastRevoke;
  if (elapsed < RATE_LIMIT_MS) {
    const waitSec = Math.ceil((RATE_LIMIT_MS - elapsed) / 1000);
    return NextResponse.json(
      { error: `Rate limited. Please wait ${waitSec}s before revoking again.` },
      { status: 429 }
    );
  }

  try {
    // Get current epoch
    const epochRes = await supabaseFetch(
      `/tenant_session_epoch?tenant=eq.${encodeURIComponent(tenant_slug)}&select=epoch`
    );
    if (!epochRes.ok) return NextResponse.json({ error: "Failed to fetch epoch" }, { status: 502 });
    const epochData: { epoch: number }[] = await epochRes.json();

    if (!epochData || epochData.length === 0) {
      return NextResponse.json({ error: `No session epoch found for tenant '${tenant_slug}'` }, { status: 404 });
    }

    const newEpoch = (epochData[0].epoch ?? 0) + 1;

    // Increment epoch
    const patchRes = await supabaseFetch(
      `/tenant_session_epoch?tenant=eq.${encodeURIComponent(tenant_slug)}`,
      {
        method: "PATCH",
        headers: { Prefer: "return=minimal" },
        body: JSON.stringify({ epoch: newEpoch }),
      }
    );

    if (!patchRes.ok) {
      const text = await patchRes.text();
      return NextResponse.json({ error: `Failed to revoke: ${patchRes.status}`, details: text }, { status: 502 });
    }

    rateLimit.set(tenant_slug, Date.now());

    // Log audit event
    await supabaseFetch("/audit_events", {
      method: "POST",
      body: JSON.stringify({
        action: "epoch_revoke",
        tenant_slug,
        result: "success",
        actor_email: "admin",
        context: { old_epoch: epochData[0].epoch, new_epoch: newEpoch },
      }),
    });

    return NextResponse.json({ success: true, new_epoch: newEpoch });
  } catch (e) {
    return NextResponse.json({ error: "Failed to revoke", details: String(e) }, { status: 500 });
  }
}
