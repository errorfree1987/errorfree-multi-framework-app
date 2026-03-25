import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { supabaseFetch } from "@/lib/supabase";

async function requireAuth() {
  const cookieStore = await cookies();
  const session = cookieStore.get("admin_session");
  if (!session?.value || !session.value.startsWith("admin_")) return null;
  return true;
}

function last7Days(): string[] {
  const days: string[] = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    days.push(d.toISOString().slice(0, 10));
  }
  return days;
}

export async function GET() {
  const auth = await requireAuth();
  if (!auth) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  try {
    const days = last7Days();
    const from = days[0] + "T00:00:00Z";

    const [tenantsRes, capsRes, usageRes] = await Promise.all([
      supabaseFetch("/tenants?select=id,slug,name&is_active=eq.true&order=name.asc"),
      supabaseFetch("/tenant_usage_caps?select=tenant_id,daily_review_cap,daily_download_cap"),
      supabaseFetch(
        `/tenant_usage_events?select=tenant_id,created_at&created_at=gte.${from}&order=created_at.asc&limit=5000`
      ),
    ]);

    if (!tenantsRes.ok) return NextResponse.json({ error: "Failed to fetch tenants" }, { status: 502 });

    const tenants: { id: string; slug: string; name?: string }[] = await tenantsRes.json();
    const caps: { tenant_id: string; daily_review_cap?: number | null; daily_download_cap?: number | null }[] =
      capsRes.ok ? await capsRes.json() : [];
    const usageEvents: { tenant_id: string; created_at: string }[] =
      usageRes.ok ? await usageRes.json() : [];

    // Build caps map
    const capsMap = new Map(caps.map((c) => [c.tenant_id, c]));

    // Group usage by tenant_id + date
    const usageMap = new Map<string, Map<string, number>>();
    for (const e of usageEvents) {
      const dateKey = e.created_at.slice(0, 10);
      if (!usageMap.has(e.tenant_id)) usageMap.set(e.tenant_id, new Map());
      const m = usageMap.get(e.tenant_id)!;
      m.set(dateKey, (m.get(dateKey) ?? 0) + 1);
    }

    const result = tenants.map((t) => {
      const cap = capsMap.get(t.id);
      const dailyCap = cap?.daily_review_cap ?? null;
      const tenantUsage = usageMap.get(t.id) ?? new Map<string, number>();
      const trend = days.map((d) => ({ date: d, count: tenantUsage.get(d) ?? 0 }));
      const todayCount = tenantUsage.get(days[6]) ?? 0;
      const overCap = dailyCap !== null && dailyCap > 0 && todayCount >= dailyCap;
      const nearCap = dailyCap !== null && dailyCap > 0 && !overCap && todayCount >= dailyCap * 0.8;

      return {
        tenant_id: t.id,
        slug: t.slug,
        name: t.name ?? t.slug,
        daily_review_cap: dailyCap,
        daily_download_cap: cap?.daily_download_cap ?? null,
        today_count: todayCount,
        trend,
        overCap,
        nearCap,
      };
    });

    return NextResponse.json({ days, tenants: result });
  } catch (e) {
    return NextResponse.json({ error: "Failed to fetch usage", details: String(e) }, { status: 500 });
  }
}
