import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { supabaseFetch } from "@/lib/supabase";

async function requireAuth() {
  const cookieStore = await cookies();
  const session = cookieStore.get("admin_session");
  if (!session?.value || !session.value.startsWith("admin_")) return null;
  return true;
}

function buildDays(n: number): string[] {
  const days: string[] = [];
  for (let i = n - 1; i >= 0; i--) {
    const d = new Date(); d.setDate(d.getDate() - i);
    days.push(d.toISOString().slice(0, 10));
  }
  return days;
}

export async function GET(req: NextRequest) {
  const auth = await requireAuth();
  if (!auth) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { searchParams } = new URL(req.url);
  const period = Math.min(parseInt(searchParams.get("period") ?? "7"), 30);

  try {
    const days = buildDays(period);
    const from = days[0] + "T00:00:00Z";

    const [tenantsRes, capsRes, usageRes, memberUsageRes] = await Promise.all([
      supabaseFetch("/tenants?select=id,slug,name&is_active=eq.true&order=name.asc"),
      supabaseFetch("/tenant_usage_caps?select=tenant_id,daily_review_cap,daily_download_cap"),
      supabaseFetch(`/tenant_usage_events?select=tenant_id,created_at&created_at=gte.${from}&order=created_at.asc&limit=10000`),
      // Top users: member email + usage count
      supabaseFetch(`/tenant_usage_events?select=tenant_id,member_email:email&created_at=gte.${from}&limit=10000`)
        .catch(() => ({ ok: false } as Response)),
    ]);

    if (!tenantsRes.ok) return NextResponse.json({ error: "Failed to fetch tenants" }, { status: 502 });

    const tenants: { id: string; slug: string; name?: string }[] = await tenantsRes.json();
    const caps: { tenant_id: string; daily_review_cap?: number | null; daily_download_cap?: number | null }[] =
      capsRes.ok ? await capsRes.json() : [];
    const usageEvents: { tenant_id: string; created_at: string }[] =
      usageRes.ok ? await usageRes.json() : [];
    const memberEvents: { tenant_id: string; member_email?: string }[] =
      memberUsageRes.ok ? await memberUsageRes.json() : [];

    const capsMap = new Map(caps.map((c) => [c.tenant_id, c]));

    // Usage by tenant + date
    const usageMap = new Map<string, Map<string, number>>();
    for (const e of usageEvents) {
      const dk = e.created_at.slice(0, 10);
      if (!usageMap.has(e.tenant_id)) usageMap.set(e.tenant_id, new Map());
      const m = usageMap.get(e.tenant_id)!;
      m.set(dk, (m.get(dk) ?? 0) + 1);
    }

    // Top users per tenant
    const memberMap = new Map<string, Map<string, number>>();
    for (const e of memberEvents) {
      if (!e.member_email) continue;
      if (!memberMap.has(e.tenant_id)) memberMap.set(e.tenant_id, new Map());
      const m = memberMap.get(e.tenant_id)!;
      m.set(e.member_email, (m.get(e.member_email) ?? 0) + 1);
    }

    const today = days[days.length - 1];

    const result = tenants.map((t) => {
      const cap = capsMap.get(t.id);
      const dailyCap = cap?.daily_review_cap ?? null;
      const tenantUsage = usageMap.get(t.id) ?? new Map<string, number>();
      const trend = days.map((d) => ({ date: d, count: tenantUsage.get(d) ?? 0 }));
      const todayCount = tenantUsage.get(today) ?? 0;
      const periodTotal = trend.reduce((s, d) => s + d.count, 0);
      const peakDay = trend.reduce((best, d) => d.count > best.count ? d : best, { date: "", count: 0 });
      const capHitDays = dailyCap && dailyCap > 0 ? trend.filter((d) => d.count >= dailyCap).length : 0;

      const overCap = dailyCap !== null && dailyCap > 0 && todayCount >= dailyCap;
      const nearCap = dailyCap !== null && dailyCap > 0 && !overCap && todayCount >= dailyCap * 0.8;

      // Top 5 users
      const userMap = memberMap.get(t.id) ?? new Map<string, number>();
      const topUsers = [...userMap.entries()]
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([email, count]) => ({ email, count }));

      return {
        tenant_id: t.id, slug: t.slug, name: t.name ?? t.slug,
        daily_review_cap: dailyCap,
        daily_download_cap: cap?.daily_download_cap ?? null,
        today_count: todayCount, period_total: periodTotal,
        peak_day: peakDay.count > 0 ? peakDay : null,
        cap_hit_days: capHitDays,
        trend, overCap, nearCap, top_users: topUsers,
      };
    });

    return NextResponse.json({ days, period, tenants: result });
  } catch (e) {
    return NextResponse.json({ error: "Failed to fetch usage", details: String(e) }, { status: 500 });
  }
}
