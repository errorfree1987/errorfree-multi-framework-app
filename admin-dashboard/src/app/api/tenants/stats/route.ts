import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { supabaseFetch } from "@/lib/supabase";

async function requireAuth() {
  const cookieStore = await cookies();
  const session = cookieStore.get("admin_session");
  if (!session?.value || !session.value.startsWith("admin_")) {
    return null;
  }
  return true;
}

export async function GET(request: NextRequest) {
  const auth = await requireAuth();
  if (!auth) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { searchParams } = new URL(request.url);
  const tenantId = searchParams.get("tenant_id");
  const tenantSlug = searchParams.get("tenant_slug");

  if (!tenantId || !tenantSlug) {
    return NextResponse.json(
      { error: "tenant_id and tenant_slug required" },
      { status: 400 }
    );
  }

  try {
    const [membersRes, usageRes, epochRes, capsRes] = await Promise.all([
      supabaseFetch(
        `/tenant_members?tenant_id=eq.${tenantId}&select=id&is_active=eq.true`
      ),
      supabaseFetch(
        `/tenant_usage_events?tenant_id=eq.${tenantId}&created_at=gte.${new Date().toISOString().slice(0, 10)}T00:00:00Z&select=id`
      ),
      supabaseFetch(
        `/tenant_session_epoch?tenant=eq.${tenantSlug}&select=epoch`
      ),
      supabaseFetch(
        `/tenant_usage_caps?tenant_id=eq.${tenantId}&select=id,daily_review_cap,daily_download_cap`
      ),
    ]);

    const members = membersRes.ok ? await membersRes.json() : [];
    const usage = usageRes.ok ? await usageRes.json() : [];
    const epochData = epochRes.ok ? await epochRes.json() : [];
    const capsData = capsRes.ok ? await capsRes.json() : [];

    const caps = capsData[0] || {};
    return NextResponse.json({
      memberCount: members.length,
      todayUsage: usage.length,
      epoch: epochData[0]?.epoch ?? 0,
      daily_review_cap: caps.daily_review_cap ?? null,
      daily_download_cap: caps.daily_download_cap ?? null,
      caps_id: caps.id ?? null,
    });
  } catch (e) {
    return NextResponse.json(
      { error: "Failed to fetch stats", details: String(e) },
      { status: 500 }
    );
  }
}
