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

/**
 * GET /api/members
 * Query: tenant_slug (optional), search (optional, fuzzy search on email/display_name)
 */
export async function GET(req: NextRequest) {
  const auth = await requireAuth();
  if (!auth) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { searchParams } = new URL(req.url);
  const tenantSlug = searchParams.get("tenant_slug")?.trim() || null;
  const search = searchParams.get("search")?.trim() || null;

  try {
    let url = "/tenant_members?select=id,tenant_id,email,display_name,phone,role,is_active,created_at&order=created_at.desc";
    if (tenantSlug) {
      const tenantRes = await supabaseFetch(
        `/tenants?slug=eq.${encodeURIComponent(tenantSlug)}&select=id`
      );
      if (!tenantRes.ok) {
        return NextResponse.json(
          { error: "Failed to find tenant" },
          { status: 400 }
        );
      }
      const tenants = await tenantRes.json();
      if (!Array.isArray(tenants) || tenants.length === 0) {
        return NextResponse.json([]);
      }
      const tenantId = tenants[0].id;
      url = `/tenant_members?tenant_id=eq.${tenantId}&select=id,tenant_id,email,display_name,phone,role,is_active,created_at&order=created_at.desc`;
    }

    const res = await supabaseFetch(url);
    if (!res.ok) {
      const text = await res.text();
      return NextResponse.json(
        { error: `Supabase error: ${res.status}`, details: text },
        { status: 502 }
      );
    }
    let members: Array<Record<string, unknown>> = await res.json();

    // Add tenant_slug to each member
    const tenantIds = [...new Set(members.map((m) => m.tenant_id as string))];
    const tenantMap: Record<string, string> = {};
    for (const tid of tenantIds) {
      const tRes = await supabaseFetch(
        `/tenants?id=eq.${tid}&select=slug`
      );
      if (tRes.ok) {
        const arr = await tRes.json();
        if (Array.isArray(arr) && arr[0]) tenantMap[tid] = arr[0].slug as string;
      }
    }
    members = members.map((m) => ({
      ...m,
      tenant_slug: tenantMap[(m.tenant_id as string) || ""] || "unknown",
    }));

    // Client-side fuzzy search (Supabase ilike on multiple columns is complex)
    if (search) {
      const q = search.toLowerCase();
      members = members.filter((m) => {
        const email = (m.email as string) || "";
        const displayName = (m.display_name as string) || "";
        const phone = (m.phone as string) || "";
        return (
          email.toLowerCase().includes(q) ||
          displayName.toLowerCase().includes(q) ||
          phone.includes(q)
        );
      });
    }

    return NextResponse.json(members);
  } catch (e) {
    return NextResponse.json(
      { error: "Failed to fetch members", details: String(e) },
      { status: 500 }
    );
  }
}
