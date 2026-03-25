import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { supabaseFetch } from "@/lib/supabase";

async function requireAuth() {
  const cookieStore = await cookies();
  const session = cookieStore.get("admin_session");
  if (!session?.value || !session.value.startsWith("admin_")) return null;
  return true;
}

export async function GET(req: NextRequest) {
  const auth = await requireAuth();
  if (!auth) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { searchParams } = new URL(req.url);
  const tenant_slug = searchParams.get("tenant_slug") || "";
  const action = searchParams.get("action") || "";
  const from = searchParams.get("from") || "";
  const to = searchParams.get("to") || "";
  const limit = Math.min(parseInt(searchParams.get("limit") || "200"), 500);

  let query = `/audit_events?select=*&order=created_at.desc&limit=${limit}`;
  if (tenant_slug) query += `&tenant_slug=eq.${encodeURIComponent(tenant_slug)}`;
  if (action) query += `&action=eq.${encodeURIComponent(action)}`;
  if (from) query += `&created_at=gte.${encodeURIComponent(from)}`;
  if (to) {
    // to date: include full day
    const toDate = new Date(to);
    toDate.setDate(toDate.getDate() + 1);
    query += `&created_at=lt.${encodeURIComponent(toDate.toISOString())}`;
  }

  const res = await supabaseFetch(query);
  if (!res.ok) {
    return NextResponse.json({ error: "Failed to fetch audit logs" }, { status: 502 });
  }
  const data = await res.json();
  return NextResponse.json(data);
}
