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

export async function GET() {
  const auth = await requireAuth();
  if (!auth) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const res = await supabaseFetch(
      "/tenants?select=id,slug,name,display_name,status,trial_start,trial_end,is_active,created_at&order=created_at.desc"
    );
    if (!res.ok) {
      const text = await res.text();
      return NextResponse.json(
        { error: `Supabase error: ${res.status}`, details: text },
        { status: 502 }
      );
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch (e) {
    return NextResponse.json(
      { error: "Failed to fetch tenants", details: String(e) },
      { status: 500 }
    );
  }
}
