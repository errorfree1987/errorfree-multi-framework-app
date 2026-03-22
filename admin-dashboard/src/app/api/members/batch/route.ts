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

type MemberRow = { email: string; phone?: string; display_name?: string };

/**
 * POST /api/members/batch
 * Body: { tenant_slug: string, members: MemberRow[], role?: string }
 */
export async function POST(req: NextRequest) {
  const auth = await requireAuth();
  if (!auth) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const body = await req.json();
    const { tenant_slug, members, role = "user" } = body;
    if (!tenant_slug || !Array.isArray(members) || members.length === 0) {
      return NextResponse.json(
        { error: "tenant_slug and members array required" },
        { status: 400 }
      );
    }

    const tenantRes = await supabaseFetch(
      `/tenants?slug=eq.${encodeURIComponent(tenant_slug)}&select=id`
    );
    if (!tenantRes.ok) {
      return NextResponse.json(
        { error: "Failed to fetch tenant" },
        { status: 400 }
      );
    }
    const tenants = await tenantRes.json();
    if (!Array.isArray(tenants) || tenants.length === 0) {
      return NextResponse.json(
        { error: `Tenant '${tenant_slug}' not found` },
        { status: 404 }
      );
    }
    const tenantId = tenants[0].id;

    const normalized: MemberRow[] = members
      .map((r: string | MemberRow) => {
        if (typeof r === "string") {
          return { email: r.trim(), phone: undefined, display_name: undefined };
        }
        const email = (r.email || "").trim();
        return {
          email,
          phone: (r.phone || "").trim() || undefined,
          display_name: (r.display_name || "").trim() || undefined,
        };
      })
      .filter((r) => r.email && r.email.includes("@"));

    if (normalized.length === 0) {
      return NextResponse.json(
        { error: "No valid emails (must contain @)" },
        { status: 400 }
      );
    }

    const existingRes = await supabaseFetch(
      "/tenant_members?select=email"
    );
    const existingEmails = new Set<string>();
    if (existingRes.ok) {
      const arr = await existingRes.json();
      (arr || []).forEach((m: { email?: string }) => {
        if (m.email) existingEmails.add(m.email.toLowerCase());
      });
    }

    const toAdd = normalized.filter(
      (r) => !existingEmails.has(r.email.toLowerCase())
    );
    const duplicates = normalized
      .filter((r) => existingEmails.has(r.email.toLowerCase()))
      .map((r) => r.email);

    if (toAdd.length === 0) {
      return NextResponse.json({
        success: false,
        error: "All emails already exist in the system",
        duplicates,
      }, { status: 400 });
    }

    const payload = toAdd.map((r) => ({
      tenant_id: tenantId,
      email: r.email,
      phone: r.phone || null,
      display_name: r.display_name || null,
      role: role === "tenant_admin" ? "tenant_admin" : "user",
      is_active: true,
    }));

    const insertRes = await supabaseFetch("/tenant_members", {
      method: "POST",
      headers: { Prefer: "return=representation" },
      body: JSON.stringify(payload),
    });

    if (!insertRes.ok) {
      const text = await insertRes.text();
      return NextResponse.json(
        { error: `Failed to add members: ${insertRes.status}`, details: text },
        { status: 502 }
      );
    }

    const added = await insertRes.json();
    const addedCount = Array.isArray(added) ? added.length : 1;

    await supabaseFetch("/audit_events", {
      method: "POST",
      body: JSON.stringify({
        action: "members_batch_added",
        tenant_slug: tenant_slug,
        result: "success",
        actor_email: "admin",
        context: {
          count: addedCount,
          emails: toAdd.map((r) => r.email),
          role,
          duplicates: duplicates.length > 0 ? duplicates : undefined,
        },
      }),
    });

    return NextResponse.json({
      success: true,
      added: addedCount,
      duplicates: duplicates.length > 0 ? duplicates : undefined,
    });
  } catch (e) {
    return NextResponse.json(
      { error: "Failed to batch add", details: String(e) },
      { status: 500 }
    );
  }
}
