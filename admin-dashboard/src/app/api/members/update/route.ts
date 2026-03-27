import { NextRequest, NextResponse } from "next/server";
import { cookies } from "next/headers";
import { supabaseFetch } from "@/lib/supabase";

async function requireAuth() {
  const cookieStore = await cookies();
  const session = cookieStore.get("admin_session");
  if (!session?.value || !session.value.startsWith("admin_")) return null;
  return true;
}

export type MemberSettings = {
  bypass_tenant_cap?: boolean;   // true = this member's usage does NOT count toward tenant cap
  custom_daily_cap?: number | null; // per-member daily usage cap (0 = unlimited)
  track_usage?: boolean;         // false = disable usage tracking for this member
};

/**
 * PATCH /api/members/update
 * Body: { member_id, role?, is_active?, display_name?, phone?, notes?, settings? }
 */
export async function PATCH(req: NextRequest) {
  const auth = await requireAuth();
  if (!auth) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  try {
    const body = await req.json();
    const { member_id, role, is_active, display_name, phone, notes, settings } = body;

    if (!member_id) {
      return NextResponse.json({ error: "member_id required" }, { status: 400 });
    }

    // Get current member to merge settings
    const currentRes = await supabaseFetch(
      `/tenant_members?id=eq.${member_id}&select=id,email,notes,tenant_id`
    );
    if (!currentRes.ok) {
      return NextResponse.json({ error: "Member not found" }, { status: 404 });
    }
    const currentMembers = await currentRes.json();
    if (!Array.isArray(currentMembers) || currentMembers.length === 0) {
      return NextResponse.json({ error: "Member not found" }, { status: 404 });
    }
    const current = currentMembers[0] as { id: string; email: string; notes?: string; tenant_id: string };

    // Build patch payload
    const patch: Record<string, unknown> = {};
    if (role !== undefined) patch.role = role;
    if (is_active !== undefined) patch.is_active = is_active;
    if (display_name !== undefined) patch.display_name = display_name || null;
    if (phone !== undefined) patch.phone = phone || null;

    // Merge settings into notes JSON
    if (settings !== undefined) {
      let existingSettings: MemberSettings = {};
      try {
        const parsed = current.notes ? JSON.parse(current.notes) : {};
        if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
          existingSettings = parsed as MemberSettings;
        }
      } catch {
        // ignore parse errors
      }
      const mergedSettings: MemberSettings = { ...existingSettings, ...settings };
      patch.notes = JSON.stringify(mergedSettings);
    } else if (notes !== undefined) {
      // Raw notes string (plain text, not settings)
      patch.notes = notes || null;
    }

    if (Object.keys(patch).length === 0) {
      return NextResponse.json({ error: "No fields to update" }, { status: 400 });
    }

    const patchRes = await supabaseFetch(
      `/tenant_members?id=eq.${member_id}`,
      {
        method: "PATCH",
        headers: { Prefer: "return=representation" },
        body: JSON.stringify(patch),
      }
    );

    if (!patchRes.ok) {
      const text = await patchRes.text();
      return NextResponse.json(
        { error: `Failed to update member: ${patchRes.status}`, details: text },
        { status: 502 }
      );
    }

    const updated = await patchRes.json();

    // Log audit
    await supabaseFetch("/audit_events", {
      method: "POST",
      body: JSON.stringify({
        action: "member_updated",
        tenant_slug: body.tenant_slug ?? "unknown",
        result: "success",
        actor_email: "admin",
        email: current.email,
        context: { member_id, changes: Object.keys(patch) },
      }),
    });

    return NextResponse.json({ success: true, member: Array.isArray(updated) ? updated[0] : updated });
  } catch (e) {
    return NextResponse.json({ error: "Failed to update member", details: String(e) }, { status: 500 });
  }
}
