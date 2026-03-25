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

type UpdateBody = {
  tenant_id: string;
  trial_end?: string;
  extend_days?: number;
  is_active?: boolean;
  daily_review_cap?: number | null;
  daily_download_cap?: number | null;
};

export async function PATCH(req: NextRequest) {
  const auth = await requireAuth();
  if (!auth) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const body: UpdateBody = await req.json();
    const {
      tenant_id,
      trial_end,
      extend_days,
      is_active,
      daily_review_cap,
      daily_download_cap,
    } = body;

    if (!tenant_id) {
      return NextResponse.json(
        { error: "tenant_id required" },
        { status: 400 }
      );
    }

    const tenantRes = await supabaseFetch(
      `/tenants?id=eq.${tenant_id}&select=id,slug,trial_end`
    );
    if (!tenantRes.ok) {
      return NextResponse.json(
        { error: "Failed to fetch tenant" },
        { status: 404 }
      );
    }
    const tenants = await tenantRes.json();
    if (!Array.isArray(tenants) || tenants.length === 0) {
      return NextResponse.json(
        { error: "Tenant not found" },
        { status: 404 }
      );
    }
    const tenant = tenants[0];
    const tenantSlug = tenant.slug;

    if (trial_end !== undefined || extend_days !== undefined) {
      let newTrialEnd: string;
      if (trial_end) {
        const d = new Date(trial_end);
        if (isNaN(d.getTime())) {
          return NextResponse.json(
            { error: "Invalid trial_end date" },
            { status: 400 }
          );
        }
        newTrialEnd = d.toISOString();
      } else if (typeof extend_days === "number") {
        const current = tenant.trial_end
          ? new Date(tenant.trial_end)
          : new Date();
        const next = new Date(current);
        next.setDate(next.getDate() + extend_days);
        newTrialEnd = next.toISOString();
      } else {
        return NextResponse.json(
          { error: "Provide trial_end or extend_days" },
          { status: 400 }
        );
      }
      const patchRes = await supabaseFetch("/tenants", {
        method: "PATCH",
        body: JSON.stringify({ trial_end: newTrialEnd }),
        headers: { Prefer: "return=minimal" },
      });
      if (!patchRes.ok) {
        const text = await patchRes.text();
        return NextResponse.json(
          { error: `Failed to update trial: ${patchRes.status}`, details: text },
          { status: 502 }
        );
      }
      await supabaseFetch("/audit_events", {
        method: "POST",
        body: JSON.stringify({
          action: "tenant_trial_updated",
          tenant_slug: tenantSlug,
          result: "success",
          actor_email: "admin",
          context: { trial_end: newTrialEnd, extend_days },
        }),
      });
    }

    if (is_active !== undefined) {
      const res = await supabaseFetch(
        `/tenants?id=eq.${tenant_id}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            is_active,
            status: is_active ? "trial" : "suspended",
          }),
          headers: { Prefer: "return=minimal" },
        }
      );
      if (!res.ok) {
        const text = await res.text();
        return NextResponse.json(
          { error: `Failed to update status: ${res.status}`, details: text },
          { status: 502 }
        );
      }
      await supabaseFetch("/audit_events", {
        method: "POST",
        body: JSON.stringify({
          action: is_active ? "tenant_activated" : "tenant_suspended",
          tenant_slug: tenantSlug,
          result: "success",
          actor_email: "admin",
          context: { is_active },
        }),
      });
    }

    if (
      daily_review_cap !== undefined ||
      daily_download_cap !== undefined
    ) {
      const capsRes = await supabaseFetch(
        `/tenant_usage_caps?tenant_id=eq.${tenant_id}&select=id`
      );
      const capsList = capsRes.ok ? await capsRes.json() : [];

      const capPayload: Record<string, unknown> = {};
      if (daily_review_cap !== undefined)
        capPayload.daily_review_cap =
          daily_review_cap === null ? null : Number(daily_review_cap);
      if (daily_download_cap !== undefined)
        capPayload.daily_download_cap =
          daily_download_cap === null ? null : Number(daily_download_cap);

      if (capsList.length > 0) {
        const capId = capsList[0].id;
        const patchRes = await supabaseFetch(
          `/tenant_usage_caps?id=eq.${capId}`,
          {
            method: "PATCH",
            body: JSON.stringify(capPayload),
            headers: { Prefer: "return=minimal" },
          }
        );
        if (!patchRes.ok) {
          const text = await patchRes.text();
          return NextResponse.json(
            {
              error: `Failed to update caps: ${patchRes.status}`,
              details: text,
            },
            { status: 502 }
          );
        }
      } else {
        const postRes = await supabaseFetch("/tenant_usage_caps", {
          method: "POST",
          body: JSON.stringify({
            tenant_id,
            daily_review_cap:
              daily_review_cap !== undefined && daily_review_cap !== null
                ? Number(daily_review_cap)
                : 50,
            daily_download_cap:
              daily_download_cap !== undefined && daily_download_cap !== null
                ? Number(daily_download_cap)
                : 20,
          }),
          headers: { Prefer: "return=minimal" },
        });
        if (!postRes.ok) {
          const text = await postRes.text();
          return NextResponse.json(
            {
              error: `Failed to create caps: ${postRes.status}`,
              details: text,
            },
            { status: 502 }
          );
        }
      }
      await supabaseFetch("/audit_events", {
        method: "POST",
        body: JSON.stringify({
          action: "tenant_caps_updated",
          tenant_slug: tenantSlug,
          result: "success",
          actor_email: "admin",
          context: { daily_review_cap, daily_download_cap },
        }),
      });
    }

    return NextResponse.json({ success: true });
  } catch (e) {
    return NextResponse.json(
      { error: "Failed to update tenant", details: String(e) },
      { status: 500 }
    );
  }
}
