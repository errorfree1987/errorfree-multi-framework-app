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

export async function POST(request: NextRequest) {
  const auth = await requireAuth();
  if (!auth) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const body = await request.json();
    const {
      slug,
      name,
      display_name,
      trial_days = 30,
      daily_review_cap = 50,
      daily_download_cap = 20,
    } = body;

    if (!slug || !name) {
      return NextResponse.json(
        { error: "slug and name are required" },
        { status: 400 }
      );
    }

    const trialEnd = new Date();
    trialEnd.setDate(trialEnd.getDate() + Number(trial_days));

    const tenantPayload = {
      slug: String(slug).toLowerCase().trim(),
      name: String(name).trim(),
      display_name: (display_name || name).trim(),
      status: "trial",
      trial_start: new Date().toISOString(),
      trial_end: trialEnd.toISOString(),
      is_active: true,
    };

    const tenantRes = await supabaseFetch("/tenants", {
      method: "POST",
      body: JSON.stringify(tenantPayload),
      headers: { Prefer: "return=representation" },
    });

    if (!tenantRes.ok) {
      const text = await tenantRes.text();
      return NextResponse.json(
        { error: "Failed to create tenant", details: text },
        { status: 400 }
      );
    }

    const tenantData = (await tenantRes.json())[0];
    const tenantId = tenantData.id;

    await supabaseFetch("/tenant_session_epoch", {
      method: "POST",
      body: JSON.stringify({ tenant: tenantPayload.slug, epoch: 0 }),
    });

    await supabaseFetch("/tenant_usage_caps", {
      method: "POST",
      body: JSON.stringify({
        tenant_id: tenantId,
        daily_review_cap: daily_review_cap > 0 ? daily_review_cap : null,
        daily_download_cap: daily_download_cap > 0 ? daily_download_cap : null,
      }),
    });

    await supabaseFetch("/audit_events", {
      method: "POST",
      body: JSON.stringify({
        action: "tenant_created",
        tenant_slug: tenantPayload.slug,
        email: "admin",
        result: "success",
        context: {
          tenant_id: tenantId,
          trial_days,
          daily_review_cap,
          daily_download_cap,
        },
      }),
    });

    return NextResponse.json({ success: true, tenant: tenantData });
  } catch (e) {
    return NextResponse.json(
      { error: "Failed to create tenant", details: String(e) },
      { status: 500 }
    );
  }
}
