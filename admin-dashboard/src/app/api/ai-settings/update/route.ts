import { NextRequest, NextResponse } from "next/server";
import { supabaseFetch, isSupabaseConfigured } from "@/lib/supabase";

export async function POST(request: NextRequest) {
  if (!isSupabaseConfigured()) {
    return NextResponse.json({ error: "Supabase not configured" }, { status: 500 });
  }

  const body = await request.json();
  const { tenant_slug, provider, base_url, model, api_key_ref, max_tokens_per_request, modified_by } = body;

  if (!tenant_slug) {
    return NextResponse.json({ error: "tenant_slug required" }, { status: 400 });
  }

  // Check if record exists
  const existsRes = await supabaseFetch(
    `/tenant_ai_settings?tenant=eq.${encodeURIComponent(tenant_slug)}&select=tenant`
  );
  const existing = existsRes.ok ? await existsRes.json() : [];

  const payload: Record<string, unknown> = {
    tenant: tenant_slug,
    provider: provider || null,
    base_url: base_url || null,
    model: model || null,
    api_key_ref: api_key_ref || null,
    max_tokens_per_request: max_tokens_per_request ? Number(max_tokens_per_request) : null,
    last_modified_by: modified_by || "admin",
    updated_at: new Date().toISOString(),
  };

  let saveRes: Response;

  if (existing.length > 0) {
    // PATCH existing record
    saveRes = await supabaseFetch(
      `/tenant_ai_settings?tenant=eq.${encodeURIComponent(tenant_slug)}`,
      {
        method: "PATCH",
        headers: { Prefer: "return=minimal" },
        body: JSON.stringify(payload),
      }
    );
  } else {
    // INSERT new record
    saveRes = await supabaseFetch("/tenant_ai_settings", {
      method: "POST",
      headers: { Prefer: "return=minimal" },
      body: JSON.stringify(payload),
    });
  }

  if (!saveRes.ok) {
    const err = await saveRes.text();
    return NextResponse.json({ error: err }, { status: 500 });
  }

  // Write audit log
  await supabaseFetch("/audit_events", {
    method: "POST",
    headers: { Prefer: "return=minimal" },
    body: JSON.stringify({
      action: "tenant_ai_settings_updated",
      tenant_slug,
      email: modified_by || "admin",
      result: "success",
      context: { provider, model, base_url, api_key_ref },
    }),
  });

  return NextResponse.json({ success: true });
}
