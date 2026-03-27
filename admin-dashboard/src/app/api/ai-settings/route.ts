import { NextResponse } from "next/server";
import { supabaseFetch, isSupabaseConfigured } from "@/lib/supabase";

export async function GET() {
  if (!isSupabaseConfigured()) {
    return NextResponse.json({ error: "Supabase not configured" }, { status: 500 });
  }

  // Fetch all tenants
  const tenantsRes = await supabaseFetch(
    "/tenants?select=id,slug,name,is_active&order=name.asc"
  );
  if (!tenantsRes.ok) {
    return NextResponse.json({ error: "Failed to fetch tenants" }, { status: 500 });
  }
  const tenants = await tenantsRes.json();

  // Fetch all ai settings
  const aiRes = await supabaseFetch(
    "/tenant_ai_settings?select=tenant,provider,base_url,model,api_key_ref,max_tokens_per_request,last_modified_by,updated_at"
  );
  if (!aiRes.ok) {
    return NextResponse.json({ error: "Failed to fetch AI settings" }, { status: 500 });
  }
  const aiSettings: Record<string, unknown>[] = await aiRes.json();

  // Map ai settings by tenant slug for easy lookup
  const aiBySlug: Record<string, Record<string, unknown>> = {};
  for (const s of aiSettings) {
    aiBySlug[s.tenant as string] = s;
  }

  // Merge
  const result = tenants.map((t: Record<string, unknown>) => ({
    ...t,
    ai: aiBySlug[t.slug as string] ?? null,
  }));

  return NextResponse.json(result);
}
