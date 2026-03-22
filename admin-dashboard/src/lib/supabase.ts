const SUPABASE_URL = process.env.SUPABASE_URL?.trim() || "";
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY?.trim() || "";

export function getSupabaseConfig() {
  return { url: SUPABASE_URL, serviceKey: SUPABASE_SERVICE_KEY };
}

export function isSupabaseConfigured() {
  return !!(SUPABASE_URL && SUPABASE_SERVICE_KEY);
}

export async function supabaseFetch(
  path: string,
  options: RequestInit = {}
): Promise<Response> {
  const { url, serviceKey } = getSupabaseConfig();
  const fullUrl = `${url}/rest/v1${path}`;
  const headers: HeadersInit = {
    apikey: serviceKey,
    Authorization: `Bearer ${serviceKey}`,
    "Content-Type": "application/json",
    "Accept": "application/json",
    ...(options.headers as Record<string, string>),
  };
  return fetch(fullUrl, { ...options, headers });
}
