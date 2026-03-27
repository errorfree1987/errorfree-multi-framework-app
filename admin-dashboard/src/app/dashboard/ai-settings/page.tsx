"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Bot,
  Save,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Info,
} from "lucide-react";

type AiSetting = {
  provider: string | null;
  base_url: string | null;
  model: string | null;
  api_key_ref: string | null;
  max_tokens_per_request: number | null;
  last_modified_by: string | null;
  updated_at: string | null;
};

type TenantRow = {
  id: string;
  slug: string;
  name: string;
  is_active: boolean;
  ai: AiSetting | null;
};

const PROVIDER_OPTIONS = [
  { value: "copilot", label: "Microsoft Copilot / Azure OpenAI" },
  { value: "openai_compatible", label: "OpenAI Compatible" },
  { value: "deepseek", label: "DeepSeek (disabled by default)" },
];

const MODEL_SUGGESTIONS: Record<string, string[]> = {
  copilot: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
  openai_compatible: ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
  deepseek: ["deepseek-chat", "deepseek-coder"],
};

const KEY_REF_SUGGESTIONS: Record<string, string> = {
  copilot: "OPENAI_API_KEY",
  openai_compatible: "OPENAI_API_KEY",
  deepseek: "DEEPSEEK_API_KEY",
};

function providerBadge(provider: string | null) {
  if (!provider) return <span className="px-2 py-0.5 rounded text-xs border border-slate-200 text-slate-400">Not configured</span>;
  const colors: Record<string, string> = {
    copilot: "bg-blue-100 text-blue-700",
    openai_compatible: "bg-green-100 text-green-700",
    deepseek: "bg-orange-100 text-orange-700",
  };
  const labels: Record<string, string> = {
    copilot: "Copilot",
    openai_compatible: "OpenAI Compatible",
    deepseek: "DeepSeek",
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${colors[provider] ?? "bg-slate-100 text-slate-600"}`}>
      {labels[provider] ?? provider}
    </span>
  );
}

type EditState = {
  provider: string;
  base_url: string;
  model: string;
  api_key_ref: string;
  max_tokens: string;
};

function TenantAiCard({ tenant, onSaved }: { tenant: TenantRow; onSaved: () => void }) {
  const [expanded, setExpanded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");

  const [form, setForm] = useState<EditState>({
    provider: tenant.ai?.provider ?? "",
    base_url: tenant.ai?.base_url ?? "",
    model: tenant.ai?.model ?? "",
    api_key_ref: tenant.ai?.api_key_ref ?? "",
    max_tokens: tenant.ai?.max_tokens_per_request?.toString() ?? "",
  });

  function onProviderChange(val: string) {
    setForm((f) => ({
      ...f,
      provider: val,
      api_key_ref: f.api_key_ref || KEY_REF_SUGGESTIONS[val] || "",
    }));
  }

  async function handleSave() {
    setSaving(true);
    setError("");
    setSaved(false);
    try {
      const res = await fetch("/api/ai-settings/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tenant_slug: tenant.slug,
          provider: form.provider || null,
          base_url: form.base_url || null,
          model: form.model || null,
          api_key_ref: form.api_key_ref || null,
          max_tokens_per_request: form.max_tokens || null,
          modified_by: "admin",
        }),
      });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.error ?? "Save failed");
      }
      setSaved(true);
      onSaved();
      setTimeout(() => setSaved(false), 3000);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setSaving(false);
    }
  }

  const modelSuggestions = MODEL_SUGGESTIONS[form.provider] ?? [];
  const isConfigured = !!tenant.ai?.provider;

  return (
    <Card className={`border ${isConfigured ? "border-slate-200" : "border-orange-200 bg-orange-50/30"}`}>
      <CardHeader className="pb-3 cursor-pointer" onClick={() => setExpanded((e) => !e)}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white text-xs font-bold">
              {tenant.name.slice(0, 2).toUpperCase()}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-medium text-sm">{tenant.name}</span>
                {!tenant.is_active && (
                  <span className="px-1.5 py-0.5 text-xs border border-slate-200 rounded text-slate-400">Inactive</span>
                )}
              </div>
              <span className="text-xs text-slate-500 font-mono">{tenant.slug}</span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {providerBadge(tenant.ai?.provider ?? null)}
            {tenant.ai?.model && (
              <span className="text-xs text-slate-500 font-mono">{tenant.ai.model}</span>
            )}
            {expanded ? <ChevronUp className="h-4 w-4 text-slate-400" /> : <ChevronDown className="h-4 w-4 text-slate-400" />}
          </div>
        </div>
      </CardHeader>

      {expanded && (
        <CardContent className="pt-0 space-y-4 border-t">
          <div className="pt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Provider */}
            <div className="space-y-1.5">
              <Label className="text-xs font-medium">AI Provider</Label>
              <select
                className="w-full h-9 px-3 text-sm border border-input rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-ring"
                value={form.provider}
                onChange={(e) => onProviderChange(e.target.value)}
              >
                <option value="">Select provider…</option>
                {PROVIDER_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>

            {/* Model */}
            <div className="space-y-1.5">
              <Label className="text-xs font-medium">Model</Label>
              <div className="relative">
                <Input
                  className="h-9 text-sm"
                  placeholder="e.g. gpt-4o"
                  value={form.model}
                  onChange={(e) => setForm((f) => ({ ...f, model: e.target.value }))}
                />
                {modelSuggestions.length > 0 && !form.model && (
                  <div className="flex gap-1 mt-1 flex-wrap">
                    {modelSuggestions.map((m) => (
                      <button
                        key={m}
                        className="text-xs px-2 py-0.5 bg-slate-100 hover:bg-slate-200 rounded font-mono"
                        onClick={() => setForm((f) => ({ ...f, model: m }))}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* Base URL */}
            <div className="space-y-1.5">
              <Label className="text-xs font-medium">
                Base URL <span className="text-slate-400 font-normal">(optional)</span>
              </Label>
              <Input
                className="h-9 text-sm font-mono"
                placeholder="https://api.openai.com/v1"
                value={form.base_url}
                onChange={(e) => setForm((f) => ({ ...f, base_url: e.target.value }))}
              />
            </div>

            {/* API Key Ref */}
            <div className="space-y-1.5">
              <div className="flex items-center gap-1">
                <Label className="text-xs font-medium">API Key Ref</Label>
                <div className="group relative">
                  <Info className="h-3.5 w-3.5 text-slate-400 cursor-help" />
                  <div className="absolute left-0 top-5 z-10 hidden group-hover:block w-64 p-2 bg-slate-800 text-white text-xs rounded shadow-lg">
                    The name of the environment variable storing the API key (e.g. OPENAI_API_KEY). The key itself is never stored in the database.
                  </div>
                </div>
              </div>
              <Input
                className="h-9 text-sm font-mono"
                placeholder="OPENAI_API_KEY"
                value={form.api_key_ref}
                onChange={(e) => setForm((f) => ({ ...f, api_key_ref: e.target.value }))}
              />
              <p className="text-xs text-slate-500">
                Name of environment variable — key is never stored in database
              </p>
            </div>

            {/* Max Tokens */}
            <div className="space-y-1.5">
              <Label className="text-xs font-medium">
                Max Tokens / Request <span className="text-slate-400 font-normal">(optional)</span>
              </Label>
              <Input
                className="h-9 text-sm"
                type="number"
                placeholder="e.g. 4000"
                value={form.max_tokens}
                onChange={(e) => setForm((f) => ({ ...f, max_tokens: e.target.value }))}
              />
            </div>
          </div>

          {/* Last modified */}
          {tenant.ai?.updated_at && (
            <p className="text-xs text-slate-400">
              Last saved: {new Date(tenant.ai.updated_at).toLocaleString()} by {tenant.ai.last_modified_by ?? "admin"}
            </p>
          )}

          {/* Actions */}
          <div className="flex items-center gap-3 pt-1">
            <Button
              size="sm"
              onClick={handleSave}
              disabled={saving}
              className="gap-2"
            >
              {saving ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
              {saving ? "Saving…" : "Save Settings"}
            </Button>
            {saved && (
              <span className="flex items-center gap-1 text-sm text-green-600">
                <CheckCircle2 className="h-4 w-4" /> Saved — audit log written
              </span>
            )}
            {error && (
              <span className="flex items-center gap-1 text-sm text-red-600">
                <AlertCircle className="h-4 w-4" /> {error}
              </span>
            )}
          </div>
        </CardContent>
      )}
    </Card>
  );
}

export default function AiSettingsPage() {
  const [tenants, setTenants] = useState<TenantRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");

  async function load() {
    setLoading(true);
    try {
      const res = await fetch("/api/ai-settings");
      if (res.ok) setTenants(await res.json());
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  const filtered = tenants.filter(
    (t) =>
      t.name.toLowerCase().includes(search.toLowerCase()) ||
      t.slug.toLowerCase().includes(search.toLowerCase())
  );

  const configured = tenants.filter((t) => !!t.ai?.provider).length;
  const notConfigured = tenants.length - configured;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Bot className="h-6 w-6 text-indigo-600" />
            AI Provider Settings
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Configure which AI provider and model each tenant uses in the Analyzer
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={load} className="gap-2">
          <RefreshCw className="h-4 w-4" />
          Refresh
        </Button>
      </div>

      {/* Summary KPIs */}
      <div className="grid grid-cols-3 gap-4">
        <Card>
          <CardContent className="pt-4 pb-4">
            <p className="text-xs text-slate-500 uppercase tracking-wide">Total Tenants</p>
            <p className="text-2xl font-bold mt-1">{tenants.length}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4 pb-4">
            <p className="text-xs text-slate-500 uppercase tracking-wide">Configured</p>
            <p className="text-2xl font-bold mt-1 text-green-600">{configured}</p>
          </CardContent>
        </Card>
        <Card className={notConfigured > 0 ? "border-orange-200" : ""}>
          <CardContent className="pt-4 pb-4">
            <p className="text-xs text-slate-500 uppercase tracking-wide">Not Configured</p>
            <p className={`text-2xl font-bold mt-1 ${notConfigured > 0 ? "text-orange-500" : "text-slate-400"}`}>
              {notConfigured}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Info banner */}
      <div className="flex items-start gap-3 p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-800">
        <Info className="h-4 w-4 mt-0.5 flex-shrink-0" />
        <div>
          <strong>API Key Security:</strong> Only the environment variable <em>name</em> (e.g. <code className="font-mono text-xs bg-blue-100 px-1 rounded">OPENAI_API_KEY</code>) is stored in the database.
          The actual key is read from server environment variables at runtime — it never touches the database.
        </div>
      </div>

      {/* Search */}
      <Input
        placeholder="Search tenants…"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="max-w-sm"
      />

      {/* Tenant list */}
      {loading ? (
        <div className="text-center py-12 text-slate-400">
          <RefreshCw className="h-6 w-6 animate-spin mx-auto mb-2" />
          Loading…
        </div>
      ) : filtered.length === 0 ? (
        <p className="text-slate-400 text-sm">No tenants found.</p>
      ) : (
        <div className="space-y-3">
          {filtered.map((t) => (
            <TenantAiCard key={t.id} tenant={t} onSaved={load} />
          ))}
        </div>
      )}
    </div>
  );
}
