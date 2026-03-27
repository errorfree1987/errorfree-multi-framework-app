"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Search, Upload, Users, Building2, Mail, UserCircle,
  Loader2, CheckCircle2, AlertCircle, Pencil, X, Save,
  Phone, ShieldCheck, Power, Download,
} from "lucide-react";

type MemberSettings = {
  bypass_tenant_cap?: boolean;
  custom_daily_cap?: number | null;
  track_usage?: boolean;
};

type Member = {
  id: string;
  tenant_id: string;
  tenant_slug: string;
  email: string;
  display_name?: string;
  phone?: string;
  role: string;
  is_active: boolean;
  notes?: string;
  last_login_at?: string;
  last_activity_at?: string;
  created_at: string;
};

type Tenant = { id: string; slug: string; name: string };

function parseSettings(notes?: string): MemberSettings {
  if (!notes) return {};
  try {
    const p = JSON.parse(notes);
    if (typeof p === "object" && p !== null) return p as MemberSettings;
  } catch { /* ignore */ }
  return {};
}

function formatDate(s?: string) {
  return s ? new Date(s).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }) : "—";
}

function roleLabel(r: string) {
  if (r === "tenant_admin") return "Admin";
  if (r === "viewer") return "Viewer";
  return "User";
}

function roleBadge(r: string) {
  const base = "inline-block rounded-full px-2 py-0.5 text-xs font-medium";
  if (r === "tenant_admin") return `${base} bg-purple-100 text-purple-700`;
  if (r === "viewer") return `${base} bg-slate-100 text-slate-600`;
  return `${base} bg-blue-100 text-blue-700`;
}

// ── Inline edit form for a single member ───────────────────────────────────
function MemberEditRow({
  member,
  tenantSlug,
  onSave,
  onCancel,
}: {
  member: Member;
  tenantSlug: string;
  onSave: (updated: Member) => void;
  onCancel: () => void;
}) {
  const settings = parseSettings(member.notes);
  const [form, setForm] = useState({
    display_name: member.display_name ?? "",
    phone: member.phone ?? "",
    role: member.role,
    is_active: member.is_active,
    bypass_tenant_cap: settings.bypass_tenant_cap ?? false,
    custom_daily_cap: settings.custom_daily_cap ?? 0,
    track_usage: settings.track_usage ?? true,
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  async function handleSave() {
    setSaving(true);
    setError("");
    try {
      const res = await fetch("/api/members/update", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          member_id: member.id,
          tenant_slug: tenantSlug,
          role: form.role,
          is_active: form.is_active,
          display_name: form.display_name,
          phone: form.phone,
          settings: {
            bypass_tenant_cap: form.bypass_tenant_cap,
            custom_daily_cap: form.custom_daily_cap || null,
            track_usage: form.track_usage,
          },
        }),
      });
      const data = await res.json();
      if (!res.ok) { setError(data.error || "Failed to save"); return; }
      onSave({
        ...member,
        display_name: form.display_name || undefined,
        phone: form.phone || undefined,
        role: form.role,
        is_active: form.is_active,
        notes: JSON.stringify({
          bypass_tenant_cap: form.bypass_tenant_cap,
          custom_daily_cap: form.custom_daily_cap || null,
          track_usage: form.track_usage,
        }),
      });
    } catch {
      setError("Network error");
    } finally {
      setSaving(false);
    }
  }

  return (
    <tr className="border-b bg-slate-50">
      <td colSpan={7} className="px-3 py-4">
        <div className="space-y-4">
          <p className="text-sm font-medium text-slate-700">
            Editing: <span className="font-mono">{member.email}</span>
          </p>

          {/* Basic info */}
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <Label className="text-xs">Display Name</Label>
              <Input
                value={form.display_name}
                onChange={(e) => setForm((f) => ({ ...f, display_name: e.target.value }))}
                placeholder="Full name"
                className="h-8 mt-1 text-sm"
              />
            </div>
            <div>
              <Label className="text-xs">Phone</Label>
              <Input
                value={form.phone}
                onChange={(e) => setForm((f) => ({ ...f, phone: e.target.value }))}
                placeholder="+886..."
                className="h-8 mt-1 text-sm"
              />
            </div>
            <div>
              <Label className="text-xs">Role</Label>
              <select
                value={form.role}
                onChange={(e) => setForm((f) => ({ ...f, role: e.target.value }))}
                className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-sm"
              >
                <option value="user">User</option>
                <option value="tenant_admin">Admin</option>
                <option value="viewer">Viewer (Read-only)</option>
              </select>
            </div>
            <div>
              <Label className="text-xs">Status</Label>
              <select
                value={form.is_active ? "1" : "0"}
                onChange={(e) => setForm((f) => ({ ...f, is_active: e.target.value === "1" }))}
                className="mt-1 h-8 w-full rounded-md border border-input bg-background px-2 text-sm"
              >
                <option value="1">Active</option>
                <option value="0">Inactive</option>
              </select>
            </div>
          </div>

          {/* Usage cap settings */}
          <div className="rounded-md border border-slate-200 bg-white p-3 space-y-3">
            <p className="text-xs font-semibold text-slate-600 uppercase tracking-wide">Usage Cap Settings</p>
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="flex items-start gap-2">
                <input
                  type="checkbox"
                  id={`track-${member.id}`}
                  checked={form.track_usage}
                  onChange={(e) => setForm((f) => ({ ...f, track_usage: e.target.checked }))}
                  className="mt-0.5 h-4 w-4 rounded border-gray-300"
                />
                <div>
                  <label htmlFor={`track-${member.id}`} className="text-sm font-medium cursor-pointer">
                    Track Usage
                  </label>
                  <p className="text-xs text-muted-foreground">Count this member&apos;s activity in usage logs</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <input
                  type="checkbox"
                  id={`bypass-${member.id}`}
                  checked={form.bypass_tenant_cap}
                  onChange={(e) => setForm((f) => ({ ...f, bypass_tenant_cap: e.target.checked }))}
                  className="mt-0.5 h-4 w-4 rounded border-gray-300"
                />
                <div>
                  <label htmlFor={`bypass-${member.id}`} className="text-sm font-medium cursor-pointer">
                    Bypass Tenant Cap
                  </label>
                  <p className="text-xs text-muted-foreground">Usage does not count toward tenant&apos;s daily cap</p>
                </div>
              </div>
              <div>
                <Label className="text-xs">
                  Personal Daily Cap
                  <span className="ml-1 text-muted-foreground">(0 = follow tenant cap)</span>
                </Label>
                <Input
                  type="number"
                  min={0}
                  value={form.custom_daily_cap ?? 0}
                  onChange={(e) => setForm((f) => ({ ...f, custom_daily_cap: Number(e.target.value) }))}
                  className="h-8 mt-1 text-sm w-28"
                  disabled={form.bypass_tenant_cap}
                />
              </div>
            </div>
          </div>

          {error && (
            <div className="flex items-center gap-2 text-destructive text-sm">
              <AlertCircle className="h-4 w-4" />{error}
            </div>
          )}

          <div className="flex gap-2">
            <Button size="sm" onClick={handleSave} disabled={saving}>
              {saving ? <Loader2 className="h-4 w-4 mr-1 animate-spin" /> : <Save className="h-4 w-4 mr-1" />}
              Save
            </Button>
            <Button size="sm" variant="outline" onClick={onCancel}>
              <X className="h-4 w-4 mr-1" />Cancel
            </Button>
          </div>
        </div>
      </td>
    </tr>
  );
}

// ── Main page ───────────────────────────────────────────────────────────────
export default function MembersPage() {
  const [members, setMembers] = useState<Member[]>([]);
  const [tenants, setTenants] = useState<Tenant[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [tenantFilter, setTenantFilter] = useState<string>("");
  const [roleFilter, setRoleFilter] = useState<string>("");
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [togglingId, setTogglingId] = useState<string | null>(null);

  // Batch add state
  const [batchOpen, setBatchOpen] = useState(false);
  const [batchTenant, setBatchTenant] = useState("");
  const [batchRole, setBatchRole] = useState<"user" | "tenant_admin">("user");
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchPaste, setBatchPaste] = useState("");
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchProgress, setBatchProgress] = useState(0);
  const [batchError, setBatchError] = useState("");
  const [batchSuccess, setBatchSuccess] = useState("");

  const loadMembers = useCallback(() => {
    const params = new URLSearchParams();
    if (tenantFilter) params.set("tenant_slug", tenantFilter);
    if (search) params.set("search", search);
    fetch(`/api/members?${params}`)
      .then((r) => r.json())
      .then((data) => setMembers(Array.isArray(data) ? data : []))
      .catch(() => setMembers([]))
      .finally(() => setLoading(false));
  }, [tenantFilter, search]);

  async function handleQuickToggle(m: Member) {
    setTogglingId(m.id);
    try {
      await fetch("/api/members/update", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ member_id: m.id, tenant_slug: m.tenant_slug, is_active: !m.is_active }),
      });
      setMembers((prev) => prev.map((x) => x.id === m.id ? { ...x, is_active: !m.is_active } : x));
    } catch { /* silent */ }
    finally { setTogglingId(null); }
  }

  function handleExportCSV() {
    const header = ["Email", "Display Name", "Phone", "Tenant", "Role", "Status", "Joined", "Last Login"];
    const rows = displayedMembers.map((m) => [
      m.email, m.display_name ?? "", m.phone ?? "", m.tenant_slug,
      roleLabel(m.role), m.is_active ? "Active" : "Inactive",
      m.created_at, m.last_login_at ?? "",
    ]);
    const csv = [header, ...rows].map((r) => r.map((c) => `"${String(c).replace(/"/g, '""')}"`).join(",")).join("\n");
    const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8;" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `members-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  useEffect(() => { setLoading(true); loadMembers(); }, [loadMembers]);
  useEffect(() => {
    fetch("/api/tenants").then((r) => r.json()).then((d) => {
      if (Array.isArray(d)) setTenants(d);
    });
  }, []);

  function parseCSV(text: string): Array<{ email: string; phone?: string; display_name?: string }> {
    return text
      .split(/\r?\n/)
      .filter((l) => l.trim())
      .map((line) => {
        const parts = line.split(",").map((p) => p.trim().replace(/^["']|["']$/g, ""));
        return { email: parts[0] || "", phone: parts[1] || undefined, display_name: parts[2] || undefined };
      })
      .filter((r) => r.email && r.email.includes("@"));
  }

  async function handleBatchAdd() {
    setBatchError(""); setBatchSuccess("");
    const tenant = batchTenant || (tenants[0]?.slug ?? "");
    if (!tenant) { setBatchError("Please select a tenant."); return; }

    let rows: Array<{ email: string; phone?: string; display_name?: string }> = [];
    if (batchFile) {
      setBatchProgress(10);
      const text = await batchFile.text();
      setBatchProgress(40);
      rows = parseCSV(text);
      setBatchProgress(60);
    } else if (batchPaste.trim()) {
      rows = parseCSV(batchPaste);
    }
    if (rows.length === 0) { setBatchError("Please upload a CSV file or paste email list."); return; }

    setBatchLoading(true); setBatchProgress(70);
    try {
      const res = await fetch("/api/members/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tenant_slug: tenant, members: rows, role: batchRole }),
      });
      setBatchProgress(95);
      const data = await res.json();
      if (!res.ok) {
        setBatchError(data.error || "Failed to add members.");
        if (data.duplicates?.length) setBatchError((p) => p + ` Duplicates: ${data.duplicates.join(", ")}`);
        return;
      }
      setBatchProgress(100);
      setBatchSuccess(`Added ${data.added} member(s)${data.duplicates?.length ? ` (${data.duplicates.length} skipped, already exist)` : ""}`);
      setBatchFile(null); setBatchPaste("");
      loadMembers();
      setTimeout(() => { setBatchSuccess(""); setBatchProgress(0); }, 4000);
    } catch (e) {
      setBatchError("Network error: " + String(e));
    } finally {
      setBatchLoading(false);
    }
  }

  // Derived stats
  const activeCount = members.filter((m) => m.is_active).length;
  const adminCount = members.filter((m) => m.role === "tenant_admin").length;
  const recentLogin = members.filter((m) => {
    if (!m.last_login_at) return false;
    return Date.now() - new Date(m.last_login_at).getTime() < 24 * 60 * 60 * 1000;
  }).length;

  const displayedMembers = members.filter((m) => {
    if (roleFilter && m.role !== roleFilter) return false;
    if (statusFilter === "active" && !m.is_active) return false;
    if (statusFilter === "inactive" && m.is_active) return false;
    return true;
  });

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Members</h1>
          <p className="text-muted-foreground mt-1">Manage members individually or in bulk. Click Edit to customise role, usage caps, and more.</p>
        </div>
        <Button variant="outline" size="sm" onClick={handleExportCSV} disabled={members.length === 0}>
          <Download className="h-4 w-4 mr-1" />Export CSV
        </Button>
      </div>

      {/* Summary stats */}
      <div className="grid gap-3 sm:grid-cols-4">
        {[
          { label: "Total Members", value: members.length, color: "text-blue-600", bg: "bg-blue-50" },
          { label: "Active", value: activeCount, color: "text-green-600", bg: "bg-green-50" },
          { label: "Admins", value: adminCount, color: "text-purple-600", bg: "bg-purple-50" },
          { label: "Active Last 24h", value: recentLogin, color: "text-amber-600", bg: "bg-amber-50" },
        ].map((k) => (
          <Card key={k.label} className="shadow-none">
            <CardContent className="pt-3 pb-3">
              <p className="text-xs text-muted-foreground">{k.label}</p>
              <p className={`text-2xl font-bold mt-0.5 ${k.color}`}>{loading ? "—" : k.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Search + Bulk Add */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-col sm:flex-row gap-4 justify-between items-start">
            <div className="flex flex-wrap gap-3 items-center">
              <div className="relative flex-1 min-w-[180px]">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input placeholder="Search email, name, phone..." value={search} onChange={(e) => setSearch(e.target.value)} className="pl-9" />
              </div>
              <select value={tenantFilter} onChange={(e) => setTenantFilter(e.target.value)} className="h-10 rounded-md border border-input bg-background px-3 text-sm">
                <option value="">All Tenants</option>
                {tenants.map((t) => <option key={t.id} value={t.slug}>{t.name || t.slug}</option>)}
              </select>
              <select value={roleFilter} onChange={(e) => setRoleFilter(e.target.value)} className="h-10 rounded-md border border-input bg-background px-3 text-sm">
                <option value="">All Roles</option>
                <option value="user">User</option>
                <option value="tenant_admin">Admin</option>
                <option value="viewer">Viewer</option>
              </select>
              <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} className="h-10 rounded-md border border-input bg-background px-3 text-sm">
                <option value="">All Status</option>
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
              </select>
            </div>
            <Button onClick={() => setBatchOpen(!batchOpen)}>
              <Upload className="h-4 w-4 mr-2" />{batchOpen ? "Close" : "Bulk Add"}
            </Button>
          </div>
        </CardHeader>

        {batchOpen && (
          <CardContent className="border-t pt-4 space-y-4">
            <CardTitle className="text-base">Bulk Add Members</CardTitle>
            <div className="space-y-1">
              <CardDescription>CSV: email (required), phone, display_name. Or paste one email per line.</CardDescription>
              <Button type="button" variant="outline" size="sm" onClick={() => {
                const csv = "email,phone,display_name\nuser1@example.com,+886912345678,John Doe\nuser2@example.com,,Jane Smith";
                const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8" });
                const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "members_template.csv"; a.click(); URL.revokeObjectURL(a.href);
              }}>Download CSV Template</Button>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <Label>Select Tenant</Label>
                <select value={batchTenant} onChange={(e) => setBatchTenant(e.target.value)} className="mt-1 w-full h-10 rounded-md border border-input bg-background px-3 text-sm">
                  <option value="">Select...</option>
                  {tenants.map((t) => <option key={t.id} value={t.slug}>{t.name || t.slug}</option>)}
                </select>
              </div>
              <div>
                <Label>Default Role</Label>
                <select value={batchRole} onChange={(e) => setBatchRole(e.target.value as "user" | "tenant_admin")} className="mt-1 w-full h-10 rounded-md border border-input bg-background px-3 text-sm">
                  <option value="user">User</option>
                  <option value="tenant_admin">Admin</option>
                </select>
              </div>
            </div>
            <div>
              <Label>Upload CSV File</Label>
              <Input type="file" accept=".csv,.txt" onChange={(e) => { setBatchFile(e.target.files?.[0] || null); setBatchPaste(""); }} className="mt-1" />
            </div>
            <div>
              <Label>Or paste emails (one per line)</Label>
              <textarea value={batchPaste} onChange={(e) => { setBatchPaste(e.target.value); if (e.target.value) setBatchFile(null); }} placeholder={"user1@example.com\nuser2@example.com"} className="mt-1 w-full min-h-[80px] rounded-md border border-input bg-background px-3 py-2 text-sm" disabled={!!batchFile} />
            </div>
            {batchLoading && (
              <div className="space-y-1">
                <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden"><div className="h-full bg-primary transition-all" style={{ width: `${batchProgress}%` }} /></div>
                <p className="text-sm text-muted-foreground flex items-center gap-2"><Loader2 className="h-4 w-4 animate-spin" />Processing...</p>
              </div>
            )}
            {batchError && <div className="flex items-center gap-2 text-destructive text-sm"><AlertCircle className="h-4 w-4" />{batchError}</div>}
            {batchSuccess && <div className="flex items-center gap-2 text-green-600 text-sm"><CheckCircle2 className="h-4 w-4" />{batchSuccess}</div>}
            <Button onClick={handleBatchAdd} disabled={batchLoading}>
              {batchLoading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Upload className="h-4 w-4 mr-2" />}Add Members
            </Button>
          </CardContent>
        )}
      </Card>

      {/* Member list */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Member List
            <span className="text-sm font-normal text-muted-foreground">
              ({displayedMembers.length}{displayedMembers.length !== members.length ? ` of ${members.length}` : ""})
            </span>
          </CardTitle>
          <CardDescription>Click <strong>Edit</strong> to customise role, caps, and settings. Use the toggle to quickly activate/deactivate.</CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-12"><Loader2 className="h-8 w-8 animate-spin text-muted-foreground" /></div>
          ) : displayedMembers.length === 0 ? (
            <p className="text-muted-foreground py-8 text-center">No members match the current filters.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-slate-50">
                    <th className="text-left py-3 px-3 font-medium">Member</th>
                    <th className="text-left py-3 px-2 font-medium">Tenant</th>
                    <th className="text-left py-3 px-2 font-medium">Role</th>
                    <th className="text-left py-3 px-2 font-medium">Status</th>
                    <th className="text-left py-3 px-2 font-medium">Last Login</th>
                    <th className="text-left py-3 px-2 font-medium">Cap Settings</th>
                    <th className="text-left py-3 px-2 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {displayedMembers.map((m) => {
                    const s = parseSettings(m.notes);
                    const isEditing = editingId === m.id;
                    return (
                      <>
                        <tr key={m.id} className={`border-b hover:bg-slate-50 ${isEditing ? "bg-blue-50/50" : ""}`}>
                          {/* Member: avatar + email + name */}
                          <td className="py-3 px-3">
                            <div className="flex items-center gap-2.5">
                              <div className={`h-8 w-8 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${m.is_active ? "bg-primary/10 text-primary" : "bg-slate-100 text-slate-400"}`}>
                                {(m.display_name || m.email).charAt(0).toUpperCase()}
                              </div>
                              <div className="min-w-0">
                                <p className="font-medium truncate max-w-[180px]">{m.email}</p>
                                <div className="flex items-center gap-2 mt-0.5">
                                  {m.display_name && <span className="text-xs text-muted-foreground truncate max-w-[120px]">{m.display_name}</span>}
                                  {m.phone && <span className="text-xs text-muted-foreground"><Phone className="h-2.5 w-2.5 inline mr-0.5" />{m.phone}</span>}
                                </div>
                              </div>
                            </div>
                          </td>
                          <td className="py-3 px-2">
                            <span className="text-xs bg-slate-100 rounded px-1.5 py-0.5">{m.tenant_slug}</span>
                          </td>
                          <td className="py-3 px-2">
                            <span className={roleBadge(m.role)}>{roleLabel(m.role)}</span>
                          </td>
                          <td className="py-3 px-2">
                            <button
                              onClick={() => handleQuickToggle(m)}
                              disabled={togglingId === m.id}
                              className={`inline-flex items-center gap-1 text-xs font-medium rounded-full px-2.5 py-1 border transition-colors ${
                                m.is_active
                                  ? "bg-green-50 text-green-700 border-green-200 hover:bg-green-100"
                                  : "bg-slate-50 text-slate-500 border-slate-200 hover:bg-slate-100"
                              }`}
                            >
                              {togglingId === m.id
                                ? <Loader2 className="h-3 w-3 animate-spin" />
                                : <Power className="h-3 w-3" />}
                              {m.is_active ? "Active" : "Inactive"}
                            </button>
                          </td>
                          <td className="py-3 px-2 text-xs text-muted-foreground">
                            {m.last_login_at ? (
                              <span title={new Date(m.last_login_at).toLocaleString()}>
                                {formatDate(m.last_login_at)}
                              </span>
                            ) : "Never"}
                          </td>
                          <td className="py-3 px-2 text-xs text-muted-foreground space-y-0.5">
                            {s.bypass_tenant_cap && <span className="flex items-center gap-1 text-amber-600"><ShieldCheck className="h-3 w-3" />Bypass cap</span>}
                            {s.custom_daily_cap ? <span>{s.custom_daily_cap}/day</span> : null}
                            {s.track_usage === false && <span className="text-slate-400">No tracking</span>}
                            {!s.bypass_tenant_cap && !s.custom_daily_cap && s.track_usage !== false && <span className="text-slate-300">—</span>}
                          </td>
                          <td className="py-3 px-2">
                            {isEditing ? (
                              <Button size="sm" variant="ghost" onClick={() => setEditingId(null)}><X className="h-4 w-4" /></Button>
                            ) : (
                              <Button size="sm" variant="outline" onClick={() => setEditingId(m.id)}>
                                <Pencil className="h-3.5 w-3.5 mr-1" />Edit
                              </Button>
                            )}
                          </td>
                        </tr>
                        {isEditing && (
                          <MemberEditRow
                            key={`edit-${m.id}`}
                            member={m}
                            tenantSlug={m.tenant_slug}
                            onSave={(updated) => {
                              setMembers((prev) => prev.map((x) => x.id === updated.id ? updated : x));
                              setEditingId(null);
                            }}
                            onCancel={() => setEditingId(null)}
                          />
                        )}
                      </>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
