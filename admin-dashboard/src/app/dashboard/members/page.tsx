"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Search,
  Upload,
  Users,
  Building2,
  Mail,
  UserCircle,
  Loader2,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";

type Member = {
  id: string;
  tenant_id: string;
  tenant_slug: string;
  email: string;
  display_name?: string;
  phone?: string;
  role: string;
  is_active: boolean;
  created_at: string;
};

type Tenant = { id: string; slug: string; name: string };

export default function MembersPage() {
  const [members, setMembers] = useState<Member[]>([]);
  const [tenants, setTenants] = useState<Tenant[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [tenantFilter, setTenantFilter] = useState<string>("");
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

  useEffect(() => {
    setLoading(true);
    loadMembers();
  }, [loadMembers]);

  useEffect(() => {
    fetch("/api/tenants")
      .then((r) => r.json())
      .then((data) => setTenants(Array.isArray(data) ? data : []))
      .catch(() => setTenants([]));
  }, []);

  function parseCSV(text: string): Array<{ email: string; phone?: string; display_name?: string }> {
    const rows: Array<{ email: string; phone?: string; display_name?: string }> = [];
    const lines = text.split(/\r?\n/).filter((l) => l.trim());
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const parts = line.split(",").map((p) => p.trim().replace(/^["']|["']$/g, ""));
      const email = parts[0] || "";
      if (email && email.includes("@")) {
        rows.push({
          email,
          phone: parts[1] || undefined,
          display_name: parts[2] || undefined,
        });
      }
    }
    return rows;
  }

  async function handleBatchAdd() {
    setBatchError("");
    setBatchSuccess("");
    const tenant = batchTenant || (tenants[0]?.slug ?? "");
    if (!tenant) {
      setBatchError("Please select a tenant.");
      return;
    }

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

    if (rows.length === 0) {
      setBatchError("Please upload a CSV file or paste email list. Format: first column = email, optional: phone, display_name");
      return;
    }

    setBatchLoading(true);
    setBatchProgress(70);
    try {
      const res = await fetch("/api/members/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tenant_slug: tenant,
          members: rows,
          role: batchRole,
        }),
      });
      setBatchProgress(95);
      const data = await res.json();

      if (!res.ok) {
        setBatchError(data.error || data.details || "Failed to add members.");
        if (data.duplicates?.length) {
          setBatchError((prev) => prev + ` Duplicates: ${data.duplicates.join(", ")}`);
        }
        return;
      }
      setBatchProgress(100);
      setBatchSuccess(`Successfully added ${data.added} member(s)${data.duplicates?.length ? ` (${data.duplicates.length} already existed, skipped)` : ""}`);
      setBatchFile(null);
      setBatchPaste("");
      loadMembers();
      setTimeout(() => {
        setBatchSuccess("");
        setBatchProgress(0);
      }, 3000);
    } catch (e) {
      setBatchError("Network error: " + String(e));
    } finally {
      setBatchLoading(false);
    }
  }

  function formatDate(s: string) {
    return s ? new Date(s).toLocaleDateString() : "—";
  }

  function roleLabel(r: string) {
    return r === "tenant_admin" ? "Admin" : "User";
  }

  const filteredCount = members.length;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Members</h1>
        <p className="text-muted-foreground mt-1">
          Batch add and manage members. Supports CSV upload and search.
        </p>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-col sm:flex-row gap-4 justify-between items-start">
            <div className="flex flex-wrap gap-3 items-center">
              <div className="relative flex-1 min-w-[180px]">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search email, name, phone..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="pl-9"
                />
              </div>
              <select
                value={tenantFilter}
                onChange={(e) => setTenantFilter(e.target.value)}
                className="h-10 rounded-md border border-input bg-background px-3 text-sm"
              >
                <option value="">All Tenants</option>
                {tenants.map((t) => (
                  <option key={t.id} value={t.slug}>
                    {t.name || t.slug}
                  </option>
                ))}
              </select>
            </div>
            <Button onClick={() => setBatchOpen(!batchOpen)}>
              <Upload className="h-4 w-4 mr-2" />
              {batchOpen ? "Close" : "Bulk Add"}
            </Button>
          </div>
        </CardHeader>

        {batchOpen && (
          <CardContent className="border-t pt-4 space-y-4">
            <CardTitle className="text-base">Bulk Add Members</CardTitle>
            <div className="space-y-2">
              <CardDescription>
                CSV format: first column = email (required), optional: phone, display_name. Or paste one email per line.
              </CardDescription>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => {
                  const csv = "email,phone,display_name\nuser1@example.com,+886912345678,John Doe\nuser2@example.com,,Jane Smith";
                  const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8" });
                  const a = document.createElement("a");
                  a.href = URL.createObjectURL(blob);
                  a.download = "members_template.csv";
                  a.click();
                  URL.revokeObjectURL(a.href);
                }}
              >
                Download CSV Template
              </Button>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <Label>Select Tenant</Label>
                <select
                  value={batchTenant}
                  onChange={(e) => setBatchTenant(e.target.value)}
                  className="mt-1 w-full h-10 rounded-md border border-input bg-background px-3 text-sm"
                >
                  <option value="">Select...</option>
                  {tenants.map((t) => (
                    <option key={t.id} value={t.slug}>
                      {t.name || t.slug}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <Label>Role</Label>
                <select
                  value={batchRole}
                  onChange={(e) => setBatchRole(e.target.value as "user" | "tenant_admin")}
                  className="mt-1 w-full h-10 rounded-md border border-input bg-background px-3 text-sm"
                >
                  <option value="user">User</option>
                  <option value="tenant_admin">Admin</option>
                </select>
              </div>
            </div>
            <div>
              <Label>Upload CSV File</Label>
              <Input
                type="file"
                accept=".csv,.txt"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  setBatchFile(f || null);
                  setBatchPaste("");
                }}
                className="mt-1"
              />
            </div>
            <div>
              <Label>Or paste email list (one per line, or CSV)</Label>
              <textarea
                value={batchPaste}
                onChange={(e) => {
                  setBatchPaste(e.target.value);
                  if (e.target.value) setBatchFile(null);
                }}
                placeholder="user1@example.com&#10;user2@example.com"
                className="mt-1 w-full min-h-[100px] rounded-md border border-input bg-background px-3 py-2 text-sm"
                disabled={!!batchFile}
              />
            </div>
            {batchLoading && (
              <div className="space-y-2">
                <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary transition-all duration-300"
                    style={{ width: `${batchProgress}%` }}
                  />
                </div>
                <p className="text-sm text-muted-foreground flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Processing...
                </p>
              </div>
            )}
            {batchError && (
              <div className="flex items-center gap-2 text-destructive text-sm">
                <AlertCircle className="h-4 w-4 shrink-0" />
                {batchError}
              </div>
            )}
            {batchSuccess && (
              <div className="flex items-center gap-2 text-green-600 text-sm">
                <CheckCircle2 className="h-4 w-4 shrink-0" />
                {batchSuccess}
              </div>
            )}
            <Button onClick={handleBatchAdd} disabled={batchLoading}>
              {batchLoading ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Upload className="h-4 w-4 mr-2" />
              )}
              Add Members
            </Button>
          </CardContent>
        )}
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Member List
            {filteredCount >= 0 && (
              <span className="text-sm font-normal text-muted-foreground">
                ({filteredCount})
              </span>
            )}
          </CardTitle>
          <CardDescription>
            Filter by search and tenant.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : members.length === 0 ? (
            <p className="text-muted-foreground py-8 text-center">
              No members found. Use Bulk Add to upload CSV or paste a list.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-3 px-2 font-medium">Email</th>
                    <th className="text-left py-3 px-2 font-medium">Name</th>
                    <th className="text-left py-3 px-2 font-medium">Tenant</th>
                    <th className="text-left py-3 px-2 font-medium">Role</th>
                    <th className="text-left py-3 px-2 font-medium">Status</th>
                    <th className="text-left py-3 px-2 font-medium">Joined</th>
                  </tr>
                </thead>
                <tbody>
                  {members.map((m) => (
                    <tr key={m.id} className="border-b hover:bg-slate-50">
                      <td className="py-3 px-2">
                        <span className="flex items-center gap-2">
                          <Mail className="h-4 w-4 text-muted-foreground" />
                          {m.email}
                        </span>
                      </td>
                      <td className="py-3 px-2">
                        {m.display_name ? (
                          <span className="flex items-center gap-2">
                            <UserCircle className="h-4 w-4 text-muted-foreground" />
                            {m.display_name}
                          </span>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td className="py-3 px-2">
                        <span className="flex items-center gap-2">
                          <Building2 className="h-4 w-4 text-muted-foreground" />
                          {m.tenant_slug}
                        </span>
                      </td>
                      <td className="py-3 px-2">{roleLabel(m.role)}</td>
                      <td className="py-3 px-2">
                        <span
                          className={
                            m.is_active
                              ? "text-green-600 font-medium"
                              : "text-muted-foreground"
                          }
                        >
                          {m.is_active ? "Active" : "Inactive"}
                        </span>
                      </td>
                      <td className="py-3 px-2 text-muted-foreground">
                        {formatDate(m.created_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
