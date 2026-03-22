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
      setBatchError("請選擇租戶");
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
      setBatchError("請上傳 CSV 或貼上 email 清單。CSV 格式：第一欄為 email，可選 phone、display_name");
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
        setBatchError(data.error || data.details || "新增失敗");
        if (data.duplicates?.length) {
          setBatchError((prev) => prev + ` 重複：${data.duplicates.join(", ")}`);
        }
        return;
      }
      setBatchProgress(100);
      setBatchSuccess(`成功新增 ${data.added} 位成員${data.duplicates?.length ? `，${data.duplicates.length} 位已存在已略過` : ""}`);
      setBatchFile(null);
      setBatchPaste("");
      loadMembers();
      setTimeout(() => {
        setBatchSuccess("");
        setBatchProgress(0);
      }, 3000);
    } catch (e) {
      setBatchError("網路錯誤：" + String(e));
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
          批量新增與管理成員。支援 CSV 上傳、搜尋篩選。
        </p>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <div className="flex flex-col sm:flex-row gap-4 justify-between items-start">
            <div className="flex flex-wrap gap-3 items-center">
              <div className="relative flex-1 min-w-[180px]">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="搜尋 email、名稱、電話..."
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
                <option value="">全部租戶</option>
                {tenants.map((t) => (
                  <option key={t.id} value={t.slug}>
                    {t.name || t.slug}
                  </option>
                ))}
              </select>
            </div>
            <Button onClick={() => setBatchOpen(!batchOpen)}>
              <Upload className="h-4 w-4 mr-2" />
              {batchOpen ? "關閉" : "批量新增"}
            </Button>
          </div>
        </CardHeader>

        {batchOpen && (
          <CardContent className="border-t pt-4 space-y-4">
            <CardTitle className="text-base">批量新增成員</CardTitle>
            <div className="space-y-2">
              <CardDescription>
                CSV 格式：第一欄為 email（必填），可選第二欄 phone、第三欄 display_name。或直接貼上 email 清單（一行一個）。
              </CardDescription>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => {
                  const csv = "email,phone,display_name\nuser1@example.com,+886912345678,張小明\nuser2@example.com,,李美華";
                  const blob = new Blob(["\uFEFF" + csv], { type: "text/csv;charset=utf-8" });
                  const a = document.createElement("a");
                  a.href = URL.createObjectURL(blob);
                  a.download = "members_template.csv";
                  a.click();
                  URL.revokeObjectURL(a.href);
                }}
              >
                下載 CSV 範本
              </Button>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <Label>選擇租戶</Label>
                <select
                  value={batchTenant}
                  onChange={(e) => setBatchTenant(e.target.value)}
                  className="mt-1 w-full h-10 rounded-md border border-input bg-background px-3 text-sm"
                >
                  <option value="">請選擇</option>
                  {tenants.map((t) => (
                    <option key={t.id} value={t.slug}>
                      {t.name || t.slug}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <Label>角色</Label>
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
              <Label>上傳 CSV 檔案</Label>
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
              <Label>或貼上 email 清單（一行一個，或 CSV）</Label>
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
                  處理中...
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
              開始新增
            </Button>
          </CardContent>
        )}
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            成員列表
            {filteredCount >= 0 && (
              <span className="text-sm font-normal text-muted-foreground">
                （{filteredCount} 筆）
              </span>
            )}
          </CardTitle>
          <CardDescription>
            依搜尋與租戶篩選顯示。啟用/停用、編輯角色請使用 Streamlit Admin UI。
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : members.length === 0 ? (
            <p className="text-muted-foreground py-8 text-center">
              尚無成員，或搜尋/篩選無結果。請使用「批量新增」上傳 CSV 或貼上清單。
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-3 px-2 font-medium">Email</th>
                    <th className="text-left py-3 px-2 font-medium">名稱</th>
                    <th className="text-left py-3 px-2 font-medium">租戶</th>
                    <th className="text-left py-3 px-2 font-medium">角色</th>
                    <th className="text-left py-3 px-2 font-medium">狀態</th>
                    <th className="text-left py-3 px-2 font-medium">加入日期</th>
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
                          {m.is_active ? "啟用" : "停用"}
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
