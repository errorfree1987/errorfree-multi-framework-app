"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Building2, Users, Shield, BarChart3, FileText } from "lucide-react";

const quickLinks = [
  { href: "/dashboard/tenants", label: "Tenant Management", icon: Building2 },
  { href: "/dashboard/members", label: "Members", icon: Users },
  { href: "/dashboard/revoke", label: "Revoke Access", icon: Shield },
  { href: "/dashboard/usage", label: "Usage & Caps", icon: BarChart3 },
  { href: "/dashboard/audit", label: "Audit Logs", icon: FileText },
];

export default function DashboardPage() {
  const [tenantCount, setTenantCount] = useState<number | null>(null);

  useEffect(() => {
    fetch("/api/tenants")
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setTenantCount(data.length);
      })
      .catch(() => {});
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground mt-1">
          Welcome to Error-Free® Admin. Manage tenants, members, and usage.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Total Tenants</CardTitle>
            <Building2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {tenantCount !== null ? tenantCount : "—"}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
          <CardDescription>Jump to a section</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {quickLinks.map((link) => {
              const Icon = link.icon;
              return (
                <Link key={link.href} href={link.href}>
                  <Button variant="outline" className="w-full justify-start gap-2 h-auto py-4">
                    <Icon className="h-5 w-5" />
                    {link.label}
                  </Button>
                </Link>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
