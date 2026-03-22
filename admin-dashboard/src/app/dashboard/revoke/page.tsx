"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function RevokePage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Revoke Access</h1>
        <p className="text-muted-foreground mt-1">
          One-click session revocation for tenants. Under construction.
        </p>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Coming Soon</CardTitle>
          <CardDescription>
            Revoke access will be available in the next update. Please use the Streamlit Admin UI for now.
          </CardDescription>
        </CardHeader>
      </Card>
    </div>
  );
}
