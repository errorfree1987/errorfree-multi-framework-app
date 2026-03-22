"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function UsagePage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Usage & Caps</h1>
        <p className="text-muted-foreground mt-1">
          View usage and set limits. Under construction.
        </p>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Coming Soon</CardTitle>
          <CardDescription>
            Usage charts and caps management will be available in the next update. Please use the Streamlit Admin UI for now.
          </CardDescription>
        </CardHeader>
      </Card>
    </div>
  );
}
