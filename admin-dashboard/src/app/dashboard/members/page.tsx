"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function MembersPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Members</h1>
        <p className="text-muted-foreground mt-1">
          Batch add and manage members. This page is under construction.
        </p>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Coming Soon</CardTitle>
          <CardDescription>
            Members management will be available in the next update. Please use the Streamlit Admin UI for now.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            Run: <code className="bg-slate-100 px-2 py-1 rounded">streamlit run admin_ui.py</code>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
