import { NextResponse } from "next/server";
import { cookies } from "next/headers";

export async function GET() {
  const cookieStore = await cookies();
  const session = cookieStore.get("admin_session");
  const valid = !!session?.value && session.value.startsWith("admin_");
  return NextResponse.json({ authenticated: valid });
}
