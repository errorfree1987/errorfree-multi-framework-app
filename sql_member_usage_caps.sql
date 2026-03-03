-- ============================================
-- Member-Level Usage Caps Migration
-- ============================================
-- 用途：支援 per-member 的 review/download 上限（覆寫 tenant caps）
-- 建立日期：2026-03-03
-- 執行：在 Supabase SQL Editor 執行
-- ============================================

CREATE TABLE IF NOT EXISTS public.member_usage_caps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    email TEXT NOT NULL,
    daily_review_cap INTEGER,
    daily_download_cap INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_modified_by TEXT,
    notes TEXT,
    UNIQUE(tenant_id, email)
);

CREATE INDEX IF NOT EXISTS idx_member_usage_caps_tenant_email
    ON public.member_usage_caps(tenant_id, email);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'fk_member_usage_caps_tenant'
    ) THEN
        ALTER TABLE public.member_usage_caps
            ADD CONSTRAINT fk_member_usage_caps_tenant
            FOREIGN KEY (tenant_id)
            REFERENCES public.tenants(id)
            ON DELETE CASCADE;
    END IF;
END $$;

COMMENT ON TABLE public.member_usage_caps IS 'Per-member usage caps (overrides tenant caps when present)';
COMMENT ON COLUMN public.member_usage_caps.daily_review_cap IS 'NULL=inherit tenant, 0=disabled, >0=limit';
COMMENT ON COLUMN public.member_usage_caps.daily_download_cap IS 'NULL=inherit tenant, 0=disabled, >0=limit';

-- ============================================
-- 驗收查詢
-- ============================================

-- 查看所有 member caps
SELECT
    t.slug,
    muc.email,
    muc.daily_review_cap,
    muc.daily_download_cap,
    muc.updated_at
FROM member_usage_caps muc
JOIN tenants t ON t.id = muc.tenant_id
ORDER BY t.slug, muc.email;
