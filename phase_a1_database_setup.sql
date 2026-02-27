-- ============================================
-- Phase A1: Mode A 營運資料庫表結構
-- ============================================
-- 用途：完整的 Phase A1 資料庫設置（已驗收通過）
-- 最後更新：2026-02-27
-- ============================================

-- ===== 總覽 =====
-- Phase A1 包含以下表：
-- 1. tenants (已存在，已補充欄位)
-- 2. tenant_members (已存在，已補充欄位)
-- 3. tenant_session_epoch (已存在，D3 完成)
-- 4. audit_events (新建)
-- 5. tenant_entitlements (已存在，已補充欄位)
-- 6. tenant_usage_caps (新建)
-- 7. tenant_usage_events (新建)

-- ===== 注意事項 =====
-- - 所有 SQL 使用 IF NOT EXISTS / IF EXISTS 檢查，可安全重複執行
-- - 測試資料使用 ON CONFLICT DO UPDATE，可安全重複執行
-- - 現有表保留原有欄位名稱（例如 slug, framework_key）

-- ============================================
-- 1. tenants 表（已存在，補充驗收查詢）
-- ============================================
-- 驗收查詢：查看所有租戶及其狀態
SELECT 
    slug AS tenant,
    name,
    display_name,
    status,
    is_active,
    trial_start,
    trial_end,
    CASE 
        WHEN trial_end IS NULL THEN 'No expiry'
        WHEN trial_end > NOW() THEN 'Active (' || EXTRACT(DAY FROM trial_end - NOW()) || ' days left)'
        ELSE 'Expired (' || EXTRACT(DAY FROM NOW() - trial_end) || ' days ago)'
    END AS trial_status,
    notes,
    created_at,
    updated_at
FROM public.tenants
ORDER BY created_at DESC;

-- ============================================
-- 2. tenant_members 表（已存在，補充驗收查詢）
-- ============================================
-- 驗收查詢：查看所有成員，按租戶分組
SELECT 
    t.slug AS tenant,
    t.name AS tenant_name,
    t.status AS tenant_status,
    t.is_active AS tenant_is_active,
    tm.email,
    tm.display_name,
    tm.is_active AS member_is_active,
    tm.role,
    tm.notes,
    tm.created_at,
    tm.last_login_at,
    tm.last_activity_at
FROM public.tenant_members tm
LEFT JOIN public.tenants t ON tm.tenant_id = t.id
ORDER BY t.slug, tm.created_at;

-- 統計每個租戶的成員數量
SELECT 
    t.slug AS tenant,
    t.name AS tenant_name,
    t.status,
    t.is_active AS tenant_active,
    COUNT(tm.id) AS total_members,
    COUNT(CASE WHEN tm.is_active THEN 1 END) AS active_members,
    COUNT(CASE WHEN NOT tm.is_active THEN 1 END) AS inactive_members,
    COUNT(CASE WHEN tm.role = 'tenant_admin' THEN 1 END) AS admin_count,
    COUNT(CASE WHEN tm.role = 'user' THEN 1 END) AS user_count
FROM public.tenants t
LEFT JOIN public.tenant_members tm ON t.id = tm.tenant_id
GROUP BY t.slug, t.name, t.status, t.is_active
ORDER BY total_members DESC;

-- ============================================
-- 3. audit_events 表（新建）
-- ============================================
CREATE TABLE IF NOT EXISTS public.audit_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action TEXT NOT NULL,
    tenant_slug TEXT,
    email TEXT,
    result TEXT NOT NULL,
    deny_reason TEXT,
    context JSONB,
    actor_email TEXT,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_events_created_at ON public.audit_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_events_tenant_slug ON public.audit_events(tenant_slug);
CREATE INDEX IF NOT EXISTS idx_audit_events_email ON public.audit_events(email);
CREATE INDEX IF NOT EXISTS idx_audit_events_action ON public.audit_events(action);
CREATE INDEX IF NOT EXISTS idx_audit_events_result ON public.audit_events(result);
CREATE INDEX IF NOT EXISTS idx_audit_events_actor_email ON public.audit_events(actor_email);
CREATE INDEX IF NOT EXISTS idx_audit_events_tenant_action ON public.audit_events(tenant_slug, action, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_events_email_action ON public.audit_events(email, action, created_at DESC);

COMMENT ON TABLE public.audit_events IS 'Mode A 審計表：記錄所有關鍵操作（login/verify/launch/revoke/deny）';
COMMENT ON COLUMN public.audit_events.action IS 'sso_verify / analyzer_launch / epoch_revoke / access_denied / member_created / member_deactivated / trial_extended';
COMMENT ON COLUMN public.audit_events.result IS 'success / denied / error';
COMMENT ON COLUMN public.audit_events.deny_reason IS 'tenant_inactive / member_inactive / trial_expired / epoch_mismatch / invalid_token / entitlement_denied';

-- 驗收查詢
SELECT 
    created_at,
    action,
    tenant_slug,
    email,
    result,
    deny_reason,
    actor_email,
    context,
    notes
FROM public.audit_events
ORDER BY created_at DESC
LIMIT 100;

-- 按 action 統計事件數量
SELECT 
    action,
    result,
    COUNT(*) AS event_count
FROM public.audit_events
GROUP BY action, result
ORDER BY action, result;

-- 查看所有拒絕事件（按原因分組）
SELECT 
    deny_reason,
    COUNT(*) AS denied_count,
    array_agg(DISTINCT tenant_slug) AS affected_tenants
FROM public.audit_events
WHERE result = 'denied'
GROUP BY deny_reason
ORDER BY denied_count DESC;

-- ============================================
-- 4. tenant_entitlements 表（已存在，補充驗收查詢）
-- ============================================
-- 驗收查詢：查看所有租戶的權限
SELECT 
    t.slug AS tenant,
    t.name AS tenant_name,
    te.framework_key,
    te.is_enabled,
    te.last_modified_by,
    te.notes,
    te.created_at,
    te.updated_at
FROM public.tenant_entitlements te
LEFT JOIN public.tenants t ON te.tenant_id = t.id
ORDER BY t.slug, te.framework_key;

-- ============================================
-- 5. tenant_usage_caps 表（新建）
-- ============================================
CREATE TABLE IF NOT EXISTS public.tenant_usage_caps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL UNIQUE,
    daily_review_cap INTEGER,
    daily_download_cap INTEGER,
    max_file_size_mb INTEGER,
    max_rounds INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_modified_by TEXT,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_tenant_usage_caps_tenant_id ON public.tenant_usage_caps(tenant_id);

DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'fk_tenant_usage_caps_tenant'
    ) THEN
        ALTER TABLE public.tenant_usage_caps
            ADD CONSTRAINT fk_tenant_usage_caps_tenant 
            FOREIGN KEY (tenant_id) 
            REFERENCES public.tenants(id) 
            ON DELETE CASCADE;
    END IF;
END $$;

COMMENT ON TABLE public.tenant_usage_caps IS 'Mode A 限流表：租戶的使用上限設定';
COMMENT ON COLUMN public.tenant_usage_caps.daily_review_cap IS '每日最多 review 次數（NULL = 無限制）';
COMMENT ON COLUMN public.tenant_usage_caps.daily_download_cap IS '每日最多下載次數（NULL = 無限制）';

-- 驗收查詢
SELECT 
    t.slug AS tenant,
    t.name AS tenant_name,
    tc.daily_review_cap,
    tc.daily_download_cap,
    tc.max_file_size_mb,
    tc.max_rounds,
    tc.last_modified_by,
    tc.notes,
    tc.created_at,
    tc.updated_at
FROM public.tenant_usage_caps tc
LEFT JOIN public.tenants t ON tc.tenant_id = t.id
ORDER BY t.slug;

-- ============================================
-- 6. tenant_usage_events 表（新建）
-- ============================================
CREATE TABLE IF NOT EXISTS public.tenant_usage_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    tenant_id UUID NOT NULL,
    email TEXT NOT NULL,
    usage_type TEXT NOT NULL,
    quantity INTEGER DEFAULT 1,
    context JSONB,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_tenant_usage_events_created_at ON public.tenant_usage_events(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tenant_usage_events_tenant_id ON public.tenant_usage_events(tenant_id);
CREATE INDEX IF NOT EXISTS idx_tenant_usage_events_usage_type ON public.tenant_usage_events(usage_type);
CREATE INDEX IF NOT EXISTS idx_tenant_usage_events_tenant_created ON public.tenant_usage_events(tenant_id, created_at DESC);

DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'fk_tenant_usage_events_tenant'
    ) THEN
        ALTER TABLE public.tenant_usage_events
            ADD CONSTRAINT fk_tenant_usage_events_tenant 
            FOREIGN KEY (tenant_id) 
            REFERENCES public.tenants(id) 
            ON DELETE CASCADE;
    END IF;
END $$;

COMMENT ON TABLE public.tenant_usage_events IS 'Mode A 用量表：記錄實際使用量（用於限流計算）';
COMMENT ON COLUMN public.tenant_usage_events.usage_type IS 'review / download / export';

-- 驗收查詢：查看今日使用量（按租戶統計）
SELECT 
    t.slug AS tenant,
    t.name AS tenant_name,
    tue.usage_type,
    COUNT(*) AS usage_count,
    SUM(tue.quantity) AS total_quantity
FROM public.tenant_usage_events tue
LEFT JOIN public.tenants t ON tue.tenant_id = t.id
WHERE tue.created_at >= CURRENT_DATE
GROUP BY t.slug, t.name, tue.usage_type
ORDER BY t.slug, tue.usage_type;

-- 租戶使用狀況總覽（caps vs actual usage）
SELECT 
    t.slug AS tenant,
    t.name AS tenant_name,
    tc.daily_review_cap,
    COALESCE(SUM(CASE WHEN tue.usage_type = 'review' AND tue.created_at >= CURRENT_DATE THEN tue.quantity END), 0) AS today_review_count,
    tc.daily_download_cap,
    COALESCE(SUM(CASE WHEN tue.usage_type = 'download' AND tue.created_at >= CURRENT_DATE THEN tue.quantity END), 0) AS today_download_count,
    CASE 
        WHEN tc.daily_review_cap IS NULL THEN 'Unlimited'
        WHEN COALESCE(SUM(CASE WHEN tue.usage_type = 'review' AND tue.created_at >= CURRENT_DATE THEN tue.quantity END), 0) >= tc.daily_review_cap THEN 'Cap reached'
        ELSE 'Under limit'
    END AS review_status
FROM public.tenants t
LEFT JOIN public.tenant_usage_caps tc ON t.id = tc.tenant_id
LEFT JOIN public.tenant_usage_events tue ON t.id = tue.tenant_id
GROUP BY t.slug, t.name, tc.daily_review_cap, tc.daily_download_cap
ORDER BY t.slug;

-- ============================================
-- 快速診斷查詢（營運常用）
-- ============================================

-- 查詢 1：誰在線上（最近 24 小時有活動）
SELECT DISTINCT
    t.slug AS tenant,
    ae.email,
    MAX(ae.created_at) AS last_seen
FROM public.audit_events ae
LEFT JOIN public.tenants t ON ae.tenant_slug = t.slug
WHERE ae.created_at > NOW() - INTERVAL '24 hours'
  AND ae.result = 'success'
  AND ae.action IN ('sso_verify', 'analyzer_launch')
GROUP BY t.slug, ae.email
ORDER BY last_seen DESC;

-- 查詢 2：哪些租戶即將到期（7 天內）
SELECT 
    slug AS tenant,
    name,
    status,
    trial_end,
    EXTRACT(DAY FROM trial_end - NOW()) AS days_remaining
FROM public.tenants
WHERE trial_end IS NOT NULL
  AND trial_end > NOW()
  AND trial_end < NOW() + INTERVAL '7 days'
  AND status = 'trial'
ORDER BY trial_end;

-- 查詢 3：今日被拒絕的登入嘗試
SELECT 
    created_at,
    tenant_slug,
    email,
    deny_reason,
    context
FROM public.audit_events
WHERE result = 'denied'
  AND created_at >= CURRENT_DATE
ORDER BY created_at DESC;

-- 查詢 4：哪些租戶今日用量接近上限（80% 以上）
SELECT 
    t.slug AS tenant,
    t.name AS tenant_name,
    tc.daily_review_cap,
    COUNT(CASE WHEN tue.usage_type = 'review' AND tue.created_at >= CURRENT_DATE THEN 1 END) AS today_review_count,
    ROUND(
        (COUNT(CASE WHEN tue.usage_type = 'review' AND tue.created_at >= CURRENT_DATE THEN 1 END)::NUMERIC / NULLIF(tc.daily_review_cap, 0)) * 100,
        1
    ) AS usage_percent
FROM public.tenants t
LEFT JOIN public.tenant_usage_caps tc ON t.id = tc.tenant_id
LEFT JOIN public.tenant_usage_events tue ON t.id = tue.tenant_id
WHERE tc.daily_review_cap IS NOT NULL
  AND tc.daily_review_cap > 0
GROUP BY t.slug, t.name, tc.daily_review_cap
HAVING COUNT(CASE WHEN tue.usage_type = 'review' AND tue.created_at >= CURRENT_DATE THEN 1 END)::NUMERIC / NULLIF(tc.daily_review_cap, 0) >= 0.8
ORDER BY usage_percent DESC;

-- ============================================
-- 結束
-- ============================================
