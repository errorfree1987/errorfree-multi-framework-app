-- ============================================
-- tenant_ai_settings 表（Analyzer 多 Provider：Copilot / DeepSeek）
-- ============================================
-- 用途：依租戶設定使用的 AI 平台（大陸→DeepSeek，其他→Copilot/OpenAI）
-- 最後更新：2026-03-03
-- ============================================

-- 建立表（tenant = 租戶 slug，與 tenants.slug 對應）
CREATE TABLE IF NOT EXISTS public.tenant_ai_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant TEXT NOT NULL UNIQUE,
    provider TEXT,
    base_url TEXT,
    model TEXT,
    api_key_ref TEXT,
    max_tokens_per_request INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_modified_by TEXT,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_tenant_ai_settings_tenant ON public.tenant_ai_settings(tenant);

COMMENT ON TABLE public.tenant_ai_settings IS 'Analyzer 多 Provider：各租戶使用的 AI 平台設定';
COMMENT ON COLUMN public.tenant_ai_settings.tenant IS '租戶 slug（與 tenants.slug 一致）';
COMMENT ON COLUMN public.tenant_ai_settings.provider IS 'copilot | openai_compatible | deepseek';
COMMENT ON COLUMN public.tenant_ai_settings.base_url IS 'OpenAI-compatible API base URL（可選）';
COMMENT ON COLUMN public.tenant_ai_settings.model IS '模型名稱（可選，覆寫 role-based 預設）';
COMMENT ON COLUMN public.tenant_ai_settings.api_key_ref IS 'API key 的環境變數名稱，如 OPENAI_API_KEY、DEEPSEEK_API_KEY';

-- 驗收查詢：查看所有租戶 AI 設定
SELECT
    tas.tenant,
    t.name AS tenant_name,
    tas.provider,
    tas.base_url,
    tas.model,
    tas.api_key_ref,
    tas.max_tokens_per_request,
    tas.updated_at,
    tas.last_modified_by
FROM public.tenant_ai_settings tas
LEFT JOIN public.tenants t ON t.slug = tas.tenant
ORDER BY tas.tenant;
