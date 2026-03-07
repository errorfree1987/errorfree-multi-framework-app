-- ============================================
-- tenant_members: 新增 phone 欄位（支援國際手機號碼）
-- ============================================
-- 用途：學員/成員資料收集，支援不同國家區域格式
-- 範例：+1 2345678900, +886 912345678, +86 13800138000, +44 20 7946 0958
-- 最後更新：2026-03-04
-- ============================================

ALTER TABLE public.tenant_members
    ADD COLUMN IF NOT EXISTS phone TEXT;

COMMENT ON COLUMN public.tenant_members.phone IS 'Phone number, international format (e.g. +1 2345678900, +886 912345678). Supports all regions.';

-- 驗收查詢：確認欄位已新增
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'tenant_members'
  AND column_name = 'phone';
