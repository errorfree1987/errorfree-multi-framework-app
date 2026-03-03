-- ============================================
-- 新增 guest 角色支援 (tenant_members.role)
-- ============================================
-- 用途：允許 Individual (Guest) 個人使用者
-- 錯誤：HTTP 400, code 23514 (CHECK constraint violation)
-- 執行：在 Supabase SQL Editor 中執行此檔案
-- ============================================

-- 移除現有的 role CHECK 約束（如果存在）
ALTER TABLE public.tenant_members 
DROP CONSTRAINT IF EXISTS tenant_members_role_check;

-- 新增包含 guest 的 CHECK 約束
ALTER TABLE public.tenant_members 
ADD CONSTRAINT tenant_members_role_check 
CHECK (role IN ('user', 'tenant_admin', 'guest'));

-- 驗證
-- SELECT DISTINCT role FROM tenant_members;
