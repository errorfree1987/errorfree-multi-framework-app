-- ============================================
-- Mode A 營運 Runbook SQL 模板
-- ============================================
-- 用途：日常營運操作的快速參考 SQL 模板
-- 使用方式：複製需要的 SQL，替換參數後在 Supabase SQL Editor 執行
-- 最後更新：2026-02-27
-- ============================================

-- ============================================
-- 📋 目錄
-- ============================================
-- 1. 租戶管理 (Tenant Management)
-- 2. 成員管理 (Member Management)
-- 3. 撤權操作 (Revocation)
-- 4. 權限管理 (Entitlements)
-- 5. 用量管理 (Usage & Caps)
-- 6. 審計查詢 (Audit Logs)
-- 7. 診斷查詢 (Diagnostics)

-- ============================================
-- 1. 租戶管理 (Tenant Management)
-- ============================================

-- 1.1 建立新租戶（完整流程）
-- ==========================================
-- 步驟 1: 建立租戶基本資訊
INSERT INTO tenants (slug, name, display_name, status, trial_start, trial_end, is_active, notes)
VALUES (
    'new_company',                          -- 替換：租戶 slug（唯一識別）
    'New Company',                          -- 替換：簡短名稱
    'New Company Inc.',                     -- 替換：顯示名稱
    'trial',                                -- 狀態：trial / active / suspended
    NOW(),                                  -- 試用開始時間
    NOW() + INTERVAL '30 days',             -- 替換：試用期限（例如 30 天）
    TRUE,                                   -- 是否啟用
    'Phase A3 測試：新建租戶'                -- 替換：備註
);

-- 步驟 2: 初始化 epoch（用於撤權）
INSERT INTO tenant_session_epoch (tenant, epoch, updated_at)
VALUES ('new_company', 0, NOW())            -- 替換：租戶 slug
ON CONFLICT (tenant) DO NOTHING;

-- 步驟 3: 設定預設 caps（可選）
INSERT INTO tenant_usage_caps (
    tenant_id, 
    daily_review_cap, 
    daily_download_cap, 
    max_file_size_mb, 
    max_rounds,
    last_modified_by,
    notes
)
VALUES (
    (SELECT id FROM tenants WHERE slug = 'new_company'),  -- 替換：租戶 slug
    50,                                     -- 替換：每日 review 上限
    20,                                     -- 替換：每日 download 上限
    10,                                     -- 替換：最大檔案大小 (MB)
    5,                                      -- 替換：最大輪次
    'admin@platform.com',                   -- 替換：操作者 email
    'Phase A3 測試：預設 caps'
);

-- 步驟 4: 啟用預設框架/功能（可選）
INSERT INTO tenant_entitlements (tenant_id, framework_key, is_enabled, last_modified_by, notes)
VALUES 
    ((SELECT id FROM tenants WHERE slug = 'new_company'), 'react', TRUE, 'system', 'Phase A3 測試'),
    ((SELECT id FROM tenants WHERE slug = 'new_company'), 'vue', TRUE, 'system', 'Phase A3 測試'),
    ((SELECT id FROM tenants WHERE slug = 'new_company'), 'analyzer_code_review', TRUE, 'system', 'Phase A3 測試');

-- 驗收：確認租戶建立成功
SELECT 
    slug,
    name,
    status,
    trial_end,
    is_active,
    created_at
FROM tenants
WHERE slug = 'new_company';  -- 替換：租戶 slug


-- 1.2 延長試用期限
-- ==========================================
UPDATE tenants
SET 
    trial_end = NOW() + INTERVAL '30 days',  -- 替換：延長天數
    updated_at = NOW()
WHERE slug = 'abc';                          -- 替換：租戶 slug

-- 驗收
SELECT slug, trial_end, 
       EXTRACT(DAY FROM trial_end - NOW()) AS days_remaining
FROM tenants
WHERE slug = 'abc';  -- 替換：租戶 slug


-- 1.3 轉為付費租戶
-- ==========================================
UPDATE tenants
SET 
    status = 'active',
    trial_end = NULL,  -- 移除試用期限
    updated_at = NOW()
WHERE slug = 'abc';  -- 替換：租戶 slug


-- 1.4 停用/啟用租戶
-- ==========================================
-- 停用租戶（立即阻止所有存取）
UPDATE tenants
SET 
    is_active = FALSE,
    status = 'suspended',
    updated_at = NOW()
WHERE slug = 'suspended';  -- 替換：租戶 slug

-- 啟用租戶
UPDATE tenants
SET 
    is_active = TRUE,
    status = 'trial',  -- 或 'active'
    updated_at = NOW()
WHERE slug = 'suspended';  -- 替換：租戶 slug


-- 1.5 查看所有租戶狀態
-- ==========================================
SELECT 
    slug,
    name,
    status,
    is_active,
    trial_start,
    trial_end,
    CASE 
        WHEN trial_end IS NULL THEN 'No expiry'
        WHEN trial_end > NOW() THEN 'Active (' || EXTRACT(DAY FROM trial_end - NOW())::INTEGER || ' days left)'
        ELSE 'Expired (' || EXTRACT(DAY FROM NOW() - trial_end)::INTEGER || ' days ago)'
    END AS trial_status,
    created_at
FROM tenants
ORDER BY created_at DESC;


-- ============================================
-- 2. 成員管理 (Member Management)
-- ============================================

-- 2.1 批量新增成員（單一租戶）
-- ==========================================
-- 方法 1：逐行插入（適合少量 <10 個）
INSERT INTO tenant_members (tenant_id, email, display_name, is_active, role, notes)
VALUES 
    ((SELECT id FROM tenants WHERE slug = 'abc'), 'user1@abc.com', 'User One', TRUE, 'user', 'Phase A3 批量新增'),
    ((SELECT id FROM tenants WHERE slug = 'abc'), 'user2@abc.com', 'User Two', TRUE, 'user', 'Phase A3 批量新增'),
    ((SELECT id FROM tenants WHERE slug = 'abc'), 'admin@abc.com', 'Admin User', TRUE, 'tenant_admin', 'Phase A3 批量新增')
ON CONFLICT (tenant_id, email) DO UPDATE SET
    display_name = EXCLUDED.display_name,
    is_active = EXCLUDED.is_active,
    updated_at = NOW();

-- 方法 2：使用 DO 區塊（適合大量 10-100 個）
DO $$ 
DECLARE
    tenant_uuid UUID;
    emails TEXT[] := ARRAY[
        'user1@company.com',
        'user2@company.com',
        'user3@company.com'
        -- 繼續新增更多 email...
    ];
    email_item TEXT;
BEGIN
    -- 取得租戶 UUID
    SELECT id INTO tenant_uuid FROM tenants WHERE slug = 'abc';  -- 替換：租戶 slug
    
    -- 迴圈插入
    FOREACH email_item IN ARRAY emails
    LOOP
        INSERT INTO tenant_members (tenant_id, email, is_active, role)
        VALUES (tenant_uuid, email_item, TRUE, 'user')
        ON CONFLICT (tenant_id, email) DO NOTHING;
    END LOOP;
END $$;

-- 驗收：查看新增的成員
SELECT 
    tm.email,
    tm.display_name,
    tm.is_active,
    tm.role,
    tm.created_at
FROM tenant_members tm
JOIN tenants t ON tm.tenant_id = t.id
WHERE t.slug = 'abc'  -- 替換：租戶 slug
ORDER BY tm.created_at DESC;


-- 2.2 停用/啟用成員
-- ==========================================
-- 停用單一成員
UPDATE tenant_members
SET 
    is_active = FALSE,
    updated_at = NOW()
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'abc')  -- 替換：租戶 slug
  AND email = 'user3@abc.com';  -- 替換：成員 email

-- 啟用成員
UPDATE tenant_members
SET 
    is_active = TRUE,
    updated_at = NOW()
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'abc')  -- 替換：租戶 slug
  AND email = 'user3@abc.com';  -- 替換：成員 email

-- 批量停用多個成員
UPDATE tenant_members
SET 
    is_active = FALSE,
    updated_at = NOW()
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'abc')  -- 替換：租戶 slug
  AND email IN ('user1@abc.com', 'user2@abc.com');  -- 替換：email 清單


-- 2.3 查看租戶的所有成員
-- ==========================================
SELECT 
    tm.email,
    tm.display_name,
    tm.is_active,
    tm.role,
    tm.last_login_at,
    tm.created_at
FROM tenant_members tm
JOIN tenants t ON tm.tenant_id = t.id
WHERE t.slug = 'abc'  -- 替換：租戶 slug
ORDER BY tm.created_at DESC;


-- 2.4 統計各租戶的成員數量
-- ==========================================
SELECT 
    t.slug,
    t.name,
    COUNT(tm.id) AS total_members,
    COUNT(CASE WHEN tm.is_active THEN 1 END) AS active_members,
    COUNT(CASE WHEN NOT tm.is_active THEN 1 END) AS inactive_members
FROM tenants t
LEFT JOIN tenant_members tm ON t.id = tm.tenant_id
GROUP BY t.slug, t.name
ORDER BY total_members DESC;


-- ============================================
-- 3. 撤權操作 (Revocation)
-- ============================================

-- 3.1 單一租戶撤權（立即踢下線）
-- ==========================================
-- 步驟 1: Bump epoch
UPDATE tenant_session_epoch
SET 
    epoch = epoch + 1,
    updated_at = NOW()
WHERE tenant = 'abc';  -- 替換：租戶 slug

-- 步驟 2: 確認撤權成功
SELECT 
    tenant,
    epoch,
    updated_at,
    NOW() - updated_at AS time_since_revoke
FROM tenant_session_epoch
WHERE tenant = 'abc';  -- 替換：租戶 slug

-- 步驟 3: 記錄撤權原因（手動記錄，供未來參考）
-- 在你的文件或註解中記錄：
-- 2026-02-27 | Amanda | Revoked tenant 'abc' epoch 6→7 | Reason: Security incident


-- 3.2 批量撤權（多個租戶）
-- ==========================================
UPDATE tenant_session_epoch
SET 
    epoch = epoch + 1,
    updated_at = NOW()
WHERE tenant IN ('abc', 'acme', 'demo');  -- 替換：租戶 slug 清單

-- 確認批量撤權成功
SELECT 
    tenant,
    epoch,
    updated_at
FROM tenant_session_epoch
WHERE tenant IN ('abc', 'acme', 'demo')  -- 替換：租戶 slug 清單
ORDER BY tenant;


-- 3.3 查看最近撤權記錄（24 小時內）
-- ==========================================
SELECT 
    tenant,
    epoch,
    updated_at,
    NOW() - updated_at AS time_since_revoke
FROM tenant_session_epoch
WHERE updated_at > NOW() - INTERVAL '24 hours'
ORDER BY updated_at DESC;


-- 3.4 查看撤權頻率高的租戶
-- ==========================================
SELECT 
    tenant,
    epoch,
    updated_at,
    CASE 
        WHEN epoch = 0 THEN 'Never revoked'
        WHEN epoch < 5 THEN 'Low frequency'
        WHEN epoch < 20 THEN 'Medium frequency'
        ELSE 'High frequency (check for issues)'
    END AS revoke_frequency
FROM tenant_session_epoch
ORDER BY epoch DESC
LIMIT 20;


-- ============================================
-- 4. 權限管理 (Entitlements)
-- ============================================

-- 4.1 查看租戶的所有權限
-- ==========================================
SELECT 
    t.slug AS tenant,
    te.framework_key,
    te.is_enabled,
    te.last_modified_by,
    te.updated_at
FROM tenant_entitlements te
JOIN tenants t ON te.tenant_id = t.id
WHERE t.slug = 'abc'  -- 替換：租戶 slug
ORDER BY te.framework_key;


-- 4.2 啟用/停用特定功能
-- ==========================================
-- 停用功能
UPDATE tenant_entitlements
SET 
    is_enabled = FALSE,
    last_modified_by = 'admin@platform.com',  -- 替換：操作者 email
    updated_at = NOW()
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'abc')  -- 替換：租戶 slug
  AND framework_key = 'react';  -- 替換：功能 key

-- 啟用功能
UPDATE tenant_entitlements
SET 
    is_enabled = TRUE,
    last_modified_by = 'admin@platform.com',  -- 替換：操作者 email
    updated_at = NOW()
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'abc')  -- 替換：租戶 slug
  AND framework_key = 'react';  -- 替換：功能 key


-- 4.3 為租戶新增功能
-- ==========================================
INSERT INTO tenant_entitlements (tenant_id, framework_key, is_enabled, last_modified_by, notes)
VALUES (
    (SELECT id FROM tenants WHERE slug = 'abc'),  -- 替換：租戶 slug
    'angular',  -- 替換：功能 key
    TRUE,
    'admin@platform.com',  -- 替換：操作者 email
    'Phase A3 測試：新增功能'
)
ON CONFLICT (tenant_id, framework_key) DO UPDATE SET
    is_enabled = EXCLUDED.is_enabled,
    last_modified_by = EXCLUDED.last_modified_by,
    updated_at = NOW();


-- ============================================
-- 5. 用量管理 (Usage & Caps)
-- ============================================

-- 5.1 查看租戶的 caps 設定
-- ==========================================
SELECT 
    t.slug AS tenant,
    tc.daily_review_cap,
    tc.daily_download_cap,
    tc.max_file_size_mb,
    tc.max_rounds,
    tc.updated_at
FROM tenant_usage_caps tc
JOIN tenants t ON tc.tenant_id = t.id
WHERE t.slug = 'abc';  -- 替換：租戶 slug


-- 5.2 調整租戶的 caps
-- ==========================================
UPDATE tenant_usage_caps
SET 
    daily_review_cap = 100,  -- 替換：新的上限
    daily_download_cap = 50,  -- 替換：新的上限
    last_modified_by = 'admin@platform.com',  -- 替換：操作者 email
    updated_at = NOW()
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'abc');  -- 替換：租戶 slug


-- 5.3 查看今日用量（單一租戶）
-- ==========================================
SELECT 
    tue.usage_type,
    COUNT(*) AS usage_count,
    SUM(tue.quantity) AS total_quantity,
    MIN(tue.created_at) AS first_usage,
    MAX(tue.created_at) AS last_usage
FROM tenant_usage_events tue
JOIN tenants t ON tue.tenant_id = t.id
WHERE t.slug = 'abc'  -- 替換：租戶 slug
  AND tue.created_at >= CURRENT_DATE
GROUP BY tue.usage_type;


-- 5.4 查看所有租戶今日用量總覽
-- ==========================================
SELECT 
    t.slug AS tenant,
    tc.daily_review_cap,
    COUNT(CASE WHEN tue.usage_type = 'review' AND tue.created_at >= CURRENT_DATE THEN 1 END) AS today_review,
    tc.daily_download_cap,
    COUNT(CASE WHEN tue.usage_type = 'download' AND tue.created_at >= CURRENT_DATE THEN 1 END) AS today_download,
    CASE 
        WHEN tc.daily_review_cap IS NULL THEN 'Unlimited'
        WHEN COUNT(CASE WHEN tue.usage_type = 'review' AND tue.created_at >= CURRENT_DATE THEN 1 END) >= tc.daily_review_cap THEN '⚠️ Cap reached'
        WHEN COUNT(CASE WHEN tue.usage_type = 'review' AND tue.created_at >= CURRENT_DATE THEN 1 END)::FLOAT / tc.daily_review_cap >= 0.8 THEN '⚠️ Near limit'
        ELSE '✅ Under limit'
    END AS review_status
FROM tenants t
LEFT JOIN tenant_usage_caps tc ON t.id = tc.tenant_id
LEFT JOIN tenant_usage_events tue ON t.id = tue.tenant_id
GROUP BY t.slug, tc.daily_review_cap, tc.daily_download_cap
ORDER BY today_review DESC;


-- 5.5 清空租戶今日用量（測試用）
-- ==========================================
-- ⚠️ 警告：這會刪除今日的使用記錄，僅用於測試環境
DELETE FROM tenant_usage_events
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'abc')  -- 替換：租戶 slug
  AND created_at >= CURRENT_DATE;


-- ============================================
-- 6. 審計查詢 (Audit Logs)
-- ============================================

-- 6.1 查看特定租戶的所有活動（最近 100 筆）
-- ==========================================
SELECT 
    created_at,
    action,
    email,
    result,
    deny_reason,
    context
FROM audit_events
WHERE tenant_slug = 'abc'  -- 替換：租戶 slug
ORDER BY created_at DESC
LIMIT 100;


-- 6.2 查看所有拒絕事件（今日）
-- ==========================================
SELECT 
    created_at,
    action,
    tenant_slug,
    email,
    deny_reason,
    context
FROM audit_events
WHERE result = 'denied'
  AND created_at >= CURRENT_DATE
ORDER BY created_at DESC;


-- 6.3 按拒絕原因統計（今日）
-- ==========================================
SELECT 
    deny_reason,
    COUNT(*) AS denied_count,
    array_agg(DISTINCT tenant_slug) AS affected_tenants
FROM audit_events
WHERE result = 'denied'
  AND created_at >= CURRENT_DATE
GROUP BY deny_reason
ORDER BY denied_count DESC;


-- 6.4 查看特定用戶的活動歷史
-- ==========================================
SELECT 
    created_at,
    action,
    tenant_slug,
    result,
    deny_reason,
    context->>'ip' AS ip_address
FROM audit_events
WHERE email = 'user1@abc.com'  -- 替換：用戶 email
ORDER BY created_at DESC
LIMIT 50;


-- 6.5 查看管理操作（有 actor 的事件）
-- ==========================================
SELECT 
    created_at,
    action,
    tenant_slug,
    email,
    result,
    actor_email,
    context,
    notes
FROM audit_events
WHERE actor_email IS NOT NULL
ORDER BY created_at DESC
LIMIT 50;


-- ============================================
-- 7. 診斷查詢 (Diagnostics)
-- ============================================

-- 7.1 查詢誰在線上（最近 24 小時）
-- ==========================================
SELECT DISTINCT
    t.slug AS tenant,
    ae.email,
    MAX(ae.created_at) AS last_seen,
    NOW() - MAX(ae.created_at) AS time_since_last_seen
FROM audit_events ae
LEFT JOIN tenants t ON ae.tenant_slug = t.slug
WHERE ae.created_at > NOW() - INTERVAL '24 hours'
  AND ae.result = 'success'
  AND ae.action IN ('sso_verify', 'analyzer_launch')
GROUP BY t.slug, ae.email
ORDER BY last_seen DESC;


-- 7.2 即將到期的租戶（7 天內）
-- ==========================================
SELECT 
    slug AS tenant,
    name,
    status,
    trial_end,
    EXTRACT(DAY FROM trial_end - NOW())::INTEGER AS days_remaining
FROM tenants
WHERE trial_end IS NOT NULL
  AND trial_end > NOW()
  AND trial_end < NOW() + INTERVAL '7 days'
  AND status = 'trial'
ORDER BY trial_end;


-- 7.3 已過期但仍啟用的租戶（需處理）
-- ==========================================
SELECT 
    slug,
    name,
    status,
    trial_end,
    is_active,
    EXTRACT(DAY FROM NOW() - trial_end)::INTEGER AS days_expired
FROM tenants
WHERE trial_end < NOW()
  AND is_active = TRUE
  AND status = 'trial'
ORDER BY trial_end;


-- 7.4 接近用量上限的租戶（80% 以上）
-- ==========================================
SELECT 
    t.slug AS tenant,
    tc.daily_review_cap,
    COUNT(CASE WHEN tue.usage_type = 'review' AND tue.created_at >= CURRENT_DATE THEN 1 END) AS today_usage,
    ROUND(
        (COUNT(CASE WHEN tue.usage_type = 'review' AND tue.created_at >= CURRENT_DATE THEN 1 END)::NUMERIC 
         / NULLIF(tc.daily_review_cap, 0)) * 100,
        1
    ) AS usage_percent
FROM tenants t
LEFT JOIN tenant_usage_caps tc ON t.id = tc.tenant_id
LEFT JOIN tenant_usage_events tue ON t.id = tue.tenant_id
WHERE tc.daily_review_cap IS NOT NULL
  AND tc.daily_review_cap > 0
GROUP BY t.slug, tc.daily_review_cap
HAVING COUNT(CASE WHEN tue.usage_type = 'review' AND tue.created_at >= CURRENT_DATE THEN 1 END)::NUMERIC 
       / NULLIF(tc.daily_review_cap, 0) >= 0.8
ORDER BY usage_percent DESC;


-- 7.5 沒有成員的租戶（需補充）
-- ==========================================
SELECT 
    t.slug,
    t.name,
    t.status,
    t.created_at,
    COUNT(tm.id) AS member_count
FROM tenants t
LEFT JOIN tenant_members tm ON t.id = tm.tenant_id
GROUP BY t.slug, t.name, t.status, t.created_at
HAVING COUNT(tm.id) = 0
ORDER BY t.created_at DESC;


-- ============================================
-- 📌 快速參考：常用組合操作
-- ============================================

-- 組合 1：完整 onboarding 新租戶
-- ==========================================
-- 步驟 1-4 見上面「1.1 建立新租戶」
-- 步驟 5: 批量新增成員（見「2.1 批量新增成員」）
-- 步驟 6: 驗收
SELECT 
    t.slug,
    t.name,
    t.status,
    t.trial_end,
    COUNT(tm.id) AS member_count,
    tc.daily_review_cap
FROM tenants t
LEFT JOIN tenant_members tm ON t.id = tm.tenant_id
LEFT JOIN tenant_usage_caps tc ON t.id = tc.tenant_id
WHERE t.slug = 'new_company'  -- 替換：租戶 slug
GROUP BY t.slug, t.name, t.status, t.trial_end, tc.daily_review_cap;


-- 組合 2：緊急撤權 + 停用租戶
-- ==========================================
-- 步驟 1: Bump epoch（踢下線）
UPDATE tenant_session_epoch
SET epoch = epoch + 1, updated_at = NOW()
WHERE tenant = 'abc';  -- 替換：租戶 slug

-- 步驟 2: 停用租戶（阻止重新登入）
UPDATE tenants
SET is_active = FALSE, status = 'suspended', updated_at = NOW()
WHERE slug = 'abc';  -- 替換：租戶 slug

-- 步驟 3: 確認
SELECT 
    t.slug,
    t.is_active,
    t.status,
    tse.epoch,
    tse.updated_at AS epoch_updated_at
FROM tenants t
LEFT JOIN tenant_session_epoch tse ON t.slug = tse.tenant
WHERE t.slug = 'abc';  -- 替換：租戶 slug


-- 組合 3：月初重置所有租戶用量（如需要）
-- ==========================================
-- ⚠️ 警告：這會刪除所有歷史 usage events，建議只刪除舊資料
-- 選項 1：刪除超過 90 天的舊資料
DELETE FROM tenant_usage_events
WHERE created_at < NOW() - INTERVAL '90 days';

-- 選項 2：查看各月用量統計後再決定（推薦）
SELECT 
    DATE_TRUNC('month', created_at) AS month,
    COUNT(*) AS total_events,
    COUNT(DISTINCT tenant_id) AS active_tenants
FROM tenant_usage_events
GROUP BY month
ORDER BY month DESC;


-- ============================================
-- 結束
-- ============================================
-- 📝 使用提示：
-- 1. 所有 SQL 都經過驗收測試，可直接使用
-- 2. 替換標記為「替換：」的參數值
-- 3. 執行前建議先在測試環境驗證
-- 4. 重要操作（撤權、刪除）建議先備份或在註解中記錄原因
-- ============================================
