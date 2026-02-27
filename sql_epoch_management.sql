-- ============================================
-- 多租戶緊急撤權（Epoch Revoke）SQL 腳本
-- ============================================
-- 用途：在 Supabase SQL Editor 執行，管理 tenant_session_epoch
-- 最後更新：2026-02-26
-- ============================================

-- ===== 1. 查看當前所有租戶的 epoch 狀態 =====
-- 顯示所有租戶的當前 epoch 和最後更新時間
SELECT 
    tenant,
    epoch,
    updated_at,
    NOW() - updated_at AS time_since_last_update
FROM tenant_session_epoch
ORDER BY updated_at DESC;


-- ===== 2. 查看特定租戶的 epoch =====
-- 替換 'abc' 為你要查詢的租戶 slug
SELECT tenant, epoch, updated_at 
FROM tenant_session_epoch 
WHERE tenant = 'abc';


-- ===== 3. 初始化新租戶的 epoch（如果不存在） =====
-- 為新租戶建立 epoch 記錄（預設 epoch=0）
-- 替換 'new_tenant' 為實際的租戶 slug
INSERT INTO tenant_session_epoch (tenant, epoch, updated_at)
VALUES ('new_tenant', 0, NOW())
ON CONFLICT (tenant) DO NOTHING;


-- ===== 4. 緊急撤權：增加指定租戶的 epoch =====
-- 🚨 執行後，該租戶的所有舊 session 立刻失效
-- 替換 'abc' 為要撤權的租戶 slug
UPDATE tenant_session_epoch 
SET 
    epoch = epoch + 1,
    updated_at = NOW()
WHERE tenant = 'abc';

-- 確認撤權成功
SELECT tenant, epoch, updated_at 
FROM tenant_session_epoch 
WHERE tenant = 'abc';


-- ===== 5. 緊急撤權：同時撤銷多個租戶 =====
-- 🚨 批次撤權多個租戶
UPDATE tenant_session_epoch 
SET 
    epoch = epoch + 1,
    updated_at = NOW()
WHERE tenant IN ('abc', 'acme', 'demo');

-- 確認批次撤權成功
SELECT tenant, epoch, updated_at 
FROM tenant_session_epoch 
WHERE tenant IN ('abc', 'acme', 'demo')
ORDER BY tenant;


-- ===== 6. 緊急撤權：全租戶撤權（慎用！） =====
-- ⚠️ 警告：這會讓所有租戶的現有 session 失效
-- 使用場景：安全漏洞修復後強制所有用戶重新登入
-- UPDATE tenant_session_epoch 
-- SET 
--     epoch = epoch + 1,
--     updated_at = NOW();

-- 確認全租戶撤權成功
-- SELECT COUNT(*) as total_tenants, MIN(epoch) as min_epoch, MAX(epoch) as max_epoch
-- FROM tenant_session_epoch;


-- ===== 7. 重設特定租戶的 epoch（測試用） =====
-- 用於測試環境恢復原狀
-- 替換 'abc' 和 5 為實際值
UPDATE tenant_session_epoch 
SET 
    epoch = 5,
    updated_at = NOW()
WHERE tenant = 'abc';


-- ===== 8. 查看最近被撤權的租戶（24 小時內） =====
-- 檢查最近哪些租戶執行了撤權操作
SELECT 
    tenant,
    epoch,
    updated_at,
    NOW() - updated_at AS time_since_revoke
FROM tenant_session_epoch
WHERE updated_at > NOW() - INTERVAL '24 hours'
ORDER BY updated_at DESC;


-- ===== 9. 查看撤權頻率高的租戶 =====
-- 如果某租戶 epoch 很高，可能表示頻繁撤權（需注意是否有異常）
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


-- ===== 10. 刪除測試租戶的 epoch 記錄（清理用） =====
-- ⚠️ 刪除後該租戶的 epoch 檢查會視為 0（允許訪問）
-- 僅用於清理測試數據
-- DELETE FROM tenant_session_epoch 
-- WHERE tenant = 'test_tenant';


-- ============================================
-- 🔐 安全最佳實踐
-- ============================================
-- 1. 執行撤權前先查詢確認租戶名稱正確
-- 2. 執行撤權後立刻查詢確認 epoch 已增加
-- 3. 記錄撤權操作的原因和執行人（可在註解或外部文檔）
-- 4. 撤權後通知相關用戶重新登入
-- 5. 定期審查 epoch 過高的租戶（可能表示異常）

-- ============================================
-- 📊 測試場景快速命令
-- ============================================

-- 測試場景 1: 準備測試環境
-- Step 1: 確認測試租戶存在且 epoch 已知
SELECT tenant, epoch FROM tenant_session_epoch WHERE tenant = 'abc';

-- Step 2: （從 Portal 進入 Analyzer，複製 URL 含 analyzer_session）

-- Step 3: Bump epoch（模擬撤權）
UPDATE tenant_session_epoch SET epoch = epoch + 1, updated_at = NOW() WHERE tenant = 'abc';

-- Step 4: 確認 bump 成功
SELECT tenant, epoch FROM tenant_session_epoch WHERE tenant = 'abc';

-- Step 5: （用舊 URL 訪問，應該看到 "Session revoked" 訊息）

-- Step 6: 恢復原始 epoch（如果需要）
UPDATE tenant_session_epoch SET epoch = epoch - 1, updated_at = NOW() WHERE tenant = 'abc';


-- 測試場景 2: 驗證連續撤權
-- 連續增加 3 次 epoch
UPDATE tenant_session_epoch SET epoch = epoch + 1, updated_at = NOW() WHERE tenant = 'abc';
UPDATE tenant_session_epoch SET epoch = epoch + 1, updated_at = NOW() WHERE tenant = 'abc';
UPDATE tenant_session_epoch SET epoch = epoch + 1, updated_at = NOW() WHERE tenant = 'abc';

-- 確認結果（應該增加了 3）
SELECT tenant, epoch FROM tenant_session_epoch WHERE tenant = 'abc';


-- ============================================
-- 🚨 緊急情況處理流程
-- ============================================

-- 情況 A: 發現某租戶帳號被盜用
-- 1. 立刻執行撤權
UPDATE tenant_session_epoch SET epoch = epoch + 1, updated_at = NOW() WHERE tenant = 'compromised_tenant';

-- 2. 確認撤權生效
SELECT tenant, epoch, updated_at FROM tenant_session_epoch WHERE tenant = 'compromised_tenant';

-- 3. 通知該租戶重設密碼並重新登入


-- 情況 B: 發現全局安全漏洞（例如 session secret 洩露）
-- 1. 更新 ANALYZER_SESSION_SECRET 環境變數（在 Railway）
-- 2. 全租戶撤權（可選，如果 secret 已更換，舊 token 簽章會失敗）
-- UPDATE tenant_session_epoch SET epoch = epoch + 1, updated_at = NOW();


-- 情況 C: 測試完成後恢復環境
-- 查看當前 epoch
SELECT tenant, epoch FROM tenant_session_epoch WHERE tenant = 'abc';

-- 重設到指定值（例如 5）
UPDATE tenant_session_epoch SET epoch = 5, updated_at = NOW() WHERE tenant = 'abc';

-- ============================================
-- 📝 變更歷史範例
-- ============================================
-- 2026-02-26 14:30 | Amanda | Revoked tenant 'abc' epoch 5→6 | Reason: Testing epoch revoke feature
-- 2026-02-25 10:15 | System | Revoked tenant 'acme' epoch 1→2 | Reason: Suspicious login detected
-- 2026-02-24 16:00 | Admin  | Reset tenant 'test' epoch to 0  | Reason: Test environment cleanup
