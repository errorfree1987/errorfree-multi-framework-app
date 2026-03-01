# 📋 Mode A 營運快速參考卡

> **目的**：最常用的 SQL 操作，可快速複製貼上執行
> 
> **完整版本**：`RUNBOOK_MODE_A_OPERATIONS.sql`（包含所有操作和詳細說明）

---

## 🚀 新租戶 Onboarding（5 分鐘）

```sql
-- 1. 建立租戶
INSERT INTO tenants (slug, name, display_name, status, trial_start, trial_end, is_active)
VALUES ('new_co', 'New Co', 'New Company Inc.', 'trial', NOW(), NOW() + INTERVAL '30 days', TRUE);

-- 2. 初始化 epoch
INSERT INTO tenant_session_epoch (tenant, epoch) VALUES ('new_co', 0) ON CONFLICT DO NOTHING;

-- 3. 設定 caps
INSERT INTO tenant_usage_caps (tenant_id, daily_review_cap, daily_download_cap)
VALUES ((SELECT id FROM tenants WHERE slug = 'new_co'), 50, 20);

-- 4. 批量新增成員
INSERT INTO tenant_members (tenant_id, email, is_active, role)
SELECT (SELECT id FROM tenants WHERE slug = 'new_co'), email, TRUE, 'user'
FROM (VALUES ('user1@new_co.com'), ('user2@new_co.com'), ('user3@new_co.com')) AS t(email)
ON CONFLICT DO NOTHING;

-- 5. 驗收
SELECT t.slug, COUNT(tm.id) AS members, tc.daily_review_cap
FROM tenants t
LEFT JOIN tenant_members tm ON t.id = tm.tenant_id
LEFT JOIN tenant_usage_caps tc ON t.id = tc.tenant_id
WHERE t.slug = 'new_co'
GROUP BY t.slug, tc.daily_review_cap;
```

---

## 🚨 緊急撤權（30 秒）

```sql
-- 1. Bump epoch（立即踢下線）
UPDATE tenant_session_epoch SET epoch = epoch + 1, updated_at = NOW() WHERE tenant = 'abc';

-- 2. 確認
SELECT tenant, epoch, updated_at FROM tenant_session_epoch WHERE tenant = 'abc';
```

---

## 👥 批量管理成員

### 新增成員（10-50 個）
```sql
DO $$ 
DECLARE
    tid UUID := (SELECT id FROM tenants WHERE slug = 'abc');
BEGIN
    INSERT INTO tenant_members (tenant_id, email, is_active, role) VALUES
        (tid, 'user1@abc.com', TRUE, 'user'),
        (tid, 'user2@abc.com', TRUE, 'user'),
        (tid, 'admin@abc.com', TRUE, 'tenant_admin')
    ON CONFLICT (tenant_id, email) DO NOTHING;
END $$;
```

### 停用成員
```sql
UPDATE tenant_members
SET is_active = FALSE, updated_at = NOW()
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'abc')
  AND email = 'user3@abc.com';
```

---

## 📊 查看使用狀況

### 今日用量總覽
```sql
SELECT 
    t.slug,
    tc.daily_review_cap,
    COUNT(CASE WHEN tue.usage_type = 'review' AND tue.created_at >= CURRENT_DATE THEN 1 END) AS today_usage,
    ROUND(100.0 * COUNT(CASE WHEN tue.usage_type = 'review' AND tue.created_at >= CURRENT_DATE THEN 1 END) / NULLIF(tc.daily_review_cap, 0), 1) AS usage_pct
FROM tenants t
LEFT JOIN tenant_usage_caps tc ON t.id = tc.tenant_id
LEFT JOIN tenant_usage_events tue ON t.id = tue.tenant_id
WHERE tc.daily_review_cap > 0
GROUP BY t.slug, tc.daily_review_cap
ORDER BY usage_pct DESC;
```

### 誰在線上（24h）
```sql
SELECT DISTINCT
    ae.tenant_slug,
    ae.email,
    MAX(ae.created_at) AS last_seen
FROM audit_events ae
WHERE ae.created_at > NOW() - INTERVAL '24 hours'
  AND ae.result = 'success'
  AND ae.action IN ('sso_verify', 'analyzer_launch')
GROUP BY ae.tenant_slug, ae.email
ORDER BY last_seen DESC;
```

---

## 🔧 調整設定

### 延長試用期
```sql
UPDATE tenants
SET trial_end = NOW() + INTERVAL '30 days', updated_at = NOW()
WHERE slug = 'abc';
```

### 調整 Caps
```sql
UPDATE tenant_usage_caps
SET daily_review_cap = 100, daily_download_cap = 50, updated_at = NOW()
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'abc');
```

### 停用/啟用租戶
```sql
-- 停用
UPDATE tenants SET is_active = FALSE, status = 'suspended' WHERE slug = 'abc';

-- 啟用
UPDATE tenants SET is_active = TRUE, status = 'trial' WHERE slug = 'abc';
```

---

## 🔍 診斷查詢

### 即將到期（7 天內）
```sql
SELECT slug, name, trial_end, EXTRACT(DAY FROM trial_end - NOW())::INTEGER AS days_left
FROM tenants
WHERE trial_end BETWEEN NOW() AND NOW() + INTERVAL '7 days'
  AND status = 'trial'
ORDER BY trial_end;
```

### 今日被拒絕的登入
```sql
SELECT created_at, tenant_slug, email, deny_reason
FROM audit_events
WHERE result = 'denied' AND created_at >= CURRENT_DATE
ORDER BY created_at DESC;
```

### 接近上限的租戶（80%+）
```sql
SELECT 
    t.slug,
    tc.daily_review_cap AS cap,
    COUNT(CASE WHEN tue.created_at >= CURRENT_DATE THEN 1 END) AS usage,
    ROUND(100.0 * COUNT(CASE WHEN tue.created_at >= CURRENT_DATE THEN 1 END) / tc.daily_review_cap, 1) AS pct
FROM tenants t
JOIN tenant_usage_caps tc ON t.id = tc.tenant_id
LEFT JOIN tenant_usage_events tue ON t.id = tue.tenant_id AND tue.usage_type = 'review'
WHERE tc.daily_review_cap > 0
GROUP BY t.slug, tc.daily_review_cap
HAVING COUNT(CASE WHEN tue.created_at >= CURRENT_DATE THEN 1 END)::FLOAT / tc.daily_review_cap >= 0.8
ORDER BY pct DESC;
```

---

## 📌 Checklist：日常營運

### 每日檢查（5 分鐘）
- [ ] 查看今日被拒絕的登入（是否有異常）
- [ ] 查看接近上限的租戶（是否需要通知）
- [ ] 查看即將到期的租戶（提前 7 天通知）

### 每週檢查（15 分鐘）
- [ ] 查看所有租戶的用量趨勢
- [ ] 查看撤權頻率高的租戶（是否有問題）
- [ ] 查看沒有成員的租戶（是否需要補充）

### 按需操作
- [ ] 新租戶 onboarding（使用上面的 5 步驟）
- [ ] 延長試用期 / 轉付費
- [ ] 緊急撤權（安全事件）
- [ ] 調整 caps（升級/降級）

---

## 🆘 緊急情況處理

### 情境 A：租戶帳號被盜用
```sql
-- 1. 立即撤權
UPDATE tenant_session_epoch SET epoch = epoch + 1 WHERE tenant = 'abc';

-- 2. 停用租戶
UPDATE tenants SET is_active = FALSE, status = 'suspended' WHERE slug = 'abc';

-- 3. 記錄原因（文件或註解）
-- 2026-02-27 | Amanda | Revoked + suspended 'abc' | Reason: Account compromised
```

### 情境 B：用量異常暴增
```sql
-- 1. 查看詳細用量
SELECT email, COUNT(*) AS count, MIN(created_at) AS first, MAX(created_at) AS last
FROM tenant_usage_events
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'abc')
  AND created_at >= CURRENT_DATE
GROUP BY email
ORDER BY count DESC;

-- 2. 如需緊急限制，降低 cap
UPDATE tenant_usage_caps
SET daily_review_cap = 10, updated_at = NOW()
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'abc');
```

---

**更多操作請參考**：`RUNBOOK_MODE_A_OPERATIONS.sql`

**最後更新**：2026-02-27 | **維護者**：Amanda Chiu
