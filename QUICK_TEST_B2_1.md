# 🧪 Phase B2.1 快速測試指南

## ⏱️ 預計時間：5-10 分鐘

---

## 🚀 部署狀態

**程式碼已推送**：
```
Commit: 9560fe9
Message: "Phase B2.1: Implement Tenant Management"
Branch: staging-portal-sso
```

**Railway 會自動部署**（約 1-2 分鐘）

---

## 📋 快速測試清單

### 步驟 1：等待部署（1-2 分鐘）

1. **進入 Railway Dashboard**
   - https://railway.app/dashboard
   - 選擇 **errorfree-techincal review-app-staging** 專案內的服務

2. **查看部署狀態**
   - Deployments 標籤
   - 確認最新部署狀態為 "Active"

3. **檢查 Build Logs**（選填）
   - 確認沒有錯誤
   - 確認 `streamlit run admin_ui.py` 正常執行

---

### 步驟 2：訪問 Tenants 頁面（1 分鐘）

1. **訪問 Admin UI**
   ```
   https://errorfree-techincal-review-app-staging-production.up.railway.app
   ```

2. **登入**（如果還沒登入）
   - 輸入 ADMIN_PASSWORD

3. **點擊左側導覽**
   - 點擊 "🏢 Tenants"

4. **確認頁面顯示**
   - ✅ 看到 "🏢 Tenant Management" 標題
   - ✅ 看到兩個 tabs："📋 Tenant List" 和 "➕ Create New Tenant"
   - ✅ 不再是「Coming in Phase B2」的佔位符

---

### 步驟 3：測試建立租戶（2-3 分鐘）⭐

1. **切換到 "➕ Create New Tenant" tab**

2. **填寫表單**：
   ```
   Tenant Slug: test-demo
   Name: Test Demo Company
   Display Name: Test Demo Inc.
   Trial Days: 30
   Daily Review Cap: 50
   Daily Download Cap: 20
   ```

3. **點擊 "🚀 Create Tenant"**

4. **✅ 確認成功訊息**：
   - "✅ Tenant created: test-demo"
   - "✅ Epoch initialized"
   - "✅ Usage caps configured"
   - "🎉 Tenant setup complete!"
   - 🎈 看到氣球動畫

5. **頁面自動重新載入**
   - 自動切換到 "📋 Tenant List" tab
   - 看到新建立的租戶

---

### 步驟 4：測試租戶列表（1-2 分鐘）

1. **查看租戶列表**
   - ✅ 顯示 "✅ Found X tenant(s)"
   - ✅ 看到 `test-demo` 租戶

2. **展開租戶詳情**
   - 點擊租戶的 expander
   - ✅ 看到基本資訊（Status, Trial Period, Created）
   - ✅ 看到統計（Members: 0, Today's Usage: 0, Epoch: 0）
   - ✅ 看到操作按鈕（Extend Trial, Disable Tenant）

---

### 步驟 5：測試延長試用期（1 分鐘）

1. **在租戶詳情中**
   - 找到 "Extend Trial" 區域
   - 輸入天數：`30`
   - 點擊 "📅 Extend"

2. **✅ 確認成功**：
   - 看到 "✅ Trial extended by 30 days"
   - 看到 "New end date: YYYY-MM-DD"
   - 頁面重新載入
   - Trial Period 已更新（結束日期往後 30 天）

---

### 步驟 6：測試停用/啟用租戶（1 分鐘）

1. **停用租戶**
   - 展開租戶
   - 點擊 "🔴 Disable Tenant"
   - ✅ 看到 "✅ Tenant disabled"
   - Status 變成 🔴 suspended
   - 按鈕變成 "🟢 Enable Tenant"

2. **啟用租戶**
   - 點擊 "🟢 Enable Tenant"
   - ✅ 看到 "✅ Tenant enabled"
   - Status 變回 🟢 trial
   - 按鈕變回 "🔴 Disable Tenant"

---

## ✅ 測試成功標準

完成以上步驟後，確認：

- [ ] ✅ Tenants 頁面正常顯示
- [ ] ✅ 可以建立新租戶
- [ ] ✅ 租戶列表顯示正確
- [ ] ✅ 可以展開租戶詳情
- [ ] ✅ 統計資訊正確（Members, Usage, Epoch）
- [ ] ✅ 可以延長試用期
- [ ] ✅ 可以停用租戶
- [ ] ✅ 可以啟用租戶
- [ ] ✅ 所有操作有成功訊息
- [ ] ✅ 頁面重新載入後資料持久

---

## 🔍 進階驗證（選填）

### 驗證 Supabase 資料

1. **進入 Supabase Dashboard**
   - https://supabase.com/dashboard

2. **開啟 SQL Editor**
   - 左側選單 → SQL Editor

3. **執行查詢**：

```sql
-- 查看新建立的租戶
SELECT * FROM tenants WHERE slug = 'test-demo';

-- 查看 epoch
SELECT * FROM tenant_session_epoch WHERE tenant = 'test-demo';

-- 查看 usage caps
SELECT tc.* 
FROM tenant_usage_caps tc
JOIN tenants t ON tc.tenant_id = t.id
WHERE t.slug = 'test-demo';

-- 查看 audit log
SELECT 
    created_at,
    action,
    tenant_slug,
    email,
    result,
    context
FROM audit_events
WHERE tenant_slug = 'test-demo'
ORDER BY created_at DESC;
```

**✅ 預期結果**：
- 租戶存在且資料正確
- Epoch = 0
- Usage caps 已設定
- Audit log 包含：
  - `tenant_created`
  - `tenant_trial_extended`
  - `tenant_disabled`
  - `tenant_enabled`

---

## 🎉 測試完成！

如果所有測試都通過，恭喜你！**Phase B2.1 完成並驗收通過！** ✅

**現在你的 Admin UI 有**：
- ✅ 登入/權限（Phase B1.1）
- ✅ 租戶管理（Phase B2.1）

---

## 🚀 下一步

**Phase B3.1: Members 批量管理**（預計 0.5-1 天）

準備好了告訴我，我會開始實作：
- 成員列表（按租戶篩選）
- 批量新增成員（貼上 email 清單）
- 批量停用/啟用成員
- 角色設定

---

## 🆘 遇到問題？

### Q: 部署失敗，Deployments 顯示錯誤

**A**: 
1. 查看 Deploy Logs
2. 常見問題：
   - 語法錯誤（檢查 Python 語法）
   - Import 錯誤（檢查 imports）
   - 環境變數錯誤

### Q: 頁面顯示 "Supabase not configured"

**A**: 
1. 確認 Railway Variables 有：
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_KEY`
2. 重新部署

### Q: 建立租戶時顯示 HTTP 409 錯誤

**A**: 
- Slug 已存在
- 換一個不同的 slug

### Q: 統計資訊都顯示 0

**A**: 
- 這是正常的（新租戶沒有成員和用量）
- 可以之後在 Phase B3 新增成員測試

---

**建立時間**：2026-03-01  
**維護者**：Amanda Chiu
