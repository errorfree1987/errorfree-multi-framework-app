# 🔧 Phase B2.1 修復和新功能測試指南

## 📋 本次更新內容

### 修復的問題：
1. ✅ Usage caps configured 訊息現在會顯示
2. ✅ Tenant setup complete 訊息現在會顯示  
3. ✅ 延長顯示時間到 3 秒
4. ✅ 改進錯誤處理（顯示詳細錯誤訊息）

### 新增功能：
5. ✅ 刪除租戶功能（需要輸入 slug 確認）

---

## 🚀 部署狀態

**程式碼已推送**：
```
Commit: 2a87269
Message: "Fix: Improve tenant creation feedback and add delete function"
Branch: staging-portal-sso
```

**Railway 會自動部署**（約 1-2 分鐘）

---

## 🧪 測試步驟

### 步驟 1：等待部署（1-2 分鐘）

1. 進入 Railway Dashboard
2. 查看 `trustworthy-analysis` 服務
3. 確認最新部署狀態為 "Active"

---

### 步驟 2：測試建立租戶（修復驗證）⭐

1. **訪問 Admin UI 並登入**
2. **點擊 "🏢 Tenants"**
3. **切換到 "➕ Create New Tenant" tab**
4. **填寫表單**：
   ```
   Tenant Slug: test-fixed
   Name: Test Fixed Company
   Display Name: Test Fixed Inc.
   Trial Days: 30
   Daily Review Cap: 50
   Daily Download Cap: 20
   ```
5. **點擊 "🚀 Create Tenant"**

**✅ 現在應該會看到（按順序）**：
1. ✅ Tenant created: test-fixed
2. ✅ Epoch initialized
3. ✅ **Usage caps configured** ← 之前沒顯示，現在會顯示
4. 🎉 **Tenant setup complete!** ← 之前沒顯示，現在會顯示
5. 🎈 氣球動畫
6. （等待 3 秒）
7. 自動切換到 "📋 Tenant List" tab（雖然 Streamlit 不支援程式化切換，但重新載入後會在列表 tab）

**如果某個步驟失敗**：
- 現在會顯示 ⚠️ 警告訊息
- 會顯示 HTTP 錯誤碼和詳細錯誤

---

### 步驟 3：測試刪除租戶（新功能）🗑️

1. **在租戶列表中展開剛才建立的租戶** (`test-fixed`)
2. **找到第 4 個欄位："⚠️ Delete Tenant"**
3. **在文字框中輸入租戶 slug**：`test-fixed`
4. **點擊 "🗑️ Delete" 按鈕**

**✅ 預期結果**：
- ✅ 顯示 "✅ Tenant 'test-fixed' deleted successfully"
- （等待 2 秒）
- 頁面重新載入
- 租戶從列表中消失

**如果輸入錯誤的 slug**：
- ❌ 顯示 "❌ Slug mismatch. Type 'test-fixed' to confirm."
- 租戶不會被刪除

---

### 步驟 4：測試誤操作防護

1. **建立一個新租戶** (slug: `test-delete-protection`)
2. **展開租戶詳情**
3. **在刪除欄位輸入錯誤的 slug**（例如：`wrong-slug`）
4. **點擊 "🗑️ Delete"**

**✅ 預期結果**：
- ❌ 顯示錯誤訊息
- 租戶沒有被刪除
- 保護機制有效

---

### 步驟 5：測試其他功能（確認沒有破壞）

#### 5.1 延長試用期
1. 展開任一租戶
2. 輸入天數：30
3. 點擊 "📅 Extend"
4. ✅ 應該看到成功訊息並更新

#### 5.2 停用/啟用租戶
1. 展開任一租戶
2. 點擊 "🔴 Disable" 或 "🟢 Enable"
3. ✅ 應該看到成功訊息並更新狀態

---

## 📊 完整測試清單

- [ ] 步驟 1：部署完成 ✅
- [ ] 步驟 2：建立租戶
  - [ ] ✅ Tenant created 訊息
  - [ ] ✅ Epoch initialized 訊息
  - [ ] ✅ **Usage caps configured 訊息**（修復重點）
  - [ ] 🎉 **Tenant setup complete 訊息**（修復重點）
  - [ ] 🎈 氣球動畫
  - [ ] 等待 3 秒後重新載入
- [ ] 步驟 3：刪除租戶
  - [ ] 輸入正確 slug 可以刪除
  - [ ] 顯示成功訊息
  - [ ] 租戶從列表消失
- [ ] 步驟 4：誤操作防護
  - [ ] 輸入錯誤 slug 無法刪除
  - [ ] 顯示錯誤訊息
- [ ] 步驟 5：其他功能
  - [ ] 延長試用期正常
  - [ ] 停用/啟用正常

---

## 🔍 驗證 Audit Log（選填）

在 Supabase SQL Editor 執行：

```sql
-- 查看租戶操作歷史
SELECT 
    created_at,
    action,
    tenant_slug,
    email AS actor,
    result,
    context
FROM audit_events
WHERE action IN ('tenant_created', 'tenant_deleted')
ORDER BY created_at DESC
LIMIT 10;
```

**✅ 預期結果**：
- 看到 `tenant_created` 記錄
- 看到 `tenant_deleted` 記錄
- Context 包含 tenant_id 和相關資訊

---

## 🎯 關鍵改進說明

### 1. 改進錯誤處理

**之前**：
```python
# 如果失敗，沒有訊息
resp = requests.post(...)
if resp.status_code == 201:
    st.success("✅ Usage caps configured")
```

**現在**：
```python
# 如果失敗，顯示警告和詳細資訊
try:
    resp = requests.post(...)
    if resp.status_code == 201:
        st.success("✅ Usage caps configured")
    else:
        st.warning(f"⚠️ Usage caps setup: HTTP {resp.status_code} - {resp.text}")
except Exception as e:
    st.warning(f"⚠️ Usage caps error: {str(e)}")
```

### 2. 刪除功能的安全設計

**防護措施**：
1. 需要輸入 slug 確認（防止誤點）
2. 按順序刪除關聯資料（避免外鍵錯誤）
3. 記錄到 audit log（追蹤責任）
4. 顯示成功訊息（確認操作完成）

**刪除順序**：
```
1. tenant_usage_events    (使用記錄)
2. tenant_usage_caps      (用量上限)
3. tenant_members         (成員)
4. tenant_session_epoch   (session 版本)
5. tenants                (主表)
```

---

## 🆘 常見問題

### Q: 為什麼還是看不到 "Usage caps configured"？

**A**: 可能的原因：
1. **Supabase API 錯誤**：
   - 檢查 Supabase 是否正常運作
   - 檢查 SERVICE_KEY 權限
   
2. **資料庫限制**：
   - `tenant_usage_caps` 表可能有唯一限制
   - 如果 tenant_id 已存在，會失敗
   
3. **解決方法**：
   - 現在會顯示 ⚠️ 警告訊息和詳細錯誤
   - 查看錯誤訊息來診斷問題

### Q: 刪除租戶後還能恢復嗎？

**A**: 不能。刪除是永久性的。
- 所有關聯資料都會被刪除
- 沒有恢復機制
- 這就是為什麼需要輸入 slug 確認

### Q: 可以刪除有成員的租戶嗎？

**A**: 可以。
- 刪除程序會先刪除所有成員
- 然後刪除租戶主表
- 確保沒有孤立資料

---

## ✨ 測試成功後

如果所有測試都通過：

**Phase B2.1 功能完整清單**：
- ✅ 租戶列表
- ✅ 建立新租戶（含完整反饋）
- ✅ 延長試用期
- ✅ 停用/啟用租戶
- ✅ **刪除租戶**（新增）
- ✅ 基本統計
- ✅ Audit logging

**準備下一步**：Phase B3.1 - Members 批量管理

---

**建立時間**：2026-03-01  
**維護者**：Amanda Chiu  
**版本**：Phase B2.1 v2 (with delete feature)
