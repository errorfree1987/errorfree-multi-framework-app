# Phase B2.1: Tenant 管理實作說明

## 📋 變更摘要

本次更新實作了 **Phase B2.1：Tenant 管理（MVP 版本）**。

新增功能：
1. ✅ 租戶列表（顯示所有租戶及統計資訊）
2. ✅ 建立新租戶（表單輸入 + 自動初始化）
3. ✅ Trial 延期（延長試用期）
4. ✅ 停用/啟用租戶（切換狀態）
5. ✅ 基本統計（成員數、今日用量、epoch）
6. ✅ 所有操作記錄到 audit_events

---

## 🏗️ 功能詳細說明

### 1. 租戶列表 📋

**UI 設計**：
- 使用 Tabs 分離「列表」和「建立」
- 使用 Expander 顯示租戶詳情
- 每個租戶顯示：
  - 基本資訊（Status, Trial Period, Created）
  - 統計資訊（Members, Today's Usage, Epoch）
  - 操作按鈕（Extend Trial, Enable/Disable）

**顯示欄位**：
```python
- slug: 租戶識別碼
- name: 公司名稱
- display_name: 顯示名稱
- status: 狀態（trial / suspended / active）
- trial_start: 試用開始日期
- trial_end: 試用結束日期
- is_active: 是否啟用
- created_at: 建立時間
```

**統計資訊**：
- **Members**: 從 `tenant_members` 表計算
- **Today's Usage**: 從 `tenant_usage_events` 表計算（今日 review 次數）
- **Epoch**: 從 `tenant_session_epoch` 表讀取

---

### 2. 建立新租戶 ➕

**表單欄位**：

**基本資訊**：
- **Tenant Slug*** (必填)
  - 唯一識別碼
  - 只能包含小寫字母、數字、連字號、底線
  - 範例：`acme-corp`, `new_company`
  
- **Name*** (必填)
  - 公司名稱
  - 範例：`Acme Corporation`
  
- **Display Name** (選填)
  - 顯示名稱
  - 未填寫時使用 Name
  
- **Trial Days**
  - 試用天數
  - 預設：30 天
  - 範圍：1-365 天

**預設設定**：
- **Daily Review Cap**
  - 每日 review 上限
  - 預設：50
  - 0 = 停用，空白 = 無限制
  
- **Daily Download Cap**
  - 每日 download 上限
  - 預設：20
  - 0 = 停用，空白 = 無限制

**自動初始化**：
建立租戶時會自動：
1. 建立 `tenants` 記錄
2. 初始化 `tenant_session_epoch` (epoch = 0)
3. 設定 `tenant_usage_caps`
4. 記錄 `tenant_created` audit event

**驗證規則**：
- Slug 必填
- Name 必填
- Slug 只能包含字母、數字、連字號、底線
- Slug 必須小寫

---

### 3. Trial 延期 📅

**功能**：
- 在租戶詳情中，可以延長試用期
- 輸入延長天數（1-365 天）
- 自動計算新的結束日期

**操作流程**：
1. 展開租戶詳情
2. 在 "Extend Trial" 區域輸入天數
3. 點擊 "📅 Extend" 按鈕
4. 系統更新 `trial_end` 欄位
5. 記錄 `tenant_trial_extended` audit event
6. 顯示成功訊息和新日期

**範例**：
```
當前結束日期：2026-03-31
延長天數：30
新結束日期：2026-04-30
```

---

### 4. 停用/啟用租戶 🔄

**功能**：
- 快速切換租戶的啟用狀態
- 停用後，租戶無法登入使用
- 啟用後，租戶可以正常使用

**操作流程**：
1. 展開租戶詳情
2. 點擊 "🔴 Disable Tenant" 或 "🟢 Enable Tenant" 按鈕
3. 系統更新：
   - `is_active`: true/false
   - `status`: trial/suspended
4. 記錄 `tenant_enabled` 或 `tenant_disabled` audit event

**狀態顯示**：
- 🟢 Active（is_active = true）
- 🔴 Suspended（is_active = false）

---

### 5. 基本統計 📊

**統計項目**：

#### Members（成員數）
```sql
SELECT COUNT(*) 
FROM tenant_members 
WHERE tenant_id = ?
```

#### Today's Usage（今日用量）
```sql
SELECT COUNT(*) 
FROM tenant_usage_events 
WHERE tenant_id = ? 
  AND usage_type = 'review'
  AND created_at >= CURRENT_DATE
```

#### Epoch（版本號）
```sql
SELECT epoch 
FROM tenant_session_epoch 
WHERE tenant = ?
```

**顯示方式**：
使用 `st.metric()` 顯示，清晰易讀。

---

## 🔧 技術實作

### 程式碼結構

```python
admin_ui.py
├── show_tenants()                    # 主要函數
│   ├── Tab 1: Tenant List
│   └── Tab 2: Create New Tenant
├── show_tenant_details()             # 顯示租戶詳情
├── get_tenant_member_count()         # 獲取成員數
├── get_tenant_today_usage()          # 獲取今日用量
├── get_tenant_epoch()                # 獲取 epoch
├── create_tenant()                   # 建立租戶
├── extend_tenant_trial()             # 延長試用期
└── toggle_tenant_status()            # 切換狀態
```

### 使用的 Supabase 表

| 表名 | 操作 | 用途 |
|------|------|------|
| `tenants` | SELECT, INSERT, PATCH | 租戶主表 |
| `tenant_session_epoch` | SELECT, INSERT | Epoch 管理 |
| `tenant_usage_caps` | INSERT | 用量上限設定 |
| `tenant_members` | SELECT | 成員統計 |
| `tenant_usage_events` | SELECT | 用量統計 |
| `audit_events` | INSERT | 審計日誌 |

### API 呼叫範例

#### 獲取所有租戶
```python
GET {SUPABASE_URL}/rest/v1/tenants
Headers:
  - apikey: {SERVICE_KEY}
  - Authorization: Bearer {SERVICE_KEY}
Params:
  - select: id,slug,name,display_name,status,trial_start,trial_end,is_active,created_at
  - order: created_at.desc
```

#### 建立租戶
```python
POST {SUPABASE_URL}/rest/v1/tenants
Headers:
  - apikey: {SERVICE_KEY}
  - Authorization: Bearer {SERVICE_KEY}
  - Content-Type: application/json
  - Prefer: return=representation
Body:
{
  "slug": "acme",
  "name": "Acme Corp",
  "display_name": "Acme Corporation",
  "status": "trial",
  "trial_start": "2026-03-01T00:00:00Z",
  "trial_end": "2026-03-31T00:00:00Z",
  "is_active": true
}
```

#### 更新租戶
```python
PATCH {SUPABASE_URL}/rest/v1/tenants?id=eq.{tenant_id}
Headers:
  - apikey: {SERVICE_KEY}
  - Authorization: Bearer {SERVICE_KEY}
  - Content-Type: application/json
Body:
{
  "trial_end": "2026-04-30T00:00:00Z"
}
```

---

## 🧪 測試驗收

### 測試案例 1：建立新租戶 ✅

**步驟**：
1. 登入 Admin UI
2. 點擊 "🏢 Tenants" 導覽
3. 切換到 "➕ Create New Tenant" tab
4. 填寫表單：
   - Tenant Slug: `test-company`
   - Name: `Test Company`
   - Display Name: `Test Company Inc.`
   - Trial Days: `30`
   - Daily Review Cap: `50`
   - Daily Download Cap: `20`
5. 點擊 "🚀 Create Tenant"

**預期結果**：
- ✅ 顯示 "✅ Tenant created: test-company"
- ✅ 顯示 "✅ Epoch initialized"
- ✅ 顯示 "✅ Usage caps configured"
- ✅ 顯示 "🎉 Tenant setup complete!"
- ✅ 頁面重新載入，在租戶列表中看到新租戶

**驗證 Supabase**：
```sql
-- 檢查租戶
SELECT * FROM tenants WHERE slug = 'test-company';

-- 檢查 epoch
SELECT * FROM tenant_session_epoch WHERE tenant = 'test-company';

-- 檢查 caps
SELECT * FROM tenant_usage_caps 
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'test-company');

-- 檢查 audit log
SELECT * FROM audit_events 
WHERE action = 'tenant_created' 
  AND tenant_slug = 'test-company';
```

---

### 測試案例 2：查看租戶列表 ✅

**步驟**：
1. 登入 Admin UI
2. 點擊 "🏢 Tenants"
3. 查看 "📋 Tenant List" tab

**預期結果**：
- ✅ 顯示所有租戶
- ✅ 每個租戶有 Expander
- ✅ Expander 標題格式：`slug - name`
- ✅ 顯示租戶數量："✅ Found X tenant(s)"

**驗證租戶詳情**：
1. 展開任一租戶
2. 確認顯示：
   - ✅ Status（🟢 或 🔴）
   - ✅ Trial Period（開始 → 結束日期）
   - ✅ Created（建立日期）
   - ✅ Members（成員數）
   - ✅ Today's Usage（今日用量）
   - ✅ Epoch（版本號）

---

### 測試案例 3：延長試用期 ✅

**步驟**：
1. 在租戶列表中展開任一租戶
2. 在 "Extend Trial" 區域輸入天數：`30`
3. 點擊 "📅 Extend"

**預期結果**：
- ✅ 顯示 "✅ Trial extended by 30 days"
- ✅ 顯示 "New end date: YYYY-MM-DD"
- ✅ 頁面重新載入
- ✅ Trial Period 已更新

**驗證 Supabase**：
```sql
-- 檢查 trial_end 已更新
SELECT slug, trial_end FROM tenants WHERE slug = 'test-company';

-- 檢查 audit log
SELECT * FROM audit_events 
WHERE action = 'tenant_trial_extended' 
  AND tenant_slug = 'test-company'
ORDER BY created_at DESC LIMIT 1;
```

---

### 測試案例 4：停用租戶 ✅

**步驟**：
1. 在租戶列表中展開任一租戶（is_active = true）
2. 點擊 "🔴 Disable Tenant"

**預期結果**：
- ✅ 顯示 "✅ Tenant disabled"
- ✅ 頁面重新載入
- ✅ Status 變成 🔴 suspended
- ✅ 按鈕變成 "🟢 Enable Tenant"

**驗證 Supabase**：
```sql
-- 檢查狀態已更新
SELECT slug, is_active, status FROM tenants WHERE slug = 'test-company';

-- 檢查 audit log
SELECT * FROM audit_events 
WHERE action = 'tenant_disabled' 
  AND tenant_slug = 'test-company'
ORDER BY created_at DESC LIMIT 1;
```

---

### 測試案例 5：啟用租戶 ✅

**步驟**：
1. 在租戶列表中展開已停用的租戶
2. 點擊 "🟢 Enable Tenant"

**預期結果**：
- ✅ 顯示 "✅ Tenant enabled"
- ✅ 頁面重新載入
- ✅ Status 變成 🟢 trial
- ✅ 按鈕變成 "🔴 Disable Tenant"

**驗證 Supabase**：
```sql
-- 檢查狀態已更新
SELECT slug, is_active, status FROM tenants WHERE slug = 'test-company';

-- 檢查 audit log
SELECT * FROM audit_events 
WHERE action = 'tenant_enabled' 
  AND tenant_slug = 'test-company'
ORDER BY created_at DESC LIMIT 1;
```

---

### 測試案例 6：表單驗證 ✅

**測試 6.1：Slug 必填**
- 步驟：不填寫 Slug，點擊 Create
- 預期：❌ Slug and Name are required

**測試 6.2：Slug 格式錯誤**
- 步驟：Slug 輸入 `Test Company`（有空格和大寫）
- 預期：❌ Slug must contain only letters, numbers, hyphens, and underscores

**測試 6.3：Slug 必須小寫**
- 步驟：Slug 輸入 `TestCompany`（大寫）
- 預期：❌ Slug must be lowercase

**測試 6.4：重複的 Slug**
- 步驟：Slug 輸入已存在的值
- 預期：❌ Failed to create tenant: HTTP 409（Supabase 會返回衝突錯誤）

---

## 📊 Audit Events

所有操作都會記錄到 `audit_events` 表：

| Action | 觸發時機 | Context 內容 |
|--------|---------|-------------|
| `tenant_created` | 建立租戶成功 | tenant_id, trial_days, daily_review_cap, daily_download_cap |
| `tenant_trial_extended` | 延長試用期 | extend_days, new_trial_end |
| `tenant_enabled` | 啟用租戶 | new_status: true |
| `tenant_disabled` | 停用租戶 | new_status: false |

**查詢所有租戶操作**：
```sql
SELECT 
    created_at,
    action,
    tenant_slug,
    email AS actor,
    result,
    context
FROM audit_events
WHERE action IN ('tenant_created', 'tenant_trial_extended', 'tenant_enabled', 'tenant_disabled')
ORDER BY created_at DESC
LIMIT 20;
```

---

## 🔄 與現有功能整合

### Phase A 的關聯

Phase B2.1 使用 Phase A 建立的表：
- ✅ `tenants` - Phase A1 建立
- ✅ `tenant_session_epoch` - Phase A1 建立
- ✅ `tenant_usage_caps` - Phase A1 建立
- ✅ `tenant_members` - Phase A1 建立
- ✅ `tenant_usage_events` - Phase A1 建立
- ✅ `audit_events` - Phase A1 建立

### Enforcement 的影響

當租戶被停用（is_active = false）：
- ✅ Phase A2-1 會阻擋 Portal SSO 登入
- ✅ 顯示錯誤："Access denied: Your organization's account is currently inactive"
- ✅ 記錄 `sso_verify` denied audit event

---

## 🎯 Phase B2.1 驗收清單

- [ ] 可以查看所有租戶列表
- [ ] 可以建立新租戶（表單驗證正確）
- [ ] 租戶建立後自動初始化 epoch 和 caps
- [ ] 可以延長租戶試用期
- [ ] 可以停用/啟用租戶
- [ ] 統計資訊正確顯示（Members, Usage, Epoch）
- [ ] 所有操作記錄到 audit_events
- [ ] UI 清晰易用
- [ ] 錯誤處理完整

---

## 🚀 下一步：Phase B3

完成 Phase B2.1 驗收後，下一步將實作：

### Phase B3.1: Members 批量管理（預計 0.5-1 天）

**功能**：
- 成員列表（按租戶篩選）
- 批量新增成員（貼上 email 清單）
- 批量停用/啟用成員
- 角色設定（user / tenant_admin）

---

## 📁 相關檔案

| 檔案 | 說明 |
|------|------|
| `admin_ui.py` | 主程式（已修改 show_tenants 函數）|
| `README_PHASE_B2_1.md` | 本檔案 |
| `ROADMAP.md` | 完整路線圖（需要更新）|

---

**最後更新**：2026-03-01  
**維護者**：Amanda Chiu  
**Phase**：B2.1 (MVP Tenant Management)
