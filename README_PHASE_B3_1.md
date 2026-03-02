# Phase B3.1: Members 批量管理（MVP 版本）

## 📋 概述

**目標**：實作成員批量管理功能，讓 Admin/OPS 能快速管理租戶成員

**預計時間**：0.5-1 天

**狀態**：✅ 已完成（2026-03-02）

---

## 🎯 功能需求

### 核心功能

1. **成員列表**
   - 按租戶篩選查看
   - 顯示成員狀態（Active/Inactive）
   - 顯示統計資訊（總數、啟用、停用）

2. **批量新增成員**
   - 貼上 email 清單（一行一個）
   - 手動單個新增
   - 設定角色（user/admin）
   - 自動設定為 Active

3. **批量操作**
   - 多選成員
   - 批量啟用/停用
   - 防止誤操作

4. **單個成員管理**
   - 查看詳細資訊
   - 切換啟用/停用狀態
   - 更改角色

---

## 🏗️ 架構設計

### UI 結構

```
👥 Members
├── 📋 Member List (Tab 1)
│   ├── 租戶篩選器
│   ├── 統計資訊
│   └── 成員列表（Expander）
│       ├── 基本資訊
│       └── 操作按鈕
│
├── ➕ Batch Add Members (Tab 2)
│   ├── 選擇租戶
│   ├── 輸入方式選擇
│   │   ├── Paste Emails（文字區域）
│   │   └── Manual Entry（表單）
│   └── 新增按鈕
│
└── ⚙️ Batch Operations (Tab 3)
    ├── 選擇租戶
    ├── 多選成員
    └── 批量操作按鈕
        ├── Enable Selected
        └── Disable Selected
```

### 資料庫操作

**使用的表**：`tenant_members`

**欄位**：
- `id` (uuid, PK)
- `tenant_slug` (text, FK)
- `email` (text)
- `role` (text: 'user' / 'admin')
- `is_active` (boolean)
- `created_at` (timestamp)

---

## 💻 實作細節

### 主要函數

#### 1. `show_members()`
主入口函數，顯示 3 個分頁。

#### 2. `show_member_list()`
顯示成員列表頁面：
- 租戶篩選器（All Tenants / 特定租戶）
- 統計資訊（Total / Active / Inactive）
- 成員列表（Expander 顯示詳細資訊）

#### 3. `show_member_details()`
顯示單個成員的詳細資訊和操作：
- 基本資訊（Email, Tenant, Role, Status, Created）
- 操作按鈕（Enable/Disable, Change Role）

#### 4. `show_batch_add_members()`
批量新增成員頁面：
- 選擇租戶
- 兩種輸入方式：
  - **Paste Emails**: `st.text_area()`，一行一個 email
  - **Manual Entry**: `st.form()`，單個新增
- 自動過濾無效 email（基本驗證：包含 `@`）

#### 5. `show_batch_operations()`
批量操作頁面：
- 選擇租戶
- `st.multiselect()` 多選成員
- 批量 Enable/Disable 按鈕

### Helper Functions

#### `get_all_tenants()`
獲取所有租戶列表（用於下拉選單）。

#### `get_members()`
獲取成員列表：
- 可選參數 `tenant_slug`
- 如果提供 `tenant_slug`，只返回該租戶的成員
- 如果不提供，返回所有成員

#### `batch_add_members()`
批量新增成員：
- 接受 email 列表
- 批量插入到 `tenant_members`
- 處理重複 email（HTTP 409）
- 記錄 `members_batch_added` audit event
- 成功後顯示 `st.balloons()`

#### `toggle_member_status()`
切換單個成員的狀態：
- 更新 `is_active` 欄位
- 記錄 `member_enabled` / `member_disabled` audit event
- 延遲 1.5 秒後重載

#### `update_member_role()`
更新成員角色：
- 更新 `role` 欄位
- 記錄 `member_role_updated` audit event
- 延遲 1.5 秒後重載

#### `batch_toggle_members()`
批量切換成員狀態：
- 迭代處理每個選中的 email
- 記錄 `members_batch_enabled` / `members_batch_disabled` audit event
- 顯示成功數量

---

## 🔒 安全考量

### 1. Email 驗證
- 基本驗證：檢查是否包含 `@`
- 自動轉換為小寫
- 去除首尾空白

### 2. 重複處理
- Supabase 會自動處理重複 email（unique constraint）
- HTTP 409 時顯示警告訊息
- 不會中斷流程

### 3. Audit Logging
所有操作都記錄到 `audit_events`：
- `members_batch_added`
- `member_enabled`
- `member_disabled`
- `member_role_updated`
- `members_batch_enabled`
- `members_batch_disabled`

Context 包含：
- 操作的 email 列表
- 成員數量
- 角色變更（如適用）

---

## 🎨 UI/UX 設計

### 視覺元素

1. **狀態指示器**
   - ✅ Active（綠色勾）
   - ❌ Inactive（紅色叉）
   - 🟢 Enable 按鈕（綠色）
   - 🔴 Disable 按鈕（紅色）

2. **統計卡片**
   - `st.metric()` 顯示：
     - Total Members
     - Active Members
     - Inactive Members

3. **反饋訊息**
   - ✅ 成功：`st.success()` + `st.balloons()`
   - ⚠️ 警告：`st.warning()`（重複 email）
   - ❌ 錯誤：`st.error()`
   - ℹ️ 提示：`st.info()`

### 用戶流程

#### 流程 1：批量新增成員（貼上 email）
```
1. 進入 "Batch Add Members" 分頁
2. 選擇租戶
3. 選擇 "Paste Emails"
4. 貼上 email 清單（一行一個）
5. 點擊 "Add All Members"
6. 看到成功訊息 + 氣球動畫
7. 頁面自動重載，表單清空
```

#### 流程 2：批量停用成員
```
1. 進入 "Batch Operations" 分頁
2. 選擇租戶
3. 用 multiselect 選擇要停用的成員
4. 點擊 "Disable Selected"
5. 看到成功訊息（例如："✅ 5 member(s) disabled!"）
6. 頁面自動重載
```

#### 流程 3：單個成員管理
```
1. 進入 "Member List" 分頁
2. 選擇租戶（或 "All Tenants"）
3. 展開成員的 expander
4. 查看詳細資訊
5. 點擊 "Enable" / "Disable" 或更改角色
6. 看到成功訊息
7. 頁面自動重載
```

---

## 🧪 測試用例

### 測試 1：批量新增成員（貼上 email）

**前置條件**：
- 至少有一個租戶存在（例如：`test-demo`）

**步驟**：
1. 進入 "👥 Members" 頁面
2. 切換到 "➕ Batch Add Members" 分頁
3. 選擇租戶：`test-demo`
4. 選擇 "Paste Emails"
5. 貼上以下 email：
   ```
   alice@example.com
   bob@example.com
   carol@example.com
   dave@example.com
   eve@example.com
   ```
6. 點擊 "➕ Add All Members"

**預期結果**：
- ✅ 看到：「✅ Successfully added 5 member(s)!」
- 🎈 看到氣球動畫
- ⏱️ 2 秒後頁面自動重載
- 📋 切換到 "Member List" 可看到新成員

---

### 測試 2：批量新增成員（重複 email）

**前置條件**：
- 測試 1 已完成，`alice@example.com` 已存在

**步驟**：
1. 再次進入 "Batch Add Members"
2. 選擇相同租戶：`test-demo`
3. 貼上包含重複 email 的清單：
   ```
   alice@example.com
   frank@example.com
   ```
4. 點擊 "Add All Members"

**預期結果**：
- ⚠️ 看到：「⚠️ Some members already exist. Skipping duplicates...」
- ℹ️ 看到：「💡 Tip: Existing members were not modified.」
- `frank@example.com` 應該成功新增

---

### 測試 3：按租戶篩選成員

**前置條件**：
- 至少有 2 個租戶，每個租戶有成員

**步驟**：
1. 進入 "Member List"
2. 選擇 "All Tenants"
3. 記下總成員數量（例如：10）
4. 選擇特定租戶（例如：`test-demo`）
5. 記下該租戶的成員數量（例如：5）

**預期結果**：
- ✅ "All Tenants" 顯示所有成員
- ✅ 選擇特定租戶後只顯示該租戶的成員
- ✅ 統計資訊正確顯示

---

### 測試 4：單個成員停用/啟用

**前置條件**：
- `alice@example.com` 存在且為 Active

**步驟**：
1. 進入 "Member List"
2. 選擇租戶（或 "All Tenants"）
3. 找到並展開 `alice@example.com`
4. 點擊 "🔴 Disable"

**預期結果**：
- ✅ 看到：「✅ Member disabled!」
- ⏱️ 1.5 秒後頁面重載
- ❌ 成員 expander 標題顯示 "❌ alice@example.com"
- 🟢 按鈕變為 "🟢 Enable"

**再次測試（啟用）**：
5. 再次展開 `alice@example.com`
6. 點擊 "🟢 Enable"

**預期結果**：
- ✅ 看到：「✅ Member enabled!」
- ✅ 成員恢復 Active 狀態

---

### 測試 5：更改成員角色

**前置條件**：
- `alice@example.com` 存在，當前角色為 `user`

**步驟**：
1. 展開 `alice@example.com`
2. 在 "Change Role" 表單中選擇 `admin`
3. 點擊 "Update Role"

**預期結果**：
- ✅ 看到：「✅ Role updated to 'admin'!」
- ⏱️ 1.5 秒後頁面重載
- 📝 成員詳情顯示 "Role: admin"

---

### 測試 6：批量停用成員

**前置條件**：
- 租戶 `test-demo` 有至少 3 個 Active 成員

**步驟**：
1. 進入 "Batch Operations"
2. 選擇租戶：`test-demo`
3. 在 multiselect 中選擇 3 個成員：
   - `alice@example.com`
   - `bob@example.com`
   - `carol@example.com`
4. 點擊 "🔴 Disable Selected"

**預期結果**：
- ✅ 看到：「✅ 3 member(s) disabled!」
- ⏱️ 2 秒後頁面重載
- ❌ 3 個成員都顯示為 Inactive

---

### 測試 7：批量啟用成員

**前置條件**：
- 測試 6 已完成，3 個成員為 Inactive

**步驟**：
1. 進入 "Batch Operations"
2. 選擇租戶：`test-demo`
3. 選擇相同的 3 個成員
4. 點擊 "🟢 Enable Selected"

**預期結果**：
- ✅ 看到：「✅ 3 member(s) enabled!」
- ⏱️ 2 秒後頁面重載
- ✅ 3 個成員都恢復為 Active

---

### 測試 8：手動新增單個成員

**步驟**：
1. 進入 "Batch Add Members"
2. 選擇租戶：`test-demo`
3. 選擇 "Manual Entry"
4. 輸入 email：`grace@example.com`
5. 選擇 Role：`admin`
6. 點擊 "➕ Add Member"

**預期結果**：
- ✅ 看到：「✅ Successfully added 1 member(s)!」
- 📋 在 Member List 可看到 `grace@example.com`，角色為 `admin`

---

### 測試 9：無效 email 驗證

**步驟**：
1. 進入 "Batch Add Members"
2. 選擇租戶
3. 貼上包含無效 email 的清單：
   ```
   valid@example.com
   invalid-email
   another-invalid
   also.valid@test.com
   ```
4. 點擊 "Add All Members"

**預期結果**：
- ✅ 只有 `valid@example.com` 和 `also.valid@test.com` 被新增
- ℹ️ 無效的 email（不含 `@`）被自動過濾

---

### 測試 10：Audit Log 驗證

**SQL 查詢**：
```sql
SELECT 
    created_at,
    action,
    tenant_slug,
    email AS actor,
    result,
    context
FROM audit_events
WHERE action LIKE 'member%'
ORDER BY created_at DESC
LIMIT 20;
```

**預期結果**：
應該看到以下 audit events：
- `members_batch_added` - 批量新增
- `member_enabled` - 單個啟用
- `member_disabled` - 單個停用
- `member_role_updated` - 角色更新
- `members_batch_enabled` - 批量啟用
- `members_batch_disabled` - 批量停用

**Context 應包含**：
- `count`: 操作的成員數量
- `emails`: 操作的 email 列表
- `member_email`: 單個成員的 email
- `old_role`, `new_role`: 角色變更時

---

## 📊 驗收標準

### 功能完整性
- ✅ 可以查看所有成員（按租戶篩選）
- ✅ 可以批量新增成員（貼上 30+ email）
- ✅ 可以批量停用/啟用成員
- ✅ 可以更改成員角色
- ✅ 可以手動新增單個成員
- ✅ 統計資訊正確顯示

### UI/UX
- ✅ 3 個分頁清晰易用
- ✅ 成功訊息正確顯示
- ✅ 錯誤處理友善
- ✅ 重複 email 有警告
- ✅ 操作後自動重載

### 安全性
- ✅ 所有操作記錄到 audit_events
- ✅ Email 驗證和清理
- ✅ 重複 email 處理
- ✅ 批量操作防止誤操作

### 效能
- ✅ 批量操作支援 30+ email
- ✅ API 呼叫有 timeout 設定
- ✅ 錯誤處理完善

---

## 📁 相關檔案

- `admin_ui.py` - 主實作檔案
  - `show_members()` - 主入口
  - `show_member_list()` - 成員列表
  - `show_batch_add_members()` - 批量新增
  - `show_batch_operations()` - 批量操作
  - Helper functions (9 個)

---

## 🚀 部署

**Railway 會自動部署**（約 1-2 分鐘）

**驗證部署**：
1. 訪問 Admin UI URL
2. 登入
3. 進入 "👥 Members" 頁面
4. 確認 3 個分頁都正常顯示

---

## 🔜 下一步

**Phase B4.1**：一鍵撤權（預計 0.5 天）
- Per-tenant revoke 按鈕
- 二次確認（輸入 tenant slug）
- Epoch bump
- Audit logging

---

**文件建立時間**：2026-03-02  
**維護者**：Amanda Chiu  
**版本**：Phase B3.1 MVP (Complete)
