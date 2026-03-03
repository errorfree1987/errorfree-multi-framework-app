# 🧪 Phase B3.1 快速測試指南

## 📋 測試結果總結與修正（2026-03-02 更新）

### 已修正問題

| 問題 | 原因 | 修正 |
|------|------|------|
| **KeyError: 'tenant_slug'** | `tenant_members` 表使用 `tenant_id` 而非 `tenant_slug` | 在 `get_members()` 中透過 `tenant_id` 查詢 `tenants` 取得 slug，動態加入 `member['tenant_slug']` |
| **Failed to update role: HTTP 400** | 資料庫角色為 `tenant_admin`，程式傳送 `admin` | 將選項改為 `user` / `tenant_admin`，UI 顯示為 User / Admin，使用 `format_func` 對應 |
| **SQL 範例錯誤** | `tenant_members` 使用 `tenant_id`，非 `tenant_slug` | 更新 DELETE / SELECT 範例，改為 JOIN tenants 或子查詢 |

### 新功能（全面改進 - 2026-03-02）

| 項目 | 改進內容 |
|------|----------|
| **測試 1** | 批量新增成功後，Email 輸入框自動清空（使用 batch_add_counter） |
| **測試 5/6** | 批量操作成功後，Select Members 自動清空 |
| **測試 5/6** | 新增搜尋框、Select All / Select Active / Select Inactive / Clear 快速按鈕 |
| **測試 5/6** | Select Members 選項顯示 ✅ Active / ❌ Inactive 標記 |
| **測試 8** | 新增 Individual (Guest) 選項 — 可新增不屬於公司租戶的個人使用者，Role 為 Guest |
| **Member List** | 新增搜尋框，可依 email 快速搜尋 |
| **Member List** | 新增批量刪除（multiselect + Delete Selected 按鈕） |
| **Member List** | 成員詳情頁新增 Delete 按鈕 |
| **Member List** | 新增 Active only / Inactive only 篩選（radio） |

---

## 🚀 測試準備

**前置條件**：
1. ✅ Admin UI 已部署到 Railway
2. ✅ 至少有 1 個測試租戶（例如：`test-demo`）
3. ✅ 登入 Admin UI

---

## ⚡ 快速測試流程（5 分鐘）

### 測試 1：批量新增成員 📝

**步驟**：
1. 進入 **"👥 Members"** 頁面
2. 切換到 **"➕ Batch Add Members"** 分頁
3. 選擇租戶：`test-demo`（或你的測試租戶）
4. 確認 **"Paste Emails"** 已選中
5. 貼上以下測試 email：

```
alice@example.com
bob@example.com
carol@example.com
dave@example.com
eve@example.com
```

6. 點擊 **"➕ Add All Members"**

**✅ 預期結果**：
- 看到：「✅ Successfully added 5 member(s)!」
- 🎈 看到氣球動畫
- 2 秒後頁面自動重載

---

### 測試 2：查看成員列表 👀

**步驟**：
1. 切換到 **"📋 Member List"** 分頁
2. 確認租戶篩選器選擇正確（或選 "All Tenants"）
3. 查看統計資訊

**✅ 預期結果**：
- **Total Members**: 5（或更多）
- **Active Members**: 5
- **Inactive Members**: 0
- 看到 5 個成員的 expander，都顯示 ✅ 圖標

---

### 測試 3：單個成員停用 🔴

**步驟**：
1. 展開 `alice@example.com` 的 expander
2. 點擊 **"🔴 Disable"** 按鈕

**✅ 預期結果**：
- 看到：「✅ Member disabled!」
- 1.5 秒後頁面重載
- `alice@example.com` 的 expander 顯示 ❌ 圖標
- **Inactive Members** 變為 1

---

### 測試 4：單個成員啟用 🟢

**步驟**：
1. 再次展開 `alice@example.com` 的 expander
2. 點擊 **"🟢 Enable"** 按鈕

**✅ 預期結果**：
- 看到：「✅ Member enabled!」
- `alice@example.com` 恢復 ✅ 圖標
- **Inactive Members** 變回 0

---

### 測試 5：批量停用成員 🚫

**步驟**：
1. 切換到 **"⚙️ Batch Operations"** 分頁
2. 確認租戶選擇正確
3. 在 **"Select Members"** multiselect 中選擇：
   - `bob@example.com`
   - `carol@example.com`
   - `dave@example.com`
4. 點擊 **"🔴 Disable Selected"**

**✅ 預期結果**：
- 看到：「✅ 3 member(s) disabled!」
- 2 秒後頁面重載
- 切換到 "Member List" 確認這 3 個成員顯示 ❌

---

### 測試 6：批量啟用成員 ✅

**步驟**：
1. 保持在 **"Batch Operations"** 分頁
2. 再次選擇相同的 3 個成員
3. 點擊 **"🟢 Enable Selected"**

**✅ 預期結果**：
- 看到：「✅ 3 member(s) enabled!」
- 3 個成員都恢復 Active 狀態

---

### 測試 7：更改成員角色 👤

**步驟**：
1. 切換到 **"Member List"**
2. 展開 `eve@example.com`
3. 在 **"Change Role"** 表單中選擇 **`Admin`**
4. 點擊 **"Update Role"**

**✅ 預期結果**：
- 看到：「✅ Role updated to 'Admin'!」
- 重載後，`eve@example.com` 的 Role 顯示為 `Admin`（DB 實際儲存為 `tenant_admin`）

---

### 測試 8：手動新增單個成員 ➕

**步驟 A - 新增至租戶**：
1. 切換到 **"Batch Add Members"**
2. 選擇 **"Tenant"**
3. 選擇 **"Manual Entry"**
4. 輸入 Email：`frank@example.com`
5. 選擇 Role：`user`
6. 點擊 **"➕ Add Member"**

**✅ 預期結果**：
- 看到：「✅ Successfully added 1 member(s)!」
- 在 Member List 可看到 `frank@example.com`

**步驟 B - 新增個人使用者 (Guest)**：
1. 選擇 **"Individual (Guest)"**
2. 選擇 **"Manual Entry"**
3. 輸入 Email：`guest@example.com`
4. Role 預設為 Guest
5. 點擊 **"➕ Add Member"**

**✅ 預期結果**：
- 看到：「✅ Successfully added 1 member(s)!」
- 在 Member List 可看到 `guest@example.com`，Tenant 顯示為 `individual`

---

### 測試 9：重複 email 處理 ⚠️

**步驟**：
1. 切換到 **"Batch Add Members"**
2. 選擇 **"Paste Emails"**
3. 貼上已存在的 email：
```
alice@example.com
bob@example.com
```
4. 點擊 **"Add All Members"**

**✅ 預期結果**：
- 看到：「⚠️ Some members already exist. Skipping duplicates...」
- 看到：「💡 Tip: Existing members were not modified.」
- 沒有錯誤發生

---

### 測試 10：按租戶篩選 🔍

**前置條件**：需要 2 個租戶，每個有成員

**步驟**：
1. 進入 **"Member List"**
2. 選擇 **"All Tenants"**
3. 記下總成員數量
4. 選擇特定租戶（例如：`test-demo`）
5. 確認只顯示該租戶的成員

**✅ 預期結果**：
- "All Tenants" 顯示所有成員
- 選擇特定租戶後只顯示該租戶的成員
- 統計資訊正確

---

## 🔍 Audit Log 驗證

在 **Supabase SQL Editor** 執行：

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

**✅ 預期結果**：
應該看到以下 audit events：
- `members_batch_added` - 批量新增（測試 1, 8）
- `member_enabled` - 單個啟用（測試 4）
- `member_disabled` - 單個停用（測試 3）
- `member_role_updated` - 角色更新（測試 7）
- `members_batch_disabled` - 批量停用（測試 5）
- `members_batch_enabled` - 批量啟用（測試 6）

**Context 範例**：
```json
{
  "count": 5,
  "emails": [
    "alice@example.com",
    "bob@example.com",
    "carol@example.com",
    "dave@example.com",
    "eve@example.com"
  ],
  "role": "user"
}
```

---

## 🎯 完整測試清單

- [ ] **測試 1**: 批量新增成員（貼上 5 個 email）✅
- [ ] **測試 2**: 查看成員列表和統計 ✅
- [ ] **測試 3**: 單個成員停用 🔴
- [ ] **測試 4**: 單個成員啟用 🟢
- [ ] **測試 5**: 批量停用成員（3 個）🚫
- [ ] **測試 6**: 批量啟用成員（3 個）✅
- [ ] **測試 7**: 更改成員角色 👤
- [ ] **測試 8**: 手動新增單個成員 ➕
- [ ] **測試 9**: 重複 email 處理 ⚠️
- [ ] **測試 10**: 按租戶篩選 🔍
- [ ] **Audit Log 驗證** ✅

---

## 💡 測試技巧

### 快速重置測試數據

如果需要清空測試成員，在 Supabase SQL Editor 執行：

```sql
-- ⚠️ 警告：這會刪除指定租戶的所有成員！
-- 注意：tenant_members 使用 tenant_id（UUID），需先透過 tenants 表查詢
DELETE FROM tenant_members 
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'test-demo');
```

### 查看特定租戶的成員

```sql
SELECT 
    tm.email,
    tm.role,
    tm.is_active,
    tm.created_at
FROM tenant_members tm
JOIN tenants t ON tm.tenant_id = t.id
WHERE t.slug = 'test-demo'
ORDER BY tm.created_at DESC;
```

### 統計成員數量

```sql
SELECT 
    t.slug AS tenant_slug,
    COUNT(*) as total_members,
    COUNT(*) FILTER (WHERE tm.is_active = true) as active_members,
    COUNT(*) FILTER (WHERE tm.is_active = false) as inactive_members
FROM tenant_members tm
JOIN tenants t ON tm.tenant_id = t.id
GROUP BY t.slug
ORDER BY total_members DESC;
```

---

## 🚨 常見問題排查

### 問題 1：批量新增時沒有成功訊息

**可能原因**：
- Railway 部署尚未完成
- Supabase 連線問題

**解決方法**：
1. 檢查 Railway 部署狀態
2. 檢查 Supabase 環境變數是否正確
3. 查看瀏覽器 Console 是否有錯誤

---

### 問題 2：重複 email 出現錯誤而非警告

**可能原因**：
- 資料庫 unique constraint 設定

**解決方法**：
- 這是正常行為，應該顯示警告而非錯誤
- 如果看到 HTTP 409，檢查錯誤處理邏輯

---

### 問題 3：批量操作沒有反應

**可能原因**：
- 沒有選擇成員

**解決方法**：
- 確認在 multiselect 中至少選擇了一個成員
- 應該看到「Selected: X member(s)」提示

---

### 問題 4：更改角色時出現 HTTP 400

**可能原因**：
- 資料庫 `tenant_members.role` 欄位使用 `tenant_admin`，而非 `admin`

**解決方法**：
- 已修正：UI 選項為 User / Admin，後端會正確傳送 `user` / `tenant_admin`
- 若仍失敗，檢查 Supabase 的 role 欄位是否有 CHECK 約束或其他限制

---

### 問題 5：新增 Guest 時出現 HTTP 400

**可能原因**：
- 資料庫 `tenant_members.role` 可能僅允許 `user`、`tenant_admin`

**解決方法**：
- 在 Supabase SQL Editor 執行：`ALTER TABLE tenant_members DROP CONSTRAINT IF EXISTS tenant_members_role_check;`
- 或新增 CHECK 包含 guest：`ALTER TABLE tenant_members ADD CONSTRAINT tenant_members_role_check CHECK (role IN ('user','tenant_admin','guest'));`

---

## 🎉 測試完成

如果所有測試都通過：

**✅ Phase B3.1 (Members 批量管理) 完成！**

**下一步**：
- 🚀 準備開始 **Phase B4.1**：一鍵撤權
- 📝 更新 ROADMAP.md
- 🎯 準備下一個功能

---

**測試指南建立時間**：2026-03-02  
**維護者**：Amanda Chiu  
**預計測試時間**：5-10 分鐘
