# Phase B1.1: MVP Admin UI - 登入/權限實作說明

## 📋 變更摘要

本次更新實作了 **Phase B1.1：MVP Admin UI 登入/權限（Streamlit 版本）**。

新增功能：
1. ✅ 簡單密碼保護（環境變數 `ADMIN_PASSWORD`）
2. ✅ Session 狀態管理（重新整理不需重新登入）
3. ✅ 登出功能
4. ✅ 記錄所有登入/登出到 `audit_events`
5. ✅ 獨立的 Admin UI（不影響現有 Analyzer）

---

## 🏗️ 架構說明

### 檔案結構

```
errorfree-multi-framework-app/
├── app.py                      # 現有的 Analyzer（保持不變）
├── admin_ui.py                 # 新增：Admin UI（本次新增）
├── requirements.txt            # 依賴套件（無需修改）
└── README_PHASE_B1_1.md        # 本檔案
```

### 關鍵設計決策

#### 1. **獨立檔案設計**
- `admin_ui.py` 完全獨立於 `app.py`
- 不會影響現有的 Portal SSO 流程
- 方便未來遷移到 Next.js（Phase B2）

#### 2. **簡單密碼保護**
- 使用環境變數 `ADMIN_PASSWORD`（單一密碼）
- 使用 `hmac.compare_digest()` 防止時序攻擊
- Phase B1.2（完美版）會升級到 Supabase Auth + RBAC

#### 3. **Session 管理**
- 使用 `st.session_state["authenticated"]` 記錄登入狀態
- 重新整理頁面不需要重新登入
- 登出時清除所有 session state

#### 4. **Audit Logging**
- 所有登入/登出都記錄到 `audit_events` 表
- 記錄 action: `admin_login` / `admin_logout`
- 記錄 actor_email（目前固定為 `admin@errorfree.com`）

---

## 🚀 部署步驟

### Step 1: 設定 Railway 環境變數

在 Railway 專案中新增以下環境變數：

```bash
# Admin UI 密碼（必須）
ADMIN_PASSWORD=your_secure_password_here

# Supabase 連線（應該已存在）
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGc...
```

**重要**：
- `ADMIN_PASSWORD` 請使用強密碼（建議 16+ 字元，包含大小寫、數字、符號）
- 不要將密碼寫入 Git（使用環境變數）

### Step 2: 修改 Railway 啟動指令（選項 A 或 B）

#### 選項 A：使用不同的 port（建議）

在 Railway 中，可以部署兩個服務：
- **Analyzer**（現有）: `streamlit run app.py --server.port 8501`
- **Admin UI**（新增）: `streamlit run admin_ui.py --server.port 8502`

這樣可以同時運行，使用不同的 URL：
- `https://your-app.up.railway.app` → Analyzer（用戶使用）
- `https://your-app-admin.up.railway.app` → Admin UI（營運使用）

#### 選項 B：單一部署（臨時測試用）

如果只想快速測試，可以暫時替換啟動指令：

```bash
# 原本（Analyzer）
streamlit run app.py --server.port 8501

# 改為（Admin UI）
streamlit run admin_ui.py --server.port 8501
```

**注意**：這樣會暫時無法使用 Analyzer，只適合測試期間。

### Step 3: 推送程式碼到 Railway

```bash
# 確認檔案已新增
git status

# 新增檔案
git add admin_ui.py README_PHASE_B1_1.md

# 提交
git commit -m "Phase B1.1: Add MVP Admin UI with login/auth"

# 推送到 Railway（如果已設定自動部署）
git push
```

### Step 4: 訪問 Admin UI

部署完成後，訪問 Admin UI：

```
https://your-admin-app.up.railway.app
```

輸入設定的 `ADMIN_PASSWORD` 即可進入。

---

## 🧪 驗收測試

### 測試案例 1：登入成功 ✅

**步驟**：
1. 訪問 Admin UI
2. 輸入正確的 `ADMIN_PASSWORD`
3. 按下 Enter

**預期結果**：
- 成功進入 Dashboard
- 顯示 "🏠 Dashboard Overview"
- 右上角顯示 "👤 admin@errorfree.com"
- 顯示 "🚪 Logout" 按鈕

**驗證 Audit Log**：
```sql
SELECT 
    created_at,
    action,
    email,
    result,
    context
FROM audit_events
WHERE action = 'admin_login'
  AND result = 'success'
ORDER BY created_at DESC
LIMIT 1;
```

**預期結果**：
- 應該看到一筆新的 `admin_login` 記錄
- `result = 'success'`
- `context` 包含 `"source": "admin_ui"`

---

### 測試案例 2：登入失敗（密碼錯誤）❌

**步驟**：
1. 訪問 Admin UI
2. 輸入錯誤的密碼
3. 按下 Enter

**預期結果**：
- 顯示錯誤訊息：❌ Incorrect password. Please try again.
- 保持在登入頁面
- 可以重新嘗試

**驗證 Audit Log**：
```sql
SELECT 
    created_at,
    action,
    email,
    result,
    deny_reason,
    context
FROM audit_events
WHERE action = 'admin_login'
  AND result = 'denied'
ORDER BY created_at DESC
LIMIT 1;
```

**預期結果**：
- 應該看到一筆 `admin_login` 記錄
- `result = 'denied'`
- `deny_reason = 'incorrect_password'`
- `email = 'unknown'`

---

### 測試案例 3：Session 持久化（重新整理）🔄

**步驟**：
1. 成功登入 Admin UI
2. 瀏覽 Dashboard
3. 按下瀏覽器的「重新整理」（F5）

**預期結果**：
- 不需要重新登入
- 保持在 Dashboard 頁面
- Session 狀態維持（顯示相同的登入時間）

---

### 測試案例 4：登出 🚪

**步驟**：
1. 成功登入 Admin UI
2. 點擊右上角的 "🚪 Logout" 按鈕

**預期結果**：
- 跳回登入頁面
- 所有 session state 被清除
- 需要重新輸入密碼才能進入

**驗證 Audit Log**：
```sql
SELECT 
    created_at,
    action,
    email,
    result,
    context
FROM audit_events
WHERE action = 'admin_logout'
  AND result = 'success'
ORDER BY created_at DESC
LIMIT 1;
```

**預期結果**：
- 應該看到一筆 `admin_logout` 記錄
- `result = 'success'`
- `email = 'admin@errorfree.com'`

---

### 測試案例 5：環境變數未設定 ⚠️

**步驟**：
1. 暫時移除 Railway 的 `ADMIN_PASSWORD` 環境變數
2. 重新部署
3. 訪問 Admin UI

**預期結果**：
- 顯示錯誤訊息：⚠️ Admin password not configured. Please set ADMIN_PASSWORD environment variable.
- 無法登入

**修復**：
- 重新設定 `ADMIN_PASSWORD` 環境變數
- 重新部署

---

## 📊 監控與診斷

### 查看今日 Admin 登入歷史

```sql
SELECT 
    created_at,
    action,
    email,
    result,
    deny_reason,
    context->>'timestamp' AS login_time
FROM audit_events
WHERE action IN ('admin_login', 'admin_logout')
  AND created_at >= CURRENT_DATE
ORDER BY created_at DESC;
```

### 查看失敗的登入嘗試（安全監控）

```sql
SELECT 
    created_at,
    action,
    result,
    deny_reason,
    context->>'timestamp' AS attempt_time
FROM audit_events
WHERE action = 'admin_login'
  AND result = 'denied'
  AND created_at >= NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;
```

**注意**：如果看到大量失敗嘗試，可能是暴力破解攻擊，建議：
- 更換 `ADMIN_PASSWORD`
- 考慮加入 IP 白名單（Railway 設定）
- Phase B1.2 升級到 Supabase Auth

---

## 🔒 安全考量

### 目前實作（Phase B1.1 MVP）

✅ **已做到**：
- 密碼使用環境變數（不在程式碼中）
- 使用 `hmac.compare_digest()` 防止時序攻擊
- 所有登入/登出記錄到 audit log
- HTTPS 傳輸（Railway 預設）

⚠️ **限制**：
- 單一密碼（所有 admin 共用）
- 沒有速率限制（可能被暴力破解）
- 沒有 RBAC（所有 admin 權限相同）
- Session 無過期時間

### Phase B1.2（完美版）會加強：

- ✅ Supabase Auth（email + password）
- ✅ RBAC 角色（admin / ops）
- ✅ 速率限制（失敗 N 次後鎖定）
- ✅ Session 過期時間（例如 8 小時）
- ✅ 多因素認證（MFA）考慮

---

## 🎯 Phase B1.1 驗收清單

- [ ] `admin_ui.py` 檔案已創建
- [ ] Railway 環境變數 `ADMIN_PASSWORD` 已設定
- [ ] Railway 環境變數 `SUPABASE_URL` 已存在
- [ ] Railway 環境變數 `SUPABASE_SERVICE_KEY` 已存在
- [ ] Admin UI 可成功部署（無啟動錯誤）
- [ ] 測試案例 1：登入成功 ✅
- [ ] 測試案例 2：登入失敗（密碼錯誤）❌
- [ ] 測試案例 3：Session 持久化 🔄
- [ ] 測試案例 4：登出 🚪
- [ ] 所有 audit log 正確記錄
- [ ] Dashboard 顯示 "Phase B1.1 Complete!" 訊息

---

## 🔄 下一步：Phase B2

完成 Phase B1.1 驗收後，下一步將實作：

### Phase B2.1: Tenant 管理（預計 0.5-1 天）

**功能**：
- [ ] 租戶列表（`st.dataframe`）
- [ ] 建立新租戶（`st.form`）
- [ ] Trial 延期（`st.date_input`）
- [ ] 停用/啟用租戶（`st.button`）
- [ ] 基本統計（成員數、今日用量）

**實作方式**：
- 在 `admin_ui.py` 中實作 `show_tenants()` 函數
- 使用 Supabase REST API 讀取/寫入 `tenants` 表
- 所有寫入操作記錄到 `audit_events`

---

## 📁 相關檔案

| 檔案 | 說明 |
|------|------|
| `admin_ui.py` | Admin UI 主程式（本次新增）|
| `README_PHASE_B1_1.md` | 本檔案 |
| `app.py` | Analyzer 主程式（保持不變）|
| `ROADMAP.md` | 完整路線圖 |

---

## 🆘 常見問題（FAQ）

### Q1: 為什麼不直接在 `app.py` 中加入 Admin UI？

**A**: 為了保持關注點分離（Separation of Concerns）：
- `app.py`: 用戶使用的 Analyzer
- `admin_ui.py`: 營運使用的 Admin UI

這樣可以：
- 避免相互干擾（Portal SSO vs Admin Auth）
- 方便未來遷移（Admin UI → Next.js）
- 獨立部署和縮放

### Q2: 為什麼不使用 Supabase Auth？

**A**: Phase B1.1 是 MVP 版本，目標是「快速可用」：
- 簡單密碼保護：1 小時實作
- Supabase Auth：需要 1-2 天（前端整合、權限設定、測試）

Phase B1.2（完美版）會升級到 Supabase Auth。

### Q3: 可以同時運行 Analyzer 和 Admin UI 嗎？

**A**: 可以！有兩種方式：
1. **不同服務**（建議）：在 Railway 建立兩個服務，使用不同 URL
2. **不同 port**：在同一服務中運行兩個 Streamlit app（需要 process manager 如 `supervisor`）

目前建議使用選項 1（不同服務），部署更簡單。

### Q4: 如何修改 Admin 的 email？

**A**: 目前固定為 `admin@errorfree.com`，可在 `admin_ui.py` 第 88 行修改：

```python
st.session_state["admin_email"] = "your-email@company.com"
```

Phase B1.2 會支援多用戶（從 Supabase Auth 讀取）。

---

**最後更新**：2026-02-28  
**維護者**：Amanda Chiu  
**Phase**：B1.1 (MVP Admin UI - Login/Auth)
