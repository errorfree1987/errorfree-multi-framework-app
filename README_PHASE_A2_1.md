# Phase A2-1: Portal SSO Enforcement 實作說明

## 📋 變更摘要

本次更新實作了 **Phase A2 的第一步：Portal SSO Verify 強制執行（Enforcement）**。

在用戶通過 Portal SSO 驗證後，系統會立即檢查：
1. ✅ 租戶是否啟用（`tenants.is_active`）
2. ✅ 租戶試用是否過期（`tenants.trial_end`）
3. ✅ 成員是否啟用（`tenant_members.is_active`）
4. ✅ 記錄所有驗證結果到 `audit_events`（成功或拒絕）

---

## 🔧 程式碼變更

### 新增函數（在 `app.py` 第 350 行之前）

#### 1. `_check_tenant_and_member_access(tenant, email)`
- **用途**：檢查租戶和成員的存取權限
- **返回**：`(allow: bool, deny_reason: str)`
- **檢查項目**：
  - 租戶 `is_active` 是否為 `true`
  - 租戶 `trial_end` 是否已過期
  - 成員 `is_active` 是否為 `true`
- **Fail-open 策略**：
  - 如果 Supabase 連線失敗或環境變數未設定，**允許存取**（避免因暫時性問題鎖住所有用戶）
  - 如果成員在 `tenant_members` 表中不存在，**允許存取**（向後相容未填入成員資料的租戶）

#### 2. `_log_audit_event(action, tenant, email, result, ...)`
- **用途**：記錄審計日誌到 `audit_events` 表
- **參數**：
  - `action`: 操作類型（例如 `sso_verify`）
  - `result`: 結果（`success` / `denied` / `error`）
  - `deny_reason`: 拒絕原因（可選，例如 `tenant_inactive`）
  - `context`: JSON 上下文（可選）
- **Best-effort**：如果記錄失敗（網路錯誤、API down），不會中斷主流程

### 修改位置（在 `try_portal_sso_login()` 函數內）

#### 位置 1：Portal verify 成功後，epoch 檢查之前（約第 810 行之後）
```python
# Phase A2 Enforcement: Check tenant and member access
allow, deny_reason = _check_tenant_and_member_access(tenant, verified_email)
if not allow:
    # Log denial
    _log_audit_event(
        action="sso_verify",
        tenant=tenant,
        email=verified_email,
        result="denied",
        deny_reason=deny_reason,
        context={"source": "portal_sso"}
    )
    
    # Block access with specific message
    st.session_state["_portal_sso_checked"] = True
    st.session_state["is_authenticated"] = False
    
    deny_messages = {
        "tenant_inactive": "Access denied: Your organization's account is currently inactive. Please contact your administrator.",
        "trial_expired": "Access denied: Your trial period has expired. Please contact support to upgrade your account.",
        "member_inactive": "Access denied: Your account has been deactivated. Please contact your administrator.",
    }
    message = deny_messages.get(deny_reason, f"Access denied: {deny_reason}")
    _render_portal_only_block(message)
```

#### 位置 2：設定 session_state 之後（約第 847 行之後）
```python
# Phase A2: Log successful verification
_log_audit_event(
    action="sso_verify",
    tenant=tenant,
    email=verified_email,
    result="success",
    context={"source": "portal_sso", "epoch": int(current_epoch)}
)
```

---

## 🧪 驗收方式

### 準備工作
確保 Railway 環境變數已設定：
```bash
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGc...
```

### 測試案例

#### 案例 1：正常登入（應該成功）
1. 從 Portal 進入 Analyzer（租戶 `abc`，用戶 `user1@abc.com`）
2. **預期結果**：成功進入 Analyzer
3. **驗證**：
   ```sql
   SELECT * FROM audit_events 
   WHERE action = 'sso_verify' 
     AND result = 'success' 
     AND tenant_slug = 'abc' 
     AND email = 'user1@abc.com'
   ORDER BY created_at DESC LIMIT 1;
   ```
   應該看到一筆 `result='success'` 的記錄

#### 案例 2：租戶已停用（應該被拒絕）
1. 在 Supabase 將租戶 `suspended` 的 `is_active` 設為 `false`
2. 從 Portal 進入 Analyzer（租戶 `suspended`，用戶 `user@suspended.com`）
3. **預期結果**：看到錯誤訊息 "Access denied: Your organization's account is currently inactive. Please contact your administrator."
4. **驗證**：
   ```sql
   SELECT * FROM audit_events 
   WHERE action = 'sso_verify' 
     AND result = 'denied' 
     AND deny_reason = 'tenant_inactive'
   ORDER BY created_at DESC LIMIT 1;
   ```

#### 案例 3：試用已過期（應該被拒絕）
1. 租戶 `expired` 的 `trial_end` 已設為過去時間
2. 從 Portal 進入 Analyzer（租戶 `expired`，用戶 `admin@expired.com`）
3. **預期結果**：看到錯誤訊息 "Access denied: Your trial period has expired. Please contact support to upgrade your account."
4. **驗證**：
   ```sql
   SELECT * FROM audit_events 
   WHERE action = 'sso_verify' 
     AND result = 'denied' 
     AND deny_reason = 'trial_expired'
   ORDER BY created_at DESC LIMIT 1;
   ```

#### 案例 4：成員已停用（應該被拒絕）
1. 在 Supabase 將 `abc` 租戶的 `user3@abc.com` 的 `is_active` 設為 `false`
2. 從 Portal 進入 Analyzer（租戶 `abc`，用戶 `user3@abc.com`）
3. **預期結果**：看到錯誤訊息 "Access denied: Your account has been deactivated. Please contact your administrator."
4. **驗證**：
   ```sql
   SELECT * FROM audit_events 
   WHERE action = 'sso_verify' 
     AND result = 'denied' 
     AND deny_reason = 'member_inactive'
   ORDER BY created_at DESC LIMIT 1;
   ```

---

## 📊 監控與診斷

### 查看今日被拒絕的登入
```sql
SELECT 
    created_at,
    tenant_slug,
    email,
    deny_reason,
    context
FROM audit_events
WHERE result = 'denied'
  AND action = 'sso_verify'
  AND created_at >= CURRENT_DATE
ORDER BY created_at DESC;
```

### 查看特定租戶的登入歷史
```sql
SELECT 
    created_at,
    email,
    result,
    deny_reason,
    context
FROM audit_events
WHERE action = 'sso_verify'
  AND tenant_slug = 'abc'
ORDER BY created_at DESC
LIMIT 20;
```

---

## 🔄 下一步：Phase A2-2

完成 Phase A2-1 驗收後，下一步將實作：
- **Analyzer 啟動時的 entitlement 檢查**（檢查租戶是否有權限使用特定功能）
- **基本成本控制 v1**（檢查今日用量是否已達上限）

---

## 📁 相關檔案

| 檔案 | 說明 |
|------|------|
| `app.py` | 主程式（已修改） |
| `phase_a1_database_setup.sql` | Phase A1 完整 SQL（包含所有表結構和驗收查詢） |
| `sql_epoch_management.sql` | Epoch 撤權管理 SQL（D3 已完成，保持不變） |
| `QUICK_REFERENCE.md` | Epoch 快速參考卡（D3 已完成，保持不變） |
| `README_PHASE_A2_1.md` | 本檔案 |

---

**最後更新：** 2026-02-27  
**維護者：** Amanda Chiu
