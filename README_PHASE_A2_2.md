# Phase A2-2: 基本成本控制 v1 實作說明

## 📋 變更摘要

本次更新實作了 **Phase A2-2：基本成本控制 v1（Caps Enforcement）**。

新增功能：
1. ✅ Analyzer 啟動時記錄 `analyzer_launch` audit event
2. ✅ 執行 review 前檢查今日用量是否已達上限
3. ✅ 記錄每次 review 使用到 `tenant_usage_events`
4. ✅ 顯示用量狀態（已用/上限）
5. ✅ 達到上限時顯示友善錯誤訊息並記錄 audit event

---

## 🔧 程式碼變更

### 新增函數（在 `app.py` 第 439 行附近）

#### 1. `_check_usage_cap(tenant, usage_type)`
- **用途**：檢查租戶今日用量是否已達上限
- **返回**：`(allow: bool, cap: int, current_usage: int, message: str)`
- **檢查邏輯**：
  1. 從 `tenant_usage_caps` 讀取 `daily_review_cap` 或 `daily_download_cap`
  2. 從 `tenant_usage_events` 統計今日已用量（`created_at >= 今日 00:00`）
  3. 如果 `current_usage >= cap`，返回 `(False, cap, usage, "錯誤訊息")`
  4. 如果 `cap = NULL`，視為無限制（unlimited）
  5. 如果 `cap = 0`，視為完全禁止（disabled）
- **Fail-open 策略**：如果 Supabase 連線失敗，允許存取（避免因暫時性問題鎖住所有用戶）

#### 2. `_record_usage_event(tenant, email, usage_type, quantity, context)`
- **用途**：記錄使用事件到 `tenant_usage_events` 表
- **參數**：
  - `usage_type`: `'review'` 或 `'download'`
  - `quantity`: 數量（預設 1）
  - `context`: JSON 上下文（可選，例如 `{"framework": "react", "step": "step5"}`）
- **Best-effort**：如果記錄失敗，不會中斷主流程

### 修改位置

#### 位置 1：Analyzer 啟動時（main() 函數，第 3136 行附近）
```python
# Phase A2-2: Analyzer launch logging + caps check
if st.session_state.get("is_authenticated"):
    tenant = st.session_state.get("tenant", "")
    email = st.session_state.get("user_email", "")
    
    # Only log once per session
    if "_analyzer_launch_logged" not in st.session_state:
        _log_audit_event(
            action="analyzer_launch",
            tenant=tenant,
            email=email,
            result="success",
            context={"source": "main_app", "session_epoch": ...}
        )
        st.session_state["_analyzer_launch_logged"] = True
```

#### 位置 2：Step 5 執行前（第 3925 行附近）
```python
# Phase A2-2: Check usage cap before proceeding
allow, cap, current_usage, cap_message = _check_usage_cap(tenant, "review")

if not allow:
    # Cap reached - show error and log denial
    st.error(cap_message)
    _log_audit_event(
        action="review_denied",
        tenant=tenant,
        email=email,
        result="denied",
        deny_reason="usage_cap_reached",
        context={"cap": cap, "current_usage": current_usage}
    )
    st.stop()

# Show usage status if cap is set
if cap > 0:
    remaining = cap - current_usage
    if remaining <= 5:
        st.warning(f"⚠️ Daily review limit: {current_usage}/{cap} used. {remaining} remaining.")
    else:
        st.info(f"📊 Daily review usage: {current_usage}/{cap}")
```

#### 位置 3：Step 5 完成後（第 3979 行附近）
```python
# Phase A2-2: Record usage event
_record_usage_event(
    tenant=tenant,
    email=email,
    usage_type="review",
    quantity=1,
    context={"framework": selected_key, "step": "step5_main_analysis"}
)
```

---

## 🧪 驗收方式

### 準備工作
確保 Supabase 中已設定測試租戶的 caps：

```sql
-- 查看現有 caps 設定
SELECT 
    t.slug AS tenant,
    tc.daily_review_cap,
    tc.daily_download_cap
FROM tenant_usage_caps tc
LEFT JOIN tenants t ON tc.tenant_id = t.id;

-- 如果需要，調整 abc 租戶的 daily_review_cap 為較小的值（例如 3）以便測試
UPDATE tenant_usage_caps
SET daily_review_cap = 3
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'abc');
```

### 測試案例

#### 案例 1：Analyzer 啟動記錄 ✅

**步驟**：
1. 從 Portal 進入 Analyzer（租戶 `abc`）
2. 確認能正常進入

**驗證**：
```sql
SELECT 
    created_at,
    action,
    tenant_slug,
    email,
    result,
    context
FROM audit_events
WHERE action = 'analyzer_launch'
  AND tenant_slug = 'abc'
ORDER BY created_at DESC
LIMIT 5;
```

**預期結果**：應該看到一筆新的 `analyzer_launch` 記錄，`context` 包含 `"source": "main_app"`

---

#### 案例 2：正常使用（未達上限）📊

**步驟**：
1. 上傳文件 → 選擇類型 → 選擇框架
2. 點擊 "Run analysis（主文件）"
3. 應該顯示用量狀態（例如：`📊 Daily review usage: 1/3`）
4. 分析成功完成

**驗證 1：Usage Events**
```sql
SELECT 
    created_at,
    tenant_id,
    email,
    usage_type,
    quantity,
    context
FROM tenant_usage_events
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'abc')
  AND created_at >= CURRENT_DATE
ORDER BY created_at DESC;
```

**預期結果**：
- 應該看到一筆新的 `usage_type='review'` 記錄
- `context` 包含 `{"framework": "...", "step": "step5_main_analysis"}`

**驗證 2：Audit Events**
```sql
SELECT 
    created_at,
    action,
    result
FROM audit_events
WHERE tenant_slug = 'abc'
  AND action IN ('analyzer_launch', 'review_denied')
ORDER BY created_at DESC
LIMIT 10;
```

**預期結果**：
- 應該看到 `analyzer_launch` 成功記錄
- **不應該**看到 `review_denied` 記錄

---

#### 案例 3：達到上限（應該被拒絕）🚫

**步驟**：
1. 重複執行 review 直到達到上限（例如 3 次）
2. 第 4 次嘗試時，應該顯示錯誤訊息：
   - **"Daily review limit reached (3/3). Please try again tomorrow or contact support to upgrade."**

**驗證 1：Audit Events**
```sql
SELECT 
    created_at,
    action,
    tenant_slug,
    email,
    result,
    deny_reason,
    context
FROM audit_events
WHERE action = 'review_denied'
  AND tenant_slug = 'abc'
ORDER BY created_at DESC
LIMIT 1;
```

**預期結果**：
- 應該看到一筆 `review_denied` 記錄
- `deny_reason = 'usage_cap_reached'`
- `context` 包含 `{"cap": 3, "current_usage": 3, "usage_type": "review"}`

**驗證 2：Usage Events（確認沒有超額記錄）**
```sql
SELECT 
    COUNT(*) AS today_review_count
FROM tenant_usage_events
WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'abc')
  AND usage_type = 'review'
  AND created_at >= CURRENT_DATE;
```

**預期結果**：
- `today_review_count` 應該等於 `daily_review_cap`（例如 3）
- **不應該**有第 4 筆記錄（因為被拒絕了）

---

#### 案例 4：警告提示（接近上限）⚠️

**步驟**：
1. 確保租戶 cap 設為較大值（例如 10）
2. 執行到剩餘 5 次或更少時
3. 應該顯示警告訊息：
   - **"⚠️ Daily review limit: 6/10 used. 4 remaining."**

**預期結果**：
- 剩餘 > 5 時：藍色 info 訊息（`📊 Daily review usage: 1/10`）
- 剩餘 ≤ 5 時：橙色 warning 訊息（`⚠️ ... 4 remaining.`）
- 剩餘 = 0 時：紅色 error 訊息並阻擋

---

#### 案例 5：無限制租戶（cap = NULL）∞

**步驟**：
1. 為某租戶設定 `daily_review_cap = NULL`：
   ```sql
   UPDATE tenant_usage_caps
   SET daily_review_cap = NULL
   WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'acme');
   ```
2. 執行多次 review（例如 10 次）

**預期結果**：
- 不顯示用量狀態訊息
- 不受限制，可無限執行
- 每次都正常記錄到 `tenant_usage_events`

---

#### 案例 6：完全禁止（cap = 0）🔒

**步驟**：
1. 為某租戶設定 `daily_review_cap = 0`：
   ```sql
   UPDATE tenant_usage_caps
   SET daily_review_cap = 0
   WHERE tenant_id = (SELECT id FROM tenants WHERE slug = 'suspended');
   ```
2. 嘗試執行 review

**預期結果**：
- 顯示錯誤訊息：**"Your organization's review access has been disabled. Please contact your administrator."**
- 記錄 `review_denied` audit event，`deny_reason='usage_cap_reached'`

---

## 📊 監控與診斷

### 查看今日各租戶的用量
```sql
SELECT 
    t.slug AS tenant,
    t.name AS tenant_name,
    tc.daily_review_cap,
    COUNT(CASE WHEN tue.usage_type = 'review' THEN 1 END) AS today_review_count,
    tc.daily_download_cap,
    COUNT(CASE WHEN tue.usage_type = 'download' THEN 1 END) AS today_download_count
FROM tenants t
LEFT JOIN tenant_usage_caps tc ON t.id = tc.tenant_id
LEFT JOIN tenant_usage_events tue ON t.id = tue.tenant_id AND tue.created_at >= CURRENT_DATE
GROUP BY t.slug, t.name, tc.daily_review_cap, tc.daily_download_cap
ORDER BY today_review_count DESC;
```

### 查看達到上限的拒絕記錄
```sql
SELECT 
    created_at,
    tenant_slug,
    email,
    deny_reason,
    context->>'cap' AS cap,
    context->>'current_usage' AS usage
FROM audit_events
WHERE action = 'review_denied'
  AND deny_reason = 'usage_cap_reached'
  AND created_at >= CURRENT_DATE
ORDER BY created_at DESC;
```

---

## 🔄 下一步：Phase A2-3

完成 Phase A2-2 驗收後，下一步可以：
- **Phase A2-3**：Download 的 caps enforcement（類似 review，但用 `daily_download_cap`）
- **Phase A3**：固定化 Runbook SQL 模板
- **Phase B**：MVP Admin UI

---

## 📁 相關檔案

| 檔案 | 說明 |
|------|------|
| `app.py` | 主程式（已修改） |
| `README_PHASE_A2_2.md` | 本檔案 |
| `README_PHASE_A2_1.md` | Phase A2-1 說明（Portal SSO Enforcement） |
| `phase_a1_database_setup.sql` | Phase A1 完整 SQL |

---

**最後更新：** 2026-02-27  
**維護者：** Amanda Chiu
