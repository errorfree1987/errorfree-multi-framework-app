# Member-Level Caps 實作規劃

> **目的**：讓 Individual 等租戶可針對**個別 member** 設定不同的 review/download 上限，真正做到 per-user 控管。
> 
> **建立日期**：2026-03-03
> 
> **✅ 實作完成**：2026-03-03

---

## 0. 前置步驟（必須先執行）

在 Supabase SQL Editor 執行 `sql_member_usage_caps.sql`，建立 `member_usage_caps` 表。

---

## 1. 現況與目標

### 現況
- Caps 僅支援 **tenant 層級**（`tenant_usage_caps`）
- Individual 租戶下所有 members 共用同一組 caps
- 無法依個別使用者（如 coco@gmail.com）設定不同上限

### 目標
- 支援 **member 層級** caps（可覆寫 tenant 預設）
- 檢查邏輯：優先使用 member cap，沒有則 fallback 到 tenant cap
- Admin UI 可搜尋 member 並設定/調整 per-member caps

---

## 2. Schema 設計

### 新增表：`member_usage_caps`

```sql
CREATE TABLE IF NOT EXISTS public.member_usage_caps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    email TEXT NOT NULL,
    daily_review_cap INTEGER,
    daily_download_cap INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_modified_by TEXT,
    notes TEXT,
    UNIQUE(tenant_id, email)
);

CREATE INDEX IF NOT EXISTS idx_member_usage_caps_tenant_email 
    ON public.member_usage_caps(tenant_id, email);

ALTER TABLE public.member_usage_caps
    ADD CONSTRAINT fk_member_usage_caps_tenant
    FOREIGN KEY (tenant_id) REFERENCES public.tenants(id) ON DELETE CASCADE;

COMMENT ON TABLE public.member_usage_caps IS 'Per-member usage caps (overrides tenant caps when present)';
COMMENT ON COLUMN public.member_usage_caps.daily_review_cap IS 'NULL=inherit tenant, 0=disabled, >0=limit';
COMMENT ON COLUMN public.member_usage_caps.daily_download_cap IS 'NULL=inherit tenant, 0=disabled, >0=limit';
```

### 語意
| 欄位值 | 意義 |
|--------|------|
| `NULL` | 繼承 tenant cap（或 unlimited 若 tenant 也為 NULL）|
| `0` |  disabled（該 member 無法使用）|
| `> 0` | 該 member 專屬上限 |

---

## 3. 檢查邏輯（app.py）

### 3.1 修改 `_check_usage_cap` 簽名

```python
def _check_usage_cap(tenant: str, usage_type: str, email: str = "") -> tuple[bool, int, int, str]:
```

- 新增 `email` 參數（選填）
- 當 `email` 有值時，先查 `member_usage_caps`，再 fallback 到 `tenant_usage_caps`
- 用量統計：member 層級時，只統計該 `email` 在該 `tenant_id` 下的今日用量

### 3.2 檢查流程

```
1. 取得 tenant_id（依 tenant slug）
2. 若 email 有值：
   a. 查 member_usage_caps WHERE tenant_id=? AND email=?
   b. 若有記錄：
      - 取得 member 的 daily_{usage_type}_cap
      - 若為 0 → 拒絕
      - 若為 NULL → 繼承 tenant cap（跳到 3）
      - 若 > 0 → 使用此 cap，用量統計 WHERE tenant_id=? AND email=?
   c. 若無記錄 → 繼承 tenant cap
3. 查 tenant_usage_caps
4. 取得 cap（member 或 tenant）
5. 統計用量（member 層級：tenant_id + email；tenant 層級：tenant_id）
6. 比較並回傳
```

### 3.3 呼叫點修改

```python
# app.py 約 3935 行
email = st.session_state.get("user_email", "")
allow, cap, current_usage, cap_message = _check_usage_cap(tenant, "review", email=email)
```

---

## 4. Admin UI 變更

### 4.1 Usage & Caps 頁面

- 在 tenant expander 的 **Members & Today's Usage** 區塊下方，每個 member 旁新增 **「Set Cap」** 按鈕
- 點擊後展開表單：Daily Review Cap、Daily Download Cap、Save
- 或：新增子分頁 **「Per-Member Caps」**，列出該租戶所有 members，可個別設定

### 4.2 建議 UI 結構

```
📊 Usage & Caps
├── 🔍 Search tenants
└── [Tenant expander]
    ├── 👥 Members & Today's Usage
    │   • coco@gmail.com: Review 3, Download 1  [Set Cap]
    │   • alice@example.com: Review 0, Download 0  [Set Cap]
    ├── 📈 Today's Usage (Total)
    └── ⚙️ Set Caps (tenant-level)
```

### 4.3 新增 Helper

- `get_member_usage_caps(tenant_id, email)` → 取得 member caps
- `update_member_usage_caps(tenant_id, email, daily_review_cap, daily_download_cap)` → 更新
- `init_member_usage_caps(tenant_id, email, ...)` → 建立（若無記錄）

---

## 5. Audit Events

- 新增 action：`member_caps_updated`、`member_caps_initialized`
- context：`member_email`, `daily_review_cap`, `daily_download_cap`

---

## 6. 實作步驟（建議順序）

| 步驟 | 內容 | 預估時間 |
|------|------|----------|
| 1 | 建立 migration SQL `sql_member_usage_caps.sql`，在 Supabase 執行 | 0.5h |
| 2 | 修改 app.py `_check_usage_cap`：新增 email 參數、查 member_usage_caps、用量依 email 篩選 | 1h |
| 3 | 修改 app.py 呼叫點，傳入 `user_email` | 0.5h |
| 4 | 在 admin_ui 新增 `get_member_usage_caps`、`update_member_usage_caps` | 0.5h |
| 5 | 在 Usage & Caps 的 Members 區塊加入 per-member cap 設定 UI | 1h |
| 6 | 記錄 audit events | 0.5h |
| 7 | 測試：Individual member 設定不同 caps，驗證 enforcement | 1h |

**總計約 5 小時**

---

## 7. 測試驗證

1. **Member cap 生效**：為 coco@gmail.com 設 Review=3，登入該帳號執行 3 次 review 後應被拒絕
2. **繼承 tenant**：無 member cap 時，使用 tenant cap
3. **Member cap 覆寫**：有 member cap 時，忽略 tenant cap（針對該 member）
4. **Admin UI**：搜尋 coco@gmail.com → 找到 individual → 設定 caps → 儲存成功

---

## 8. 檔案清單

| 檔案 | 變更 |
|------|------|
| `sql_member_usage_caps.sql` | 新建（migration）|
| `app.py` | 修改 `_check_usage_cap`、呼叫點 |
| `admin_ui.py` | 新增 helpers、Members 區塊 per-member UI |
| `phase_a1_database_setup.sql` | 選填：附加 member_usage_caps 建表 |
| `ROADMAP.md` | 更新 Phase B6 或新增 Phase B7 |

---

## 9. 風險與注意

- **效能**：每次 review 前多一次 DB 查詢（member_usage_caps），可接受
- **資料一致性**：member 刪除後，member_usage_caps 可保留（供 audit）或 CASCADE 刪除（需加 FK 到 tenant_members，較複雜，建議保留）
- **Individual 預設**：新 member 若無 member cap，仍用 tenant (individual) cap；可選在新增 member 時自動建立 member_usage_caps 預設值
