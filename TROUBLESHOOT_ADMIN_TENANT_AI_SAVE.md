# Admin UI：Tenant AI Settings Save 故障排除指南

> 當 Admin UI → Tenant AI Settings → Save 失敗時，依此指南排查。

---

## 1. 確認表已建立

**步驟**：在 Supabase SQL Editor 執行：

```sql
SELECT 1 FROM public.tenant_ai_settings LIMIT 1;
```

- **若報錯**：`relation "tenant_ai_settings" does not exist` → 請執行 `sql_tenant_ai_settings.sql` 整段 SQL
- **若成功**：表已存在，繼續下一步

---

## 2. 讀取 Save 錯誤訊息（2026-03 已強化）

Save 失敗時，Admin UI 會顯示具體錯誤，例如：

| 錯誤 | 可能原因 | 處理方式 |
|------|----------|----------|
| `HTTP 404` | 表不存在或 URL 錯誤 | 執行 `sql_tenant_ai_settings.sql`；確認 `SUPABASE_URL` 正確 |
| `HTTP 401` / `JWT expired` | Service key 無效或過期 | 檢查 `SUPABASE_SERVICE_KEY` |
| `HTTP 403` / `permission denied` | RLS 阻擋 | 見下方「RLS 設定」 |
| `HTTP 422` / `invalid input` | 欄位型別或約束不符 | 對照 `sql_tenant_ai_settings.sql` 結構 |
| `Connection error` / `timeout` | 連線問題 | 檢查 Supabase URL、防火牆、網路 |

---

## 3. RLS（Row Level Security）設定

若 Supabase 專案啟用 RLS 且 `tenant_ai_settings` 未設 policy，`SUPABASE_SERVICE_KEY` 通常仍可 bypass RLS。

若仍被擋，可在 Supabase SQL Editor 執行：

```sql
-- 檢視 tenant_ai_settings 是否啟用 RLS
SELECT relname, relrowsecurity
FROM pg_class
WHERE relname = 'tenant_ai_settings';

-- 若 relrowsecurity = true 且 service_role 仍無法寫入，可為 service_role 允許所有操作
-- （service_role 預設 bypass RLS，通常不需額外 policy）

-- 若要對 anon 或其他 role 開放，需建立 policy，例如：
-- CREATE POLICY "Allow service role full access"
--   ON public.tenant_ai_settings
--   FOR ALL
--   USING (true)
--   WITH CHECK (true);
```

**建議**：`tenant_ai_settings` 僅由 Admin UI（使用 service role）存取，無需對外開放；維持現有 RLS 或關閉 RLS 即可。

---

## 4. 用 SQL 直接寫入（驗證用）

若 Admin Save 持續失敗，可先用 SQL 驗證資料是否能寫入：

```sql
INSERT INTO public.tenant_ai_settings (tenant, provider, base_url, model, api_key_ref, last_modified_by)
VALUES ('your-tenant-slug', 'openai_compatible', NULL, NULL, 'OPENAI_API_KEY', 'admin');
```

- 成功 → 表與權限正常，多半為 Admin UI 程式或環境變數問題
- 失敗 → 檢查 Supabase 權限、RLS、表結構

---

## 5. 環境變數確認

Admin UI 需以下變數：

| 變數 | 用途 |
|------|------|
| `SUPABASE_URL` | 例：`https://xxxxx.supabase.co` |
| `SUPABASE_SERVICE_KEY` | Service role key（可 bypass RLS） |

取得方式：Supabase Dashboard → Settings → API → Project URL、service_role key

---

## 6. 相關檔案

| 檔案 | 說明 |
|------|------|
| `sql_tenant_ai_settings.sql` | 建立 `tenant_ai_settings` 表 |
| `admin_ui.py` | `upsert_tenant_ai_settings()`、`show_tenant_ai_settings()` |
| `GUIDE_TENANT_AI_COPILOT_DEEPSEEK.md` | Copilot / DeepSeek 設定流程 |

---

**最後更新**：2026-03-12
