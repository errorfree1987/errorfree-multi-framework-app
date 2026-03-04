# Tenant AI 設定：先 Copilot，再 DeepSeek

本指南協助你一步一步完成「方案 B」：讓不同租戶使用不同 AI（先做 Copilot，再做 DeepSeek）。

---

## 步驟一：在 Supabase 建立 `tenant_ai_settings` 表（只需做一次）

1. 打開 **Supabase Dashboard** → 你的專案 → **SQL Editor**。
2. 複製並貼上 `sql_tenant_ai_settings.sql` 檔案裡的 **整段 SQL**（從 `CREATE TABLE` 到最後的 `SELECT` 都可以貼上）。
3. 按 **Run** 執行。
4. 若沒有錯誤，表 `tenant_ai_settings` 就已建立完成。

---

## 步驟二：先做 Copilot（給「其他客戶」用的租戶）

### 2.1 在 Railway 確認環境變數

- 若 Copilot / OpenAI 使用同一個 API Key，確認已有 **`OPENAI_API_KEY`**（Analyzer 本來就在用）。
- 若某租戶要用自己的 key，可在 Railway 新增一個環境變數，例如：  
  **`OPENAI_API_KEY_TENANT_A`** = 該租戶的 API key（名稱可自訂）。

### 2.2 在 Admin UI 為該租戶設定 Copilot

1. 啟動 **Admin UI**（例如：`streamlit run admin_ui.py`，或打開你部署的 Admin 網址）。
2. 左側選 **🤖 Tenant AI Settings**。
3. 切到 **✏️ Edit / Add** 分頁。
4. **Select tenant**：選要使用 Copilot 的租戶（例如「其他客戶」的公司）。
5. 設定：
   - **Provider**：選 **Copilot**（或 **OpenAI compatible**）。
   - **Base URL**：可留空（會用預設 OpenAI endpoint）；若 Copilot 有給專用網址再填。
   - **Model**：可留空（用系統預設），或填例如 `gpt-4o`。
   - **API Key (env variable name)**：選 **OPENAI_API_KEY**（或你為該租戶設的 env 名稱，如 `OPENAI_API_KEY_TENANT_A`）。
   - **Max tokens per request**：可填 0（用預設）或依需要填數字。
6. 按 **💾 Save**。
7. 到 **📋 Current Settings** 分頁確認該租戶的 Provider 顯示為 Copilot。

### 2.3 驗證 Copilot 是否生效

1. 用該租戶的帳號登入 **Analyzer**（Portal SSO 或你現有登入方式）。
2. 上傳一份文件，執行 Step 5 主分析。
3. 若分析正常產出，且 Sidebar 顯示 **Provider: copilot**（或 openai_compatible），即表示 Copilot 設定成功。

---

## 步驟三：再做 DeepSeek（給「大陸客戶」用的租戶）

### 3.1 在 Railway 設定 DeepSeek 環境變數

1. 到 **Railway**（或你部署的環境）→ 對應的 Service → **Variables**。
2. 新增：
   - **名稱**：`DEEPSEEK_API_KEY`
   - **值**：你的 DeepSeek API key（從 DeepSeek 後台取得）。
3. （可選）若 DeepSeek 有指定不同 endpoint，可再新增 **`DEEPSEEK_BASE_URL`**，例如 `https://api.deepseek.com`；未設則程式會用預設。

### 3.2 在 Admin UI 為大陸客戶租戶設定 DeepSeek

1. 在 Admin UI 左側選 **🤖 Tenant AI Settings** → **✏️ Edit / Add**。
2. **Select tenant**：選「大陸客戶」對應的租戶。
3. 設定：
   - **Provider**：選 **DeepSeek**。
   - **Base URL**：可留空（會用 `DEEPSEEK_BASE_URL` 或預設 `https://api.deepseek.com`）。
   - **Model**：可留空或填 DeepSeek 模型名，例如 `deepseek-chat`。
   - **API Key (env variable name)**：選 **DEEPSEEK_API_KEY**。
   - **Max tokens per request**：可填 0 或依需要填。
4. 按 **💾 Save**。
5. 到 **📋 Current Settings** 確認該租戶的 Provider 為 **deepseek**。

### 3.3 驗證 DeepSeek 是否生效

1. 用大陸客戶租戶的帳號登入 Analyzer。
2. 上傳文件並執行 Step 5 主分析。
3. 若分析正常產出，且 Sidebar 顯示 **Provider: deepseek**，即表示 DeepSeek 設定成功。

---

## 常見問題

- **沒有設定任何 Tenant AI 的租戶會怎樣？**  
  會自動使用原本的 **OPENAI_API_KEY** 與 OpenAI 行為，不需改動。

- **Admin UI 儲存時出現錯誤？**  
  請確認步驟一已執行 `sql_tenant_ai_settings.sql`，且 Supabase 連線與 `SUPABASE_SERVICE_KEY` 正確。

- **Analyzer Sidebar 顯示 source: not_found？**  
  代表該租戶在 `tenant_ai_settings` 尚無一筆設定；不影響使用，會 fallback 到全域 OPENAI_API_KEY。

---

## 檔案對照

| 檔案 | 用途 |
|------|------|
| `sql_tenant_ai_settings.sql` | 在 Supabase 建立 `tenant_ai_settings` 表（步驟一） |
| Admin UI → 🤖 Tenant AI Settings | 為各租戶設定 Provider、Base URL、Model、API Key Ref（步驟二、三） |
| `app.py` | Analyzer 依 `tenant_ai_settings` 選擇 Copilot / DeepSeek / 預設 |

完成以上步驟後，即可做到：**先 Copilot（其他客戶），再 DeepSeek（大陸客戶）**。
