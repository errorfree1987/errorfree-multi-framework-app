# Error-Free® 千人企業 BYOK 與多 Provider 規劃（consolidated plan）

> **用途**：給千人以上大客戶使用的整體規劃，包含 BYOK、多 Provider、Company Admin 自管設定。  
> **維護者**：Amanda Chiu  
> **最後更新**：2026-03-12

---

## 一、客戶場景與需求

### 1.1 千人企業情境

- 客戶為千人以上大公司，使用微軟生態（如 Copilot、Azure、M365）
- 公司環境封閉：防火牆、資料合規、IT 審批
- 希望 **自己管理 AI 設定**（BYOK），不依賴 Error-Free 運維手動操作

### 1.2 企業使用流程（目標）

1. 企業取得 Portal 連結（如 `https://portal.errorfree.com`）
2. 員工用瀏覽器開啟連結 → 依企業 SSO 或 Magic Link 登入
3. 進入 Analyzer 上傳文件、執行審閱（LLM 在後端依 tenant 設定切換）
4. **Company Admin** 登入後台，自行設定本公司的 AI Provider、Base URL、API Key Ref 等（不需找工程師）

---

## 二、現況 vs 尚待完成

### 2.1 已完成 ✅

| 項目 | 說明 |
|------|------|
| **Analyzer 多 Provider** | `_get_llm_client_for_tenant()`、`chat.completions` 統一呼叫層，依 `tenant_ai_settings` 切換 Copilot / OpenAI / DeepSeek |
| **tenant_ai_settings 表** | Supabase 儲存 provider、base_url、model、api_key_ref |
| **Admin UI：Tenant AI 設定** | Error-Free 運維可在 Admin UI 為各租戶設定 Provider、Base URL、Model、API Key Ref |
| **Copilot / ChatGPT 測試** | 測試通過，provider=copilot 顯示正確，共用 `OPENAI_API_KEY` |
| **Company Admin 角色** | `company_admin` 角色已存在，可登入 Admin UI 看到 Company Admin Dashboard |
| **DeepSeek 預留** | 程式已支援，但 **暫時不開放**；有客戶需求時再啟用 |

### 2.2 尚待完成：Company Admin BYOK 自管設定 ⏳

| 項目 | 說明 |
|------|------|
| **Company Admin 設定頁** | 僅 `company_admin` 可見，可為「本公司」設定 tenant_ai_settings |
| **API Key 輸入 / 驗證** | 企業輸入自家 API Key（或 key ref），不存明文；驗證 key 有效 |
| **權限與安全** | Company Admin 僅能改自己租戶，不能改其他租戶；所有修改寫入 audit_events |

---

## 三、Provider 策略（2026-03-12 決策）

| Provider | 狀態 | 使用情境 |
|----------|------|----------|
| **Copilot / OpenAI compatible** | ✅ 現行使用 | 一般企業（含千人公司），使用 `OPENAI_API_KEY` 或企業自備 key |
| **DeepSeek** | 🔒 程式已支援，暫不開放 | 大陸客戶；有需求時再在 Admin 為該 tenant 啟用 |
| **未來（如 Azure OpenAI / Gemini）** | 規劃中 | 視企業要求擴充 |

---

## 四、BYOK 與 Company Admin 設定：實作計畫（Phase C）

### 4.1 目標

讓企業的 **Company Admin** 登入後，在介面中自行設定本公司租戶的 AI Provider / Base URL / API Key，**無需 Error-Free 工程師介入**。

### 4.2 功能範圍

- **入口**：Admin UI（或 Portal 內嵌）中「Company Admin Settings」或「本公司 AI 設定」
- **權限**：僅 `company_admin` 角色可見、可編輯
- **可設定項目**：
  - Provider（copilot / openai_compatible / 未來 DeepSeek）
  - Base URL（例如 Azure OpenAI endpoint；可選）
  - Model（例如 gpt-4o；可選）
  - API Key Ref 或 API Key 輸入（**不存明文**，僅存 ref 或加密後指標）

### 4.3 建議實作步驟

1. **資料層**：沿用 `tenant_ai_settings` 結構；必要時擴充欄位（如 `encrypted_key_ref`）
2. **權限**：Company Admin 只能讀寫 `tenant = 本公司 slug` 的 tenant_ai_settings
3. **UI**：在 Admin UI 的 Company Admin Dashboard 新增「AI Provider 設定」區塊
4. **API Key 安全**：
   - 方案 A：企業提供環境變數名稱（api_key_ref），Error-Free 在 server 端從 env 讀取
   - 方案 B：企業輸入 key，即時驗證後加密存於密鑰管理（如 Vault），或僅存 ref
5. **Audit**：每次儲存寫入 `audit_events`（action: `tenant_ai_settings_updated`, actor: company_admin email）

### 4.4 驗收標準

- [ ] Company Admin 登入後可見「本公司 AI 設定」
- [ ] 可選擇 Provider、填寫 Base URL、Model
- [ ] API Key 以安全方式處理（不存明文）
- [ ] 儲存後 Analyzer 依新設定切換 provider / model
- [ ] 變更記錄至 audit_events

---

## 五、與 ROADMAP 對照

| ROADMAP 章節 | 對應內容 |
|--------------|----------|
| **Phase B 已完成** | Admin UI（租戶、成員、用量、Tenant AI 設定） |
| **Analyzer 多 Provider** | 已完成；Copilot / DeepSeek 程式已支援 |
| **Company Admin BYOK 設定頁** | 本 plan 第四章；Phase C 優先項目 |
| **C1 BYOK Onboarding** | 含 API Key 輸入 / 驗證、key rotation |
| **C2 企業級隔離** | RLS、tenant-claim JWT |

---

## 六、檔案與快速連結

| 檔案 | 用途 |
|------|------|
| `ROADMAP.md` | 完整路線圖、Phase A/B/C、變更日誌 |
| `GUIDE_TENANT_AI_COPILOT_DEEPSEEK.md` | Copilot / DeepSeek 設定手順（運維操作） |
| `CHECKLIST_PRE_LAUNCH.md` | 上線前檢查清單 |
| `TROUBLESHOOT_ADMIN_TENANT_AI_SAVE.md` | Admin Save 故障排除 |
| `sql_tenant_ai_settings.sql` | tenant_ai_settings 表結構 |

---

## 七、給 AI 的起手文（新對話用）

```
【Error-Free BYOK / 千人企業規劃】

專案：errorfree-multi-framework-app
請讀取 PLAN_BYOK_ENTERPRISE.md 了解 BYOK、多 Provider、Company Admin 自管設定的整體規劃。

已完成：
- Analyzer 多 Provider（Copilot / OpenAI / DeepSeek 預留）
- Admin UI Tenant AI 設定（運維操作）
- DeepSeek 暫不開放，先專注 Copilot / ChatGPT

下一步：
- Company Admin BYOK 設定頁：讓企業 Company Admin 自行設定本公司 AI Provider / Base URL / API Key
- 參考 ROADMAP Phase C、本 plan 第四章
```
