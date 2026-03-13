# 新對話無縫銜接 — Analyzer 開發起點

> **用途**：開啟新對話時，將下方「給 AI 的起手文」整段貼上，即可延續上下文，不需重複說明。

---

## 一、專案架構（快速回顧）

```
errorfree-multi-framework-app/
├── errorfree-portal/       ← Portal（登入、catalog、launch）— 已更新 Magic Link
└── errorfree-multi-framework-app/  ← Analyzer（本 repo）
    ├── app.py              ← Analyzer 主程式（Streamlit）
    ├── admin_ui.py         ← Admin UI（租戶、成員、用量、Tenant AI 設定）
    └── ...
```

- **Portal**：登入已完成 Magic Link + Register/Sign in，與 Analyzer 無需改動。
- **Analyzer**：`/sso/verify` 流程不變，專注 `app.py` 與 `admin_ui.py` 更新即可。

---

## 二、本對話已完成的內容（無需重做）

| 項目 | 狀態 | 說明 |
|------|------|------|
| C: tenant_ai_settings Admin Save | ✅ 完成 | `admin_ui.py` 已強化錯誤處理，Save 失敗時顯示具體訊息 |
| TROUBLESHOOT_ADMIN_TENANT_AI_SAVE.md | ✅ 新增 | Admin Save 故障排除指南 |
| CHECKLIST_PRE_LAUNCH_手把手步驟.md | ✅ 新增 | 給非技術人員的手把手檢查步驟 |
| CHECKLIST_PRE_LAUNCH.md、ROADMAP.md | ✅ 已更新 | 變更紀錄與檔案結構 |

---

## 三、下一步：Analyzer 更新（ROADMAP 建議）

### 優先：LLM 依 tenant_ai_settings 切換 Provider

**現況**：`load_tenant_ai_settings_from_supabase()` 已存在，Sidebar 有顯示 Provider，但 LLM 呼叫仍用全域 OpenAI client。

**要做的**：
1. 實作 `_get_llm_client_for_tenant(tas)`：依 provider/base_url/api_key_ref 回傳對應 client
2. 支援 `openai_compatible`、`deepseek`、`copilot`
3. 將 `run_llm_analysis`、`_openai_simple` 等改為使用 tenant 專屬 client
4. 無 tenant_ai_settings 時 fallback 到 `OPENAI_API_KEY`

**相關檔**：`app.py`、`PROMPT_ANALYZER_MULTI_PROVIDER.md`、`ROADMAP.md`（Analyzer 多 Provider 章節）

### 之後：Company Admin BYOK、Phase C 等（依 ROADMAP）

---

## 四、稍後要做：手動驗證

- 依 `CHECKLIST_PRE_LAUNCH_手把手步驟.md` 完成全流程測試
- 建議在 Analyzer 更新完成後再做

---

## 五、給 AI 的起手文（貼到新對話）

```
【Error-Free Analyzer 開發銜接】

我從上一個對話轉來，請先讀取本 repo 的 NEW_CHAT_HANDOFF.md 了解背景。

專案：errorfree-multi-framework-app（Analyzer）
下一步：依 ROADMAP 做 Analyzer 更新，優先實作 LLM 依 tenant_ai_settings 切換 Provider（大陸用 DeepSeek、其他用 Copilot/OpenAI）。

已完成（勿重複）：
- tenant_ai_settings Admin Save 強化（admin_ui.py）
- TROUBLESHOOT_ADMIN_TENANT_AI_SAVE.md、CHECKLIST_PRE_LAUNCH_手把手步驟.md

請從 Analyzer 多 Provider 實作開始，並參考 app.py、PROMPT_ANALYZER_MULTI_PROVIDER.md、ROADMAP.md。
```

---

**最後更新**：2026-03-12
