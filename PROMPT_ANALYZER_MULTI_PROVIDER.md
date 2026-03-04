# 開啟新對話時可直接貼上的內容

---

請將下方整段複製，貼到下一個 Cursor 對話視窗即可開始。

---

## 貼上內容開始

```
專案：Error-Free® Analyzer 多 Provider 支援

目標：讓不同 tenant 使用不同的 AI 平台進行分析
- 大陸客戶 → DeepSeek
- 其他客戶 → Copilot / OpenAI

現況：
- app.py 已有 tenant_ai_settings（D4），從 Supabase 讀取 provider, base_url, model, api_key_ref, max_tokens_per_request
- 登入時已載入並顯示於 sidebar
- 但 LLM 呼叫（run_llm_analysis, _openai_simple 等）仍固定使用全域 OpenAI client

請協助：
1. 建立 _get_llm_client_for_tenant(tas) 或類似函式，依 tenant_ai_settings 建立對應的 client
2. 支援 provider=deepseek（DeepSeek API，OpenAI-compatible base URL）
3. 支援 provider=openai_compatible / copilot（沿用現有 OpenAI 邏輯）
4. 將 run_llm_analysis、_openai_simple、OCR 等 LLM 呼叫改為使用 tenant 專屬 client
5. 無 tenant_ai_settings 或 provider 未設定時 fallback 到現有 OPENAI_API_KEY

參考：ROADMAP.md 中的「Analyzer 多 Provider 支援（DeepSeek / Copilot）」章節。
```

---

## 貼上後可補充的額外需求（可選）

若需一併處理 Admin UI 管理 tenant_ai_settings，可再加：

```
此外，請在 Admin UI 新增「Tenant AI 設定」頁面，讓管理員可為各 tenant 設定 provider、base_url、model、api_key_ref（或從 env 選取）。
```
