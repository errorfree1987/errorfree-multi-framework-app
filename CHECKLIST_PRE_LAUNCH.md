# 月底上線前必測清單

> **目的**：確保大公司試用前，所有關鍵流程可運作  
> **適用對象**：CS/OPS 或專案負責人執行  
> **最後更新**：2026-03-04

---

## 一、學員資料收集與系統驗證

> **重要**：必須能收集學員 email 與手機號碼，並在系統中完成驗證。

### 1.1 學員資料收集

- [ ] **建立學員資料收集表單或 Excel 範本**
  - 欄位：`Email`（必填）、`手機號碼`（必填）、`姓名`（選填）
  - 範本格式可供企業 HR/承辦人填寫後回傳

- [ ] **確認收集流程**
  - 企業提供學員清單（email + 手機）→ 承辦人整理成標準格式 → 匯入系統或由 Admin 批量加入

### 1.2 學員匯入系統（email + 手機 + 姓名）

- [ ] **先執行 schema 遷移（僅需一次）**
  - 在 Supabase SQL Editor 執行 `sql_tenant_members_phone.sql`
  - 確認 `tenant_members.phone` 欄位已新增

- [ ] **Admin UI → Members → Batch Add Members**
  - 選擇租戶
  - **Paste**：一行一筆，格式 `email` 或 `email,phone` 或 `email,phone,display_name`
    - 範例：`user@example.com`、`user2@example.com,+886912345678`、`user3@example.com,+1-234-567-8900,Jane`
  - **Manual Entry**：逐筆輸入 Email、Phone（選填）、Display name（選填）
  - 點擊「Add All Members」或「Add Member」
  - 確認成功訊息與 `st.balloons()`

- [ ] **國際手機格式**（支援不同國家區域）
  - 範例：`+1 2345678900`（美加）、`+886 912345678`（台灣）、`+86 13800138000`（中國）、`+44 20 7946 0958`（英國）
  - 可含空格、破折號，系統會原樣儲存

- [ ] **檢查重複與錯誤**
  - 重複 email 會顯示警告，不影響已成功新增的
  - 無效 email（不含 `@`）會被過濾

### 1.3 驗證手機與姓名顯示

- [ ] 成員詳情（Member List → 展開）可顯示 Phone、Display name
- [ ] 有填寫 phone 的成員，詳情中會出現 **Phone** 欄位

### 1.4 學員登入驗證

- [ ] **學員可用 email 登入 Portal**
  - 開啟 Portal 網址
  - 輸入已加入系統的 email + 密碼（若為首次，依 Portal 流程設定）
  - 登入成功，可看到 Framework Catalog

- [ ] **學員可進入 Analyzer**
  - 從 Portal 點擊進入 Analyzer
  - 無「access denied」或「tenant inactive」等錯誤

- [ ] **抽查 3–5 位學員**
  - 隨機選 3–5 筆新增的 email，實際登入測試
  - 確認都能順利進入 Analyzer

---

## 二、租戶與權限流程

### 2.1 租戶建立

- [ ] Admin UI → Tenants → Create New Tenant
- [ ] 填寫 Slug、Name、Trial Days、Caps
- [ ] 建立成功，出現在 Tenant List
- [ ] 確認該租戶有對應的 epoch、usage caps 初始化

### 2.2 試用期管理

- [ ] **延長試用**：Extend (Add Days) 或 Set End Date
- [ ] **縮短試用**：Set End Date 選較早日期
- [ ] 確認 Trial End 更新正確，且 Sidebar / 診斷可反映

### 2.3 租戶啟用/停用

- [ ] 停用租戶 → 該租戶成員登入應被拒（tenant_inactive）
- [ ] 重新啟用 → 可再登入

---

## 三、成員管理

### 3.1 批量新增

- [ ] Paste Emails：一次貼上 10+ 個 email，Add All
- [ ] Manual Entry：逐筆新增
- [ ] Individual (Guest)：新增到 individual tenant
- [ ] 全系統重複檢查：同一 email 不可跨租戶重複

### 3.2 成員啟用/停用

- [ ] 單個 Disable → 該成員登入被拒（member_inactive）
- [ ] 單個 Enable → 可再登入
- [ ] 批量 Enable/Disable 正常

### 3.3 角色與用量上限

- [ ] 設定 member-level caps（Individual / Company 模式）
- [ ] 確認用量顯示（今日 review/download、進度條）正確

---

## 四、一鍵撤權

- [ ] Admin UI → Revoke Access
- [ ] 選擇租戶，輸入 tenant slug 確認
- [ ] 撤權成功，顯示新 epoch
- [ ] 該租戶現有 session 失效，需重新登入

---

## 五、用量限制（Caps）

### 5.1 顯示與警告

- [ ] Analyzer 內顯示今日用量（如 2/10）
- [ ] 接近上限時有警告
- [ ] 達上限時拒絕執行 review，並有明確錯誤訊息

### 5.2 調整 Caps

- [ ] Admin UI → Usage & Caps
- [ ] 調整 daily_review_cap、daily_download_cap
- [ ] Save 後立即生效
- [ ] Per-member caps 可個別設定（若為 Company 模式）

---

## 六、Audit Log

- [ ] Admin UI → Audit Logs
- [ ] 可依 tenant、action、result、時間範圍篩選
- [ ] 可查看單一事件的 context JSON
- [ ] 確認關鍵操作都有記錄（login、member_added、review、epoch_revoke 等）

---

## 七、Analyzer 與 AI Provider

### 7.1 多 Provider（Copilot / OpenAI）

- [ ] 租戶有 tenant_ai_settings 時，Sidebar 顯示正確 Provider（如 openai_compatible）
- [ ] 分析可正常跑完（Step 5 主分析、Step 6/7/8 等）
- [ ] 無 tenant_ai_settings 時，fallback 到 OPENAI_API_KEY

### 7.2 Tenant AI 設定

- [ ] Admin UI → Tenant AI Settings 可查看 Current Settings
- [ ] Admin UI → Edit/Add → Save 成功（錯誤時會顯示具體 HTTP 訊息）
- [ ] （Save 若失敗，見 TROUBLESHOOT_ADMIN_TENANT_AI_SAVE.md；或改用 SQL 直接寫入，見 GUIDE_TENANT_AI_COPILOT_DEEPSEEK.md）

---

## 八、Portal 與 Analyzer 端對端

- [ ] 從 Portal 登入 → 選擇 Framework → 進入 Analyzer
- [ ] 無跨域或 session 遺失問題
- [ ] 登出後再登入，狀態正確

---

## 九、部署與環境

- [ ] Railway 上 Portal、Analyzer、Admin UI 可正常存取
- [ ] 環境變數正確：SUPABASE_URL、SUPABASE_SERVICE_KEY、OPENAI_API_KEY
- [ ] 無 5xx 錯誤或長時間無回應

---

## 十、文件與交接

- [ ] 準備「客戶使用說明」（Portal 登入 → 進 Analyzer → 上傳文件 → 下載報告）
- [ ] 準備「Admin 操作 Checklist」（租戶建立、成員批量、Caps、Tenant AI 設定）
- [ ] 學員資料收集表單/Excel 範本已提供給企業承辦人

---

## 快速勾選一覽

| 區塊 | 項目數 | 狀態 |
|------|--------|------|
| 一、學員資料收集與驗證 | 12 | ☐ |
| 二、租戶與權限 | 6 | ☐ |
| 三、成員管理 | 8 | ☐ |
| 四、一鍵撤權 | 1 | ☐ |
| 五、用量限制 | 6 | ☐ |
| 六、Audit Log | 3 | ☐ |
| 七、Analyzer 與 AI Provider | 4 | ☐ |
| 八、Portal 端對端 | 3 | ☐ |
| 九、部署與環境 | 3 | ☐ |
| 十、文件與交接 | 3 | ☐ |

---

**備註**：
- 學員「手機號碼」若需存入系統，需先執行 schema 擴充與 Admin UI 擴充（見 1.3）。
- DeepSeek 已延後，本清單不包含 DeepSeek 測試。
- 遇問題可參考 RUNBOOK_MODE_A_OPERATIONS.sql、QUICK_REFERENCE_MODE_A.md。
