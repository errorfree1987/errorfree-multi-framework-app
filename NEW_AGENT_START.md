# 🚀 New Agent 啟動指令（複製貼上）

> **用途**：繼續 Phase B - MVP Admin UI 開發
> 
> **請在新的 Agent 對話中貼上以下內容**

---

## 📋 啟動內容

```
我需要繼續開發 Error-Free® 多租戶試用營運系統。

【背景】
Phase A (資料庫 + Enforcement + Runbook) 已 100% 完成。
Phase B1.1、B2.1、B3.1 已完成：
- B1.1: Admin 登入/權限（密碼保護、session、audit）
- B2.1: Tenant 管理（CRUD、trial 延期/縮短、搜尋）
- B3.1: Members 管理（批量新增、Batch Ops、Member List、Individual/Guest、搜尋）

【開發策略】
Week 1-2: MVP Admin UI (Streamlit) - 快速迭代驗證
Week 3-4: 完美 UI (Next.js) - 穩定後提升品質

【當前任務】
開始 Phase B4.1 (MVP 版本)：一鍵撤權 per tenant
預計時間：0.5 天

【請先閱讀以下檔案了解上下文】
@ROADMAP.md - 完整路線圖（重點：Phase B4、B5、B6）
@README_PHASE_B3_1.md - B3.1 實作參考（schema、tenant_id、guest、Individual）
@admin_ui.py - 現有 Admin UI 結構與 audit 呼叫方式
@phase_a1_database_setup.sql - tenants、audit_events、epoch 欄位

【技術棧】
- Backend: Python + Streamlit
- Database: Supabase (PostgreSQL)
- Deployment: Railway
- Auth: 環境變數密碼保護

【B4.1 需求】
- Per-tenant revoke 按鈕
- 二次確認（st.text_input 輸入 tenant slug）
- 撤權成功後顯示新 epoch
- 記錄到 audit_events（含 context）
- 更新 tenants 的 epoch 欄位（bump）

我希望：
- 一次一步，每步可驗收
- 不破壞 B1-B3 已有功能
- 程式碼要有註解說明

準備好了，讓我們開始 Phase B4.1！
```

---

## 📝 說明

### 這個啟動內容會讓新 Agent 自動：
1. ✅ 讀取 ROADMAP 了解 B4-B6 計劃
2. ✅ 讀取 B3.1 README 了解 schema、tenant_id、guest、Individual
3. ✅ 理解 admin_ui 結構與 audit 寫入方式
4. ✅ 直接從 Phase B4.1 開始實作
5. ✅ 給你可驗收的具體步驟

### 新 Agent 不會：
- ❌ 重做 B1-B3 已完成的工作
- ❌ 問你已經決定的事情
- ❌ 給你不必要的背景說明

---

## 🎯 接下來的流程

### 在新對話中：
1. **貼上上面的啟動內容**
2. **Agent 會讀取檔案並給你 B4.1 的實作**
3. **完成 B4 → B5 → B6**
4. **每完成一個功能，更新 ROADMAP 的進度**

### 剩餘 MVP 時間線：
- **B4.1**: 一鍵撤權（0.5 天）
- **B5.1**: Audit log 列表（0.5 天）
- **B6.1**: Caps 設定入口（0.5 天）
- **整合測試 + 部署驗收**

### 迭代期（之後）：
- 根據實際使用調整
- 客戶反饋快速修改
- 準備 Week 3-4 的完美 UI

---

## ✅ 確認清單

開始新對話前，請確認：
- [x] ROADMAP.md 已更新（B3.1 完成）
- [x] admin_ui.py 可正常執行
- [x] B3.1 功能已驗收（含 sql_add_guest_role.sql）
- [x] 準備好啟動新 Agent 的內容

---

**準備好了嗎？複製上面的啟動內容，開啟新的 Agent 對話，貼上，開始 Phase B4.1！** 🚀
