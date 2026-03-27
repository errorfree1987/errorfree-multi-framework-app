# Phase B 完成 / Phase C 進行中 — 交接摘要

> **用途**：貼到新 Cursor Chat，讓 AI 延續工作  
> **最後更新**：2026-03-27

---

## 一、貼給 AI 的起手文（直接複製貼上）

```
【Error-Free Admin 專案交接 — Phase C 進行中】

專案工作目錄：
- iMac:    /Users/amandachiu/Projects/errorfree-multi-framework-app
- MacBook: ~/Projects/errorfree-multi-framework-app
GitHub branch: staging-portal-sso

Phase B 已全部完成。Phase C 的 C1（AI Settings 管理頁）已完成。
請讀取 NEW_CHAT_Phase_B_C_交接摘要.md 了解詳細狀況，再讀取 PLAN_BYOK_ENTERPRISE.md 了解 Phase C 規劃。

我不會 coding，請用最簡單有細節的步驟說明。
```

---

## 二、Phase B — 全部完成 ✅

| 功能 | 狀態 |
|------|------|
| B2 租戶管理（編輯、Quick Extend、KPI卡片） | ✅ |
| B3 成員批量管理（CSV、個別編輯、自訂 cap） | ✅ |
| B4 撤權管理（風險標示、Emergency Revoke All） | ✅ |
| B5 Audit Log（時間線、篩選、匯出 CSV） | ✅ |
| B6 用量圖表（7/14/30天、卡片/表格視圖） | ✅ |
| Dashboard 首頁（KPI、到期警告、活動 feed） | ✅ |
| 舊版 admin_ui.py（加 deprecation banner） | ✅ |

---

## 三、Phase C — 進行中

### C1 ✅ AI Settings 管理頁（已完成）

**新增功能：左側導航 → AI Settings**

| 項目 | 說明 |
|------|------|
| 頁面位置 | `/dashboard/ai-settings` |
| 每個 tenant 一張卡片 | 點擊展開設定 |
| Provider 選單 | Copilot / OpenAI Compatible / DeepSeek |
| Model 欄位 | 手動輸入，有常用 model 快速選項 |
| Base URL | 可選填（企業自建 API endpoint 用） |
| API Key Ref | 只存環境變數**名稱**（如 `OPENAI_API_KEY`），key 本身不進 database |
| 儲存 → Audit Log | 每次 Save 自動寫入 `audit_events`（action = `tenant_ai_settings_updated`） |
| 3 個 KPI 卡片 | Total / Configured / Not Configured |

**相關檔案：**
- `admin-dashboard/src/app/dashboard/ai-settings/page.tsx`
- `admin-dashboard/src/app/api/ai-settings/route.ts`（GET：讀取所有 tenant 的 AI 設定）
- `admin-dashboard/src/app/api/ai-settings/update/route.ts`（POST：儲存設定 + 寫 audit log）

### C1 下一步（尚未做）

**Company Admin 自助入口** — 讓企業的 company_admin 用自己的 email 登入，只能看到並修改**自己 tenant** 的 AI 設定。這需要新的 email 登入機制（比目前的 ADMIN_PASSWORD 更複雜），建議作為 C1-B 階段實作。

### C2 ⏳ 企業級隔離（尚未開始）
- Row-Level Security（RLS）
- Tenant-claim JWT

### C3 ⏳ 可觀測性（尚未開始）
- Request ID 追蹤
- Sentry 錯誤監控

---

## 四、線上網址

| 服務 | 網址 |
|------|------|
| **新版 Next.js Admin Dashboard** | `https://empathetic-quietude-production-507e.up.railway.app` |
| **Streamlit Analyzer**（保留中） | `https://trustworthy-analysis-production-7beb.up.railway.app` |

---

## 五、重要檔案清單

```
errorfree-multi-framework-app/
├── admin-dashboard/
│   ├── src/app/
│   │   ├── dashboard/
│   │   │   ├── page.tsx              # 首頁 Dashboard
│   │   │   ├── tenants/page.tsx      # B2 租戶管理
│   │   │   ├── members/page.tsx      # B3 成員管理
│   │   │   ├── audit/page.tsx        # B5 Audit Log
│   │   │   ├── usage/page.tsx        # B6 用量圖表
│   │   │   ├── revoke/page.tsx       # B4 撤權管理
│   │   │   └── ai-settings/page.tsx  # C1 AI 設定（NEW）
│   │   └── api/
│   │       ├── ai-settings/route.ts         # GET AI 設定列表（NEW）
│   │       └── ai-settings/update/route.ts  # POST 儲存 AI 設定（NEW）
│   ├── railway.toml
│   └── .env.local  ← 每台電腦需自建（不進 Git）
├── admin_ui.py              # 舊版（有 deprecation banner）
├── app.py                   # Analyzer 主程式
├── PLAN_BYOK_ENTERPRISE.md  # Phase C 完整規劃
└── NEW_CHAT_Phase_B_C_交接摘要.md  # 本檔案
```

---

## 六、本機啟動指令

```bash
cd admin-dashboard
rm -rf .next        # 若遇到 JS 載入問題，先清快取
npm run dev
```

瀏覽器開啟（看 Terminal 顯示哪個 port，通常是 3000 或 3001）：
```
http://localhost:3000/dashboard/ai-settings
```

---

## 七、環境變數（.env.local — 每台電腦需手動建立，不進 Git）

檔案位置：`admin-dashboard/.env.local`

```
SUPABASE_URL=（你的 Supabase URL）
SUPABASE_SERVICE_KEY=（你的 service key）
ADMIN_PASSWORD=（你設定的管理員密碼）
```

---

## 八、使用者背景

- 專案負責人 **Amanda Chiu**，非工程師
- 操作說明需以「複製貼上指令」＋「逐步說明」為主
- 下週開放千人企業試用，C1 AI Settings 已完成，下一個優先項目為 C1-B（Company Admin 自助登入）

---

**最後更新**：2026-03-27
