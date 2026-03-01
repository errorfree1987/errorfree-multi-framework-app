# Error-Free® Multi-Tenant Trial Operations Roadmap

> **目的**：Error-Free® Portal + Analyzer 多租戶試用營運（Mode A）+ BYOK（Mode B）並行開發路線圖
> 
> **維護者**：Amanda Chiu
> 
> **最後更新**：2026-02-27

---

## 📋 目標總覽

### 短期目標（Mode A）：1 週內可讓客戶遠端試用
- ✅ 後台控管公司/人員/期限/撤權/成本
- ✅ 完整審計追蹤
- ✅ 基本用量控制

### 中期目標（Phase B）：2 週內完成 MVP Admin UI
- 🔄 讓非工程 CS/OPS 也能安全操作
- 🔄 降低誤操作風險
- 🔄 提升操作效率

### 長期目標（Mode B BYOK）：企業級並行
- ⏳ BYOK onboarding + key rotation
- ⏳ RLS + tenant-claim JWT
- ⏳ Wrapper + tests + observability

---

## ✅ 已完成項目（Phase A: 1 週 MVP）

### Phase A1: 資料庫結構（已完成 - 2026-02-27）

#### 建立的表結構
| 表名 | 用途 | 狀態 |
|------|------|------|
| `tenants` | 租戶基本資訊、試用期限、狀態 | ✅ 完成 |
| `tenant_members` | 租戶成員管理、個別啟用/停用 | ✅ 完成 |
| `tenant_session_epoch` | 撤權版本控制（epoch） | ✅ 完成（D3） |
| `audit_events` | 審計日誌（所有關鍵操作） | ✅ 完成 |
| `tenant_entitlements` | 功能權限控制（framework/analyzer） | ✅ 完成 |
| `tenant_usage_caps` | 使用上限設定（daily caps） | ✅ 完成 |
| `tenant_usage_events` | 實際用量記錄 | ✅ 完成 |

#### 相關檔案
- `phase_a1_database_setup.sql` - 完整資料庫設置 SQL
- 驗收狀態：✅ 所有表已建立並通過驗收

---

### Phase A2: Enforcement 強制執行（已完成 - 2026-02-27）

#### A2-1: Portal SSO Enforcement
**功能**：Portal verify 成功後立即檢查租戶/成員狀態

**實作內容**：
- ✅ `_check_tenant_and_member_access()` - 檢查租戶和成員存取權限
  - 租戶 `is_active` 檢查
  - 租戶 `trial_end` 過期檢查
  - 成員 `is_active` 檢查
- ✅ `_log_audit_event()` - 記錄審計日誌
- ✅ Portal verify 拒絕時顯示友善錯誤訊息
- ✅ 所有驗證結果記錄到 `audit_events`

**驗收結果**：
- ✅ 租戶停用檢查（`tenant_inactive`）
- ✅ 試用過期檢查（`trial_expired`）
- ✅ 成員停用檢查（`member_inactive`）
- ✅ 所有拒絕都記錄到 audit log

**相關檔案**：
- `README_PHASE_A2_1.md` - 詳細實作說明和驗收方式
- `app.py` (修改行：350-447, 840-870)

---

#### A2-2: Basic Usage Caps Enforcement
**功能**：執行 review 前檢查今日用量，達到上限時拒絕

**實作內容**：
- ✅ `_check_usage_cap()` - 檢查今日用量是否達上限
  - 讀取 `tenant_usage_caps`
  - 統計今日已用量（`tenant_usage_events`）
  - 支援 unlimited (`cap=NULL`) 和 disabled (`cap=0`)
- ✅ `_record_usage_event()` - 記錄使用事件
- ✅ Analyzer 啟動時記錄 `analyzer_launch` audit event
- ✅ Review 前顯示用量狀態（info / warning）
- ✅ 達到上限時拒絕並記錄 `review_denied` audit event

**驗收結果**：
- ✅ 顯示用量狀態（`1/3`, `2/3`）
- ✅ 剩餘 ≤ 5 時顯示警告
- ✅ 達到上限時拒絕執行
- ✅ 每次 review 記錄到 `tenant_usage_events`

**相關檔案**：
- `README_PHASE_A2_2.md` - 詳細實作說明和驗收方式
- `app.py` (修改行：439-650, 3136-3160, 3961-4015)

---

### Phase A3: Runbook 固定化（已完成 - 2026-02-27）

**功能**：整理所有日常營運操作的 SQL 模板

**實作內容**：
- ✅ 租戶管理（建立、延長試用、停用/啟用）
- ✅ 成員管理（批量新增、停用/啟用）
- ✅ 撤權操作（單一/批量 epoch bump）
- ✅ 權限管理（啟用/停用功能）
- ✅ 用量管理（查看/調整 caps）
- ✅ 審計查詢（篩選 tenant/user/action）
- ✅ 診斷查詢（誰在線、即將到期、接近上限）

**相關檔案**：
- `RUNBOOK_MODE_A_OPERATIONS.sql` - 完整 SQL runbook (927 行)
- `QUICK_REFERENCE_MODE_A.md` - 快速參考卡
- `sql_epoch_management.sql` - Epoch 撤權管理 SQL（D3）
- `QUICK_REFERENCE.md` - Epoch 快速參考卡（D3）

**使用方式**：
- 複製需要的 SQL 模板
- 替換參數值
- 在 Supabase SQL Editor 執行

---

## 🔄 進行中項目（Phase B: 2 週 MVP Admin UI）

### Phase B0: 定位與範圍（規劃中）

**目標**：讓非工程 CS/OPS 也能安全操作

**範圍**：
- ✅ 只做「營運必需」功能
- ❌ 不做 BYOK（BYOK 走 Mode B 並行）
- ✅ 所有寫入操作必寫 audit
- ✅ Guardrails（防止誤操作）

---

### Phase B1: 登入/權限（預計 1-2 天）

**任務**：
- [ ] Admin UI 登入機制
  - 選項 1：走 Portal-only SSO（推薦，複用現有）
  - 選項 2：獨立 admin login
- [ ] RBAC 角色定義
  - `admin`：全權操作
  - `ops`：可操作但受 guardrail 限制
- [ ] 權限檢查邏輯
- [ ] 所有操作記錄 actor_email

**技術方案**：
- 使用 Streamlit（快速原型）
- 或 Next.js + Supabase Auth（更完整）

**驗收標準**：
- [ ] Admin 可登入並看到儀表板
- [ ] Ops 可登入但受限於特定操作
- [ ] 所有操作記錄到 audit_events

---

### Phase B2: Tenant 管理（預計 0.5-1 天）

**任務**：
- [ ] 租戶列表（顯示狀態、試用期、成員數）
- [ ] 建立新租戶（表單，含 trial 設定）
- [ ] Trial 延期/轉正
- [ ] 停用/啟用租戶
- [ ] 查看租戶詳情
  - 近 24h/7d 活躍用戶
  - Review/下載量
  - 最後活動時間
- [ ] 快速入口：一鍵進 tenant 詳情、查看 audit

**UI 元件**：
- 租戶列表表格（可排序、篩選）
- 租戶詳情頁
- 操作按鈕（延期、轉正、停用）

**驗收標準**：
- [ ] 可建立新租戶並設定 trial
- [ ] 可延長試用期
- [ ] 可停用/啟用租戶
- [ ] 可查看租戶使用狀況

---

### Phase B3: Users 批量管理（預計 0.5-1 天）

**任務**：
- [ ] 成員列表（按租戶篩選）
- [ ] 批量新增成員
  - 匯入/貼上 email 清單（30-100+）
  - 設定預設 role
- [ ] 批量停用/啟用
- [ ] 角色設定（tenant_admin / user）
- [ ] 搜尋功能
  - 依 email 查租戶
  - 查看狀態、最後活動

**UI 元件**：
- 文字框（貼上 email 清單）
- 批量操作按鈕
- 成員列表表格

**驗收標準**：
- [ ] 可貼上 30+ email 並批量新增
- [ ] 可批量停用/啟用成員
- [ ] 可搜尋成員

---

### Phase B4: 一鍵撤權（預計 0.5 天）

**任務**：
- [ ] Per-tenant revoke 按鈕
- [ ] Guardrails
  - Ops 需二次確認（輸入 tenant slug）
  - 或「原因必填」
  - 速率限制（同一 tenant 10 分鐘最多撤權 1 次）
- [ ] 撤權結果顯示
  - 目前 epoch
  - 上次撤權時間
  - 受影響 session（估算）

**UI 元件**：
- 撤權按鈕（紅色，醒目）
- 確認對話框
- 結果顯示

**驗收標準**：
- [ ] Admin 可一鍵撤權
- [ ] Ops 需二次確認或填原因
- [ ] 速率限制有效
- [ ] 撤權記錄到 audit_events

---

### Phase B5: Audit log（預計 0.5 天）

**任務**：
- [ ] Audit log 列表（可篩選）
  - Tenant
  - Email
  - Action
  - Time range
  - Result
- [ ] 重要事件高亮
  - `verify`
  - `deny_reason`
  - `revoke`
  - `member_change`
  - `trial_end_change`
  - `entitlement_change`

**UI 元件**：
- 篩選器（下拉選單、日期選擇器）
- 事件列表表格
- 詳情展開（顯示 context JSON）

**驗收標準**：
- [ ] 可按 tenant/email/action 篩選
- [ ] 可選擇時間範圍
- [ ] 可查看事件詳情

---

### Phase B6: 基本限流/成本設定入口（預計 0.5 天）

**任務**：
- [ ] Per-tenant cap 設定
  - 每日 review/download
  - 單次最大檔案大小
  - 最大輪次
- [ ] 超限處理策略設定
  - Block
  - Degrade
  - Require admin approval
- [ ] UI 顯示今日用量 vs cap

**UI 元件**：
- Caps 設定表單
- 用量儀表板（進度條）
- 保存按鈕

**驗收標準**：
- [ ] 可調整 caps
- [ ] 可查看即時用量
- [ ] 設定立即生效

---

## ⏳ 未來項目（Phase C: Mode B BYOK 並行）

### C1: BYOK Onboarding（未開始）

**任務**：
- [ ] BYOK 租戶建立流程
- [ ] API Key 輸入/驗證
- [ ] Key rotation 機制
- [ ] Per-tenant provider config

**預計時間**：1-2 週

---

### C2: 企業級隔離（未開始）

**任務**：
- [ ] Supabase RLS 設定
- [ ] Tenant-claim JWT
- [ ] 避免 service-role bypass

**預計時間**：1 週

---

### C3: Wrapper + Tests + Observability（未開始）

**任務**：
- [ ] Request ID
- [ ] Structured logs
- [ ] Sentry 整合
- [ ] 單元測試
- [ ] 整合測試

**預計時間**：2-3 週

---

## 📂 檔案結構總覽

```
errorfree-multi-framework-app/
├── app.py                              # 主程式（已修改：A2-1, A2-2）
├── README_PHASE_A2_1.md                # Phase A2-1 實作說明
├── README_PHASE_A2_2.md                # Phase A2-2 實作說明
├── phase_a1_database_setup.sql         # Phase A1 完整 SQL
├── RUNBOOK_MODE_A_OPERATIONS.sql       # Phase A3 完整 runbook (927 行)
├── QUICK_REFERENCE_MODE_A.md           # Phase A3 快速參考卡
├── sql_epoch_management.sql            # Epoch 撤權管理（D3）
├── QUICK_REFERENCE.md                  # Epoch 快速參考卡（D3）
├── test_epoch_revoke.sh                # Epoch 自動化測試腳本（D3）
├── test_epoch_revoke.md                # Epoch 測試手冊（D3）
├── CHECKLIST.txt                       # Epoch 驗收清單（D3）
└── ROADMAP.md                          # 本檔案
```

---

## 🎯 優先順序建議

### 立即優先（Phase B）
1. **B1 登入/權限** - 基礎（必須先完成）
2. **B2 Tenant 管理** - 最常用
3. **B4 一鍵撤權** - 緊急需求
4. **B3 Users 批量** - 常用操作
5. **B5 Audit log** - 診斷/合規
6. **B6 Caps 設定** - 成本控制

### 中期推進（Phase C）
- C1, C2, C3 可並行推進，不影響 Phase A/B

---

## 🔧 技術棧總覽

### 目前使用
- **Backend**: Python + Streamlit
- **Database**: Supabase (PostgreSQL)
- **Deployment**: Railway
- **Auth**: Portal-only SSO (已實作)

### Phase B 建議
- **選項 1（快速）**: Streamlit Admin UI
  - 優點：快速原型、與現有程式碼整合
  - 缺點：UI 較簡陋
- **選項 2（完整）**: Next.js + Supabase Auth
  - 優點：更好的 UI/UX、獨立部署
  - 缺點：開發時間較長

---

## 📊 進度追蹤

### 完成度總覽
- ✅ **Phase A1**: 100% (7/7 表)
- ✅ **Phase A2**: 100% (2/2 子項)
- ✅ **Phase A3**: 100% (Runbook 完成)
- 🔄 **Phase B**: 0% (0/6 子項)
- ⏳ **Phase C**: 0% (0/3 子項)

### 整體進度
- **Mode A (1 週 MVP)**: ✅ **100% 完成**
- **Phase B (2 週 Admin UI)**: 🔄 **進行中**
- **Mode B BYOK (長期)**: ⏳ **未開始**

---

## 🆘 交接注意事項

### 新工程師 Onboarding（建議閱讀順序）

#### 1. 理解目標和架構（30 分鐘）
- 閱讀本 ROADMAP.md（本檔案）
- 理解 Mode A 和 Mode B 的差異

#### 2. 了解資料庫結構（30 分鐘）
- 閱讀 `phase_a1_database_setup.sql`
- 在 Supabase Table Editor 中瀏覽各表

#### 3. 理解 Enforcement 邏輯（1 小時）
- 閱讀 `README_PHASE_A2_1.md`（Portal SSO Enforcement）
- 閱讀 `README_PHASE_A2_2.md`（Usage Caps Enforcement）
- 在 `app.py` 中搜尋 `Phase A2` 註解，查看實作

#### 4. 熟悉 Runbook 操作（1 小時）
- 閱讀 `QUICK_REFERENCE_MODE_A.md`
- 在 Supabase 執行幾個常用操作
- 嘗試建立測試租戶、批量新增成員

#### 5. 開始 Phase B 開發（持續）
- 從 B1 (登入/權限) 開始
- 每完成一個子項，更新本 ROADMAP 的進度

### 關鍵知識點

#### 資料庫設計
- **租戶識別**: 使用 `slug` (text) 作為主要識別，`id` (uuid) 作為內部主鍵
- **Epoch 機制**: `tenant_session_epoch.epoch` 只增不減，用於撤權
- **Fail-open 策略**: Supabase 連線失敗時允許存取（避免鎖死所有用戶）

#### 安全考量
- **Service key 保護**: `SUPABASE_SERVICE_KEY` 只在 server-side 使用，不暴露給前端
- **Audit 記錄**: 所有寫入操作都記錄 `actor_email`
- **Guardrails**: 危險操作（撤權、刪除）需二次確認

#### 效能考量
- **索引**: 所有常用查詢欄位都已建立索引
- **JSONB**: `context` 欄位使用 JSONB 靈活存儲額外資訊
- **避免 N+1**: 使用 JOIN 而非迴圈查詢

---

## 📞 聯絡資訊

- **維護者**: Amanda Chiu
- **Repository**: [GitHub URL]
- **Deployment**: Railway
- **Database**: Supabase

---

## 📝 變更日誌

### 2026-02-27
- ✅ 完成 Phase A1 (資料庫結構)
- ✅ 完成 Phase A2-1 (Portal SSO Enforcement)
- ✅ 完成 Phase A2-2 (Usage Caps Enforcement)
- ✅ 完成 Phase A3 (Runbook 固定化)
- 📝 建立本 ROADMAP

### 未來更新
- 請在完成每個 Phase B 子項後更新進度
- 記錄重要決策和權衡（trade-offs）

---

**最後更新**: 2026-02-27 | **維護者**: Amanda Chiu
