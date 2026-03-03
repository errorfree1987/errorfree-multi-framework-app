# Error-Free® Multi-Tenant Trial Operations Roadmap

> **目的**：Error-Free® Portal + Analyzer 多租戶試用營運（Mode A）+ BYOK（Mode B）並行開發路線圖
> 
> **維護者**：Amanda Chiu
> 
> **最後更新**：2026-03-03

---

## 📋 目標總覽

### 短期目標（Mode A）：1 週內可讓客戶遠端試用
- ✅ 後台控管公司/人員/期限/撤權/成本
- ✅ 完整審計追蹤
- ✅ 基本用量控制

### 中期目標（Phase B）：混合模式開發（3-4 週）
- **Week 1-2**: MVP Admin UI (Streamlit) - 快速迭代驗證
  - 🔄 讓非工程 CS/OPS 也能安全操作
  - 🔄 快速根據反饋調整
  - 🔄 降低誤操作風險
- **Week 3-4**: 完美 UI (Next.js) - 穩定後提升品質
  - 🔄 企業級 UI/UX
  - 🔄 視覺化和圖表
  - 🔄 響應式設計

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

## 🔄 進行中項目（Phase B: 混合模式 Admin UI）

### Phase B0: 開發策略（已確認 - 2026-02-27）

**策略**：混合模式（MVP → 完美 UI）

**時間規劃**：
- **Week 1-2 (3-5 天實作 + 5-9 天迭代)**：MVP Admin UI (Streamlit)
  - 快速開發所有核心功能
  - 根據使用反饋快速調整
  - 驗證工作流程和需求
- **Week 3-4 (10-14 天)**：完美 UI (Next.js)
  - 需求已穩定，專注 UI/UX
  - 加入視覺化、圖表、響應式設計
  - 企業級品質

**範圍**：
- ✅ 只做「營運必需」功能
- ❌ 不做 BYOK（BYOK 走 Mode B 並行）
- ✅ 所有寫入操作必寫 audit
- ✅ Guardrails（防止誤操作）

**技術棧**：
- **Phase B1 (MVP)**: Streamlit + Supabase
- **Phase B2 (完美版)**: Next.js + Tailwind CSS + shadcn/ui + Supabase

---

### Phase B1: 登入/權限

#### B1.1 MVP 版本（已完成 - 2026-02-28）

**任務**：
- ✅ 簡單密碼保護（環境變數 ADMIN_PASSWORD）
- ✅ Session 狀態管理
- ✅ 登出功能
- ✅ 記錄所有登入/登出到 audit_events
- ✅ 獨立的 Admin UI（`admin_ui.py`）

**技術方案**：
- Streamlit `st.text_input(type="password")`
- `st.session_state` 管理登入狀態
- `hmac.compare_digest()` 防止時序攻擊

**驗收標準**：
- ✅ 輸入正確密碼可進入
- ✅ 重新整理不需要重新登入（session 持久）
- ✅ 可登出
- ✅ 所有操作記錄到 audit_events

**相關檔案**：
- `admin_ui.py` - Admin UI 主程式
- `README_PHASE_B1_1.md` - 詳細實作說明和驗收方式
- `RAILWAY_DEPLOY_ADMIN.md` - Railway 部署指南

#### B1.2 完美版本（預計 1-2 天）

**任務**：
- [ ] Next.js + Supabase Auth
- [ ] RBAC 角色定義（admin / ops）
- [ ] 權限檢查中介層
- [ ] 所有操作記錄 actor_email

**驗收標準**：
- [ ] Admin/Ops 分別登入
- [ ] Ops 受限於特定操作
- [ ] 所有操作記錄到 audit_events

---

### Phase B2: Tenant 管理

#### B2.1 MVP 版本（已完成 - 2026-03-02）✅

**任務**：
- ✅ 租戶列表（顯示基本資訊、搜尋租戶）
- ✅ 建立新租戶（st.form，自動清空）
- ✅ **Trial 延期**（兩種模式）⭐
  - ✅ Extend (Add Days) - 快速加天數
  - ✅ Set End Date - 日曆選擇器（可縮短或延長）
- ✅ 停用/啟用租戶（st.button + SQL UPDATE）
- ✅ 刪除租戶（cascading delete，需確認 slug）
- ✅ 基本統計（成員數、今日用量、epoch）

**UI 元件**：
- `st.tabs()` - 分離列表和建立
- `st.expander()` - 租戶詳情
- `st.form()` - 建立租戶表單（clear_on_submit=True）
- `st.metric()` - 顯示統計
- `st.radio()` - 選擇 trial 管理模式
- `st.date_input()` - 日曆選擇器（精確設定日期）

**實作內容**：
- 完整的 CRUD 操作
- Supabase REST API 整合
- 自動初始化（epoch, caps）
- 表單驗證和自動清空
- 錯誤處理和成功訊息延遲顯示
- Audit logging
- 靈活的試用期管理（延長/縮短/精確設定）

**關鍵功能升級**（2026-03-02）：
1. **靈活試用期管理**⭐
   - 可縮短試用期（修正誤操作）
   - 可設定精確日期（商業需求變更）
   - 智能反饋訊息（自動判斷延長/縮短）
   - 新 audit event：`tenant_trial_date_updated`

2. **表單自動清空**
   - 新增 `clear_on_submit=True`
   - 成功後顯示提示訊息
   - 防止重複提交錯誤

3. **成功訊息顯示修正**
   - `time.sleep()` 延遲重載
   - 確保用戶看到反饋訊息

4. **租戶搜尋**（2026-03-03）
   - Tenant List 新增搜尋框（依 slug/name 過濾）

**驗收標準**：
- ✅ 可建立新租戶（表單自動清空）
- ✅ 可延長試用期（兩種方式）
- ✅ 可縮短試用期（新功能）⭐
- ✅ 可停用/啟用租戶
- ✅ 可刪除租戶（需確認）
- ✅ 統計資訊正確顯示
- ✅ 所有操作記錄到 audit_events
- ✅ 成功訊息正確顯示

**相關檔案**：
- `admin_ui.py` - 完整實作（show_tenants, create_tenant, delete_tenant, update_tenant_trial_date）
- `README_PHASE_B2_1.md` - 詳細實作說明和驗收方式
- `TEST_B2_1_v2.md` - 完整測試指南（含修正後功能）
- `TEST_TRIAL_DATE_FEATURE.md` - 試用期靈活調整功能測試指南

#### B2.2 完美版本（預計 1-2 天）

**任務**：
- [ ] 美化表格（shadcn/ui DataTable）
- [ ] 搜尋/篩選/排序
- [ ] 視覺化（圖表顯示用量趨勢）
- [ ] 租戶詳情頁（完整儀表板）
- [ ] 快速操作（右鍵選單）

**驗收標準**：
- [ ] 可快速搜尋租戶
- [ ] 圖表顯示 7 天用量趨勢
- [ ] 響應式設計

---

### Phase B3: Users 批量管理

#### B3.1 MVP 版本（已完成 - 2026-03-03）✅

**任務**：
- ✅ 成員列表（按租戶篩選、搜尋、狀態篩選）
- ✅ 批量新增成員（貼上 email 30+、Manual Entry）
- ✅ Individual (Guest) 個人使用者
- ✅ 批量停用/啟用（checkbox 列表 + Select All/Active/Inactive）
- ✅ 角色設定（user / tenant_admin / guest）
- ✅ 批量刪除、單個刪除
- ✅ 全系統重複檢查（跨租戶）
- ✅ 租戶搜尋（Batch Add、Batch Operations）

**UI 元件**：
- `st.tabs()` - Member List / Batch Add / Batch Operations
- `st.text_area()` - 貼上 email 清單
- `st.checkbox()` - 成員列表個別勾選
- `st.expander()` - 成員詳情
- `st.radio()` - 狀態篩選、Add to (Tenant/Individual)
- `st.metric()` - Total / Active / Inactive
- 英文介面、表單自動清空

**實作內容**：
- `ensure_individual_tenant()`、`delete_member()`、`batch_delete_members()`
- `get_all_existing_emails_global()` - 全系統重複檢查
- Guest 角色需執行 `sql_add_guest_role.sql`

**關鍵功能**：
1. **批量新增** - 貼上清單、Manual Entry、Individual (Guest)
2. **Batch Operations** - 列表 + checkbox、Select All/Active/Inactive、搜尋成員
3. **Member List** - 搜尋、狀態篩選、批量刪除、單個 Delete 按鈕
4. **重複檢查** - 同一 email 跨所有租戶不可重複
5. **租戶搜尋** - Tenant Management、Batch Add、Batch Operations 三處

**相關檔案**：
- `admin_ui.py`、`README_PHASE_B3_1.md`、`QUICK_TEST_B3_1.md`
- `sql_add_guest_role.sql` - Guest 角色 DB 遷移

#### B3.2 完美版本（預計 1 天）

**任務**：
- [ ] CSV 上傳（拖拽）
- [ ] 進度條（批量處理時）
- [ ] 成功/失敗詳情
- [ ] 搜尋功能（模糊搜尋）

**驗收標準**：
- [ ] 可上傳 CSV（100+ email）
- [ ] 顯示處理進度
- [ ] 可搜尋成員

---

### Phase B4: 一鍵撤權

#### B4.1 MVP 版本（已完成 - 2026-03-03）✅

**任務**：
- ✅ Per-tenant revoke 按鈕
- ✅ 二次確認（st.text_input 輸入 tenant slug）
- ✅ 撤權結果顯示（新 epoch）
- ✅ 記錄到 audit_events（含 context: old_epoch, new_epoch, source）
- ✅ 更新 tenant_session_epoch 的 epoch 欄位（bump）

**UI 元件**：
- `st.button(type="primary")` - 紅色撤權按鈕
- `st.text_input()` - 確認輸入
- `st.success()` - 結果顯示

**驗收標準**：
- ✅ 需輸入 tenant slug 才能撤權
- ✅ 撤權成功顯示新 epoch
- ✅ 記錄到 audit_events（action: epoch_revoke）

#### B4.2 完美版本（預計 0.5 天）

**任務**：
- [ ] 動畫效果（倒數計時）
- [ ] 速率限制（10 分鐘內不可重複）
- [ ] 受影響 session 估算
- [ ] 撤權歷史（時間線）

**驗收標準**：
- [ ] 速率限制有效
- [ ] 顯示受影響 session 數量

---

### Phase B5: Audit log

#### B5.1 MVP 版本（已完成 - 2026-03-03）✅

**任務**：
- ✅ Audit log 列表（st.dataframe）
- ✅ 基本篩選（tenant, action, result）
- ✅ 時間範圍（Today / Last 7 days / Last 30 days）
- ✅ 詳情展開（selectbox 選事件 → st.json 顯示 context）

**UI 元件**：
- `st.selectbox()` - 篩選器
- `st.dataframe()` - 事件列表
- `st.json()` - Context 詳情

**驗收標準**：
- ✅ 可按 tenant/action/result 篩選
- ✅ 可選擇時間範圍
- ✅ 可查看 context JSON

#### B5.2 完美版本（預計 1 天）

**任務**：
- [ ] 時間線視圖（類似 GitHub Activity）
- [ ] 高級篩選（多選、模糊搜尋）
- [ ] 顏色編碼（成功/失敗）
- [ ] 匯出功能（CSV）

**驗收標準**：
- [ ] 時間線視圖美觀
- [ ] 可匯出篩選結果

---

### Phase B6: 基本限流/成本設定入口

#### B6.1 MVP 版本（已完成 - 2026-03-03）✅

**任務**：
- ✅ Per-tenant cap 設定（st.number_input + Unlimited checkbox）
- ✅ 保存按鈕（PATCH tenant_usage_caps）
- ✅ 顯示今日用量（review、download）
- ✅ 用量百分比（st.progress）
- ✅ 記錄 audit_events（usage_caps_updated）

**UI 元件**：
- `st.number_input()` - Cap 設定
- `st.progress()` - 用量進度條
- `st.form()` + 保存按鈕

**驗收標準**：
- ✅ 可調整 caps（含 Unlimited 選項）
- ✅ 可查看即時用量
- ✅ 設定立即生效

#### B6.2 完美版本（預計 1 天）

**任務**：
- [ ] 圖表顯示（折線圖、餅圖）
- [ ] 7 天用量趨勢
- [ ] 超限警告（自動通知）
- [ ] 批量設定（多個租戶）

**驗收標準**：
- [ ] 圖表美觀
- [ ] 可批量調整 caps

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
├── app.py                              # 主程式（Analyzer）
├── admin_ui.py                         # Admin UI（Phase B1.1）✨
├── README_PHASE_A2_1.md                # Phase A2-1 實作說明
├── README_PHASE_A2_2.md                # Phase A2-2 實作說明
├── README_PHASE_B1_1.md                # Phase B1.1 實作說明 ✨
├── RAILWAY_DEPLOY_ADMIN.md             # Railway 部署指南 ✨
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
- ✅ **Phase B1 (MVP 登入)**: 100% (1/1 子項完成)
- ✅ **Phase B2.1 (MVP Tenant)**: 100% (1/1 子項完成)
- ✅ **Phase B3.1 (MVP Members)**: 100% (1/1 子項完成)
- ✅ **Phase B4-B6 (MVP)**: 100% (3/3 子項完成)
- ⏳ **Phase B (完美版)**: 0% (0/6 子項)
- ⏳ **Phase C**: 0% (0/3 子項)

### 整體進度
- **Mode A (1 週 MVP)**: ✅ **100% 完成** (2026-02-27)
- **Phase B1.1 (MVP Admin 登入)**: ✅ **100% 完成** (2026-02-28)
- **Phase B2.1 (MVP Tenant 管理)**: ✅ **100% 完美完成** (2026-03-02) ⭐
- **Phase B3.1 (MVP Members 管理)**: ✅ **100% 完成** (2026-03-03) ⭐
- **Phase B4-B6 (MVP Admin UI)**: ✅ **全部完成** - B4.1、B5.1、B6.1 (2026-03-03)
- **Phase B (完美 Admin UI)**: ⏳ **未開始** (Week 3-4)
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

### 2026-03-03 Phase B6.1 Usage & Caps 完成
- ✅ 完成 Phase B6.1 (MVP Usage & Caps Management)
- 📝 實作 `show_usage()`、`get_tenant_usage_caps()`、`get_tenant_today_usage_full()`、`update_tenant_usage_caps()`
- 📝 顯示今日 review/download 用量、進度條、per-tenant cap 設定（含 Unlimited）
- 📝 記錄 audit_events（action: `usage_caps_updated`）
- 🎉 **Phase B4-B6 MVP 全部完成**

### 2026-03-03 Phase B5.1 Audit Log 完成
- ✅ 完成 Phase B5.1 (MVP Audit Log Viewer)
- 📝 實作 `show_audit_logs()` 和 `get_audit_events()` helper
- 📝 篩選：時間範圍（Today/7d/30d）、tenant、action、result
- 📝 st.dataframe 顯示事件列表、selectbox 選事件查看 context JSON

### 2026-03-03 Phase B4.1 一鍵撤權完成
- ✅ 完成 Phase B4.1 (MVP 一鍵撤權 per tenant)
- 📝 實作 `show_revoke()` 和 `revoke_tenant_sessions()` helper
- 📝 Per-tenant revoke 按鈕、二次確認（輸入 tenant slug）、顯示新 epoch
- 📝 記錄 audit_events（action: `epoch_revoke`，context: old_epoch, new_epoch, source）
- 📝 更新 tenant_session_epoch 的 epoch 欄位（bump）

### 2026-03-03 Phase B3.1 完整驗收 + 租戶搜尋
- ✅ Phase B3.1 測試完成、全部修正通過
- 📝 **B3.1  refinements**：Individual (Guest)、checkbox 列表、英文 UI、Manual form 清空、全系統重複檢查、delete member、sql_add_guest_role.sql
- 📝 **租戶搜尋**：Tenant Management、Batch Add Members、Batch Operations 三處新增搜尋
- 📝 更新 ROADMAP 反映 B3.1 完整功能

### 2026-03-02 ⭐ Phase B3.1 完成
- ✅ 完成 Phase B3.1 (MVP Members 批量管理)
- 📝 實作成員列表、批量新增、批量操作功能
- 📝 新增 3 個主要頁面（Member List / Batch Add / Batch Operations）
- 📝 新增 9 個 helper functions
- 📝 支援批量新增 30+ email（貼上清單）
- 📝 支援批量 enable/disable 成員
- 📝 新增角色管理功能（user/admin）
- 📝 新增按租戶篩選功能
- 📝 新增 `README_PHASE_B3_1.md`（完整實作指南，10 個測試用例）
- 📝 新增 `QUICK_TEST_B3_1.md`（快速測試指南，5-10 分鐘）
- 🎯 關鍵功能：
  - Email 驗證和清理（lowercase, trim, @ validation）
  - 重複 email 處理（HTTP 409 → warning）
  - 成功動畫（st.balloons）
  - 6 種 audit events（batch_added, enabled, disabled, role_updated 等）

### 2026-03-02 ⭐ Phase B2.1 完美完成
- ✅ **完美完成 Phase B2.1** (MVP Tenant 管理)
- 📝 新增靈活試用期管理功能
  - Extend (Add Days) - 快速延長
  - Set End Date - 日曆選擇器（可縮短或延長）
- 📝 新增 `update_tenant_trial_date()` 函數
- 📝 新增 `tenant_trial_date_updated` audit event
- 📝 UI 重新組織（4 欄 → 2 欄，更清晰）
- 📝 智能反饋訊息（自動判斷延長/縮短）
- 📝 新增 `TEST_TRIAL_DATE_FEATURE.md`（試用期調整測試指南）
- 🎯 解決關鍵問題：
  - 可以修正誤操作（錯誤輸入天數）
  - 可以應對商業需求變更（縮短試用期）
  - 可以精確設定日期（日曆選擇器）

### 2026-03-01
- ✅ 完成 Phase B2.1 基礎（MVP Tenant 管理）
- 📝 實作租戶列表、建立、延期、停用/啟用功能
- 📝 新增統計資訊（成員數、今日用量、epoch）
- 📝 新增表單自動清空功能
- 📝 新增刪除租戶功能（cascading delete）
- 📝 修正成功訊息顯示（time.sleep）
- 📝 新增 `README_PHASE_B2_1.md`（實作說明）
- 📝 新增 `TEST_B2_1_v2.md`（完整測試指南）
- 📝 更新 ROADMAP.md

### 2026-02-28
- ✅ 完成 Phase B1.1 (MVP Admin UI - 登入/權限)
- 📝 新增 `admin_ui.py`（獨立的 Admin UI）
- 📝 URL query parameter 實作 session persistence
- 📝 修正 Streamlit rerun warning
- 📝 新增 `README_PHASE_B1_1.md`（實作說明）
- 📝 新增 `SESSION_PERSISTENCE_FIX.md`（session 修正文件）
- 📝 新增 `RAILWAY_DEPLOY_ADMIN.md`（部署指南）

### 2026-02-27
- ✅ 完成 Phase A1 (資料庫結構)
- ✅ 完成 Phase A2-1 (Portal SSO Enforcement)
- ✅ 完成 Phase A2-2 (Usage Caps Enforcement)
- ✅ 完成 Phase A3 (Runbook 固定化)
- 📝 建立本 ROADMAP
- 📝 新增 `README_PHASE_A2_1.md`, `README_PHASE_A2_2.md`, `QUICK_REFERENCE_MODE_A.md`

### 未來更新
- 請在完成每個 Phase B 子項後更新進度
- 記錄重要決策和權衡（trade-offs）

---

**最後更新**: 2026-03-03 | **維護者**: Amanda Chiu
