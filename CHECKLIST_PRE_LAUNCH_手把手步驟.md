# 上線前檢查清單 — 手把手步驟（非技術人員版）

> 本文件專為**不熟悉程式的人**設計，用最白話的方式帶你完成每一項檢查。  
> 若某步驟需要工程師協助，會特別標註 **「需請工程師幫忙」**。

---

## 事前準備：你需要知道的網址

請先準備好以下網址（若沒有，請向工程師索取）：

| 名稱 | 網址範例 | 用途 |
|------|----------|------|
| **Portal** | 例如 `https://xxx.up.railway.app` | 學員登入、選擇 Framework、進入 Analyzer |
| **Admin UI** | 例如 `https://yyy.up.railway.app` | 管理員後台（建立租戶、加學員、設定 AI 等） |
| **Supabase 後台** | https://supabase.com/dashboard | 資料庫管理（部分步驟需在此操作） |

Admin 密碼、Portal 測試帳號也請一併取得。

---

## 一、Supabase 資料庫設定（需請工程師幫忙，只需做一次）

有些檢查需要先確保資料庫表已建立。**若你完全沒碰過 Supabase**，這一步請交給工程師。

**要請工程師做的事**：

1. 登入 Supabase Dashboard → 選擇專案
2. 左側點「SQL Editor」
3. 新增一個 Query，貼上 `sql_tenant_members_phone.sql` 檔案裡的全部內容
4. 執行（Run）
5. 若有 `sql_tenant_ai_settings.sql`，同樣方式執行一次

完成後，就可以繼續下面的 Admin UI 與 Portal 檢查。

---

## 二、Admin UI 登入（你可以在瀏覽器完成）

### 步驟 2-1：打開 Admin UI

1. 開啟瀏覽器（Chrome、Safari、Edge 皆可）
2. 在網址列輸入 **Admin UI 的網址**
3. 按 Enter

**你會看到**：一個登入頁，標題類似「Error-Free Admin」或「Operations Dashboard」，下面有一個密碼輸入框。

---

### 步驟 2-2：輸入密碼並登入

1. 在密碼框輸入你的 **Admin 密碼**（工程師提供）
2. 按 Enter 或點擊登入按鈕

**若成功**：會進入管理儀表板，左側有選單（Tenants、Members、Revoke Access 等）。

**若失敗**：會出現紅色錯誤訊息「Incorrect password」，請確認密碼是否正確。

---

### 步驟 2-3：確認可以登出

1. 找到右上角或側邊欄的「Logout」或「登出」按鈕
2. 點擊

**預期**：回到登入頁，需要再次輸入密碼才能進入。

---

## 三、租戶建立（在 Admin UI 操作）

### 步驟 3-1：建立新租戶

1. 登入 Admin UI
2. 在左側選單點「**Tenants**」或「租戶管理」
3. 找到「Create New Tenant」或「建立新租戶」按鈕
4. 點擊後會出現表單

---

### 步驟 3-2：填寫表單

依照畫面提示填寫：

- **Slug**：租戶代碼，英文小寫＋底線，例如 `company_abc`
- **Name**：公司或專案名稱，例如「測試公司」
- **Trial Days**：試用天數，例如 30
- **Caps**：用量上限（可先用預設值）

填好後點「Create」或「建立」。

---

### 步驟 3-3：確認建立成功

**預期**：會看到成功訊息，且新租戶出現在租戶列表中。

---

### 步驟 3-4：調整試用期（選做）

1. 在租戶列表中點選你要調整的租戶
2. 找到「Extend」或「Set End Date」相關按鈕
3. 試著延長或縮短試用期
4. 確認「Trial End」日期有正確更新

---

## 四、成員管理（把學員加入租戶）

### 步驟 4-1：進入批量新增成員

1. 在 Admin UI 左側選單點「**Members**」或「成員管理」
2. 找到「Batch Add Members」或「批量新增」
3. 點擊進入

---

### 步驟 4-2：選擇租戶

1. 在「Select tenant」下拉選單中選擇你要加人的租戶（例如 `company_abc`）
2. 確認選對

---

### 步驟 4-3：新增成員（兩種方式二選一）

**方式 A：貼上 email 清單**

1. 找到「Paste Emails」或「貼上 Email」的區塊
2. 在一行一個 email 的格式下貼上，例如：
   ```
   user1@example.com
   user2@example.com
   user3@example.com
   ```
3. 點擊「Add All Members」或「Add All」

**方式 B：手動一筆一筆輸入**

1. 找到「Manual Entry」或「手動新增」
2. 輸入 Email（必填）
3. 若有欄位，可選填 Phone、Display name
4. 點擊「Add Member」

---

### 步驟 4-4：確認成功

**預期**：會出現成功訊息（可能伴隨動畫），成員列表會顯示剛加入的 email。

---

### 步驟 4-5：抽查成員詳情

1. 在 Member List 中找到某位成員
2. 點擊展開或查看詳情
3. 確認可以見到 Phone、Display name 等欄位（若當初有填）

---

## 五、Tenant AI 設定（七、7.2 對應項目）

### 步驟 5-1：進入 Tenant AI Settings

1. 登入 Admin UI
2. 在左側選單點「**Tenant AI Settings**」或「租戶 AI 設定」
3. 會看到兩個分頁：「Current Settings」與「Edit / Add」

---

### 步驟 5-2：查看現有設定

1. 點選「**Current Settings**」分頁
2. 若有資料，會看到一個表格，列出各租戶的 Provider、Model 等
3. 若沒有資料，會顯示「No tenant AI settings yet」，這是正常的

---

### 步驟 5-3：試著儲存一筆新設定

1. 點選「**Edit / Add**」分頁
2. 在「Select tenant」選擇一個租戶
3. 在 Provider 選「(use default)」或「OpenAI compatible」
4. 其他欄位可先不填
5. 點擊「**Save**」按鈕

**若成功**：會看到綠色 success 訊息「Saved AI settings for tenant xxx」，且 Current Settings 會更新。

**若失敗**：會顯示紅色錯誤訊息。請把**完整錯誤內容**複製下來，交給工程師，並請他參考 `TROUBLESHOOT_ADMIN_TENANT_AI_SAVE.md`。

---

## 六、Portal 與 Analyzer 端對端測試

### 步驟 6-1：用學員帳號登入 Portal

1. 開新分頁，輸入 **Portal 網址**
2. 應會看到登入頁（Register / Sign in 按鈕）
3. 點「**Sign in**」
4. 輸入一個**已加入租戶的學員 email** 及密碼
   - 若是首次使用，需先走 Register → 驗證信箱 → 設定密碼 → 再 Sign in
5. 登入成功後，應看到 Framework Catalog（框架目錄）

---

### 步驟 6-2：進入 Analyzer

1. 在 Catalog 中點選任一 Framework（例如 Project Management）
2. 應該會跳轉到 Analyzer 畫面
3. **預期**：不會出現 access denied、tenant inactive 等錯誤

---

### 步驟 6-3：抽查多位學員

1. 登出 Portal
2. 用另外 3–5 個學員 email 重複步驟 6-1、6-2
3. 確認每一位都能登入並進入 Analyzer

---

### 步驟 6-4：確認用量顯示

1. 在 Analyzer 中查看側邊欄或上方
2. 應可看到今日用量（例如 2/10）
3. 若接近或達上限，應有提示

---

## 七、一鍵撤權（選做，可請工程師協助）

1. 登入 Admin UI
2. 點「**Revoke Access**」或「一鍵撤權」
3. 選擇要撤權的租戶
4. 依畫面指示輸入 tenant slug 確認
5. 執行撤權

**預期**：顯示成功，且該租戶學員需重新登入。

---

## 八、用量與 Caps 調整（選做）

1. 登入 Admin UI
2. 點「**Usage & Caps**」或「用量與上限」
3. 選擇租戶，試著調整 daily_review_cap、daily_download_cap
4. 點 Save
5. 確認儲存成功

---

## 九、Audit Log 檢查（選做）

1. 登入 Admin UI
2. 點「**Audit Logs**」
3. 依 tenant、action、時間等篩選
4. 隨機點幾個事件，看能否查看 context JSON
5. 確認有 login、member_added、review、epoch_revoke 等記錄

---

## 十、檢查完成後的回報方式

建議用表格整理給工程師，例如：

| 項目 | 結果 | 備註 |
|------|------|------|
| Admin 登入 | ✅ 通過 | |
| 租戶建立 | ✅ 通過 | |
| 成員批量新增 | ✅ 通過 | |
| Tenant AI Save | ✅ 通過 或 ❌ 失敗 | 若失敗，請附錯誤訊息 |
| Portal 學員登入 | ✅ 通過 | |
| 進入 Analyzer | ✅ 通過 | |
| 抽查 5 位學員 | ✅ 通過 | |

若有**需工程師處理**的項目，請一併列出，並附上你看到的錯誤訊息或畫面描述。

---

## 快速對照：CHECKLIST 與本文件的對應

| CHECKLIST 區塊 | 對應本文段落 |
|----------------|--------------|
| 一、學員資料收集與驗證 | 一（Supabase）、四（成員）、六（Portal 學員登入） |
| 二、租戶與權限 | 三 |
| 三、成員管理 | 四 |
| 四、一鍵撤權 | 七 |
| 五、用量限制 | 八、六-4 |
| 六、Audit Log | 九 |
| 七、Analyzer 與 AI Provider、Tenant AI 設定 | 五、六 |
| 八、Portal 端對端 | 六 |
| 九、部署與環境 | 由工程師在 Railway 檢查 |
| 十、文件與交接 | 準備給客戶的說明文件 |

---

**最後更新**：2026-03-12  
**適用對象**：非技術背景的 CS/OPS / 專案負責人
