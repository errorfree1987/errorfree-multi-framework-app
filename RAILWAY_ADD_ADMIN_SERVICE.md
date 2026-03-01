# 🚀 Railway 新增獨立 Admin UI 服務 - 詳細步驟

## 📋 目標

創建一個獨立的 Admin UI 服務，與現有的 Analyzer 服務並行運行。

---

## 步驟 1：新增服務（2 分鐘）

### 1.1 進入 Railway Dashboard

1. 開啟瀏覽器
2. 訪問：https://railway.app/dashboard
3. 選擇你的專案：**Error-Free® Technical Review**

### 1.2 新增服務

1. **點擊右上角的 "+ New" 按鈕**
   - 位置：專案標題旁邊
   - 或者點擊專案內的 "+ New Service" 按鈕

2. **選擇 "GitHub Repo"**
   - 會彈出 GitHub repository 選擇框

3. **選擇 Repository**
   - 找到並點擊：`errorfree-multi-framework-app`
   - 如果看不到，點擊 "Configure GitHub App" 重新授權

4. **選擇 Branch**
   - Branch: `staging-portal-sso`
   - 點擊 "Add Service" 或 "Deploy"

5. **等待初始部署**
   - Railway 會開始自動部署
   - 這個初始部署會失敗或運行錯誤的檔案（app.py）
   - **這是正常的！** 我們接下來會修正

---

## 步驟 2：重新命名服務（1 分鐘）

### 2.1 修改服務名稱

1. **點擊新創建的服務**
   - 在左側邊欄應該會看到新的服務卡片
   - 預設名稱可能是 `errorfree-multi-framework-app` 或類似

2. **進入 Settings**
   - 點擊 "Settings" 標籤

3. **修改名稱**
   - 找到 "Service Name" 欄位
   - 改為：`errorfree-admin-ui` 或 `admin-ui`
   - 點擊外部區域保存（或按 Enter）

---

## 步驟 3：設定啟動指令（2 分鐘）⭐ 重要！

### 3.1 設定 Start Command

1. **在 Settings 標籤中**
   - 找到 "Deploy" 區段
   - 找到 "Start Command" 欄位

2. **輸入啟動指令**
   ```bash
   streamlit run admin_ui.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
   ```

3. **重要提醒**：
   - ✅ 確認是 `admin_ui.py`（不是 `app.py`）
   - ✅ 確認有 `$PORT`（不是固定的 8501）
   - ✅ 確認有 `--server.address 0.0.0.0`

4. **保存**
   - 點擊外部區域或按 Tab 鍵保存

---

## 步驟 4：設定環境變數（3 分鐘）

### 4.1 進入 Variables 標籤

1. **點擊 "Variables" 標籤**
   - 在服務的主選單中

### 4.2 新增 ADMIN_PASSWORD

1. **點擊 "+ New Variable" 按鈕**

2. **輸入資訊**：
   ```
   Variable Name: ADMIN_PASSWORD
   Value: [你的強密碼]
   ```
   
3. **密碼建議**：
   - 至少 16 個字元
   - 包含大小寫字母、數字、符號
   - 例如：`ErrorFree2026Admin!@#`

4. **點擊 "Add" 按鈕**

### 4.3 新增 SUPABASE_URL

1. **點擊 "+ New Variable" 按鈕**

2. **輸入資訊**：
   ```
   Variable Name: SUPABASE_URL
   Value: [從現有服務複製]
   ```

3. **如何複製現有服務的值**：
   - 方法 A：如果你記得值，直接輸入
   - 方法 B：
     1. 另開一個分頁
     2. 進入 `errorfree-techincal review-app-staging` 服務
     3. 點擊 Variables
     4. 找到 `SUPABASE_URL`
     5. 點擊值旁邊的 "眼睛" 圖示顯示完整值
     6. 複製後回到 Admin UI 服務貼上

4. **點擊 "Add" 按鈕**

### 4.4 新增 SUPABASE_SERVICE_KEY

1. **點擊 "+ New Variable" 按鈕**

2. **輸入資訊**：
   ```
   Variable Name: SUPABASE_SERVICE_KEY
   Value: [從現有服務複製]
   ```

3. **同樣從現有服務複製**（如上述方法）

4. **點擊 "Add" 按鈕**

### 4.5 確認環境變數

確認你現在有 **3 個 Service Variables**：
- ✅ `ADMIN_PASSWORD`
- ✅ `SUPABASE_URL`
- ✅ `SUPABASE_SERVICE_KEY`

---

## 步驟 5：重新部署（2 分鐘）

### 5.1 觸發重新部署

**方法 A：自動觸發**
- 新增環境變數後，Railway 通常會自動重新部署
- 查看左側邊欄，服務卡片應該顯示 "Building..." 或 "Deploying..."

**方法 B：手動觸發**
- 如果沒有自動部署：
  1. 點擊 "Deployments" 標籤
  2. 點擊右上角的 "Deploy" 按鈕
  3. 選擇 "Redeploy"

### 5.2 監控部署進度

1. **點擊 "Deployments" 標籤**

2. **點擊最新的部署**
   - 會看到部署時間和狀態

3. **查看 Build Logs**
   - 點擊 "Build Logs" 標籤
   - 應該會看到類似：
     ```
     $ streamlit run admin_ui.py --server.port=$PORT ...
     ```
   - ✅ 確認是 `admin_ui.py`（不是 `app.py`）

4. **等待部署完成**
   - 通常需要 1-2 分鐘
   - 狀態會從 "Building" → "Deploying" → "Active"
   - 左側邊欄的服務卡片會顯示 "Online" ✅

---

## 步驟 6：生成 Public URL（1 分鐘）

### 6.1 設定網域

1. **回到 Settings 標籤**

2. **找到 "Networking" 區段**
   - 向下捲動到 "Public Networking"

3. **生成域名**
   - 點擊 "Generate Domain" 按鈕
   - Railway 會自動生成一個 URL
   - 格式類似：`https://errorfree-admin-ui-production-xxxx.up.railway.app`

4. **複製 URL**
   - 點擊 URL 旁邊的複製圖示
   - 或手動選取複製

---

## 步驟 7：測試 Admin UI（5 分鐘）🎉

### 7.1 訪問 Admin UI

1. **開啟新的瀏覽器分頁**

2. **貼上剛才複製的 URL**
   - 例如：`https://errorfree-admin-ui-production-xxxx.up.railway.app`

3. **等待載入**
   - 第一次訪問可能需要 5-10 秒

### 7.2 預期結果：登入頁面

你應該看到：
```
🔐 Error-Free® Admin Login
━━━━━━━━━━━━━━━━━━━━━━━━━━

[密碼輸入框]

💡 Note: This password is configured via the ADMIN_PASSWORD 
environment variable in Railway.
```

### 7.3 測試登入

1. **輸入密碼**
   - 輸入你設定的 `ADMIN_PASSWORD`
   - 按下 Enter

2. **預期結果：成功進入 Dashboard**
   ```
   🔐 Error-Free® Operations Dashboard
   
   右上角顯示：
   👤 admin@errorfree.com
   🚪 Logout
   
   左側邊欄顯示：
   📋 Navigation
   🏠 Dashboard
   🏢 Tenants
   👥 Members
   ...
   ```

3. **測試 Supabase 連線**
   - Dashboard 應該顯示：
     ```
     ✅ Supabase connected
     ```
   - 或如果環境變數有問題：
     ```
     ⚠️ Supabase not configured
     ```

### 7.4 測試其他功能

1. **測試重新整理**
   - 按 F5 或 Cmd+R
   - 應該保持登入狀態（不需重新輸入密碼）

2. **測試登出**
   - 點擊右上角 "🚪 Logout"
   - 應該跳回登入頁面

3. **測試錯誤密碼**
   - 輸入錯誤的密碼
   - 應該顯示：`❌ Incorrect password. Please try again.`

---

## 步驟 8：驗證 Audit Log（3 分鐘）

### 8.1 進入 Supabase

1. **開啟 Supabase Dashboard**
   - https://supabase.com/dashboard

2. **選擇你的專案**

3. **開啟 SQL Editor**
   - 左側選單 → "SQL Editor"

### 8.2 執行查詢

```sql
SELECT 
    created_at,
    action,
    email,
    result,
    deny_reason,
    context
FROM audit_events
WHERE action IN ('admin_login', 'admin_logout')
ORDER BY created_at DESC
LIMIT 10;
```

### 8.3 預期結果

應該看到：
- ✅ `admin_login` (result: 'success') - 成功登入
- ✅ `admin_login` (result: 'denied') - 登入失敗（如果測試過錯誤密碼）
- ✅ `admin_logout` (result: 'success') - 登出
- ✅ context 包含 `{"source": "admin_ui", "timestamp": "..."}`

---

## ✅ 驗收清單

完成後，確認以下項目：

### Railway 設定
- [ ] 新服務已創建（名稱：`errorfree-admin-ui` 或類似）
- [ ] 啟動指令正確（`streamlit run admin_ui.py ...`）
- [ ] 環境變數已設定（ADMIN_PASSWORD, SUPABASE_URL, SUPABASE_SERVICE_KEY）
- [ ] Public URL 已生成
- [ ] 服務狀態顯示 "Online" ✅

### 功能測試
- [ ] 可以訪問 Admin UI URL
- [ ] 登入頁面正常顯示
- [ ] 正確密碼可以登入
- [ ] 錯誤密碼顯示錯誤訊息
- [ ] 重新整理保持登入狀態
- [ ] 可以登出
- [ ] Dashboard 顯示正常
- [ ] Supabase 連線狀態正確

### Audit Log
- [ ] Supabase 中有 `admin_login` 記錄
- [ ] Supabase 中有 `admin_logout` 記錄
- [ ] Context 包含正確的資訊

### 並行運行確認
- [ ] Analyzer 仍可正常訪問（原有 URL）
- [ ] Admin UI 可正常訪問（新的 URL）
- [ ] 兩個服務互不干擾

---

## 🎉 完成！

恭喜你！現在你有：

✅ **Analyzer**（用戶使用）
- URL: `https://errorfree-techincal-review-app-staging-production.up.railway.app`
- 啟動指令：`streamlit run app.py`

✅ **Admin UI**（營運使用）
- URL: `https://errorfree-admin-ui-production-xxxx.up.railway.app`
- 啟動指令：`streamlit run admin_ui.py`

兩個服務可以同時運行，互不干擾！

---

## 🆘 常見問題

### Q: 部署失敗，顯示 "ModuleNotFoundError: No module named 'streamlit'"

**A**: 
1. 確認 `requirements.txt` 包含 `streamlit` 和 `requests`
2. 在 Settings → Deploy → 確認 "Install Command" 是否正確
3. 查看 Build Logs 確認套件是否安裝成功

### Q: 訪問 URL 顯示 "Application Error"

**A**: 
1. 檢查 Deploy Logs（Deployments → 最新部署 → Deploy Logs）
2. 常見原因：
   - 環境變數未設定
   - 啟動指令錯誤
   - Port 設定錯誤

### Q: Build Logs 仍顯示 "streamlit run app.py"

**A**: 
1. 確認啟動指令已正確設定為 `admin_ui.py`
2. 重新部署（Deployments → Deploy → Redeploy）
3. 清除快取後重新部署

### Q: 登入後顯示 "Supabase not configured"

**A**: 
1. 檢查 Variables 標籤
2. 確認 `SUPABASE_URL` 和 `SUPABASE_SERVICE_KEY` 已設定
3. 確認值是否正確（沒有多餘空格）
4. 重新部署

---

## 📞 需要協助？

如果遇到任何問題：
1. 截圖 Build Logs / Deploy Logs
2. 截圖 Variables 設定（遮蔽敏感資訊）
3. 告訴我錯誤訊息

我會協助你排查！

---

**建立時間**：2026-02-28  
**維護者**：Amanda Chiu  
**下一步**：完成驗收後，準備 Phase B2.1（Tenant 管理）🚀
