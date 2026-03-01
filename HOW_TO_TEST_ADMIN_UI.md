# 🔍 如何在 Railway 找到並測試 Admin UI

## 步驟 1：找到 Admin UI 的 URL

### 在 Railway Dashboard：

1. **選擇服務**
   - 進入 Railway Dashboard
   - 在左側邊欄找到部署 Admin UI 的服務
   - 可能是：
     - `errorfree-admin` 或 `admin-ui`（如果有新增獨立服務）
     - `errorfree-multi-framework-app`（如果使用現有服務）

2. **查看部署狀態**
   - 點擊服務卡片
   - 點擊 "**Deployments**" 標籤
   - 確認最新部署狀態是 "✅ Success"
   - 查看部署時間（應該是最近幾分鐘）

3. **獲取 Public URL**
   - 點擊 "**Settings**" 標籤
   - 找到 "**Networking**" 區段
   - 在 "Public Networking" 下查看：
     ```
     example-production-xxxx.up.railway.app
     ```
   - 如果沒有 URL，點擊 "**Generate Domain**" 按鈕

4. **確認啟動指令（重要！）**
   - 在 "Settings" 標籤
   - 找到 "**Deploy**" 區段
   - 查看 "Start Command"：
     ```bash
     # Admin UI 應該是：
     streamlit run admin_ui.py --server.port $PORT --server.address 0.0.0.0
     
     # Analyzer 是：
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0
     ```

5. **如果啟動指令不對**
   - 有兩個選擇：
     
     **選擇 A：修改現有服務（快速測試）**
     - 將啟動指令改為 `streamlit run admin_ui.py ...`
     - 點擊 "Deploy" 重新部署
     - 注意：這會暫時無法使用 Analyzer
     
     **選擇 B：新增獨立服務（建議）**
     - 點擊 Railway 左上角 "New" → "GitHub Repo"
     - 選擇相同的 `errorfree-multi-framework-app` repo
     - 設定啟動指令為 `streamlit run admin_ui.py ...`
     - 這樣 Analyzer 和 Admin UI 可以同時運行

---

## 步驟 2：設定環境變數

在 Admin UI 服務中，確認以下環境變數已設定：

1. **進入 Variables 標籤**
   - 點擊服務
   - 點擊 "**Variables**" 標籤

2. **必需的環境變數**：

   ```bash
   # Admin UI 密碼（必須新增）
   ADMIN_PASSWORD=你的強密碼
   
   # Supabase（應該已存在）
   SUPABASE_URL=https://xxxxx.supabase.co
   SUPABASE_SERVICE_KEY=eyJhbGc...
   ```

3. **如果 ADMIN_PASSWORD 不存在**
   - 點擊 "+ New Variable"
   - Variable Name: `ADMIN_PASSWORD`
   - Value: 設定一個強密碼（例如：`SecureAdminPass2026!`）
   - 點擊 "Add"

4. **重新部署**
   - 修改環境變數後，Railway 會自動重新部署
   - 等待部署完成（約 1-2 分鐘）

---

## 步驟 3：測試 Admin UI

### 測試案例 1：訪問登入頁面

1. **開啟瀏覽器**
   - 訪問你的 Admin UI URL（從 Step 1 獲取）
   - 例如：`https://errorfree-admin-production.up.railway.app`

2. **預期結果**：
   ```
   ✅ 看到 "🔐 Error-Free® Admin Login" 標題
   ✅ 看到密碼輸入框
   ✅ 看到提示文字："Enter the admin password to access..."
   ```

3. **如果出現錯誤**：
   - Application Error → 檢查 Logs（Deployments → 最新部署 → View Logs）
   - 404 Not Found → 確認 URL 正確
   - Connection Timeout → 確認服務正在運行（應該顯示 "Online"）

---

### 測試案例 2：登入成功

1. **輸入正確密碼**
   - 在密碼框輸入你設定的 `ADMIN_PASSWORD`
   - 按下 Enter

2. **預期結果**：
   ```
   ✅ 成功進入 Dashboard
   ✅ 看到標題："🔐 Error-Free® Operations Dashboard"
   ✅ 右上角顯示："👤 admin@errorfree.com"
   ✅ 右上角有 "🚪 Logout" 按鈕
   ✅ 左側邊欄有導覽選單
   ✅ 主區域顯示 Dashboard 或 Supabase 連線狀態
   ```

---

### 測試案例 3：登入失敗（密碼錯誤）

1. **輸入錯誤密碼**
   - 登出（如果已登入）
   - 輸入錯誤的密碼（例如：`wrong123`）
   - 按下 Enter

2. **預期結果**：
   ```
   ✅ 顯示紅色錯誤訊息："❌ Incorrect password. Please try again."
   ✅ 保持在登入頁面
   ✅ 密碼框已清空
   ✅ 可以重新嘗試
   ```

---

### 測試案例 4：Session 持久化

1. **登入後重新整理**
   - 成功登入 Admin UI
   - 按下瀏覽器的重新整理按鈕（F5 或 Cmd+R）

2. **預期結果**：
   ```
   ✅ 不需要重新登入
   ✅ 保持在 Dashboard 頁面
   ✅ Session 狀態維持
   ✅ 右上角仍顯示用戶資訊
   ```

---

### 測試案例 5：登出

1. **點擊登出按鈕**
   - 點擊右上角的 "🚪 Logout" 按鈕

2. **預期結果**：
   ```
   ✅ 跳回登入頁面
   ✅ 所有 session state 清除
   ✅ 需要重新輸入密碼才能進入
   ```

---

## 步驟 4：驗證 Audit Log（在 Supabase）

### 檢查登入/登出記錄

1. **進入 Supabase Dashboard**
   - https://supabase.com/dashboard
   - 選擇你的專案

2. **開啟 SQL Editor**
   - 左側選單 → "SQL Editor"
   - 點擊 "New query"

3. **執行查詢**：
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

4. **預期結果**：
   ```
   ✅ 看到 admin_login (result: 'success') - 登入成功
   ✅ 看到 admin_login (result: 'denied') - 登入失敗（如果測試過）
   ✅ 看到 admin_logout (result: 'success') - 登出
   ✅ context 包含 {"source": "admin_ui", "timestamp": "..."}
   ```

---

## 🎉 驗收完成清單

測試完成後，確認以下項目：

- [ ] ✅ 可以訪問 Admin UI URL
- [ ] ✅ 登入頁面正常顯示
- [ ] ✅ 正確密碼可以登入
- [ ] ✅ 錯誤密碼顯示錯誤訊息
- [ ] ✅ 重新整理不需要重新登入
- [ ] ✅ 可以登出
- [ ] ✅ Dashboard 顯示正常
- [ ] ✅ Supabase 連線狀態顯示（連接或警告）
- [ ] ✅ Audit log 正確記錄所有操作

---

## 🆘 常見問題排查

### Q: 找不到 Admin UI 的 URL

**A**: 檢查步驟：
1. 確認服務已部署（左側邊欄應該顯示 "Online"）
2. 進入 Settings → Networking
3. 如果沒有 Public URL，點擊 "Generate Domain"
4. 等待 DNS 生效（約 1-2 分鐘）

### Q: 訪問 URL 顯示 "Application Error"

**A**: 檢查步驟：
1. 進入 Deployments → 最新部署 → View Logs
2. 查找錯誤訊息（ModuleNotFoundError, 環境變數錯誤等）
3. 常見原因：
   - 啟動指令錯誤
   - 環境變數未設定
   - Port 設定錯誤

### Q: 顯示 "Admin password not configured"

**A**: 
1. 進入 Variables 標籤
2. 確認 `ADMIN_PASSWORD` 環境變數已設定
3. 如果沒有，新增該變數
4. 重新部署

### Q: 登入後顯示 "Supabase not configured"

**A**: 這是正常的（如果只測試登入功能）
- Supabase 連線用於後續功能（Tenant 管理、Audit log 等）
- 登入功能不需要 Supabase
- Phase B2 會用到 Supabase 連線

---

## ✨ 測試成功後

恭喜！Phase B1.1 驗收完成！🎉

**下一步**：
- 準備好了就告訴我
- 我們可以開始 Phase B2.1（Tenant 管理）
- 預計時間：0.5-1 天

---

**建立時間**：2026-02-28  
**維護者**：Amanda Chiu
