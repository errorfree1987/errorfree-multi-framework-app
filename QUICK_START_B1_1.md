# 🚀 Phase B1.1 快速開始指南

## 你已經完成了什麼？

✅ **Admin UI 程式碼**（`admin_ui.py`）- 已創建
✅ **實作文檔**（`README_PHASE_B1_1.md`）- 已完成
✅ **部署指南**（`RAILWAY_DEPLOY_ADMIN.md`）- 已完成
✅ **路線圖更新**（`ROADMAP.md`）- 已更新

---

## 接下來要做什麼？

### 第一步：本地測試（5 分鐘）

在推送到 Railway 之前，先在本地測試：

```bash
# 1. 設定環境變數（臨時）
export ADMIN_PASSWORD="test123"
export SUPABASE_URL="https://xxxxx.supabase.co"
export SUPABASE_SERVICE_KEY="eyJhbGc..."

# 2. 啟動 Admin UI
streamlit run admin_ui.py

# 3. 訪問（應該自動開啟瀏覽器）
# http://localhost:8501
```

**預期結果**：
- 看到登入頁面
- 輸入 `test123` 可以登入
- 看到 Dashboard

---

### 第二步：推送到 Git（2 分鐘）

```bash
# 查看變更
git status

# 應該看到：
# - admin_ui.py（新增）
# - README_PHASE_B1_1.md（新增）
# - RAILWAY_DEPLOY_ADMIN.md（新增）
# - ROADMAP.md（修改）

# 新增檔案
git add admin_ui.py README_PHASE_B1_1.md RAILWAY_DEPLOY_ADMIN.md ROADMAP.md QUICK_START_B1_1.md

# 提交
git commit -m "Phase B1.1: Add MVP Admin UI with login/auth

- Add admin_ui.py (independent Admin UI)
- Add login/logout with password protection
- Add session management
- Add audit logging for all admin operations
- Update ROADMAP.md with Phase B1.1 completion
"

# 推送
git push
```

---

### 第三步：部署到 Railway（5 分鐘）

#### 選項 A：新增獨立服務（建議）

1. **進入 Railway 專案**
   - https://railway.app/dashboard

2. **新增服務**
   - 點擊 "New" → "GitHub Repo"
   - 選擇 `errorfree-multi-framework-app`
   - Service Name: `errorfree-admin` 或 `admin-ui`

3. **設定環境變數**
   - Settings → Variables
   - 新增：
     ```
     ADMIN_PASSWORD=YourSecurePassword123!@#
     SUPABASE_URL=https://xxxxx.supabase.co
     SUPABASE_SERVICE_KEY=eyJhbGc...
     ```

4. **設定啟動指令**
   - Settings → Deploy → Start Command
   ```bash
   streamlit run admin_ui.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
   ```

5. **生成域名**
   - Settings → Networking → Generate Domain
   - 例如：`https://errorfree-admin-production.up.railway.app`

6. **重新部署**
   - 點擊 "Deploy" 或等待自動部署

7. **訪問測試**
   - 使用生成的域名
   - 輸入 `ADMIN_PASSWORD`
   - 成功進入 Dashboard！

#### 選項 B：暫時替換現有服務（測試用）

如果只想快速測試，可以暫時修改現有服務：

1. **修改啟動指令**
   ```bash
   streamlit run admin_ui.py --server.port $PORT --server.address 0.0.0.0
   ```

2. **新增環境變數**
   ```
   ADMIN_PASSWORD=test123
   ```

3. **重新部署**

4. **測試完成後恢復**
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

---

### 第四步：驗收測試（10 分鐘）

按照 `README_PHASE_B1_1.md` 中的驗收清單：

- [ ] 測試案例 1：登入成功 ✅
- [ ] 測試案例 2：登入失敗（密碼錯誤）❌
- [ ] 測試案例 3：Session 持久化 🔄
- [ ] 測試案例 4：登出 🚪
- [ ] 測試案例 5：環境變數未設定 ⚠️

**驗證 Audit Log**：
```sql
-- 在 Supabase SQL Editor 執行
SELECT 
    created_at,
    action,
    email,
    result,
    context
FROM audit_events
WHERE action IN ('admin_login', 'admin_logout')
ORDER BY created_at DESC
LIMIT 10;
```

---

## 完成！🎉

Phase B1.1 已完成！現在你有：

✅ 獨立的 Admin UI（不影響 Analyzer）
✅ 簡單密碼保護
✅ Session 管理（重新整理不需重新登入）
✅ Audit Logging（所有操作記錄）
✅ Railway 部署（可立即使用）

---

## 下一步：Phase B2.1（Tenant 管理）

準備好了嗎？讓我知道，我會幫你實作：

**Phase B2.1 功能**：
- 租戶列表（查看所有租戶）
- 建立新租戶（表單）
- Trial 延期（日期選擇器）
- 停用/啟用租戶（按鈕）
- 基本統計（成員數、今日用量）

**預計時間**：0.5-1 天
**檔案修改**：只需修改 `admin_ui.py`（在 `show_tenants()` 函數中實作）

---

## 需要幫助？

### 常見問題

**Q: 本地測試時，Supabase 連線失敗？**

A: 確認環境變數已設定：
```bash
echo $SUPABASE_URL
echo $SUPABASE_SERVICE_KEY
```

如果沒有顯示，請重新執行 `export` 指令。

**Q: Railway 部署後，顯示 "Application Error"？**

A: 檢查 Logs：
1. Railway Dashboard → 選擇服務
2. Deployments → 最新部署 → View Logs
3. 查看錯誤訊息

常見原因：
- 環境變數未設定
- 啟動指令錯誤
- Port 設定錯誤

**Q: 可以同時使用 Analyzer 和 Admin UI 嗎？**

A: 可以！使用「選項 A：新增獨立服務」，兩個服務會有不同的 URL。

---

## 資源連結

- 📖 **完整實作說明**：`README_PHASE_B1_1.md`
- 🚂 **Railway 部署指南**：`RAILWAY_DEPLOY_ADMIN.md`
- 🗺️ **完整路線圖**：`ROADMAP.md`
- 💻 **Admin UI 程式碼**：`admin_ui.py`

---

**準備好開始 Phase B2 了嗎？讓我知道！** 🚀

---

**最後更新**：2026-02-28  
**維護者**：Amanda Chiu
