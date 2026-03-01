# 🚂 Railway 部署指南 - Admin UI

## 快速部署步驟（5 分鐘）

### 選項 A：新增獨立的 Admin UI 服務（建議）

這個方式可以讓 Analyzer 和 Admin UI 同時運行，互不干擾。

#### 步驟：

1. **在 Railway 專案中新增服務**
   - 進入你的 Railway 專案
   - 點擊 "New" → "GitHub Repo"
   - 選擇相同的 repo（errorfree-multi-framework-app）

2. **設定環境變數**（在新服務中）
   ```bash
   # Admin UI 必須
   ADMIN_PASSWORD=your_secure_password_here
   
   # Supabase（應該已存在，確認一下）
   SUPABASE_URL=https://xxxxx.supabase.co
   SUPABASE_SERVICE_KEY=eyJhbGc...
   ```

3. **設定啟動指令**
   - 在 Railway 服務設定中
   - Settings → Deploy → Start Command
   ```bash
   streamlit run admin_ui.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
   ```

4. **生成域名**
   - Settings → Networking → Generate Domain
   - 會得到類似：`https://admin-production-xxxx.up.railway.app`

5. **部署**
   - 點擊 "Deploy" 或推送程式碼
   - 等待部署完成（約 1-2 分鐘）

6. **訪問 Admin UI**
   - 使用生成的域名訪問
   - 輸入 `ADMIN_PASSWORD` 登入

---

### 選項 B：單一服務（測試用）

如果只想快速測試，可以暫時替換現有服務的啟動指令。

#### 步驟：

1. **新增環境變數**（在現有服務中）
   ```bash
   ADMIN_PASSWORD=your_secure_password_here
   ```

2. **暫時修改啟動指令**
   - Settings → Deploy → Start Command
   ```bash
   # 原本（Analyzer）
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   
   # 改為（Admin UI）
   streamlit run admin_ui.py --server.port $PORT --server.address 0.0.0.0
   ```

3. **重新部署**
   - 會暫時無法使用 Analyzer

4. **測試完成後恢復**
   - 將啟動指令改回 `streamlit run app.py`

**注意**：這個方式只適合測試期間，正式使用請用選項 A。

---

## 環境變數說明

### 必須設定

| 變數名稱 | 說明 | 範例 |
|---------|------|------|
| `ADMIN_PASSWORD` | Admin UI 登入密碼 | `MySecurePass123!@#` |
| `SUPABASE_URL` | Supabase 專案 URL | `https://xxxxx.supabase.co` |
| `SUPABASE_SERVICE_KEY` | Supabase service_role key | `eyJhbGc...` |

### 安全提醒

- ⚠️ `ADMIN_PASSWORD` 請使用強密碼（16+ 字元）
- ⚠️ 不要將密碼寫入程式碼或 Git
- ⚠️ `SUPABASE_SERVICE_KEY` 是敏感資訊，不要暴露

---

## 啟動指令參數說明

```bash
streamlit run admin_ui.py \
  --server.port $PORT \              # Railway 自動設定的 port
  --server.address 0.0.0.0 \         # 監聽所有 IP（Railway 需要）
  --server.enableCORS false \        # 允許 CORS
  --server.enableXsrfProtection false  # 關閉 XSRF 保護（Railway 環境）
```

---

## 驗證部署成功

1. **訪問 Admin UI URL**
   - 應該看到登入頁面
   - 標題：🔐 Error-Free® Admin Login

2. **輸入密碼**
   - 使用設定的 `ADMIN_PASSWORD`
   - 應該成功進入 Dashboard

3. **檢查 Audit Log**
   ```sql
   SELECT * FROM audit_events 
   WHERE action = 'admin_login' 
   ORDER BY created_at DESC LIMIT 1;
   ```
   - 應該看到一筆 `admin_login` 記錄

---

## 常見問題

### Q: 部署失敗，顯示 "ModuleNotFoundError"？

**A**: 確認 `requirements.txt` 包含所有依賴：
```
streamlit
requests
```

### Q: 無法訪問 Admin UI，顯示 "Application Error"？

**A**: 檢查啟動指令是否正確：
- 確認 `--server.port $PORT`（使用 Railway 的 PORT 變數）
- 確認 `--server.address 0.0.0.0`（監聽所有 IP）

### Q: 可以同時運行 Analyzer 和 Admin UI 嗎？

**A**: 可以！請使用「選項 A：新增獨立服務」。

---

## 下一步

部署完成後：
1. ✅ 完成 Phase B1.1 驗收（參考 `README_PHASE_B1_1.md`）
2. 🔄 開始 Phase B2.1（Tenant 管理）

---

**最後更新**：2026-02-28
