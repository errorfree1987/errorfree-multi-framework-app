# 🚀 Epoch Revoke 快速參考卡

## 📝 常用 SQL 命令（Supabase SQL Editor）

### 查看租戶當前 epoch
```sql
SELECT tenant, epoch, updated_at 
FROM tenant_session_epoch 
WHERE tenant = 'abc';
```

### 緊急撤權（單一租戶）
```sql
UPDATE tenant_session_epoch 
SET epoch = epoch + 1, updated_at = NOW() 
WHERE tenant = 'abc';
```

### 緊急撤權（批次）
```sql
UPDATE tenant_session_epoch 
SET epoch = epoch + 1, updated_at = NOW() 
WHERE tenant IN ('abc', 'acme', 'demo');
```

### 查看最近撤權記錄（24h）
```sql
SELECT tenant, epoch, updated_at, NOW() - updated_at AS ago
FROM tenant_session_epoch
WHERE updated_at > NOW() - INTERVAL '24 hours'
ORDER BY updated_at DESC;
```

---

## 🧪 手動測試流程（5 分鐘）

### 1. 取得舊 token
- 從 Portal 進入 Analyzer
- URL 加上 `&debug_epoch=1`
- 複製完整 URL（保存到文字檔）
- 確認看到 `token_epoch=X current_epoch=X`（相等）

### 2. Bump epoch
```sql
UPDATE tenant_session_epoch 
SET epoch = epoch + 1, updated_at = NOW() 
WHERE tenant = 'abc';

-- 確認
SELECT epoch FROM tenant_session_epoch WHERE tenant = 'abc';
```

### 3. 驗證撤權
- **開新無痕視窗**
- **貼上舊 URL**
- **預期：** 看到 `Session revoked (tenant epoch mismatch)`
- **Debug：** 看到 `token_epoch=X current_epoch=X+1`（不相等）

---

## 🛠️ 自動化腳本使用

### 測試腳本
```bash
# 設定環境變數（一次性）
export SUPABASE_URL="https://xxxxx.supabase.co"
export SUPABASE_SERVICE_KEY="eyJhbGc..."

# 執行測試（互動式）
./test_epoch_revoke.sh abc
```

### 提交部署腳本
```bash
# 驗收通過後執行
./commit_and_deploy.sh
```

---

## 🔍 Debug 技巧

### 檢查 token 內容（不驗證簽章）
```bash
# 提取 analyzer_session 的 payload（第一段 base64）
TOKEN="eyJ0eXAi...base64...signature"
echo $TOKEN | cut -d'.' -f1 | base64 -d 2>/dev/null | jq .
```

### 在 Railway Logs 看 debug 輸出
```bash
# 在本地測試時可以看到的 print 輸出
railway logs --follow
```

---

## ⚠️ 常見錯誤與解決

| 錯誤訊息 | 原因 | 解決方法 |
|---------|------|---------|
| `No valid Portal SSO parameters` | 簽章錯誤 / 過期 / 必填欄位缺失 | 這是正常的安全拒絕，不是 epoch 問題 |
| `Session revoked (tenant epoch mismatch)` | Epoch 不匹配 | 這是預期的撤權訊息 ✅ |
| `Epoch check unavailable` | Supabase 連線失敗 | 檢查 `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` |

---

## 📊 驗收檢查清單

- [ ] 舊 token (epoch < current) 被拒絕
- [ ] 顯示 "Session revoked (tenant epoch mismatch)"
- [ ] Debug 顯示 token_epoch ≠ current_epoch
- [ ] 新 token (epoch = current) 可正常訪問
- [ ] 過期 token 仍顯示 generic error（不是 epoch 訊息）
- [ ] 簽章錯誤 token 仍顯示 generic error（不是 epoch 訊息）

---

## 🚨 緊急撤權 SOP

### 情境 A：單一租戶帳號異常
1. 執行 SQL：`UPDATE tenant_session_epoch SET epoch = epoch + 1 WHERE tenant = 'xxx';`
2. 確認：所有該租戶的舊 session 立刻失效
3. 通知：該租戶用戶重新從 Portal 登入

### 情境 B：安全漏洞（全局）
1. 更新環境變數：`ANALYZER_SESSION_SECRET`（在 Railway）
2. （可選）全租戶撤權：`UPDATE tenant_session_epoch SET epoch = epoch + 1;`
3. 通知：所有用戶重新登入

---

## 📁 相關檔案

| 檔案 | 用途 |
|------|------|
| `test_epoch_revoke.md` | 完整測試手冊（詳細） |
| `sql_epoch_management.sql` | SQL 命令集合（複製貼上用） |
| `test_epoch_revoke.sh` | 自動化測試腳本（互動式） |
| `commit_and_deploy.sh` | Git 提交部署腳本 |
| `QUICK_REFERENCE.md` | 本檔案（快速參考） |

---

## 🎯 核心修改說明（給新接手的開發者）

**問題：** 舊的 `verify_analyzer_session()` 在 epoch mismatch 時返回 `None`，導致顯示 generic 錯誤訊息。

**解決：** 將 epoch 檢查從 `verify_analyzer_session()` 移到 `_enforce_epoch_or_block()`，使其顯示專用的撤權訊息。

**安全性：** 所有基礎驗證（簽章、過期、必填欄位）完全保留，epoch 檢查仍然是 strict (fail-closed)。

**影響範圍：** 只修改 1 個函數（`verify_analyzer_session`），不影響 DB / Portal / 環境變數。

---

**最後更新：** 2026-02-26  
**版本：** 1.0  
**維護者：** Amanda Chiu
