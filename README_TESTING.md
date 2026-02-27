# 📋 Epoch Revoke 測試協助包 - 總覽

## ✅ 已完成項目

### 1. 程式碼修改
- ✅ **檔案**: `app.py`
- ✅ **函數**: `verify_analyzer_session()` (行 423-466)
- ✅ **變更**: 移除內部 epoch 檢查，改由 `_enforce_epoch_or_block()` 統一處理
- ✅ **安全性**: 完全保留（簽章、過期、必填欄位驗證不變）
- ✅ **Linter**: 無錯誤

### 2. 測試文檔與腳本（5 個檔案）

| 檔案 | 大小 | 用途 | 使用方式 |
|------|------|------|---------|
| **QUICK_REFERENCE.md** | 4.3K | 🎯 快速參考卡 | 日常查詢 SQL / 測試流程 |
| **test_epoch_revoke.md** | 8.1K | 📖 完整測試手冊 | 第一次測試時詳細閱讀 |
| **sql_epoch_management.sql** | 6.5K | 🗄️ SQL 命令集合 | 在 Supabase SQL Editor 複製執行 |
| **test_epoch_revoke.sh** | 7.2K | 🤖 自動化測試腳本 | `./test_epoch_revoke.sh abc` |
| **commit_and_deploy.sh** | 4.0K | 🚀 提交部署腳本 | 驗收通過後執行 |

---

## 🚀 快速開始指南

### 方案 A：手動測試（推薦新手，約 5 分鐘）

#### Step 1: 準備環境
打開 Supabase SQL Editor：
```sql
SELECT tenant, epoch FROM tenant_session_epoch WHERE tenant = 'abc';
```

#### Step 2: 取得舊 token
1. 從 Portal 進入 Analyzer（租戶 `abc`）
2. URL 加上 `&debug_epoch=1`
3. 確認看到 `token_epoch=5 current_epoch=5`
4. 複製完整 URL 到文字檔

#### Step 3: Bump epoch
```sql
UPDATE tenant_session_epoch 
SET epoch = epoch + 1, updated_at = NOW() 
WHERE tenant = 'abc';

SELECT epoch FROM tenant_session_epoch WHERE tenant = 'abc';
-- 應該顯示 6
```

#### Step 4: 驗證撤權
1. **開新無痕視窗**
2. **貼上 Step 2 的舊 URL**
3. **預期看到**: `Session revoked (tenant epoch mismatch)`

✅ **成功！** 繼續到「提交程式碼」

---

### 方案 B：自動化測試（推薦熟手，約 3 分鐘）

#### Step 1: 設定環境變數（一次性）
```bash
export SUPABASE_URL="https://xxxxx.supabase.co"
export SUPABASE_SERVICE_KEY="eyJhbGc..."
```

#### Step 2: 執行測試腳本
```bash
./test_epoch_revoke.sh abc
```

腳本會引導你完成所有測試步驟。

✅ **成功！** 繼續到「提交程式碼」

---

## 📤 提交程式碼

### 選項 1: 使用部署腳本（推薦）
```bash
./commit_and_deploy.sh
```

腳本會：
1. 顯示變更摘要
2. 確認測試完成
3. 執行 git commit
4. 詢問是否 push
5. 提示後續步驟

### 選項 2: 手動提交
```bash
# 只提交 app.py
git add app.py
git commit -m "Fix epoch revoke UX: show specific 'Session revoked' message"
git push origin main

# 或包含測試文檔
git add app.py test_epoch_revoke.md sql_epoch_management.sql test_epoch_revoke.sh QUICK_REFERENCE.md commit_and_deploy.sh
git commit -m "Fix epoch revoke UX: show specific 'Session revoked' message

- Move epoch mismatch check from verify_analyzer_session to _enforce_epoch_or_block
- Add comprehensive testing documentation and automation scripts"
git push origin main
```

---

## 📚 文檔導覽

### 我該讀哪個檔案？

#### 🎯 日常使用（記不住 SQL 命令時）
→ **QUICK_REFERENCE.md**（本檔案位於專案根目錄）

#### 📖 第一次測試（需要詳細步驟）
→ **test_epoch_revoke.md**

#### 🗄️ 需要複製 SQL 命令
→ **sql_epoch_management.sql**（在 Supabase SQL Editor 打開）

#### 🤖 想自動化測試
→ **test_epoch_revoke.sh**（需要設定環境變數）

#### 🚀 準備部署
→ **commit_and_deploy.sh**

---

## 🔍 驗收標準（必須全部通過）

| # | 測試項目 | 預期結果 | 狀態 |
|---|---------|---------|------|
| 1 | 舊 token (epoch < current) | 被拒絕 | ⬜ |
| 2 | 錯誤訊息 | 顯示 "Session revoked (tenant epoch mismatch)" | ⬜ |
| 3 | Debug 輸出 | token_epoch ≠ current_epoch | ⬜ |
| 4 | 新 token (epoch = current) | 可正常訪問 | ⬜ |
| 5 | 過期 token | 顯示 generic "No valid Portal SSO" | ⬜ |
| 6 | 簽章錯誤 token | 顯示 generic "No valid Portal SSO" | ⬜ |

**全部打勾 ✅ → 可以部署！**

---

## 🎯 核心修改摘要（30 秒版）

**問題：**  
舊 token (epoch 不匹配) 顯示誤導性的 "No valid Portal SSO parameters"

**解決：**  
將 epoch 檢查從 `verify_analyzer_session()` 移到 `_enforce_epoch_or_block()`

**結果：**  
顯示精確的 "Session revoked (tenant epoch mismatch)" 訊息

**安全性：**  
✅ 簽章驗證（保持）  
✅ 過期驗證（保持）  
✅ 必填欄位驗證（保持）  
✅ Epoch 檢查（strict, fail-closed）  

**影響範圍：**  
1 檔案、1 函數、-7 行程式碼

---

## ⚡ 超快速測試（1 分鐘版）

如果你已經很熟悉，只需要：

```bash
# 1. 查 epoch
SELECT epoch FROM tenant_session_epoch WHERE tenant = 'abc';  # 假設 5

# 2. 取舊 URL（從 Portal 進入，複製 URL）

# 3. Bump
UPDATE tenant_session_epoch SET epoch = 6 WHERE tenant = 'abc';

# 4. 無痕視窗貼舊 URL → 看到 "Session revoked" ✅
```

---

## 🆘 遇到問題？

### 常見問題快速排查

| 問題 | 可能原因 | 查看章節 |
|------|---------|---------|
| 舊 URL 仍能訪問 | 不小心走 Portal 產生新 token | test_epoch_revoke.md § Q1 |
| "Epoch check unavailable" | 環境變數未設定 | test_epoch_revoke.md § Q2 |
| Debug 不顯示 | 缺少 `debug_epoch=1` 參數 | test_epoch_revoke.md § Q3 |
| 腳本無法執行 | 權限問題 | `chmod +x test_epoch_revoke.sh` |

---

## 📞 支援資訊

**文檔位置：** `/errorfree-multi-framework-app/`  
**主要修改：** `app.py` 行 423-466  
**相關表：** `tenant_session_epoch` (Supabase)  
**環境變數：** `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`  

**維護者：** Amanda Chiu  
**建立日期：** 2026-02-26  
**版本：** 1.0

---

## ✨ 下一步

1. ✅ 執行測試（手動或自動化）
2. ✅ 確認所有驗收項目通過
3. ✅ 提交程式碼（使用腳本或手動）
4. ✅ Push 到 Railway
5. ✅ 在生產環境重新驗收
6. ✅ 記錄測試結果
7. 🎉 完成！

**預估總時間：** 10-15 分鐘（含部署等待）

---

**Good luck! 🚀**
