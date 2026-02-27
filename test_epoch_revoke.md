# 多租戶緊急撤權（Epoch Revoke）測試指南

## 📋 測試目標
驗證當 DB 的 `tenant_session_epoch.epoch` 增加後，舊的 `analyzer_session` 立刻失效並顯示專用的撤權訊息。

---

## 🔧 前置準備

### 1. 確認 Supabase 連線資訊
```bash
# 確認環境變數已設定（在 Railway 或本地 .env）
echo $SUPABASE_URL
echo $SUPABASE_SERVICE_KEY
```

### 2. 確認測試租戶存在
```sql
-- 在 Supabase SQL Editor 執行
SELECT tenant, epoch, updated_at 
FROM tenant_session_epoch 
WHERE tenant IN ('abc', 'acme')
ORDER BY tenant;
```

**預期結果：**
```
tenant | epoch | updated_at
-------|-------|------------------
abc    | 5     | 2026-02-26 10:30:00
acme   | 2     | 2026-02-25 15:00:00
```

如果租戶不存在，先建立：
```sql
-- 建立測試租戶的 epoch 記錄（如果不存在）
INSERT INTO tenant_session_epoch (tenant, epoch, updated_at)
VALUES ('abc', 0, NOW())
ON CONFLICT (tenant) DO NOTHING;
```

---

## 🧪 測試步驟

### Phase 1: 取得舊 Token URL

#### Step 1.1: 從 Portal 正常登入
1. 開啟瀏覽器，前往 Portal
2. 以 `abc` tenant 的用戶身份登入
3. 點擊進入 Analyzer

#### Step 1.2: 記錄當前 epoch 和 URL
1. 在 Analyzer URL 後面加上 `&debug_epoch=1`
   ```
   範例：
   https://your-analyzer.railway.app/?analyzer_session=eyJ...&debug_epoch=1
   ```

2. 刷新頁面，應該在側邊欄看到：
   ```
   [debug_epoch] tenant=abc token_epoch=5 current_epoch=5
   ```
   ✅ **確認兩個 epoch 相等（正常狀態）**

3. **複製完整 URL 到文字檔保存**
   ```
   檔名：old_token_url.txt
   內容：https://your-analyzer.railway.app/?analyzer_session=eyJ...&debug_epoch=1
   ```

#### Step 1.3: 驗證當前可正常訪問
- 確認頁面正常顯示 Analyzer 介面
- 確認左上角顯示正確的 tenant 和 email

---

### Phase 2: Bump Epoch（模擬緊急撤權）

#### Step 2.1: 執行 SQL 增加 epoch
在 Supabase SQL Editor 執行：

```sql
-- 將 abc 租戶的 epoch 增加 1（模擬撤權操作）
UPDATE tenant_session_epoch 
SET 
    epoch = epoch + 1,
    updated_at = NOW()
WHERE tenant = 'abc';

-- 確認更新成功
SELECT tenant, epoch, updated_at 
FROM tenant_session_epoch 
WHERE tenant = 'abc';
```

**預期結果：**
```
tenant | epoch | updated_at
-------|-------|------------------
abc    | 6     | 2026-02-26 14:25:30  ← epoch 從 5 變成 6
```

---

### Phase 3: 驗證撤權生效

#### Step 3.1: 測試舊 token 被拒絕
1. **開啟新的無痕視窗/私密瀏覽**（重要！避免 session 干擾）
2. **直接貼上 Step 1.2 保存的舊 URL**
3. 按下 Enter

#### Step 3.2: 確認預期結果

**✅ 預期看到的畫面：**
```
🔴 請從 Error-Free® Portal 進入此分析框架（Portal-only）。

Reason: Session revoked (tenant epoch mismatch). Please re-enter from Portal.

[回到 Portal / Back to Portal] ← 按鈕
```

**✅ Debug 輸出（如果 URL 有 debug_epoch=1）：**
```
[debug_epoch] tenant=abc token_epoch=5 current_epoch=6
                                  ↑不相等↑
```

**❌ 不應該看到的錯誤訊息：**
```
Reason: No valid Portal SSO parameters  ← 這是修改前的錯誤行為
```

---

### Phase 4: 驗證新 Token 仍可正常訪問

#### Step 4.1: 重新從 Portal 進入
1. 回到 Portal
2. 再次點擊進入 Analyzer
3. 應該正常顯示（因為 Portal 會用最新的 epoch=6 mint 新 token）

#### Step 4.2: 確認新 token 的 epoch
1. 在 URL 加上 `&debug_epoch=1`
2. 刷新，應該看到：
   ```
   [debug_epoch] tenant=abc token_epoch=6 current_epoch=6
                                  ↑相等↑
   ```

---

## 🔄 額外測試場景

### 測試場景 A: 過期 Token（非 epoch 問題）

**目的：** 確認過期 token 仍顯示 generic error（不是 epoch revoked）

1. 等待 token 過期（通常是 24 小時，或查看 `exp` claim）
2. 或手動產生一個過期的 token（修改 exp）
3. 訪問時應該顯示：
   ```
   Reason: No valid Portal SSO parameters
   ```
   （不是 epoch mismatch 訊息）

### 測試場景 B: 簽章錯誤

**目的：** 確認簽章錯誤仍顯示 generic error

1. 複製一個 analyzer_session URL
2. 手動修改 token 的最後幾個字元（破壞簽章）
3. 訪問時應該顯示：
   ```
   Reason: No valid Portal SSO parameters
   ```
   （不是 epoch mismatch 訊息）

### 測試場景 C: 多次 Bump Epoch

**目的：** 確認可以連續撤權

```sql
-- 連續增加 epoch 3 次
UPDATE tenant_session_epoch 
SET epoch = epoch + 1, updated_at = NOW() 
WHERE tenant = 'abc';

UPDATE tenant_session_epoch 
SET epoch = epoch + 1, updated_at = NOW() 
WHERE tenant = 'abc';

UPDATE tenant_session_epoch 
SET epoch = epoch + 1, updated_at = NOW() 
WHERE tenant = 'abc';

-- 確認結果
SELECT tenant, epoch FROM tenant_session_epoch WHERE tenant = 'abc';
-- 應該顯示 epoch = 9（如果原本是 6）
```

所有舊 token（epoch < 9）都應該被拒絕。

---

## 📊 測試結果記錄表

| 測試項目 | 預期結果 | 實際結果 | 狀態 |
|---------|---------|---------|------|
| 取得舊 token (epoch=5) | URL 可正常訪問 | | ⬜ |
| Debug 顯示 epoch 相等 | token_epoch=5 current_epoch=5 | | ⬜ |
| Bump epoch 到 6 | SQL 成功，epoch=6 | | ⬜ |
| 舊 token 被拒絕 | 顯示 "Session revoked" | | ⬜ |
| Debug 顯示 epoch 不等 | token_epoch=5 current_epoch=6 | | ⬜ |
| 重新從 Portal 進入 | 成功訪問（新 token epoch=6） | | ⬜ |
| 過期 token | 顯示 "No valid Portal SSO" | | ⬜ |
| 簽章錯誤 token | 顯示 "No valid Portal SSO" | | ⬜ |

---

## 🛠️ 常見問題排查

### Q1: 刷新舊 URL 後仍然能訪問（沒有被 block）

**可能原因：**
- 不小心走了 Portal 重新產生了新 token（新 token 會用最新 epoch）
- Streamlit session 快取了 authentication 狀態

**解決方法：**
1. 確保使用「無痕視窗」測試
2. 確保直接貼上 URL，不要從 Portal 點擊進入
3. 檢查 debug_epoch 輸出，確認 token_epoch 確實小於 current_epoch

### Q2: 出現 "Epoch check unavailable" 錯誤

**可能原因：**
- `SUPABASE_URL` 或 `SUPABASE_SERVICE_KEY` 環境變數未設定
- Supabase 服務暫時無法連線

**解決方法：**
```bash
# 檢查環境變數（在 Railway 的 Variables 頁面）
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGc...（service_role key）
```

### Q3: Debug 不顯示 epoch 資訊

**可能原因：**
- URL 缺少 `debug_epoch=1` 參數

**解決方法：**
```
在 URL 後面加上：
?analyzer_session=xxx&debug_epoch=1
或
?analyzer_session=xxx&lang=en&debug_epoch=1
```

---

## 🎯 驗收通過標準

✅ **必須全部滿足：**

1. ✅ 舊 token (epoch < current) 被拒絕
2. ✅ 錯誤訊息顯示 "Session revoked (tenant epoch mismatch)"
3. ✅ Debug 顯示 token_epoch ≠ current_epoch
4. ✅ 新 token (epoch = current) 可正常訪問
5. ✅ 過期/簽章錯誤的 token 仍顯示 generic error（不是 epoch 訊息）

---

## 📝 測試完成後

### 恢復測試環境（可選）
```sql
-- 如果需要恢復原始 epoch（例如從 9 改回 5）
UPDATE tenant_session_epoch 
SET epoch = 5, updated_at = NOW() 
WHERE tenant = 'abc';
```

### 提交程式碼
```bash
cd "/Users/amandachiu/Dropbox (Personal)/errorfree web app/errorfree-multi-framework-app/errorfree-multi-framework-app"

git add app.py
git commit -m "Fix epoch revoke UX: show specific 'Session revoked' message

- Move epoch mismatch check from verify_analyzer_session to _enforce_epoch_or_block
- Old tokens with mismatched epoch now show 'Session revoked (tenant epoch mismatch)' instead of generic 'No valid Portal SSO parameters'
- Maintains all security checks: signature, expiration, required fields
- Verified: tenant epoch revoke immediately blocks old sessions"

git push origin main
```

---

## 📞 需要協助？

如果測試遇到問題，請記錄：
1. 實際看到的錯誤訊息（截圖）
2. Debug 輸出的 epoch 數值
3. Supabase 中的 epoch 值（SQL 查詢結果）
4. 完整的測試步驟

---

**測試日期：** _______________  
**測試人員：** _______________  
**測試結果：** ⬜ 通過 / ⬜ 失敗  
**備註：** _______________________________________________
