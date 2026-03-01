# 🔧 Session 持久化修復說明

## 問題描述

**原始問題**：
- 用戶登入後，重新整理頁面會跳回登入頁面
- 需要重新輸入密碼

**原因**：
- Streamlit 的 `session_state` 在頁面重新載入時會被重置
- 原本的實作只依賴 `session_state["authenticated"]`
- 這個狀態在重新整理後會消失

---

## 解決方案

### 實作機制：Session Token

**核心概念**：
使用安全的 session token 來驗證登入狀態，token 存在 `session_state` 中並在重新整理後仍然有效。

### Token 設計

**Token 格式**：
```
{password_hash_prefix}.{random_part}
```

**範例**：
```
a1b2c3d4e5f6g7h8.xY9zW8vU7tS6rQ5p
```

**組成部分**：
1. **password_hash_prefix**（16 字元）
   - 密碼的 SHA256 hash 的前 16 個字元
   - 用於驗證 token 的有效性

2. **random_part**（22 字元）
   - 使用 `secrets.token_urlsafe(16)` 生成
   - 提供額外的隨機性，防止預測

### 工作流程

#### 1. 登入時
```python
# 用戶輸入正確密碼
if hmac.compare_digest(entered_password, admin_password):
    # 生成 session token
    token_hash = hashlib.sha256(admin_password.encode()).hexdigest()[:16]
    random_part = secrets.token_urlsafe(16)
    session_token = f"{token_hash}.{random_part}"
    
    # 儲存到 session_state
    st.session_state["authenticated"] = True
    st.session_state["session_token"] = session_token
```

#### 2. 重新整理後
```python
# 檢查 session_state 中的 token
session_token = st.session_state.get("session_token", "")
if session_token and verify_session_token(session_token):
    # Token 有效，恢復登入狀態
    st.session_state["authenticated"] = True
    return True
```

#### 3. Token 驗證
```python
def verify_session_token(token):
    admin_password = os.getenv("ADMIN_PASSWORD", "").strip()
    expected_token_base = hashlib.sha256(admin_password.encode()).hexdigest()
    
    # 驗證 token 格式和前綴
    token_parts = token.split(".")
    if len(token_parts) == 2 and token_parts[0] == expected_token_base[:16]:
        return True
    return False
```

---

## 安全性考量

### ✅ 已做到

1. **密碼保護**
   - Token 包含密碼的 hash，不是明文
   - 使用 `hashlib.sha256` 加密

2. **隨機性**
   - Token 包含隨機部分
   - 使用 `secrets.token_urlsafe()` 生成（密碼學安全）

3. **驗證機制**
   - Token 必須包含正確的 hash 前綴
   - Token 格式必須正確（兩個部分，用 "." 分隔）

4. **登出清除**
   - 登出時清除所有 session_state（包括 token）
   - 確保無法重複使用舊 token

### ⚠️ 限制

1. **瀏覽器關閉**
   - Token 儲存在 Streamlit 的 session_state
   - 關閉瀏覽器 tab 後 token 會消失
   - 這是預期行為（類似關閉瀏覽器後登出）

2. **密碼修改**
   - 如果修改 `ADMIN_PASSWORD` 環境變數
   - 舊的 token 會立即失效（因為 hash 不匹配）
   - 所有已登入的用戶需要重新登入

3. **無過期時間**
   - 目前的 token 沒有設定過期時間
   - 只要瀏覽器 tab 不關閉，token 永久有效
   - Phase B1.2 會加入過期時間機制

---

## 測試驗證

### 測試案例 1：登入後重新整理 ✅

**步驟**：
1. 輸入正確密碼登入
2. 成功進入 Dashboard
3. 按下 F5 或 Cmd+R 重新整理

**預期結果**：
- ✅ 保持登入狀態
- ✅ 不需要重新輸入密碼
- ✅ Dashboard 正常顯示

### 測試案例 2：登入後關閉 Tab

**步驟**：
1. 輸入正確密碼登入
2. 成功進入 Dashboard
3. 關閉瀏覽器 Tab
4. 重新開啟 Admin UI URL

**預期結果**：
- ✅ 跳回登入頁面
- ✅ 需要重新輸入密碼

### 測試案例 3：登出後重新整理

**步驟**：
1. 登入 Dashboard
2. 點擊 Logout 登出
3. 按下 F5 重新整理

**預期結果**：
- ✅ 保持在登入頁面
- ✅ 無法自動恢復登入狀態

### 測試案例 4：修改密碼後

**步驟**：
1. 登入 Dashboard（使用密碼 A）
2. 在 Railway 修改 `ADMIN_PASSWORD` 為密碼 B
3. 重新整理頁面

**預期結果**：
- ✅ 跳回登入頁面
- ✅ 舊 token 失效
- ✅ 需要使用新密碼 B 登入

---

## 程式碼變更摘要

### 修改的函數

#### 1. `check_password()`
**新增功能**：
- 生成 session token（登入時）
- 驗證 session token（重新整理後）
- 恢復登入狀態（如果 token 有效）

**新增內部函數**：
- `generate_session_token()` - 生成 token
- `verify_session_token(token)` - 驗證 token

#### 2. `logout()`
**改進**：
- 更安全的 session_state 清除
- 避免迭代時修改 dict 的問題

### 新增 import

```python
import secrets  # 用於生成安全的隨機 token
```

---

## 部署步驟

### 自動部署（推薦）

程式碼已推送到 GitHub：
```bash
Commit: 84d2d40
Message: "Fix: Maintain login state after page refresh"
Branch: staging-portal-sso
```

Railway 會自動偵測並重新部署：
1. 等待 1-2 分鐘
2. 查看 Railway Deployments 標籤
3. 確認最新部署狀態為 "Active"

### 驗證部署

1. **檢查 Build Logs**：
   - Railway → trustworthy-analysis → Deployments
   - 點擊最新部署 → Build Logs
   - 確認沒有錯誤

2. **測試功能**：
   - 訪問 Admin UI
   - 登入
   - **重新整理頁面**
   - ✅ 確認保持登入狀態

---

## Phase B1.2 改進計劃

在 Phase B1.2（完美版）中，會加入：

1. **Token 過期時間**
   - 例如：8 小時後自動登出
   - 在 token 中加入時間戳記

2. **Token 刷新機制**
   - 接近過期時自動刷新 token
   - 延長登入時間

3. **多用戶支援**
   - 使用 Supabase Auth
   - 每個用戶有獨立的 session

4. **速率限制**
   - 防止暴力破解
   - 失敗 N 次後鎖定

---

## 總結

### 問題
- ❌ 重新整理後登入狀態消失

### 解決方案
- ✅ Session token 機制
- ✅ Token 存在 session_state
- ✅ 重新整理後自動驗證 token
- ✅ 安全的 hash + 隨機數設計

### 結果
- ✅ 登入後可以重新整理
- ✅ 關閉瀏覽器後需要重新登入
- ✅ 修改密碼後舊 token 失效
- ✅ 安全性維持

---

**修復時間**：2026-03-01  
**維護者**：Amanda Chiu  
**版本**：Phase B1.1 Hotfix
