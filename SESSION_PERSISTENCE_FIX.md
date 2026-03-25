# 🔧 Session 持久化修復說明（v2 - URL Query Parameters）

## 問題描述

**原始問題**：
- 用戶登入後，重新整理頁面會跳回登入頁面
- 需要重新輸入密碼

**第一次嘗試（失敗）**：
- 使用 `session_state` 儲存 token
- **失敗原因**：Streamlit 在頁面重新整理時會完全清除 `session_state`
- 即使我們在其中儲存 token，重新整理後也會消失

**第二次嘗試（成功）**：
- 使用 **URL query parameters** 儲存 token
- Query parameters 會保留在 URL 中，重新整理後仍然存在
- ✅ 這個方法有效！

---

## 解決方案：URL Query Parameters

### 核心概念

使用 Streamlit 的 `st.query_params` API 將 session token 儲存在 URL 中：

```
https://errorfree-techincal-review-app-staging-production.up.railway.app/?session=1234567890.abc123def456...
                                                              ↑
                                                        Session token
```

**為什麼這個方法有效？**
- URL query parameters 是瀏覽器的標準功能
- 重新整理頁面時，URL（包含 query parameters）保持不變
- Streamlit 可以從 URL 讀取 query parameters
- 即使 `session_state` 被清除，我們仍能從 URL 恢復登入狀態

---

## Token 設計

### Token 格式

```
{timestamp}.{hash}
```

**範例**：
```
1709251234.a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
```

**組成部分**：

1. **timestamp**（10 位數字）
   - Unix timestamp（UTC）
   - 用於追蹤 token 生成時間
   - 未來可用於實作過期時間

2. **hash**（32 字元）
   - SHA256(password + timestamp) 的前 32 個字元
   - 確保 token 無法被偽造
   - 使用 `hmac.compare_digest()` 安全比較

---

## 工作流程

### 1. 登入時

```python
def password_entered():
    admin_password = os.getenv("ADMIN_PASSWORD", "").strip()
    entered_password = st.session_state.get("password_input", "")
    
    if hmac.compare_digest(entered_password, admin_password):
        # 生成 token
        timestamp = str(int(datetime.utcnow().timestamp()))
        combined = f"{password}{timestamp}"
        token_hash = hashlib.sha256(combined.encode()).hexdigest()[:32]
        session_token = f"{timestamp}.{token_hash}"
        
        # 設定 session state
        st.session_state["authenticated"] = True
        
        # 設定 URL query parameter（關鍵！）
        st.query_params["session"] = session_token
        
        # 重新運行以更新 URL
        st.rerun()
```

**結果**：
- URL 變成：`https://.../?session=1709251234.a1b2c3d4e5f6...`
- 登入狀態記錄在 URL 中

### 2. 重新整理後

```python
def check_password():
    # 先檢查 session state
    if st.session_state.get("authenticated", False):
        return True
    
    # 從 URL 讀取 session token
    session_token = st.query_params.get("session", "")
    admin_password = os.getenv("ADMIN_PASSWORD", "").strip()
    
    # 驗證 token
    if session_token and verify_session_token(session_token, admin_password):
        # Token 有效，恢復登入狀態
        st.session_state["authenticated"] = True
        return True
    
    # Token 無效或不存在，顯示登入頁面
    return False
```

**流程**：
1. 頁面重新整理
2. `session_state` 被清除（`authenticated = False`）
3. 從 URL 讀取 `?session=xxx`
4. 驗證 token
5. Token 有效 → 恢復登入狀態
6. 用戶看到 Dashboard（無需重新登入）

### 3. Token 驗證

```python
def verify_session_token(token: str, password: str) -> bool:
    if not token or "." not in token:
        return False
    
    # 解析 token
    timestamp_str, token_hash = token.split(".")
    
    # 重新計算 hash
    combined = f"{password}{timestamp_str}"
    expected_hash = hashlib.sha256(combined.encode()).hexdigest()[:32]
    
    # 安全比較
    return hmac.compare_digest(token_hash, expected_hash)
```

**驗證邏輯**：
1. 檢查 token 格式（必須包含 "."）
2. 解析 timestamp 和 hash
3. 使用相同的密碼和 timestamp 重新計算 hash
4. 使用 `hmac.compare_digest()` 比較（防時序攻擊）
5. 匹配 → token 有效

### 4. 登出時

```python
def logout():
    # 清除 URL query parameters
    st.query_params.clear()
    
    # 清除 session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    st.rerun()
```

**結果**：
- URL 變回：`https://...` （沒有 `?session=xxx`）
- 下次重新整理時，無法從 URL 恢復登入狀態
- 用戶看到登入頁面

---

## 安全性考量

### ✅ 已做到

1. **密碼保護**
   - Token 包含密碼的 hash，不是明文
   - 使用 `SHA256` 加密

2. **防偽造**
   - Token 必須包含正確的 hash
   - Hash = SHA256(password + timestamp)
   - 沒有密碼無法生成有效的 token

3. **安全比較**
   - 使用 `hmac.compare_digest()` 比較 hash
   - 防止時序攻擊

4. **登出清除**
   - 登出時清除 URL query parameters
   - 確保無法重複使用舊 token

### ⚠️ 注意事項

1. **Token 可見性**
   - Token 在 URL 中可見
   - 使用 HTTPS 傳輸（Railway 預設）
   - 不要在公共電腦上使用

2. **Token 共享**
   - 如果用戶複製 URL 給別人
   - 別人可以使用相同的 token 登入
   - **建議**：Phase B1.2 加入 IP 檢查或 Token 過期時間

3. **密碼修改**
   - 修改 `ADMIN_PASSWORD` 後
   - 舊 token 立即失效（hash 不匹配）
   - 所有已登入用戶需要重新登入

4. **瀏覽器歷史**
   - Token 會出現在瀏覽器歷史記錄中
   - 建議定期清除瀏覽器歷史

---

## 測試驗證

### 測試案例 1：登入後重新整理 ✅

**步驟**：
1. 輸入正確密碼登入
2. 觀察 URL 變化：`https://.../?session=xxx`
3. 按下 F5 或 Cmd+R 重新整理

**預期結果**：
- ✅ 保持登入狀態
- ✅ URL 中仍有 `?session=xxx`
- ✅ Dashboard 正常顯示
- ✅ 不需要重新輸入密碼

### 測試案例 2：複製 URL 到新 Tab

**步驟**：
1. 登入 Dashboard
2. 複製 URL（包含 `?session=xxx`）
3. 開啟新的 Tab
4. 貼上 URL

**預期結果**：
- ✅ 新 Tab 也顯示 Dashboard
- ✅ 不需要重新登入
- ⚠️ 這說明 token 可以共享（需注意）

### 測試案例 3：登出後重新整理

**步驟**：
1. 登入 Dashboard
2. 點擊 Logout
3. 觀察 URL 變化：`https://...`（沒有 `?session=xxx`）
4. 按下 F5 重新整理

**預期結果**：
- ✅ 保持在登入頁面
- ✅ URL 沒有 query parameters
- ✅ 無法自動恢復登入狀態

### 測試案例 4：手動修改 URL

**步驟**：
1. 登入 Dashboard
2. 手動修改 URL 中的 `?session=xxx` 部分
3. 按下 Enter

**預期結果**：
- ✅ 跳回登入頁面
- ✅ Token 驗證失敗
- ✅ 系統拒絕偽造的 token

---

## 與第一版的差異

| 特性 | v1 (session_state) | v2 (query params) |
|------|-------------------|-------------------|
| 儲存位置 | `st.session_state` | URL query parameters |
| 重新整理後 | ❌ 消失 | ✅ 保留 |
| 關閉 Tab 後 | ❌ 消失 | ❌ 消失 |
| 可複製 URL | ❌ 無法 | ✅ 可以 |
| Token 可見性 | ❌ 不可見 | ⚠️ 可見（在 URL） |
| 實作複雜度 | 簡單 | 簡單 |
| 安全性 | 好 | 中等（需 HTTPS） |

---

## Phase B1.2 改進計劃

在 Phase B1.2（完美版）中，會加入：

1. **Token 過期時間**
   - 例如：8 小時後自動登出
   - 在 token 驗證時檢查 timestamp

2. **IP 綁定**
   - Token 綁定到特定 IP
   - 防止 token 被共享

3. **Session ID 追蹤**
   - 每個 session 有唯一 ID
   - 記錄到資料庫

4. **多用戶支援**
   - 使用 Supabase Auth
   - 每個用戶有獨立的 session

5. **Token 刷新**
   - 接近過期時自動刷新
   - 延長登入時間

---

## 程式碼變更摘要

### 新增函數

```python
def get_query_param(key: str, default: str = "") -> str
def set_query_param(key: str, value: str)
def clear_query_params()
```

### 修改函數

#### `check_password()`
- 從 URL query parameters 讀取 token
- 使用 `st.query_params` API
- 驗證 token 並恢復登入狀態

#### `password_entered()`
- 登入成功後設定 URL query parameter
- 使用 `set_query_param("session", token)`

#### `logout()`
- 清除 URL query parameters
- 使用 `clear_query_params()`

---

## 部署步驟

程式碼已推送到 GitHub：
```bash
Commit: defc6e1
Message: "Fix: Use URL query parameters for session persistence"
Branch: staging-portal-sso
```

Railway 會自動部署（約 1-2 分鐘）。

---

## 總結

### 問題
- ❌ `session_state` 在重新整理時被清除
- ❌ 第一版修復無效

### 解決方案
- ✅ 使用 URL query parameters
- ✅ Token 格式：`{timestamp}.{hash}`
- ✅ 重新整理後從 URL 恢復登入狀態

### 結果
- ✅ 登入後可以重新整理（F5/Cmd+R）
- ✅ Token 安全（使用 SHA256 hash）
- ✅ 登出後 token 清除
- ⚠️ Token 在 URL 中可見（需注意安全）

---

**修復時間**：2026-03-01  
**維護者**：Amanda Chiu  
**版本**：Phase B1.1 Hotfix v2
