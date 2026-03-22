# 從 Dropbox 搬到 GitHub 同步 — 操作手冊

> **目的**：改用 GitHub 同步專案，讓兩台電腦都能用 Cursor 開發  
> **給誰看**：非工程師  
> **最後更新**：2026-03-21

---

## 📌 總覽

| 階段 | 在這台電腦（iMac）做 | 在另一台電腦做 |
|------|----------------------|----------------|
| **第一階段** | 推送到 GitHub | — |
| **第二階段** | 把專案移出 Dropbox（可選） | 從 GitHub  clone 專案 |
| **第三階段** | — | 安裝 Node.js、npm install、建立 .env.local、npm run dev |

---

## 第一階段：在這台 iMac 推送內容到 GitHub

### 1.1 開啟 Terminal

按 **Command + 空格鍵** → 輸入 **Terminal** → 按 Enter

### 1.2 進入專案資料夾

貼上執行：

```bash
cd "/Users/amandachiu/Dropbox (Personal)/errorfree web app/errorfree-multi-framework-app/errorfree-multi-framework-app"
```

### 1.3 加入檔案

貼上執行：

```bash
git add .
```

### 1.4 提交變更

貼上執行（引號內的說明可自行修改）：

```bash
git commit -m "Add Next.js Admin Dashboard (Phase B2) + migration guide"
```

### 1.5 推送到 GitHub

貼上執行：

```bash
git push origin staging-portal-sso
```

**若被要求登入：**

- 會跳出瀏覽器或 Terminal 要求輸入 GitHub 帳密
- 若已設 SSH key，會直接成功
- 帳號：你的 GitHub 使用者名稱  
  密碼：使用 **Personal Access Token**（非 GitHub 密碼，須到 GitHub → Settings → Developer settings → Personal access tokens 建立）

---

## 第二階段：把專案移出 Dropbox（建議，避免 sync 問題）

### 2.1 關閉 Cursor

### 2.2 複製專案到非 Dropbox 資料夾

在 **Finder**：

1. 前往 `/Users/amandachiu/Dropbox (Personal)/errorfree web app/errorfree-multi-framework-app/`
2. 複製整個 **errorfree-multi-framework-app** 資料夾
3. 貼到例如：`/Users/amandachiu/Projects/`  
   （若沒有 Projects 資料夾，可在「家目錄」建立一個）

### 2.3 刪除 Dropbox 內的專案

確認新位置可以正常開啟後，再刪除 Dropbox 裡的舊專案。

### 2.4 之後在 Cursor 用新路徑開啟專案

- 舊路徑：`~/Dropbox (Personal)/errorfree web app/.../errorfree-multi-framework-app`
- 新路徑：`~/Projects/errorfree-multi-framework-app`

---

## 第三階段：在另一台電腦設定

### 3.1 安裝 Node.js

若尚未安裝，請到 https://nodejs.org 下載 LTS 版本並安裝。

### 3.2  clone 專案

1. 開啟 Terminal
2. 進入要放專案的資料夾，例如：

```bash
cd ~/Projects
```

（若沒有 Projects，可先執行 `mkdir -p ~/Projects` 建立）

3. 從 GitHub clone（請將 `你的帳號` 換成你的 GitHub 帳號）：

```bash
git clone https://github.com/errorfree1987/errorfree-multi-framework-app.git
```

4. 進入專案：

```bash
cd errorfree-multi-framework-app
```

### 3.3 切換到正確分支

```bash
git checkout staging-portal-sso
```

### 3.4 安裝 Admin Dashboard 套件

```bash
cd admin-dashboard
npm install
```

### 3.5 建立 .env.local

1. 在 Cursor 開啟專案
2. 在 `admin-dashboard` 按右鍵 → New File
3. 檔名：`.env.local`
4. 內容（改成你的實際值）：

```
SUPABASE_URL=你的Supabase網址
SUPABASE_SERVICE_KEY=你的Supabase_Service_Key
ADMIN_PASSWORD=你的Admin密碼
```

5. 儲存（Command + S）

### 3.6 啟動 Admin UI

```bash
cd admin-dashboard
npm run dev
```

瀏覽器開啟：http://localhost:3000

---

## 之後的日常使用（兩台電腦）

### 在這台電腦改完程式後

```bash
cd /你的專案路徑/errorfree-multi-framework-app
git add .
git commit -m "說明你做了什麼"
git push origin staging-portal-sso
```

### 在另一台電腦要取得最新內容

```bash
cd /你的專案路徑/errorfree-multi-framework-app
git pull origin staging-portal-sso
```

若在另一台有改 `admin-dashboard` 並執行過 `npm install`，之後每次 `git pull` 後建議再跑一次：

```bash
cd admin-dashboard
npm install
```

以確保套件與最新程式一致。

---

## 重要提醒

| 項目 | 說明 |
|------|------|
| **.env.local 不會上傳** | 每台電腦都要自己建立 `.env.local`，Git 不會同步它 |
| **node_modules 不會上傳** | 每台電腦都要執行 `npm install` |
| **Dropbox 建議移出** | 專案放在 Dropbox 易有 sync 衝突，建議移到本機資料夾 |
| **兩台都要裝 Node.js** | 執行 Admin UI 的電腦都要安裝 Node.js |

---

## 快速檢查清單

### 這台 iMac

- [ ] `git add .` 執行成功
- [ ] `git commit` 執行成功
- [ ] `git push origin staging-portal-sso` 執行成功
- [ ] （可選）專案已複製到非 Dropbox 資料夾

### 另一台電腦

- [ ] 已安裝 Node.js
- [ ] `git clone` 成功
- [ ] `git checkout staging-portal-sso` 成功
- [ ] `cd admin-dashboard` 後執行 `npm install` 成功
- [ ] 已建立 `.env.local` 並填入正確值
- [ ] `npm run dev` 可啟動，瀏覽器可開 http://localhost:3000

---

**最後更新**：2026-03-21
