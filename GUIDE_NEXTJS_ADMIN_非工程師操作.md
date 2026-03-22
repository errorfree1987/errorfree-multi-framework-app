# Next.js Admin 企業級 UI — 非工程師操作手冊

> **給誰看**：完全不會寫程式的人  
> **目標**：用「複製貼上」完成 Next.js Admin UI 的設定與使用  
> **預計時間**：第一次約 30 分鐘（含安裝）

---

## 📌 你只需要做三件事

| 步驟 | 你做什麼 | 說明 |
|------|----------|------|
| 1 | 安裝 Node.js | 若已安裝可略過（見下方檢查） |
| 2 | 在 Terminal 複製貼上指令 | 照著本手冊的指令一行一行執行 |
| 3 | 有問題就問 Cursor | 把錯誤訊息貼給 AI，它會幫你排除 |

---

## 第一步：檢查 Node.js 是否已安裝

### 如何開啟 Terminal

**在 Cursor 裡：**
1. 按鍵盤 `Ctrl + \``（反引號，在 Esc 下方）
2. 或點選上方選單 **View → Terminal**
3. 下方會出現一個黑色/白色視窗，那就是 Terminal

### 執行檢查指令

在 Terminal 裡**複製貼上**下面這行，然後按 **Enter**：

```bash
node -v
```

**結果說明：**
- 若顯示 `v18.x.x` 或 `v20.x.x` → ✅ 已安裝，跳到第二步
- 若顯示 `command not found` → 需要安裝，見下方「安裝 Node.js」

### 安裝 Node.js（若尚未安裝）

1. 打開瀏覽器，前往：https://nodejs.org
2. 點選 **LTS** 版本下載（左邊綠色按鈕）
3. 安裝完成後，**關閉並重新打開 Cursor**
4. 再執行一次 `node -v` 確認

---

## 第二步：安裝套件並啟動（專案已建立好）

> ✅ **專案已由 Cursor 建立完成**，你只需要安裝套件即可。

請在 Terminal 中**依序**執行下列指令（每次複製一行，貼上後按 Enter，等它跑完再執行下一行）。

### 2.1 進入 admin-dashboard 資料夾

```bash
cd "/Users/amandachiu/Dropbox (Personal)/errorfree web app/errorfree-multi-framework-app/errorfree-multi-framework-app/admin-dashboard"
```

### 2.2 安裝套件（約 1–2 分鐘）

```bash
npm install
```

等待完成，出現類似 `added XXX packages` 即表示成功。

---

## 第三步：設定環境變數

### 3.1 建立 .env.local 檔案

1. 在 Cursor 左側檔案列表，點開 `admin-dashboard` 資料夾  
2. 在 `admin-dashboard` 上按右鍵 → **New File**  
3. 檔名輸入：`.env.local`  
4. 在檔案中貼上以下內容（**請替換成你的實際數值**）：

```
SUPABASE_URL=你的Supabase網址
SUPABASE_SERVICE_KEY=你的Supabase_Service_Key
ADMIN_PASSWORD=你的Admin密碼
```

**如何取得這些值：**
- `SUPABASE_URL`、`SUPABASE_SERVICE_KEY`：Supabase 專案 → Settings → API
- `ADMIN_PASSWORD`：跟現在 Streamlit Admin UI 使用的密碼相同

### 3.2 儲存檔案

按 `Ctrl + S`（Windows）或 `Cmd + S`（Mac）儲存。

---

## 第四步：啟動 Admin UI

在 Terminal 中執行（若已在 admin-dashboard 資料夾，可省略 cd 那一行）：

```bash
cd "/Users/amandachiu/Dropbox (Personal)/errorfree web app/errorfree-multi-framework-app/errorfree-multi-framework-app/admin-dashboard"
npm run dev
```

**成功的話**會看到類似：
```
✓ Ready in 2.5s
○ Local: http://localhost:3000
```

### 開啟瀏覽器

1. 打開 Chrome 或 Safari  
2. 網址列輸入：`http://localhost:3000`  
3. 按 Enter  

應會看到 Admin 登入頁面。

---

## 第五步：日常使用（每次要開 Admin UI 時）

1. 打開 Cursor，開啟本專案  
2. 按 `Ctrl + \`` 開啟 Terminal  
3. 執行：

```bash
cd "/Users/amandachiu/Dropbox (Personal)/errorfree web app/errorfree-multi-framework-app/errorfree-multi-framework-app/admin-dashboard"
npm run dev
```

4. 瀏覽器開啟 `http://localhost:3000`  
5. 輸入 `ADMIN_PASSWORD` 登入  

---

## 常見問題

### Q1：Terminal 顯示 `command not found: node`

**A**：Node.js 尚未安裝或未正確設定。請依「第一步」安裝 Node.js，並重新開啟 Cursor。

### Q2：`npm run dev` 出現紅色錯誤訊息

**A**：請將整段錯誤訊息複製，貼到 Cursor 的 Chat，並告訴 AI：「執行 npm run dev 時出現這個錯誤」，AI 會協助你排除。

### Q3：瀏覽器開 localhost:3000 顯示無法連線

**A**：確認 Terminal 中 `npm run dev` 正在執行，且沒有關閉 Terminal 視窗。

### Q4：登入後畫面是空白的

**A**：可能是環境變數未正確設定。檢查 `admin-dashboard/.env.local` 是否已建立，且三行變數都有填寫。

### Q5：想部署到 Railway 讓別人也能用

**A**：等本地測試沒問題後，可依 `RAILWAY_DEPLOY_ADMIN_NEXTJS.md`（之後會產生）的步驟部署。

---

## 進度一覽（你可以隨時問 Cursor）

|  phase | 內容 | 狀態 |
|--------|------|------|
| Phase 0 | 環境準備（本手冊第一步～第四步） | 進行中 |
| Phase B2 | 租戶管理（Tenant List、建立、搜尋、詳情展開） | ✅ 已完成 |
| Phase B3 | 成員批量管理 | 待開始（暫時用 Streamlit） |
| Phase B6 | 用量與 Caps 圖表 | 待開始 |
| Phase B5 | Audit Log 時間線 | 待開始 |
| Phase B4 | 一鍵撤權增強 | 待開始 |

完成 Phase 0 後即可使用。如需其他功能，請告訴 Cursor：「請繼續建置 B3 成員管理」等。

---

**最後更新**：2026-03-21  
**維護者**：Amanda Chiu
