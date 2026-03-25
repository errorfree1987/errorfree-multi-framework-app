# Railway Staging — 唯一正式目標（請勿連錯專案）

> **最後更新**：2026-03-24  
> **用途**：所有人部署、查 Log、設 Variables 時，一律以本文件為準。

---

## ✅ 正式 Staging 專案（與你提供的畫面一致）

| 項目 | 值 |
|------|-----|
| **Railway 專案名稱** | `errorfree-techincal review-app-staging` |
| **對外 URL** | `https://errorfree-techincal-review-app-staging-production.up.railway.app` |
| **GitHub 倉庫** | `https://github.com/errorfree1987/errorfree-multi-framework-app` |
| **建議分支** | `staging-portal-sso`（或你團隊約定的 staging 分支） |

> 註：專案名稱在 Railway 顯示為 `techincal`（與 dashboard 一致，非拼錯後再行修改亦可）。

---

## GitHub 與 Railway 的連結方式

1. 登入 [Railway Dashboard](https://railway.app/dashboard)。
2. 開啟專案 **`errorfree-techincal review-app-staging`** — **不要** 使用舊的或其他名稱的專案（例如過去的 `trustworthy-analysis` 等）。
3. 確認 **Deployments** 顯示來源為上述 **GitHub 倉庫**。
4. Push 到連線的分支後，應只會觸發 **本專案** 的部署。

若你看到部署出現在別的 Railway 專案裡，請到該錯誤專案 **Settings → Disconnect** 或刪除錯誤的 Service，並只在 **本專案** 內 **New → GitHub Repo** 連同一個 repo。

---

## 此 Staging 服務通常跑什麼？

依目前設定多為 **Python / Streamlit**（例如 Analyzer：`streamlit run app.py`）。  
若另有 **Next.js Admin Dashboard**（`admin-dashboard`），應在 **同一 Railway 專案底下新增第二個 Service**，或獨立文件說明，**不要** 把不同用途的服務綁到錯誤的專案名稱下。

---

## 舊文件裡的錯誤範例（請勿再使用）

以下僅作歷史參考，**請改從本頁 URL 進入**：

- `trustworthy-analysis-production-*.up.railway.app`
- 任意未在本文件列出的 staging URL

---

## 快速檢查清單

- [ ] 瀏覽器開啟的 URL 是否為 `errorfree-techincal-review-app-staging-production.up.railway.app`？
- [ ] Railway 左上角專案名稱是否為 `errorfree-techincal review-app-staging`？
- [ ] GitHub 連線的 repo 是否為 `errorfree1987/errorfree-multi-framework-app`？

全部為「是」即為正確目標。
