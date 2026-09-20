# FuelWise 油耗領航員：架構與本機開發骨架

請先讀 `docs/architecture.md`，它定義頁面、資料、API、模型和開發順序。

## 目前已提供
- Next.js 單趟分析首頁：讀取後端固定摘要，有載入、錯誤、重試與模型未就緒狀態。
- FastAPI：健康檢查、示範摘要、數值分析、模型狀態。
- 獨立分析函式：不規則取樣時間加權、端點油耗、計數器異常檢查。
- 架構規格與模型介面文件。

## 尚未提供
資料庫、Excel 匯入、登入、真正的模型訓練與預測、語言模型、改善任務持久化及正式部署。文件中的其餘頁面/API 為規劃，沒有用假結果模擬完成。

## 本機啟動
需要 Node.js >=20.9 與 Python >=3.11。已附前端 package-lock.json；Python 套件請在選定部署環境後鎖定版本，供團隊重現。

在 VS Code 開啟此資料夾，打開兩個終端機。

### 終端機一：後端（Windows PowerShell）
```powershell
cd backend
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

後端 API 文件：http://localhost:8000/docs 。這是本機開發服務，不是公開網站。

macOS/Linux 將 Python 執行檔改成 `.venv/bin/python`，以 `python3 -m venv .venv` 建立環境。

### 終端機二：前端
```bash
cd frontend
npm ci
npm run dev
```
打開 http://localhost:3000 。預設連向 http://localhost:8000，不需要先設定環境變數。變更後端位址時，將 `.env.example` 複製為 `.env.local` 並修改。

### 分析核心檢查
在 backend 資料夾執行：
```bash
python -m unittest discover -s tests
```

## 發佈前
目前僅供本機開發，沒有帳號與車隊權限。正式部署前完成架構文件中的權限、持久儲存、資料管理與部署清單。原始車聯網資料不包含在此套件；示範首頁只使用先前已計算的單趟摘要。
