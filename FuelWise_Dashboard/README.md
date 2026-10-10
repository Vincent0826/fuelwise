# FuelWise｜商用車行程分析儀表板

這是一個能讀取實際 Excel 的本機網站。Python 提供資料 API，HTML/CSS/JavaScript 呈現互動地圖與圖表。不需要 Node.js 或雲端資料庫。AI 行程助理可選擇使用 Gemini API；其餘分析在本機執行。此版本使用歷史資料模擬監控，尚未接即時車聯網。

## 專案結構與程式職責

### 網站與資料處理

| 檔案 | 職責 |
| --- | --- |
| `server.py` | 啟動本機 HTTP 服務、提供行程／原始資料／油耗模型 API；選用的 Gemini 行程助理也由此轉送請求。 |
| `importer.py` | 讀取 XLSX／XLS、挑選含行程欄位的工作表，建立供網站使用的本機 SQLite 索引。 |
| `analytics.py` | 共用行程分析核心：清理資料、計算里程／燃油／訊號／事件與資料品質；網站及離線程式都會使用。 |
| `web/index.html` | 定義網站頁面、導覽、篩選器、圖表、地圖及各分析頁籤的 HTML 結構。 |
| `web/app.js` | 網站互動邏輯：車隊總覽、行程地圖、模型油耗估計與 SHAP 解釋、歷史資料模擬監控、AI 助理、訊號／事件／品質分析、歷史比較、游標播放及 CSV 下載。 |
| `web/style.css` | 網站主要版面、元件及不同螢幕尺寸的基礎樣式。 |
| `web/polish.css` | 在主要樣式之後載入的視覺細節與樣式微調層。 |

### 獨立分析與模型

| 檔案 | 職責 |
| --- | --- |
| `analysis/analyze_driving_fuel.py` | 獨立分析急加速／急減速事件與每趟油耗的關聯，輸出彙總、圖表及報告；不屬於網站啟動流程。 |
| `analysis/train_fuel_model.py` | 離線建立行程層級油耗模型、特徵與時間切分評估；使用既有索引，不會重新匯入 Excel 或呼叫網站 API。 |
| `analysis/vehicle_history_experiment.py` | 固定第一次實驗的樣本及切分，產生防洩漏的車輛歷史油耗特徵，對照 A/B/C 三種方法。 |
| `analysis/fuel_liter_experiment.py` | 以距離分組診斷短行程，並只訓練一次以整趟燃油 L 為目標的候選 D；依驗證集選擇後才評估測試追蹤集。 |
| `analysis/fuel_model_inference.py` | 載入網站選定模型一次，對單趟行程推論並用 XGBoost 原生 TreeSHAP 驗證及整理個別特徵貢獻。 |
| `analysis/prepare_dashboard_model.py` | 首次啟動時從本機 SQLite 索引訓練並建立網站模型包；索引未變時重用，不將活頁簿或模型資料上傳。 |

### 測試

| 檔案 | 職責 |
| --- | --- |
| `tests/test_analytics.py` | 驗證共用分析公式、資料品質、計數器及區間統計。 |
| `tests/test_ai.py` | 驗證 AI 感測器資料匿名化、輸入限制及 Gemini 請求處理。 |
| `tests/test_driving_fuel.py` | 驗證急加減速事件判定與離線油耗分析。 |
| `tests/test_fuel_model.py` | 驗證模型特徵、資料切分、歷史特徵防洩漏、模型載入／估計及 SHAP 加總一致性。 |

### 啟動與專案輔助檔

| 檔案 | 職責 |
| --- | --- |
| `start_windows.bat`、`start_mac_linux.sh` | 建立本機虛擬環境、安裝套件並啟動網站。 |
| `requirements.txt` | 列出網站、Excel 匯入及離線分析所需的 Python 套件。 |
| `VALIDATION.md` | 記錄既有匯入、API、頁面邏輯及程式檢查結果與限制。 |
| `Data_Analysis.code-workspace` | VS Code 多資料夾工作區設定；包含 FuelWise 及相鄰的 `Data_Analysis` 資料夾。 |

`data.xlsx`／`data.xls` 是本機來源資料；`.venv` 是套件環境；`.cache` 是可重建的 SQLite 索引；`analysis_output` 是離線分析與模型輸出。這些不是網站原始碼，但仍供本機工作流程使用，因此本次整理予以保留。`.cache` 的重建方式見「索引與更新」；其他資料或輸出如需清除，請先確認其用途。

## Windows 開始使用

1. 安裝 Python 3.11 或 3.12，安裝時勾選 Add Python to PATH。
2. 解壓縮，把你的完整 Excel 放在 `server.py` 同一資料夾，命名為 `data.xlsx` 或 `data.xls`。若下載名稱是 `data(1).xlsx`，改名為 `data.xlsx` 即可，不要把真正的 xlsx 轉名為 xls。若兩種檔名都有，預設讀取 data.xlsx。
3. 雙擊 `start_windows.bat`。首次會建立 .venv 並安裝套件，需網路。
4. 第一次匯入49萬筆資料與建立摘要需要數分鐘，終端機會顯示進度。請等到出現「FuelWise 已啟動」。
5. 瀏覽器會自動開啟 http://127.0.0.1:8501 。保留終端機，使用完按 Ctrl+C。

也可以在此資料夾的 VS Code 終端機執行：

```bash
python -m pip install -r requirements.txt
python server.py
```

Mac/Linux：`sh start_mac_linux.sh`。

### 啟用 AI 行程助理（選用）

AI 助理需要 Gemini API key。請在啟動伺服器前，於同一個終端機設定環境變數；不要將 key 貼進程式碼或提交到版本控制。

Windows PowerShell：

```powershell
$env:GEMINI_API_KEY = "你的 Gemini API key"
python server.py
```

預設使用 `gemini-3.8-flash`。如要改用 `gemini-3.7-flash`，請在啟動前設定 `$env:GEMINI_MODEL = "gemini-3.7-flash"`。未設定 `GEMINI_API_KEY` 時，網站仍可使用本機分析功能，但 AI 助理不會回覆。

提問時，瀏覽器會將對話內容以及匿名化的近期感測器讀值、行程加減速摘要送到本機伺服器，再轉送 Gemini。程式不會傳送車牌、車輛代碼、行程代碼或 GPS 座標。

自訂檔案與port：

```bash
python server.py --file "D:/your-folder/data.xlsx" --port 8502
```

## 網站內容

- 車隊總覽：日期範圍、車輛與行程數、有效配對里程／耗油、每車耗油及可搜尋行程表。
- 行程與地圖：車輛、行程下拉選單；GPS路徑按速度／RPM／負載著色、起終點、停止引擎運轉熱點。
- 任務即時監控：按「開始任務」進入獨立頁面，以目前選取的歷史行程每0.5秒模擬一筆感測器資料，動態繪製GPS軌跡並顯示車速、轉速、負載、溫度、電瓶、油量、里程與耗油；可暫停、繼續或結束。這是歷史資料重播，不連接真實車輛或感測器。
- 觀測游標：拖曳時間游標或點選圖表／路段定位地圖，可逐筆播放。播放非真實時間等速。
- 區間分析：輸入起終分鐘，縮放時間圖並計算觀測端點的燃油與里程增量。
- 訊號分析：累積耗油、車速、RPM、負載、水溫、電瓶V、油量%、區間平均加速度，切換時間／距離橫軸。
- 速度分布：按時間加權的區間比例，並呈現RPM×負載散點。
- 事件與停留：原始事件ID 1–20對照、內容去重、與CAN推估停留分開，可點列定位。
- 歷史比較：同車同方向，起終點各在200m／500m／1km內；顯示總耗油、百公里油耗、停止時間。中位基準排除本趟。
- 資料品質：長間隔、重複時間、GPS跳點、CAN異常、計數器倒退、可判讀時間及刻度。
- 原始資料：55個來源欄位與來源列號；分頁檢視、CSV下載，完整欄位不只限於圖表指標。

## 資料與指標

來源支援含 `journeyCode`、`time` 的工作表，自動略過欄位對照表。車輛以 enabledCode 優先分組，缺少時以 carNum；行程以車輛＋journeyCode 分組。缺失車牌可用同趟其他紀錄補顯示，但不修改來源。Excel數字識別碼完整顯示；若Excel本身已將長碼捨入，程式不能復原。

所有指標先逐趟排序後計算，禁止跨行程差分：

- 距離 = 末筆 can.totalMileage − 首筆值，單位km。
- 燃油 = 末筆 can.engine.totalFuelUsed − 首筆值，單位L。
- 百公里油耗 = 100×燃油÷距離，不是各趟比率的簡單平均。
- 前值持有估計：每筆狀態持續至下一筆；最後一筆無額外時間，>120秒空檔不納入加權與停止時長。
- 停車引擎運轉：CAN速度=0且RPM>0；連續≥60秒列停留事件。不等於可以省掉的怠速。
- CAN狀態依原始對照表：0正常、9異常。未知／異常狀態不納入CAN指標；缺整欄時保留數值但顯示未驗證。
- 累積計數器倒退或端點無效，該趟對應用油／里程顯示「—」，不以補零或正增量加總掩蓋重置。
- GPS座標超範圍、(0,0)、長間隔及推算速度>160km/h的線段切斷，不當成真實軌跡。地圖不是道路匹配結果。
- 電瓶原始單位mV，除1000後顯示V；引擎負載範圍依來源表0–250%，不截斷100%以上的有效值。
- 區間加速度只在相鄰時間差≤30秒估計，不直接認定急加速／急煞。
- 平台事件座標由事件時間120秒內最近觀測點推估；距離較遠則無事件座標。原始事件不是由模型編造。
- 限速值需正且車速有效才能比較；觀測超限時間不是法律上的違規判定。
- 粗刻度燃油不支持精準瞬時耗油與怠速用油拆帳。區間燃油為階梯計數器差，非秒級量測。
- 同OD不能證明同貨物、載重或客戶；目前沒有載重、坡度、氣象及因果節油模型。

## 索引與更新

`.cache` 內保存SQLite索引。首次讀Excel，之後同檔名、大小、修改時間與程式版本相同就直接用索引。更換Excel並重新啟動會建立新索引。想重建可在關閉程式後刪除 `.cache`；不影響來源Excel。程式不修改Excel。

本機服務只監聽127.0.0.1。不可直接將此開發伺服器當作公開雲端產品；對外部署需另外加身分驗證、權限和正式伺服器。

## 網路與資料流

試算表與計算都在本機。地圖使用OpenStreetMap圖磚，瀏覽器會請求目前地圖視野的圖磚；不會把Excel整份上傳。Leaflet 1.9.4與Plotly.js 2.35.2由CDN載入，因此需要網路。CDN無法連線時，表格／指標仍可使用，地圖或圖表會顯示錯誤提示。不是完全離線版本。

## 驗證方式

```bash
python -m unittest discover -s tests -p "test_*.py"
```

此指令會執行所有 `test_*.py`，涵蓋跨行程隔離、計數器倒退、CAN異常、長空檔切斷、事件去重、急加減速判定、AI 資料匿名化、時間切分邊界及模型輸入特徵檢查。另用所提供完整Excel驗證493,744筆、20台車、3,541趟可匯入，第一趟案例30km／7L／3378秒。

## 急加減速與油耗分析（獨立腳本）

分析腳本不會加入網站。安裝 `requirements.txt` 後，在此資料夾執行：

```powershell
python -m analysis.analyze_driving_fuel
```

預設讀取 `data.xlsx`（若不存在則讀取 `data.xls`），輸出到 `analysis_output`：逐趟彙總 CSV、急加減速頻率散佈圖、事件次數分組油耗箱型圖，以及摘要 Markdown。可用 `--file` 指定來源、`--output` 指定輸出資料夾、`--min-distance-km` 調整最小有效里程（預設 1 km）。事件定義為速度變化率超過 3 km/h/s 且連續至少 3 個有效觀測，觀測間隔最多 30 秒；油耗使用同一行程累積燃油與里程計數器差。結果為觀察性關聯，不能據此推論急加減速造成油耗改變。

## 行程層級油耗模型（離線）

### 模型概述

目前模型是**行程層級的監督式回歸模型**：每筆樣本代表一趟車程，以該趟行程資料預測平均油耗（L/100 km）。標籤依現有分析規則計算：

```text
目標油耗 (L/100 km) = 100 × 有效累積燃油差 (L) ÷ 有效累積里程差 (km)
```

燃油與里程都取同一趟行程的累積計數器端點差。計數器重置、無效端點、非正里程、非有限值或其他無效標籤會排除，不以補零或修補計數器的方式製造標籤。

目前的模型輸入是 19 個**不含燃油值**的特徵：行程距離與時間、時間加權的車速平均／標準差、RPM／引擎負載／冷卻水溫平均、引擎運轉停留比例、速度變化與加速度統計、正向動能變化代理值，以及 7 個速度區間的時間比例。每個特徵的資料涵蓋率會一併輸出作為品質診斷，**目前涵蓋率欄位只寫入特徵資料檔，不會輸入模型**。車輛與行程識別碼、來源列號、燃油值及其衍生欄位都不作為模型特徵；缺值由 XGBoost 原生處理，沒有另行擬合填補器。

目前使用 XGBoost 平方誤差回歸，最多 800 輪，依驗證集 MAE early stopping（耐心 50 輪）。固定參數及 random seed 記錄在每次執行輸出的中繼資料中。網站單趟頁會載入選定的實驗模型，估計已完成行程的油耗，並以 XGBoost 原生 `pred_contribs` TreeSHAP 解釋該趟模型輸出；使用模型實際採用的樹數，不另裝 SHAP 套件。這是行程後估計與模型解釋，不是即時預測或因果歸因；另有獨立的單特徵情境試算，限制與說明見下文。

### 一般運輸適用範圍模型（目前網站模型）

網站目前使用 `applicability-v1` 規則篩選後重訓的 C 模型。適用條件是距離至少 5 km、可判讀時間至少 20 分鐘時停車引擎運轉比例低於 80%，且觀測涵蓋率至少 70%；既有目標有效性規則仍適用。標籤可重疊，排除總數以行程去重。所有實驗沿用既有時間切分，不因篩選改變切分邊界。訓練列的歷史僅使用行程結束時間早於該列起始時間的合格訓練行程；驗證、測試及新行程使用凍結的合格訓練歷史。

四種方法在相同合格行程上的驗證 MAE（L/100 km）：

| 方法 | MAE | RMSE | MedAE | P90 | R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 A：原訓練資料車輛歷史中位數 | 3.742 | 5.644 | 2.367 | 7.694 | 0.697 |
| 新 A：合格訓練資料車輛歷史中位數 | 3.735 | 5.692 | 2.396 | 8.088 | 0.692 |
| 原 C：原模型與原特徵處理 | 4.211 | 6.448 | 2.910 | 8.056 | 0.604 |
| 新 C：適用範圍篩選後重訓 | **2.789** | 4.164 | 1.894 | 6.174 | 0.835 |

新 C 依驗證 MAE 優於原 C 與新 A 而選為展示模型。既有測試集在開發期間已多次用於追蹤，185 趟合格行程上的新 C MAE 為 2.668，僅作追蹤評估，不用來選模型。合格行程保留率：訓練 1,013/2,120（47.8%）、驗證 204/449（45.4%）、測試 185/456（40.6%）；全體合格行程占原資料總里程 93.0%、總燃油 92.9%。這些數字描述合格範圍內的模型成績，不代表被排除行程沒有業務價值。

訓練與比較的完整輸出（包括逐車及距離分組指標）保存在 `analysis_output/experiments/applicability_applicability-v1_20261009_135206/`。網站可部署模型包及推論所需歷史資料保存在 `analysis/model_bundle_applicable/`；重新啟動網站會載入此包。原模型包與先前實驗輸出仍保留於 `analysis_output/experiments/`。

### 儀表板模型面板與本趟觀察

在「AI 油耗估計與解釋」頁選取合格行程後，網站會顯示實際油耗、新 C 模型估計、實際值減估計值及該車合格訓練行程歷史中位數。成績文字顯示合格驗證行程上的新 C MAE（2.789）與新 A 歷史中位數 MAE（3.735 L/100 km）。瀑布圖呈現 SHAP 解釋起點、貢獻最大的最多 6 個特徵、其餘特徵合計及模型最終估計；表格同時列出特徵值、單位、SHAP 貢獻方向。模型版本、驗證／測試追蹤分數、樣本分組及資料限制收在展開區。不符合適用範圍的行程保留原始資料，但不顯示新模型估計與 SHAP。

若模型情境試算找到可量化的項目，AI 省油建議會同時顯示在「AI 油耗估計與解釋」及「行程與地圖」的「本趟觀察」中，列出改善方式及預估省油百分比；行車安全提醒及各項具體操作說明只放在 AI 油耗估計區，不放在「本趟觀察」。情境只針對 SHAP 推高估計且可對應駕駛行為的特徵，以同一模型將單一特徵調至合格訓練行程第 25 百分位試算，並顯示本趟換算值。沒有合格建議時不顯示空白或提示卡片。「本趟觀察」不顯示停車引擎運轉時間；其他怠速分析仍保留。情境不是因果推論或實際節油保證，彼此不可相加；加減速相關指標合併為一項建議。

SHAP 說明的是**模型輸出如何由其輸入特徵構成**，不是特徵對耗油的因果影響；歷史中位數基準與 SHAP 解釋起點是不同量。模型目前未勝過歷史基準，介面會明確標示「實驗模型」並顯示兩者驗證分數。

研究實驗輸出 `analysis_output/` 仍由 `.gitignore` 排除；網站所需的適用範圍模型包獨立保存在 `analysis/model_bundle_applicable/`，部署網站時須一併包含此目錄。模型包含用於推論的車輛歷史統計及合格訓練特徵索引；僅部署程式碼而未包含模型包時，網站無法載入本次模型。若要換用另一份來源資料或重新訓練，請依權限管理來源資料與模型包，不要將原始行程資料或含識別資訊的實驗輸出公開。

合作者日常使用不必手動訓練；使用前述啟動程式即可。以下指令是重現離線研究實驗用，第二次歷史特徵實驗依賴第一輪模型產物，第三次候選實驗依賴已保存的研究產物，不是全新工作目錄的安裝步驟：

```powershell
.\.venv\Scripts\python.exe -m analysis.vehicle_history_experiment
.\.venv\Scripts\python.exe -m analysis.fuel_liter_experiment
```

### 準備資料及執行訓練

在 FuelWise 專案根目錄（`server.py` 所在資料夾）開啟 PowerShell。使用與網站相同的 Python 虛擬環境，並先安裝專案依賴：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

將獲准使用的來源活頁簿放在專案根目錄並命名為 `data.xlsx`／`data.xls`，或先使用網站匯入流程建立 `.cache` 索引。若索引尚未建立、只想建立索引而不啟動網站：

```powershell
.\.venv\Scripts\python.exe server.py --prepare-only
```

資料索引準備好後執行完整訓練及測試集評估：

```powershell
.\.venv\Scripts\python.exe -m analysis.train_fuel_model
```

程式**只讀取既有 SQLite 索引**，不會自行重新匯入 Excel、不會呼叫網站 API，也不會覆寫來源活頁簿。預設選取 `.cache` 中最新可用索引。可以指定索引和輸出根目錄：

```powershell
.\.venv\Scripts\python.exe -m analysis.train_fuel_model --index ".cache\index.sqlite" --output "analysis_output\fuel_model_runs"
```

每次執行都會在 `analysis_output/fuel_model_runs` 下建立新的時間戳資料夾，不覆蓋前次結果。主要產物如下：

| 輸出檔案 | 用途 |
| --- | --- |
| `report.md` | 人類可讀的資料切分、排除摘要、評估指標及特徵說明。 |
| `evaluation.json` | MAE、RMSE、R²、early-stopping 最佳輪數、負預測數及各車輛測試 MAE。 |
| `metadata.json` | 特徵與目標定義、切分日期與樣本數、排除原因、來源索引資訊、套件版本及完整模型參數。 |
| `trip_features.csv` | 納入切分的逐趟特徵、目標值、資料涵蓋率、品質欄位及 train／validation／test 標記。 |
| `excluded_trips.csv` | 因目標／時間品質或跨切分邊界而排除的行程及原因。 |
| `test_predictions.csv` | 測試集實際油耗、基準與 XGBoost 預測，以及逐趟誤差。 |
| `xgboost_model.json` | 可用 XGBoost 載入的模型檔。 |
| `feature_definitions.json` | 特徵定義、單位及適用限制。 |

### 目前訓練與評估方法

1. 從索引逐趟讀取原始觀測，使用共用 `analytics.py` 重新計算標籤、特徵與資料品質；拒絕重複的車輛＋行程樣本。
2. 依行程起始時間排序，按約 **70%／15%／15%** 建立 train／validation／test 時間區段。同一趟若跨越任一時間邊界就整趟排除，不隨機切分、不退回放寬邊界的切法。有效樣本少於 20 趟或切分後任何集合為空時，訓練會停止。
3. 只用 train 訓練 XGBoost；validation 僅用於 early stopping；test 僅在模型選定後評估，不用來調參。固定的驗證規則可避免將未來測試資料洩漏回模型選擇。
4. 在同一測試集上比較 XGBoost 與簡單基準：依訓練資料計算的車輛油耗中位數；測試中未出現在訓練資料的車輛則退回全體訓練中位數。比較 MAE、RMSE、R²，也保留各車輛 MAE 與負油耗預測數量。
5. 儲存模型後重新載入，確認載入前後測試預測一致，並輸出本次完整資料、模型及軟體版本資訊。

### 一般運輸行程適用性篩選（applicability-v1）

`analysis/trip_applicability.py` 集中定義規則（版本 `applicability-v1`），供訓練、API 與前端共用。以下是產品範圍的預先指定值，不是已驗證的物理界線，也未依 MAE 調整：

- `short_distance`：有效距離 < 5.0 km。
- `stationary_dominant`：可判讀時間 ≥ 20 分鐘且可判讀區間內停車引擎運轉比例 ≥ 80%（只描述可觀測區間，不推論整趟）。
- `insufficient_observation`：可判讀時間／整趟時間 < 70%，或無法計算。
- `invalid_target`：沿用既有燃油、里程端點與數值有效性規則。
- 標籤可複選；`model_eligible` 代表無任何排除原因。原始列全部保留，被排除行程仍顯示總耗油、距離與停車資訊，只是不顯示模型估計與 SHAP。

可判讀時間＝速度與轉速皆有效的實際時間差聯集，不跨越超過 120 秒的長缺口；無效 CAN 與長缺口保留為 unknown。

重現實驗：`.\.venv\Scripts\python.exe -W ignore -m analysis.applicability_experiment`，輸出到 `analysis_output/experiments/applicability_applicability-v1_<時間>/`（含 `evaluation.json` 與 `dashboard_bundle/`）。把 `dashboard_bundle/` 複製為 `analysis_output/experiments/dashboard_model_applicable/` 後，推論會優先使用它；舊的 `dashboard_model/` 保留為備援。測試集已多次參與開發，結果僅稱為追蹤評估。

## 參考

- Leaflet: https://leafletjs.com/reference.html
- Plotly.js: https://plotly.com/javascript/plotlyjs-function-reference/
- OpenStreetMap: https://www.openstreetmap.org/copyright
- 訊號單位、車輛/CAN狀態與事件ID：來源Excel「欄位對照表」。
