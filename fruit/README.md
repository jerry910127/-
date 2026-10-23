# 水果種類與外觀新鮮度 CNN 分類

使用 Kaggle「Fruits fresh and rotten for classification」資料集（下載後的 `fruit.zip`），辨識 freshapples、freshbanana、freshoranges、rottenapples、rottenbanana、rottenoranges 六類。符合公告的資料處理、模型訓練、結果評估流程；不沿用課堂範例資料集。

## 執行

Python 3.10 以上，在儲存庫根目錄執行：

```powershell
python -m pip install -r fruit/requirements.txt
python fruit/train.py
python fruit/predict.py "照片1.jpg" "照片2.jpg"   # 或給一個照片資料夾
python fruit/make_report.py
```

- `train.py`：資料稽核 → EDA 圖 → 訓練 → 測試評估 → 學習曲線、混淆矩陣、分錯圖片。
- `predict.py`：印出每張照片的機率，並輸出 `result/demo_predictions.png`（含同名 `.json`）。
- `make_report.py`：依 `result/` 的實際結果產生 PDF，並更新根目錄 README 的結果表。
- 只想重畫圖不重新訓練：`python fruit/plot_results.py`、`python fruit/eda.py`。

將原始 `fruit.zip` 放在本資料夾，不必解壓縮。檔案過大，未提交 GitHub。ZIP 須包含 `dataset/train/<class>/` 與 `dataset/test/<class>/`；重複的 `dataset/dataset/` 完全不讀取。

輸入轉 RGB、縮至 64×64、以 (x-0.5)/0.5 正規化，僅訓練集隨機水平翻轉。移除完全相同檔案內容，manifest 記錄切分、SHA-256 與原始照片名稱。

**資料切分（避免洩漏）**：資料集的 13,599 張圖其實只來自 1,511 張原始照片，每張另有旋轉、平移、椒鹽雜訊、垂直翻轉等離線增強版本（檔名前綴 `rotated_by_*`、`translation_` 等）。資料集原本的 train / test 中，測試圖的原始照片 100% 也出現在訓練資料裡，測試分數會被高估。因此預設改為**按原始照片分組**逐類切分：20% 原始照片作測試、其餘的 15% 作驗證，同一張照片的所有變體只會在同一組。若要重現原始切分做對照：`python fruit/train.py --split original --out-dir fruit/result_original_split`。`result_original_split/` 保存了原始切分在 Colab T4 上的結果，報告會拿來比較修正前後。

CNN 三個卷積區塊 (16/32/64 通道)，搭配 BatchNorm、ReLU、Pooling、Dropout。使用 CrossEntropy、Adam、驗證 loss 選最佳模型、5 回合未改善提前停止。預設最多 15 回合，CPU/GPU 自動選擇。輸出 `result/`：最佳權重、資料稽核、分割清單、EDA 圖、訓練曲線、混淆矩陣、分錯圖片、accuracy、macro F1、逐類 precision/recall/F1、參數量、執行環境與逐張測試預測（含信心度）。

`original/` 是課堂原始三個 Python 檔案的完整副本（所有註解保留）；`previous/` 是先前 LEGO 版本的完整副本（含 `archive.zip` 資料）。新程式用獨立入口，避免舊程式重複 main 與多餘訓練迴圈混入。

PDF 位於 `output/pdf/fruit_cnn_report.pdf`，內容包含題目與動機、EDA、程式設計理由、模型架構與參數量、訓練結果、混淆分析、錯誤案例、Demo 與結論。模型只判斷資料集定義的外觀類別，不代表食物可安全食用。

產生 PDF 會依序尋找 Windows 微軟正黑體、macOS Arial Unicode、Linux 文泉驛正黑（Colab 可先 `apt install fonts-wqy-zenhei`）；都找不到時改用 reportlab 內建中文字型，仍可產生。GPU 上個別運算沒有確定性版本時只會警告、不中斷訓練。
