# 水果種類與外觀新鮮度 CNN 分類

使用 Kaggle「Fruits fresh and rotten for classification」資料集（下載後的 `fruit.zip`），辨識 freshapples、freshbanana、freshoranges、rottenapples、rottenbanana、rottenoranges 六類。符合公告的資料處理、模型訓練、結果評估流程；不沿用課堂範例資料集。

## 執行

Python 3.10 以上，在儲存庫根目錄執行：

```powershell
python -m pip install -r fruit/requirements.txt
python fruit/train.py --epochs 15
python fruit/predict.py "你的照片.jpg"
python fruit/make_report.py
```

將原始 `fruit.zip` 放在本資料夾，不必解壓縮。檔案過大，未提交 GitHub。ZIP 須包含 `dataset/train/<class>/` 與 `dataset/test/<class>/`；重複的 `dataset/dataset/` 完全不讀取。

輸入轉 RGB、縮至 64×64、以 (x-0.5)/0.5 正規化，僅訓練集隨機水平翻轉。原 test 保留作最終測試；從原 train 逐類固定種子抽 15% 作驗證。移除完全相同檔案內容，manifest 記錄切分與 SHA-256。未偵測所有近似圖及離線增強家族，仍可能高估泛化能力。

CNN 三個卷積區塊 (16/32/64 通道)，搭配 BatchNorm、ReLU、Pooling、Dropout。使用 CrossEntropy、Adam、驗證 loss 選最佳模型、5 回合未改善提前停止。預設最多 15 回合，CPU/GPU 自動選擇。輸出 `result/`：最佳權重、資料稽核、分割清單、訓練曲線、混淆矩陣、accuracy、macro F1、逐類 precision/recall/F1 與逐張測試預測。

`original/` 是課堂原始三個 Python 檔案的完整副本（所有註解保留）；`previous/` 是先前 LEGO 版本的完整副本（含 `archive.zip` 資料）。新程式用獨立入口，避免舊程式重複 main 與多餘訓練迴圈混入。

PDF 位於 `output/pdf/fruit_changes_report.pdf`，說明修改內容、原因與本次實測。模型只判斷資料集定義的外觀類別，不代表食物可安全食用。

產生 PDF 會依序尋找 Windows 微軟正黑體、macOS Arial Unicode、Linux 文泉驛正黑（Colab 可先 `apt install fonts-wqy-zenhei`）；都找不到時改用 reportlab 內建中文字型，仍可產生。GPU 上個別運算沒有確定性版本時只會警告、不中斷訓練。
