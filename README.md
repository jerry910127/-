# 新鮮 / 腐爛水果影像分類（PyTorch）

期中個人專案。資料集：[Fruits fresh and rotten for classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)，
共 6 類（freshapples、freshbanana、freshoranges、rottenapples、rottenbanana、rottenoranges），約 13,600 張圖片。

## 檔案說明

| 檔案 | 用途 |
|---|---|
| `dataset.py` | 讀取資料（`ImageFolder`）、資料增強、依類別比例切出 train/val、建立 DataLoader |
| `eda.py` | 資料探索：各類張數、每類範例圖、圖片尺寸分布 |
| `model.py` | 模型 A：自建 CNN；模型 B：ResNet18 / MobileNetV2 遷移學習 |
| `train.py` | 訓練：記錄 loss/acc、保存最佳模型、Early stopping、學習率遞減 |
| `evaluate.py` | Test set 評估：Accuracy、classification report、混淆矩陣、分錯的圖片、模型比較表 |
| `predict.py` | Demo：對任意圖片預測並輸出 Top-3 機率 |
| `make_report.py` | 把 `result/` 的數據與圖表整理成一份 HTML 結果報告（`result/report.html`） |
| `utils.py` | 共用函式（亂數種子、載入 checkpoint、反正規化） |
| `fruit_colab.ipynb` | Colab 一鍵執行與報告用 Notebook |

## 在 Colab 執行（建議）

1. 程式碼在 https://github.com/jerry910127/-
2. 在 Colab 開啟 [`fruit_colab.ipynb`](https://colab.research.google.com/github/jerry910127/-/blob/main/fruit_colab.ipynb)，並將執行階段改為 **T4 GPU**。
3. 由上往下執行（第一格會自動 clone 此 repo）。

## 在本機執行

```bash
pip install -r requirements.txt
python eda.py      --data_dir <資料夾>
python train.py    --data_dir <資料夾> --arch cnn
python train.py    --data_dir <資料夾> --arch resnet18
python evaluate.py --data_dir <資料夾> --ckpt result/best_cnn.pth
python evaluate.py --data_dir <資料夾> --ckpt result/best_resnet18.pth
python predict.py  --ckpt result/best_resnet18.pth my_photo.jpg
python make_report.py --author "姓名 學號"
```

報告 `result/report.html` 是單一檔案（圖片已內嵌），用瀏覽器開啟後可「列印 → 另存為 PDF」。
內容包含：專案概述、EDA、前處理、模型設計、訓練曲線、測試結果（各類指標、混淆矩陣）、錯誤分析、Demo、結論。
其中的數字與觀察（最佳模型、最難分的類別、最常混淆的類別、是否過擬合）都是從結果自動算出來的。

`<資料夾>` 指向 Kaggle 下載解壓後的位置即可。程式會自動往下層尋找同時包含 `train/`、`test/` 的資料夾。

所有輸出都存在 `result/`：

| 輸出檔 | 內容 |
|---|---|
| `best_<arch>.pth` | 最佳模型（內含 arch、類別、圖片尺寸） |
| `log_<arch>.csv`、`curve_<arch>.png` | 訓練紀錄與 loss/acc 曲線 |
| `report_<arch>.txt`、`confusion_<arch>.png`、`misclassified_<arch>.png` | 評估結果 |
| `summary.csv` | 兩個模型的比較表（準確率、參數量） |

## 設計重點（報告可講）

- **資料切分**：Kaggle 已提供 train/test。val 從 train 依類別比例（stratify）切出 20%；test 只在最後評估時用一次。
- **資料增強**：隨機裁切、翻轉、旋轉、亮度/對比/飽和度。刻意**不調色相（hue）**，因為顏色是判斷新鮮或腐爛的關鍵線索。
- **自建 CNN**：4 個 `Conv-BN-ReLU ×2 + MaxPool` 區塊，接 Global Average Pooling，約 1.2M 參數。
- **遷移學習**：ResNet18 載入 ImageNet 權重，換掉最後一層改為 6 類，用較小的學習率（1e-4）fine-tune；約 11M 參數。
- **訓練技巧**：CrossEntropyLoss + Adam、ReduceLROnPlateau、依 val_loss 保存最佳模型、Early stopping。
- **評估**：除了 Accuracy，也看每類的 precision/recall/F1 和混淆矩陣，並分析分錯的圖片。
