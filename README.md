# 水果新鮮度 CNN 分類（機器學習期中專案）

以 PyTorch CNN 辨識蘋果、香蕉、橘子是新鮮還是腐爛，共 6 類。
資料集：Kaggle「Fruits fresh and rotten for classification」。

專案程式、說明與結果都在 [`fruit/`](fruit/)，詳細內容請見 [fruit/README.md](fruit/README.md)。

## 目前結果

| 指標 | 數值 |
|---|---|
| Test Accuracy | 95.52% |
| Macro F1 | 0.9543 |
| 多數類別基準 Accuracy | 22.28% |
| 訓練 | 8 回合，CPU 約 9 分鐘 |

完整報告：[fruit/output/pdf/fruit_changes_report.pdf](fruit/output/pdf/fruit_changes_report.pdf)

## 快速開始

在本 repo 根目錄執行；`fruit.zip`（Kaggle 下載的資料集）需先放在 `fruit/` 裡：

```bash
python -m pip install -r fruit/requirements.txt
python fruit/train.py                  # 訓練並在 test set 評估
python fruit/predict.py "你的照片.jpg"  # 單張照片推論（Demo）
python fruit/make_report.py            # 重新產生 PDF 報告
```

## 目錄

```
fruit/
├── dataset.py        讀取 ZIP、去除重複圖、切分 train / val / test
├── cnn.py            CNN 模型
├── train.py          訓練、驗證、測試評估
├── predict.py        單張照片推論
├── plot_results.py   訓練曲線、混淆矩陣
├── make_report.py    產生 PDF 報告
├── result/           最佳模型權重與評估結果
├── output/pdf/       PDF 報告
├── original/         課堂範例原始程式（Fashion-MNIST，保留作參考）
└── previous/         先前 LEGO 版本程式與資料
```
