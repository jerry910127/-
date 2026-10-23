# 水果新鮮度 CNN 分類（機器學習期中專案）

以 PyTorch CNN 辨識蘋果、香蕉、橘子是新鮮還是腐爛，共 6 類。
資料集：Kaggle「Fruits fresh and rotten for classification」。

專案程式、說明與結果都在 [`fruit/`](fruit/)，詳細內容請見 [fruit/README.md](fruit/README.md)。

## 目前結果

<!-- 下表由 python fruit/make_report.py 依 fruit/result/metrics.json 自動更新，請勿手動修改 -->
<!-- results:start -->
| 指標 | 數值 |
|---|---|
| 資料切分 | 按原始照片分組（測試照片洩漏 0%） |
| Test Accuracy | 91.95% |
| Macro F1 | 0.9154 |
| 新鮮 / 腐爛 二分準確率 | 95.64% |
| 多數類別基準 Accuracy | 21.59% |
| 模型參數量 | 25,350 |
| 訓練 | 15 回合（最佳第 15 回合），Tesla T4 約 1.9 分鐘 |
| 外部照片（Demo） | 24 張中 10 張完全正確、14 張新鮮度正確 |
<!-- results:end -->

完整報告：[fruit/output/pdf/fruit_cnn_report.pdf](fruit/output/pdf/fruit_cnn_report.pdf)

## 在 Colab 執行（建議）

開啟 [colab_run.ipynb](https://colab.research.google.com/github/jerry910127/-/blob/main/colab_run.ipynb)，執行階段選 **T4 GPU**，由上往下執行：
自動下載資料集（免 Kaggle 帳號）→ 訓練與評估 → 上傳照片做 Demo → 產生 PDF → 下載 `fruit_results.zip`。

## 在本機執行

在本 repo 根目錄執行；`fruit.zip`（Kaggle 下載的資料集）需先放在 `fruit/` 裡：

```bash
python -m pip install -r fruit/requirements.txt
python fruit/train.py                  # EDA、訓練、test set 評估與所有圖表
python fruit/predict.py demo_photos/   # Demo：預測照片（可給多張或資料夾）並輸出結果圖
python fruit/make_report.py            # 重新產生 PDF 報告，並更新上方結果表
```

## 目錄

```
fruit/
├── dataset.py        讀取 ZIP、去除重複圖、切分 train / val / test
├── eda.py            資料探索圖：類別張數、範例圖、尺寸分布
├── cnn.py            CNN 模型
├── train.py          訓練、驗證、測試評估
├── predict.py        照片推論與 Demo 結果圖
├── plot_results.py   訓練曲線、混淆矩陣、分錯圖片
├── make_report.py    產生 PDF 報告
├── result/           最佳模型權重與評估結果
├── output/pdf/       PDF 報告
├── original/         課堂範例原始程式（Fashion-MNIST，保留作參考）
└── previous/         先前 LEGO 版本程式與資料
```
