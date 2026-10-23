# make_report.py
# 把 result/ 裡的訓練與評估結果整理成一份 HTML 報告（圖片內嵌，單一檔案即可分享 / 列印成 PDF）
# 用法：先跑完 eda.py、train.py、evaluate.py（predict.py 可選），再執行
#   python make_report.py --result_dir result --out result/report.html
import argparse
import base64
import csv
import json
from datetime import date
from html import escape
from pathlib import Path

ARCH_ORDER = ["cnn", "resnet18", "mobilenet_v2"]
ARCH_NAMES = {
    "cnn": "模型 A：自建 CNN",
    "resnet18": "模型 B：ResNet18（遷移學習）",
    "mobilenet_v2": "模型 B：MobileNetV2（遷移學習）",
}
ARCH_DESC = {
    "cnn": "4 個「Conv-BN-ReLU ×2 → MaxPool」區塊（32→64→128→256 通道），"
           "接 Global Average Pooling、Dropout(0.5) 與全連接層輸出 6 類。從零開始訓練。",
    "resnet18": "載入 ImageNet 預訓練權重的 ResNet18，把最後的全連接層換成 6 類輸出，"
                "以較小的學習率 fine-tune 整個網路。",
    "mobilenet_v2": "載入 ImageNet 預訓練權重的 MobileNetV2，把分類層換成 6 類輸出後 fine-tune。",
}
CLASS_ZH = {
    "freshapples": "新鮮蘋果", "freshbanana": "新鮮香蕉", "freshoranges": "新鮮橘子",
    "rottenapples": "腐爛蘋果", "rottenbanana": "腐爛香蕉", "rottenoranges": "腐爛橘子",
}

CSS = """
:root { --ink:#1f2933; --muted:#616e7c; --line:#e4e7eb; --bg:#ffffff; --soft:#f5f7fa;
        --accent:#2f7d4f; --bad:#b83232; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font: 16px/1.7 "Noto Sans TC","Microsoft JhengHei","PingFang TC",system-ui,sans-serif; }
main { max-width: 960px; margin: 0 auto; padding: 40px 20px 80px; }
h1 { font-size: 30px; margin: 0 0 4px; }
.sub { color: var(--muted); margin: 0 0 28px; }
h2 { font-size: 22px; margin: 44px 0 12px; padding-bottom: 6px; border-bottom: 2px solid var(--line); }
h3 { font-size: 18px; margin: 28px 0 8px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 16px; font-size: 15px; }
th, td { border: 1px solid var(--line); padding: 6px 10px; text-align: left; }
th { background: var(--soft); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
img { max-width: 100%; display: block; margin: 8px auto 16px; border: 1px solid var(--line); border-radius: 6px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0; }
.kpi { background: var(--soft); border-radius: 8px; padding: 12px 16px; }
.kpi b { display: block; font-size: 26px; color: var(--accent); }
.kpi span { color: var(--muted); font-size: 14px; }
.note { background: var(--soft); border-left: 4px solid var(--accent); padding: 10px 14px; margin: 12px 0; border-radius: 4px; }
.missing { color: var(--bad); }
ul { padding-left: 22px; }
@media print { main { padding: 0; } h2 { break-after: avoid; } img, table { break-inside: avoid; } }
"""


# ---------- 小工具 ----------
def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_log(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def img_tag(path: Path, alt: str):
    if not path.exists():
        return f'<p class="missing">（找不到 {escape(path.name)}，請先執行對應的程式）</p>'
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{escape(alt)}">'


def zh(c):
    return f"{CLASS_ZH.get(c, c)}（{c}）"


def pct(x):
    return f"{x * 100:.2f}%"


def table(headers, rows, num_cols=()):
    th = "".join(f'<th class="{"num" if i in num_cols else ""}">{escape(h)}</th>' for i, h in enumerate(headers))
    trs = "".join(
        "<tr>" + "".join(f'<td class="{"num" if i in num_cols else ""}">{v}</td>' for i, v in enumerate(r)) + "</tr>"
        for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


# ---------- 自動分析 ----------
def most_confused(m):
    """混淆矩陣中最大的非對角元素 → (真實類別, 預測類別, 張數)"""
    names, cm = m["class_names"], m["confusion"]
    best = max(((cm[i][j], i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j),
               default=(0, 0, 0))
    return names[best[1]], names[best[2]], best[0]


def fresh_rotten_acc(m):
    """把 6 類合併成「新鮮 / 腐爛」二分類時的準確率"""
    names, cm = m["class_names"], m["confusion"]
    is_fresh = [n.startswith("fresh") for n in names]
    correct = sum(cm[i][j] for i in range(len(cm)) for j in range(len(cm)) if is_fresh[i] == is_fresh[j])
    return correct / max(1, sum(map(sum, cm)))


def train_summary(log, metrics):
    best_ep = metrics["best_epoch"] if metrics else int(min(log, key=lambda r: r["val_loss"])["epoch"])
    row = next(r for r in log if int(r["epoch"]) == best_ep)
    total_sec = sum(r.get("sec", 0) for r in log)
    return {
        "epochs": len(log), "best_epoch": best_ep,
        "val_acc": row["val_acc"], "train_acc": row["train_acc"],
        "gap": row["train_acc"] - row["val_acc"], "minutes": total_sec / 60,
    }


# ---------- 各段落 ----------
def section_overview(eda, models):
    n_img = sum(eda["train"]) + sum(eda["test"]) if eda else None
    best = max(models, key=lambda a: models[a]["test_acc"]) if models else None
    kpis = []
    if n_img:
        kpis.append((f"{n_img:,}", "圖片總數"))
    kpis.append(("6", "類別數"))
    if best:
        kpis.append((pct(models[best]["test_acc"]), f"最佳 Test Accuracy（{best}）"))
        kpis.append((pct(fresh_rotten_acc(models[best])), "新鮮 / 腐爛 二分準確率"))
    kpi_html = "".join(f'<div class="kpi"><b>{v}</b><span>{escape(t)}</span></div>' for v, t in kpis)
    return f"""
<h2>1. 專案概述</h2>
<ul>
  <li><b>題目</b>：以 CNN 辨識水果（蘋果、香蕉、橘子）是新鮮還是腐爛，共 6 類的影像分類。</li>
  <li><b>動機</b>：生鮮賣場與物流的品管多半靠人工目視，若能用影像自動判別腐爛水果，可以減少人力並降低食物浪費。</li>
  <li><b>資料集</b>：Kaggle「Fruits fresh and rotten for classification」（sriramr）。</li>
  <li><b>方法</b>：比較「從零訓練的自建 CNN」與「ImageNet 預訓練模型的遷移學習」。</li>
  <li><b>工具</b>：Python、PyTorch、torchvision、scikit-learn、matplotlib，於 Google Colab GPU 上執行。</li>
</ul>
<div class="kpis">{kpi_html}</div>"""


def section_data(eda, result_dir):
    if eda:
        rows = [[escape(zh(c)), f"{tr:,}", f"{te:,}", f"{tr + te:,}"]
                for c, tr, te in zip(eda["classes"], eda["train"], eda["test"])]
        rows.append(["<b>合計</b>", f"<b>{sum(eda['train']):,}</b>", f"<b>{sum(eda['test']):,}</b>",
                     f"<b>{sum(eda['train']) + sum(eda['test']):,}</b>"])
        tbl = table(["類別", "Train", "Test", "合計"], rows, num_cols=(1, 2, 3))
        ratio = max(eda["train"]) / max(1, min(eda["train"]))
        balance = ("各類張數差距不大，不需要特別處理類別不平衡。" if ratio < 1.5 else
                   f"最多與最少類別相差約 {ratio:.1f} 倍，因此除了 Accuracy 之外，也要看各類別的 recall 與 F1。")
        size = (f"抽樣 {eda['size_sample']} 張，寬度 {eda['width'][0]}~{eda['width'][1]} px、"
                f"高度 {eda['height'][0]}~{eda['height'][1]} px，尺寸不一，所以訓練前要統一縮放。")
    else:
        tbl, balance, size = '<p class="missing">（找不到 eda_stats.json，請先執行 eda.py）</p>', "", ""
    return f"""
<h2>2. 資料集與資料探索（EDA）</h2>
{tbl}
<div class="note">{balance} {size}</div>
{img_tag(result_dir / "eda_class_counts.png", "各類別張數")}
<h3>各類別範例</h3>
{img_tag(result_dir / "eda_samples.png", "各類別範例圖")}
<h3>圖片尺寸分布</h3>
{img_tag(result_dir / "eda_sizes.png", "圖片尺寸分布")}"""


def section_preprocess(configs):
    cfg = next(iter(configs.values()), {}) if configs else {}
    val_ratio = cfg.get("val_ratio", 0.2)
    sizes = "、".join(f"{a} 為 {c['img_size']}×{c['img_size']}"
                      for a in ARCH_ORDER if (c := configs.get(a))) or "自建 CNN 128×128、預訓練模型 224×224"
    return f"""
<h2>3. 資料前處理</h2>
<ul>
  <li><b>資料切分</b>：Kaggle 已分好 train / test。再從 train 依類別比例（stratify）切出 {val_ratio:.0%} 作為 validation，
      用來挑選最佳模型與 early stopping；test set 只在最後評估時使用一次，避免資料洩漏。</li>
  <li><b>尺寸統一</b>：圖片尺寸不一，訓練前統一縮放（{sizes}）。自建 CNN 用較小尺寸以加快訓練，
      預訓練模型用 224×224 與 ImageNet 一致。</li>
  <li><b>正規化</b>：以 ImageNet 的 mean / std 正規化 RGB 三個通道。</li>
  <li><b>資料增強（只用在訓練集）</b>：RandomResizedCrop、水平與垂直翻轉、±20° 旋轉、亮度 / 對比 / 飽和度微調。
      刻意<b>不調整色相（hue）</b>，因為顏色是判斷新鮮或腐爛最重要的線索。</li>
</ul>"""


def section_models(models, configs):
    archs = [a for a in ARCH_ORDER if a in models or a in configs]
    rows = []
    for a in archs:
        m, c = models.get(a, {}), configs.get(a, {})
        rows.append([escape(ARCH_NAMES.get(a, a)),
                     f"{m['params']:,}" if m else "-",
                     f"{c.get('img_size', m.get('img_size', '-'))}",
                     f"{c['lr']:g}" if c else "-",
                     f"{c.get('batch_size', '-')}"])
    desc = "".join(f"<li><b>{escape(ARCH_NAMES.get(a, a))}</b>：{ARCH_DESC.get(a, '')}</li>" for a in archs)
    cfg = next(iter(configs.values()), {}) if configs else {}
    env = (f"訓練環境：{escape(cfg.get('gpu') or cfg.get('device', '-'))}，PyTorch {escape(cfg.get('torch', '-'))}。"
           if cfg else "")
    return f"""
<h2>4. 模型設計</h2>
<ul>{desc}</ul>
{table(["模型", "參數量", "輸入尺寸", "學習率", "Batch size"], rows, num_cols=(1, 2, 3, 4))}
<ul>
  <li><b>Loss / Optimizer</b>：CrossEntropyLoss + Adam。</li>
  <li><b>學習率調整</b>：ReduceLROnPlateau（val_loss 連續 2 個 epoch 沒進步，學習率減半）。</li>
  <li><b>模型保存</b>：只保存 val_loss 最低的 checkpoint；連續 {cfg.get('patience', 5)} 個 epoch 沒進步就 early stopping。</li>
</ul>
<p>{env}</p>"""


def section_training(models, logs, result_dir):
    parts = ["<h2>5. 訓練過程</h2>"]
    rows = []
    for a in [a for a in ARCH_ORDER if a in logs]:
        s = train_summary(logs[a], models.get(a))
        rows.append([escape(ARCH_NAMES.get(a, a)), s["epochs"], s["best_epoch"],
                     pct(s["train_acc"]), pct(s["val_acc"]), f"{s['gap'] * 100:+.2f}",
                     f"{s['minutes']:.1f}"])
        verdict = ("train 與 val 準確率接近，沒有明顯過擬合。" if s["gap"] < 0.03 else
                   "train 準確率明顯高於 val，有些過擬合，可以加強資料增強或提早停止。")
        parts.append(f"<h3>{escape(ARCH_NAMES.get(a, a))}</h3>"
                     f"{img_tag(result_dir / f'curve_{a}.png', f'{a} 訓練曲線')}"
                     f'<div class="note">共訓練 {s["epochs"]} 個 epoch，第 {s["best_epoch"]} 個 epoch 的 val_loss 最低'
                     f'（val acc {pct(s["val_acc"])}）。{verdict}</div>')
    if rows:
        parts.insert(1, table(["模型", "訓練 epoch", "最佳 epoch", "Train acc", "Val acc",
                               "差距（百分點）", "訓練時間（分）"], rows, num_cols=(1, 2, 3, 4, 5, 6)))
    else:
        parts.append('<p class="missing">（找不到 log_*.csv，請先執行 train.py）</p>')
    return "".join(parts)


def section_results(models, result_dir):
    if not models:
        return '<h2>6. 測試結果</h2><p class="missing">（找不到 metrics_*.json，請先執行 evaluate.py）</p>'
    archs = [a for a in ARCH_ORDER if a in models]
    rows = [[escape(ARCH_NAMES.get(a, a)), pct(models[a]["test_acc"]), f"{models[a]['macro_f1']:.4f}",
             pct(fresh_rotten_acc(models[a])), f"{models[a]['params']:,}"] for a in archs]
    parts = [f"""
<h2>6. 測試結果（Test set，{models[archs[0]]['n_test']:,} 張）</h2>
{table(["模型", "Accuracy", "Macro F1", "新鮮/腐爛 二分準確率", "參數量"], rows, num_cols=(1, 2, 3, 4))}"""]

    for a in archs:
        m = models[a]
        pc = m["per_class"]
        crow = [[escape(zh(c)), f"{pc[c]['precision']:.4f}", f"{pc[c]['recall']:.4f}",
                 f"{pc[c]['f1-score']:.4f}", int(pc[c]["support"])] for c in m["class_names"]]
        worst = min(m["class_names"], key=lambda c: pc[c]["f1-score"])
        t, p, n = most_confused(m)
        confused = (f"最常見的錯誤是把<b>{escape(zh(t))}</b>判成<b>{escape(zh(p))}</b>（{n} 張）。" if n else
                    "測試集沒有任何分錯的圖片。")
        parts.append(f"""
<h3>{escape(ARCH_NAMES.get(a, a))}</h3>
{table(["類別", "Precision", "Recall", "F1", "張數"], crow, num_cols=(1, 2, 3, 4))}
<div class="note">F1 最低的類別是<b>{escape(zh(worst))}</b>（{pc[worst]['f1-score']:.4f}）。{confused}</div>
{img_tag(result_dir / f"confusion_{a}.png", f"{a} 混淆矩陣")}""")
    return "".join(parts)


def section_errors(models, result_dir):
    archs = [a for a in ARCH_ORDER if a in models]
    if not archs:
        return ""
    parts = ["<h2>7. 錯誤分析</h2>",
             "<p>以下列出 test set 中被分錯的圖片（最多 12 張），標題是真實類別、預測類別與模型信心度。</p>"]
    for a in archs:
        parts.append(f"<h3>{escape(ARCH_NAMES.get(a, a))}</h3>"
                     f"{img_tag(result_dir / f'misclassified_{a}.png', f'{a} 分錯的圖片')}")
    parts.append('<div class="note">可從圖片觀察錯誤的共同原因，例如：腐爛面積很小、光線過暗或反光、'
                 '背景雜亂、圖片中有多顆水果，或者原本標註就有爭議。</div>')
    return "".join(parts)


def section_demo(result_dir):
    path = result_dir / "predict.png"
    if not path.exists():
        return ""
    return f"""
<h2>8. Demo：自行拍攝的照片</h2>
<p>用 <code>predict.py</code> 對資料集以外的照片做預測，標題為預測類別與信心度（綠色為新鮮、紅色為腐爛）。</p>
{img_tag(path, "Demo 預測結果")}"""


def section_conclusion(models, logs):
    if not models:
        return ""
    archs = [a for a in ARCH_ORDER if a in models]
    best = max(archs, key=lambda a: models[a]["test_acc"])
    items = [f"<li>最佳模型為 <b>{escape(ARCH_NAMES.get(best, best))}</b>，test accuracy 達 "
             f"<b>{pct(models[best]['test_acc'])}</b>；只看新鮮 / 腐爛二分類時為 {pct(fresh_rotten_acc(models[best]))}。</li>"]
    if "cnn" in models and best != "cnn":
        diff = (models[best]["test_acc"] - models["cnn"]["test_acc"]) * 100
        ratio = models[best]["params"] / models["cnn"]["params"]
        items.append(f"<li>遷移學習比自建 CNN 高出 {diff:.2f} 個百分點，但參數量約為 {ratio:.1f} 倍；"
                     "ImageNet 預訓練學到的邊緣、紋理特徵，能直接幫助辨識腐爛斑點。</li>")
    elif best == "cnn" and len(archs) > 1:
        items.append("<li>自建 CNN 的表現不輸遷移學習，而且參數量少很多，比較適合部署在效能有限的裝置上。</li>")
    t, p, n = most_confused(models[best])
    if n:
        items.append(f"<li>主要錯誤集中在{escape(zh(t))}與{escape(zh(p))}之間，未來可針對這兩類收集更多樣本。</li>")
    items.append("<li>未來改進方向：加入更多水果種類與真實場景照片（不同背景、光線）、"
                 "用 Grad-CAM 觀察模型關注的區域，或把模型輕量化後部署到手機上即時辨識。</li>")
    return f"<h2>9. 結論與未來改進</h2><ul>{''.join(items)}</ul>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="result")
    parser.add_argument("--out", type=str, default="result/report.html")
    parser.add_argument("--title", type=str, default="新鮮 / 腐爛水果影像分類")
    parser.add_argument("--author", type=str, default="", help="姓名 / 學號，會顯示在標題下方")
    args = parser.parse_args()

    rd = Path(args.result_dir)
    eda = load_json(rd / "eda_stats.json")
    models, configs, logs = {}, {}, {}
    for a in ARCH_ORDER:
        if (m := load_json(rd / f"metrics_{a}.json")):
            models[a] = m
        if (c := load_json(rd / f"config_{a}.json")):
            configs[a] = c
        if (lg := load_log(rd / f"log_{a}.csv")):
            logs[a] = lg

    sub = " · ".join(x for x in [escape(args.author), "機器學習期中專案", date.today().isoformat()] if x)
    body = "".join([
        f"<h1>{escape(args.title)}</h1><p class='sub'>{sub}</p>",
        section_overview(eda, models),
        section_data(eda, rd),
        section_preprocess(configs),
        section_models(models, configs),
        section_training(models, logs, rd),
        section_results(models, rd),
        section_errors(models, rd),
        section_demo(rd),
        section_conclusion(models, logs),
    ])
    html = (f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{escape(args.title)}</title><style>{CSS}</style></head>"
            f"<body><main>{body}</main></body></html>")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"[OK] 報告已輸出：{Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
