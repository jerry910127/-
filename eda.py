# eda.py
# 資料探索：各類別張數、每類範例圖、圖片尺寸分布
# 用法：python eda.py --data_dir <資料夾>
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets

from dataset import find_split_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="result")
    parser.add_argument("--per_class", type=int, default=5, help="每類顯示幾張範例")
    args = parser.parse_args()

    random.seed(42)
    root = find_split_root(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {s: datasets.ImageFolder(root / s) for s in ("train", "test")}
    classes = splits["train"].classes

    # ---- 1. 各類別張數 ----
    counts = {s: Counter(ds.targets) for s, ds in splits.items()}
    print(f"{'class':<15}{'train':>8}{'test':>8}")
    for i, c in enumerate(classes):
        print(f"{c:<15}{counts['train'][i]:>8}{counts['test'][i]:>8}")
    print(f"{'total':<15}{len(splits['train']):>8}{len(splits['test']):>8}")

    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.4
    xs = range(len(classes))
    ax.bar([x - w / 2 for x in xs], [counts["train"][i] for i in xs], w, label="train")
    ax.bar([x + w / 2 for x in xs], [counts["test"][i] for i in xs], w, label="test")
    ax.set_xticks(list(xs), classes, rotation=20)
    ax.set_ylabel("images")
    ax.set_title("Images per class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "eda_class_counts.png", dpi=150)
    plt.close(fig)

    # ---- 2. 每類範例圖 ----
    train = splits["train"]
    by_class = {i: [p for p, t in train.samples if t == i] for i in range(len(classes))}
    fig, axes = plt.subplots(len(classes), args.per_class,
                             figsize=(args.per_class * 2.2, len(classes) * 2.2))
    for i, c in enumerate(classes):
        picks = random.sample(by_class[i], min(args.per_class, len(by_class[i])))
        for j in range(args.per_class):
            ax = axes[i][j]
            ax.axis("off")
            if j < len(picks):
                ax.imshow(Image.open(picks[j]).convert("RGB"))
            if j == 0:
                ax.set_title(c, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_samples.png", dpi=120)
    plt.close(fig)

    # ---- 3. 圖片尺寸分布（抽樣 500 張，只讀檔頭，很快）----
    sample = random.sample(train.samples, min(500, len(train.samples)))
    sizes = [Image.open(p).size for p, _ in sample]
    ws, hs = zip(*sizes)
    print(f"\n尺寸（抽樣 {len(sizes)} 張）：寬 {min(ws)}~{max(ws)}，高 {min(hs)}~{max(hs)}")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(ws, hs, s=8, alpha=0.5)
    ax.set(xlabel="width", ylabel="height", title="Image sizes (sampled)")
    fig.tight_layout()
    fig.savefig(out_dir / "eda_sizes.png", dpi=150)
    plt.close(fig)

    # 數字存成 json，給 make_report.py 使用
    stats = {
        "classes": classes,
        "train": [counts["train"][i] for i in range(len(classes))],
        "test": [counts["test"][i] for i in range(len(classes))],
        "width": [min(ws), max(ws)],
        "height": [min(hs), max(hs)],
        "size_sample": len(sizes),
    }
    (out_dir / "eda_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] EDA 圖已輸出到 {out_dir.resolve()}")


if __name__ == "__main__":
    main()
