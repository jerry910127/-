"""資料探索（EDA）：各類別張數、每類範例原圖、原始尺寸分布；train.py 會自動呼叫。"""
import io
import json
import random
import statistics
import zipfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from dataset import CLASSES


def plot_eda(archive, out, metadata, per_class=5, size_sample=100, seed=42):
    out = Path(out)
    rng = random.Random(seed)
    counts = metadata['counts']

    # 1. 各類別在 train / val / test 的張數
    fig, ax = plt.subplots(figsize=(9, 4))
    w = .27
    for k, split in enumerate(('train', 'val', 'test')):
        ax.bar([i + (k-1)*w for i in range(len(CLASSES))], [counts[split][c] for c in CLASSES], w, label=split)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=20)
    ax.set(ylabel='images', title='Images per class')
    ax.legend()
    fig.tight_layout(); fig.savefig(out/'eda_class_counts.png', dpi=150); plt.close(fig)

    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.startswith('dataset/train/')
                 and Path(n).suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        by_class = {c: [n for n in names if n.split('/')[2] == c] for c in CLASSES}

        def load(name):
            with Image.open(io.BytesIO(z.read(name))) as im:
                return ImageOps.exif_transpose(im).convert('RGB')

        # 2. 每類隨機範例（原始解析度，看得出腐爛斑點）
        fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class*2, len(CLASSES)*2))
        for i, c in enumerate(CLASSES):
            picks = rng.sample(by_class[c], min(per_class, len(by_class[c])))
            for j in range(per_class):
                ax = axes[i][j]; ax.axis('off')
                if j < len(picks):
                    ax.imshow(load(picks[j]))
                if j == 0:
                    ax.set_title(c, fontsize=9, loc='left')
        fig.tight_layout(); fig.savefig(out/'eda_samples.png', dpi=110); plt.close(fig)

        # 3. 原始尺寸分布（每類抽樣，只讀檔頭）
        sizes = []
        for c in CLASSES:
            for n in rng.sample(by_class[c], min(size_sample, len(by_class[c]))):
                with Image.open(io.BytesIO(z.read(n))) as im:
                    sizes.append(im.size)
    ws, hs = zip(*sizes)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(ws, hs, s=8, alpha=.4)
    ax.set(xlabel='width (px)', ylabel='height (px)', title='Original image sizes (sampled)')
    fig.tight_layout(); fig.savefig(out/'eda_sizes.png', dpi=150); plt.close(fig)

    train = [counts['train'][c] for c in CLASSES]
    stats = {'size_sample': len(sizes),
             'width': [min(ws), statistics.median(ws), max(ws)],
             'height': [min(hs), statistics.median(hs), max(hs)],
             'imbalance_ratio': max(train)/min(train),
             'largest_class': CLASSES[train.index(max(train))],
             'smallest_class': CLASSES[train.index(min(train))]}
    (out/'eda_stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    return stats


if __name__ == '__main__':
    # 單獨重畫：python fruit/eda.py（需要 fruit.zip 與先前訓練產生的 data_audit.json）
    root = Path(__file__).resolve().parent
    meta = json.loads((root/'result/data_audit.json').read_text(encoding='utf-8'))
    print(plot_eda(root/'fruit.zip', root/'result', meta))
