# predict.py
# Demo：對任意圖片（單張、多張或整個資料夾）做預測
# 用法：python predict.py --ckpt result/best_resnet18.pth my_apple.jpg my_banana.png
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from dataset import build_transforms
from utils import get_device, load_checkpoint

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
        else:
            files.append(p)
    return files


@torch.no_grad()
def predict(model, img: Image.Image, tfm, device):
    x = tfm(img.convert("RGB")).unsqueeze(0).to(device)   # [1, 3, H, W]
    return torch.softmax(model(x), dim=1)[0].cpu()        # [num_classes]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="圖片檔或資料夾")
    parser.add_argument("--ckpt", type=str, default="result/best_cnn.pth")
    parser.add_argument("--out", type=str, default="result/predict.png")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    device = get_device()
    model, ckpt = load_checkpoint(args.ckpt, device)
    class_names = ckpt["class_names"]
    _, eval_tfm = build_transforms(ckpt["img_size"])   # 與訓練時的驗證前處理一致

    files = collect_images(args.images)
    if not files:
        raise SystemExit("找不到圖片")

    results = []
    for f in files:
        img = Image.open(f).convert("RGB")
        prob = predict(model, img, eval_tfm, device)
        top = prob.topk(min(args.topk, len(class_names)))
        ranked = [(class_names[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())]
        results.append((f, img, ranked))
        print(f"{f.name}: " + " | ".join(f"{n} {p:.1%}" for n, p in ranked))

    # 畫成格狀圖：新鮮標綠色、腐爛標紅色
    cols = min(4, len(results))
    rows = (len(results) + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3.5, rows * 3.8))
    for k, (f, img, ranked) in enumerate(results):
        name, p = ranked[0]
        ax = fig.add_subplot(rows, cols, k + 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{f.name}\n{name} ({p:.1%})", fontsize=10,
                     color="green" if name.startswith("fresh") else "red")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"[OK] 已輸出：{args.out}")


if __name__ == "__main__":
    main()
