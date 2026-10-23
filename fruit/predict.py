"""現場展示：載入最佳模型，預測一張或多張水果照片（也可給資料夾），並輸出 Demo 結果圖。"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cnn import CNN

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def collect(paths):
    files = []
    for p in paths:
        files += sorted(f for f in p.rglob('*') if f.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]
    return files


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('images', type=Path, nargs='+', help='照片檔或資料夾')
    parser.add_argument('--checkpoint', type=Path, default=root/'result/best_cnn.pth')
    parser.add_argument('--out', type=Path, default=root/'result/demo_predictions.png',
                        help='Demo 結果圖；同名 .json 存逐張預測，make_report.py 會放進報告')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    classes = checkpoint['classes']
    model = CNN(len(classes))
    model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    norm = checkpoint['normalization']

    results = []
    for path in collect(args.images):
        with Image.open(path) as im:
            img = ImageOps.exif_transpose(im).convert('RGB')
        x = np.asarray(img.resize((checkpoint['image_size'],)*2)).copy()
        x = torch.from_numpy(x).permute(2, 0, 1).float()/255
        with torch.no_grad():
            probs = model(((x-norm['mean'])/norm['std']).unsqueeze(0)).softmax(1)[0]
        order = probs.argsort(descending=True).tolist()
        print(f'== {path.name}')
        for i in order:
            print(f"   {classes[i]}: {probs[i].item():.2%}")
        results.append({'file': path.name, 'img': img,
                        'top3': [[classes[i], round(probs[i].item(), 4)] for i in order[:3]]})
    if not results:
        raise SystemExit('找不到照片')

    # Demo 結果圖：預測為新鮮標綠色、腐爛標紅色
    cols = min(4, len(results)); rows = (len(results)+cols-1)//cols
    fig = plt.figure(figsize=(cols*3, rows*3.3))
    for k, r in enumerate(results):
        name, prob = r['top3'][0]
        ax = fig.add_subplot(rows, cols, k+1); ax.axis('off')
        ax.imshow(r['img'])
        ax.set_title(f"{r['file']}\n{name} ({prob:.1%})", fontsize=9,
                     color='#2f7d4f' if name.startswith('fresh') else '#b83232')
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130); plt.close(fig)
    args.out.with_suffix('.json').write_text(json.dumps(
        [{k: v for k, v in r.items() if k != 'img'} for r in results], ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Demo 結果圖：{args.out}')


if __name__ == '__main__':
    main()
