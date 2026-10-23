"""從已保存的結果重建圖表，不必重新訓練。"""
import csv
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def plot_misclassified(out, archive, n=12):
    """從 ZIP 讀原圖，畫出測試集分錯的圖片；有信心度時先列「最有把握卻答錯」的。"""
    out = Path(out)
    with (out/'test_predictions.csv').open(encoding='utf-8-sig') as f:
        wrong = [r for r in csv.DictReader(f) if r['actual'] != r['predicted']]
    if not wrong:
        return 0
    if 'confidence' in wrong[0]:
        wrong.sort(key=lambda r: -float(r['confidence']))
    show = wrong[:n]
    cols = 4; rows = (len(show)+cols-1)//cols
    fig = plt.figure(figsize=(cols*3, rows*3.3))
    with zipfile.ZipFile(archive) as z:
        for k, r in enumerate(show):
            ax = fig.add_subplot(rows, cols, k+1); ax.axis('off')
            with Image.open(io.BytesIO(z.read(r['file']))) as im:
                ax.imshow(ImageOps.exif_transpose(im).convert('RGB'))
            conf = f" ({float(r['confidence']):.2f})" if 'confidence' in r else ''
            ax.set_title(f"true: {r['actual']}\npred: {r['predicted']}{conf}", fontsize=9, color='#b83232')
    fig.suptitle(f'Misclassified test images ({len(show)} of {len(wrong)})')
    fig.tight_layout(); fig.savefig(out/'misclassified.png', dpi=130); plt.close(fig)
    return len(wrong)


def plot_results(out, archive=None):
    out = Path(out)
    r = json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    with (out/'loss.csv').open(encoding='utf-8') as f:
        rows = list(csv.reader(f))[1:]
    h = np.asarray(rows, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, cols, title in zip(axes, ((1, 2), (3, 4)), ('Cross entropy', 'Accuracy')):
        for col, label in zip(cols, ('train', 'validation')):
            ax.plot(h[:, 0], h[:, col], label=label)
        ax.set(xlabel='Epoch', title=title); ax.legend()
    fig.tight_layout(); fig.savefig(out/'learning_curves.png', dpi=160); plt.close(fig)
    cm = np.asarray(r['confusion_matrix'])
    classes = [x['class'] for x in r['per_class']]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap='Blues')
    ax.set(xticks=range(6), yticks=range(6), xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='Actual', title='Test confusion matrix')
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
    for i in range(6):
        for j in range(6): ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
    fig.tight_layout(); fig.savefig(out/'confusion_matrix.png', dpi=150); plt.close(fig)
    if archive is not None and Path(archive).exists():
        plot_misclassified(out, archive)


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    plot_results(root/'result', root/'fruit.zip')
