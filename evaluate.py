# evaluate.py
# 用法：python evaluate.py --data_dir <資料夾> --ckpt result/best_cnn.pth
# 在 test set 上評估：Accuracy、classification report、混淆矩陣、分錯的圖片
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

from dataset import get_test_loader
from model import count_params
from utils import denormalize, get_device, load_checkpoint


@torch.no_grad()
def predict_all(model, loader, device, max_wrong=12):
    """回傳 y_true、y_pred，以及前 max_wrong 張分錯的圖（只留分錯的，避免佔滿記憶體）"""
    y_true, y_pred, wrong = [], [], []
    for x, y in loader:
        prob = torch.softmax(model(x.to(device)), dim=1).cpu()
        conf, pred = prob.max(1)
        for i in (pred != y).nonzero().flatten().tolist():
            if len(wrong) < max_wrong:
                wrong.append((x[i], y[i].item(), pred[i].item(), conf[i].item()))
        y_true += y.tolist()
        y_pred += pred.tolist()
    return y_true, y_pred, wrong


def plot_misclassified(wrong, class_names, out_path):
    if not wrong:
        print("沒有分錯的圖片")
        return
    cols = 4
    rows = (len(wrong) + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 3.2, rows * 3.4))
    for k, (img, t, p, conf) in enumerate(wrong):
        ax = fig.add_subplot(rows, cols, k + 1)
        ax.imshow(denormalize(img))
        ax.axis("off")
        ax.set_title(f"true: {class_names[t]}\npred: {class_names[p]} ({conf:.2f})",
                     fontsize=9, color="red")
    fig.suptitle("Misclassified test images")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--ckpt", type=str, default="result/best_cnn.pth")
    parser.add_argument("--out_dir", type=str, default="result")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_checkpoint(args.ckpt, device)
    arch, class_names = ckpt["arch"], ckpt["class_names"]
    print(f"[INFO] loaded {args.ckpt}（arch={arch}, epoch={ckpt['epoch']}）")

    test_loader, test_classes = get_test_loader(
        args.data_dir, ckpt["img_size"], args.batch_size, args.num_workers)
    assert test_classes == class_names, "test 類別順序與訓練時不一致"

    y_true, y_pred, wrong = predict_all(model, test_loader, device)

    # ---- 指標 ----
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print(f"\nTest Accuracy: {acc:.4f}  ({len(y_true)} 張)\n")
    print(report)
    (out_dir / f"report_{arch}.txt").write_text(
        f"Test Accuracy: {acc:.4f}\n\n{report}", encoding="utf-8")

    # ---- 混淆矩陣 ----
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion Matrix ({arch}, acc={acc:.4f})")
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{arch}.png", dpi=150)
    plt.close(fig)

    plot_misclassified(wrong, class_names, out_dir / f"misclassified_{arch}.png")

    # 完整指標存成 json，給 make_report.py 使用
    report_dict = classification_report(y_true, y_pred, target_names=class_names,
                                        digits=4, output_dict=True, zero_division=0)
    metrics = {
        "arch": arch, "test_acc": acc, "n_test": len(y_true),
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "params": count_params(model), "img_size": ckpt["img_size"], "best_epoch": ckpt["epoch"],
        "class_names": class_names,
        "per_class": {c: report_dict[c] for c in class_names},
        "confusion": cm.tolist(),
    }
    (out_dir / f"metrics_{arch}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 模型比較表（每評估一個模型就新增 / 更新一列）----
    summary_path = out_dir / "summary.csv"
    rows = []
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["arch"] != arch]
    rows.append({"arch": arch, "test_acc": f"{acc:.4f}", "params": count_params(model),
                 "img_size": ckpt["img_size"], "best_epoch": ckpt["epoch"]})
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] 結果已輸出到 {out_dir.resolve()}")


if __name__ == "__main__":
    main()
