# train.py
# 用法：
#   python train.py --data_dir <資料夾> --arch cnn
#   python train.py --data_dir <資料夾> --arch resnet18
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import default_img_size, get_dataloaders
from model import ARCHS, build_model, count_params
from utils import get_device, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    """跑一個 epoch；有給 optimizer 就是訓練，沒給就是驗證。回傳 (loss, acc)"""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)              # [B, num_classes]，未做 softmax
            loss = criterion(logits, labels)    # CrossEntropyLoss 內含 softmax

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            correct += (logits.argmax(1) == labels).sum().item()
            n += bs
    return total_loss / n, correct / n


def plot_curves(log_rows, out_path, title):
    epochs = [r["epoch"] for r in log_rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [r["train_loss"] for r in log_rows], label="train")
    ax1.plot(epochs, [r["val_loss"] for r in log_rows], label="val")
    ax1.set(xlabel="epoch", ylabel="loss", title=f"{title} - Loss")
    ax1.legend()
    ax2.plot(epochs, [r["train_acc"] for r in log_rows], label="train")
    ax2.plot(epochs, [r["val_acc"] for r in log_rows], label="val")
    ax2.set(xlabel="epoch", ylabel="accuracy", title=f"{title} - Accuracy")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="資料集目錄（含 train/ test/）")
    parser.add_argument("--out_dir", type=str, default="result", help="輸出目錄")
    parser.add_argument("--arch", type=str, default="cnn", choices=ARCHS)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None, help="預設：cnn 1e-3，預訓練模型 1e-4")
    parser.add_argument("--img_size", type=int, default=None, help="預設：cnn 128，預訓練模型 224")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping：val_loss 幾個 epoch 沒進步就停")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--freeze_backbone", action="store_true", help="遷移學習時只訓練分類層")
    parser.add_argument("--no_pretrained", action="store_true", help="不載入 ImageNet 權重")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    img_size = args.img_size or default_img_size(args.arch)
    lr = args.lr or (1e-3 if args.arch == "cnn" else 1e-4)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} | arch={args.arch} | img_size={img_size} | lr={lr}")

    # ---- 資料 ----
    train_loader, val_loader, _, class_names = get_dataloaders(
        args.data_dir, img_size, args.batch_size, args.val_ratio, args.num_workers, args.seed)

    # ---- 模型 / Loss / Optimizer ----
    model = build_model(args.arch, len(class_names),
                        pretrained=not args.no_pretrained,
                        freeze_backbone=args.freeze_backbone).to(device)
    print(f"參數量：{count_params(model):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # ---- 訓練迴圈 ----
    best_path = out_dir / f"best_{args.arch}.pth"
    log_path = out_dir / f"log_{args.arch}.csv"
    log_rows = []
    best_val_loss, bad_epochs = float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        sec = time.time() - t0
        log_rows.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                             train_acc=train_acc, val_acc=val_acc, sec=round(sec, 1)))
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} ({sec:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss, bad_epochs = val_loss, 0
            # 把 arch / 類別 / 尺寸一起存進去，evaluate.py 與 predict.py 直接讀取
            torch.save({
                "epoch": epoch,
                "arch": args.arch,
                "class_names": class_names,
                "img_size": img_size,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, best_path)
            print(f"  -> Saved new best to {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping：val_loss 已 {args.patience} 個 epoch 沒有進步")
                break

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    plot_curves(log_rows, out_dir / f"curve_{args.arch}.png", args.arch)
    # 訓練設定也存下來，報告會用到
    config = {**vars(args), "img_size": img_size, "lr": lr, "device": str(device),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "torch": torch.__version__}
    (out_dir / f"config_{args.arch}.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"訓練完成。最佳 val_loss={best_val_loss:.4f}，紀錄：{log_path}")


if __name__ == "__main__":
    main()
