# dataset.py
# Fruits fresh and rotten for classification (Kaggle: sriramr)
# 6 類：freshapples / freshbanana / freshoranges / rottenapples / rottenbanana / rottenoranges
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ImageNet 統計值：自建 CNN 與遷移學習共用同一套正規化，方便比較
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def default_img_size(arch: str) -> int:
    # 自建 CNN 用 128 訓練較快；預訓練模型用 224（與 ImageNet 一致）
    return 128 if arch == "cnn" else 224


def find_split_root(data_dir) -> Path:
    """Kaggle 解壓後常是 dataset/dataset/train，往下找同時含 train/ 與 test/ 的資料夾"""
    data_dir = Path(data_dir)
    for train_dir in [data_dir / "train", *data_dir.rglob("train")]:
        if train_dir.is_dir() and (train_dir.parent / "test").is_dir():
            return train_dir.parent
    raise FileNotFoundError(f"在 {data_dir} 底下找不到同時包含 train/ 與 test/ 的資料夾")


def build_transforms(img_size: int):
    # 訓練集做資料增強；不調 hue，避免改掉「新鮮 vs 腐爛」最關鍵的顏色線索
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # 驗證 / 測試 / Demo 只做縮放與正規化
    eval_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tfm, eval_tfm


def _loader(ds, batch_size, shuffle, num_workers):
    # persistent_workers：每個 epoch 不重開 worker（Windows 上差很多）
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


def get_dataloaders(data_dir, img_size=128, batch_size=64, val_ratio=0.2,
                    num_workers=2, seed=42):
    """回傳 train / val / test 三個 DataLoader 與類別名稱。
    val 從 train 依類別比例（stratify）切出；test 只在最後評估時使用。"""
    root = find_split_root(data_dir)
    train_tfm, eval_tfm = build_transforms(img_size)

    # 同一份 train 資料夾建兩次：一份帶增強（訓練用）、一份不帶（驗證用）
    train_aug = datasets.ImageFolder(root / "train", transform=train_tfm)
    train_plain = datasets.ImageFolder(root / "train", transform=eval_tfm)
    test_set = datasets.ImageFolder(root / "test", transform=eval_tfm)
    if train_aug.classes != test_set.classes:
        raise ValueError(f"train/test 類別不一致：{train_aug.classes} vs {test_set.classes}")

    idx = list(range(len(train_aug)))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_ratio, stratify=train_aug.targets, random_state=seed)
    train_set = Subset(train_aug, train_idx)
    val_set = Subset(train_plain, val_idx)

    print(f"資料根目錄：{root}")
    print(f"類別：{train_aug.classes}")
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    return (_loader(train_set, batch_size, True, num_workers),
            _loader(val_set, batch_size, False, num_workers),
            _loader(test_set, batch_size, False, num_workers),
            train_aug.classes)


def get_test_loader(data_dir, img_size=128, batch_size=64, num_workers=2):
    root = find_split_root(data_dir)
    _, eval_tfm = build_transforms(img_size)
    test_set = datasets.ImageFolder(root / "test", transform=eval_tfm)
    return _loader(test_set, batch_size, False, num_workers), test_set.classes


if __name__ == "__main__":
    # 快速自測：python dataset.py <data_dir>
    import sys
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        sys.argv[1] if len(sys.argv) > 1 else "data", num_workers=0)
    x, y = next(iter(train_loader))
    print("Input batch shape:", x.shape)   # [B, 3, 128, 128]
    print("Target batch shape:", y.shape)  # [B]
