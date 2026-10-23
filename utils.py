# utils.py
import random

import numpy as np
import torch

from dataset import MEAN, STD
from model import build_model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, device):
    """讀取 train.py 存的 checkpoint，回傳 (model, ckpt)。
    ckpt 內含 arch / class_names / img_size，評估與 Demo 不必再手動設定。"""
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["arch"], len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


def denormalize(img_t):
    """[3,H,W] 正規化後的張量 → [H,W,3] 0~1 的 numpy，供 matplotlib 顯示"""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (img_t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
