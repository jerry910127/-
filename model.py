# model.py
# 模型 A：自建 CNN（baseline）
# 模型 B：遷移學習（ResNet18 / MobileNetV2，ImageNet 預訓練權重）
import torch
import torch.nn as nn
from torchvision import models

ARCHS = ["cnn", "resnet18", "mobilenet_v2"]


def conv_block(cin, cout):
    # Conv-BN-ReLU ×2 → MaxPool，尺寸減半
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class FruitCNN(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        # 128x128 → 64 → 32 → 16 → 8
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        # Global Average Pooling：每個 channel 取平均 → 256 維，參數少、也不受輸入尺寸限制
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(arch: str, num_classes: int, pretrained: bool = True,
                freeze_backbone: bool = False) -> nn.Module:
    if arch == "cnn":
        return FruitCNN(num_classes)

    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        head_name = "fc"
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        head_name = "classifier"
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    else:
        raise ValueError(f"未知的 arch：{arch}，可選 {ARCHS}")

    # 凍結 backbone：只訓練最後的分類層（feature extraction）
    if freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(head_name)
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # 快速自測
    m = FruitCNN()
    y = m(torch.randn(8, 3, 128, 128))
    print(y.shape)  # [8, 6]
    print(f"FruitCNN 參數量：{count_params(m):,}")
