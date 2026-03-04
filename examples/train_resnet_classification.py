"""
ResNet-18 分类训练示例 + MixedPrecisionNanny 监控

功能演示：
  1. 用简化版 ResNet-18 在合成 CIFAR-10 数据上训练
  2. 集成 MixedPrecisionNanny 实时监控梯度、激活值
  3. 训练结束后用 CLI 查询监控结果

运行：
    cd MixedPrecisionNanny
    python examples/train_resnet_classification.py

查询结果：
    python cli.py summary --db examples/resnet_nanny_logs/metrics.db
    python cli.py alerts  --db examples/resnet_nanny_logs/metrics.db
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nanny import MixedPrecisionNanny


# ─── ResNet Building Blocks ──────────────────────────────────────────────────


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNet18(nn.Module):
    """简化版 ResNet-18，适配 32×32 输入（CIFAR 尺寸）。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(
        in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ─── Synthetic Data ──────────────────────────────────────────────────────────


def make_synthetic_cifar(
    num_samples: int = 512, num_classes: int = 10
) -> TensorDataset:
    """生成合成 CIFAR-10 格式数据 (3×32×32 图像 + 整数标签)。"""
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(images, labels)


# ─── Training Loop ───────────────────────────────────────────────────────────


def train(
    num_epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-2,
    trace_interval: int = 10,
    output_dir: str = "examples/resnet_nanny_logs",
):
    torch.manual_seed(42)
    device = torch.device("cpu")

    print("=" * 60)
    print("  ResNet-18 Classification Training with Nanny Monitoring")
    print("=" * 60)

    model = ResNet18(num_classes=10).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    dataset = make_synthetic_cifar(num_samples=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    nanny = MixedPrecisionNanny(
        model,
        trace_interval=trace_interval,
        output_dir=output_dir,
        verbose=True,
        precision="bf16"
    )

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            with nanny.step(global_step):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            global_step += 1

        avg_loss = epoch_loss / len(dataloader)
        acc = 100.0 * correct / total
        print(
            f"Epoch [{epoch + 1}/{num_epochs}]  "
            f"Loss: {avg_loss:.4f}  Acc: {acc:.1f}%"
        )

    nanny.close()

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Monitoring data saved to: {output_dir}/metrics.db")
    print("  Query results with:")
    print(f"    python cli.py summary --db {output_dir}/metrics.db")
    print(f"    python cli.py alerts  --db {output_dir}/metrics.db")
    print("=" * 60)


if __name__ == "__main__":
    train()
