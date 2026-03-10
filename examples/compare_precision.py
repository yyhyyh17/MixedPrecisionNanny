"""
混合精度 vs FP32 前向传播精度对比示例

演示功能：
  1. 用 ResNet-18 在合成数据上进行精度对比分析
  2. 生成 JSON 报告文件
  3. 启动可视化 Web 服务查看结果

运行：
    cd MixedPrecisionNanny
    python examples/compare_precision.py

    # 仅生成报告（不启动服务）
    python examples/compare_precision.py --no-server

    # 使用 BF16
    python examples/compare_precision.py --precision bf16

    # 查看已有报告
    python visualization/server.py --report nanny_logs/reports/xxx.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from analyzer.precision_diff import PrecisionDiffAnalyzer
from visualization.server import launch_server


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_layer(in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))
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


def main():
    parser = argparse.ArgumentParser(description="Precision Diff Analysis Example")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--report-dir", type=str, default="nanny_logs/reports")
    parser.add_argument("--no-server", action="store_true", help="Only generate report, do not start web server")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device("cpu")

    print("=" * 60)
    print("  Precision Diff Analysis: FP32 vs Mixed Precision")
    print("=" * 60)

    model = ResNet18(num_classes=10).to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet-18 | Params: {total_params:,} | Precision: {args.precision.upper()}")

    sample_input = torch.randn(args.batch_size, 3, 32, 32, device=device)
    print(f"Input shape: {list(sample_input.shape)}")

    print("\nRunning analysis...")
    t0 = time.time()
    analyzer = PrecisionDiffAnalyzer(model, precision=args.precision, device=device)
    report = analyzer.analyze(sample_input)
    elapsed = time.time() - t0
    print(f"Analysis completed in {elapsed:.2f}s")

    print(f"\n{'─' * 50}")
    print(f"  Results Summary")
    print(f"{'─' * 50}")
    print(f"  Total layers analyzed:       {report.total_layers}")
    print(f"  Final output cosine sim:     {report.final_output_cosine_similarity:.8f}")
    print(f"  Final output max abs diff:   {report.final_output_max_abs_diff:.6e}")
    print(f"  Final output mean abs diff:  {report.final_output_mean_abs_diff:.6e}")

    if report.layers:
        worst = max(report.layers, key=lambda l: l.max_abs_diff)
        best = max(report.layers, key=lambda l: l.cosine_similarity)
        avg_cos = sum(l.cosine_similarity for l in report.layers) / len(report.layers)
        nan_layers = [l for l in report.layers if l.mp_nan_count > 0]
        print(f"\n  Avg layer cosine sim:        {avg_cos:.8f}")
        print(f"  Worst layer (max abs diff):  {worst.layer_name} = {worst.max_abs_diff:.6e}")
        print(f"  Best layer (cosine sim):     {best.layer_name} = {best.cosine_similarity:.8f}")
        if nan_layers:
            print(f"  WARNING: {len(nan_layers)} layers have NaN in mixed precision output!")

    os.makedirs(args.report_dir, exist_ok=True)
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.report_dir, f"diff_{report.model_name}_{args.precision}_{ts_str}.json")
    report.to_json(report_path)
    print(f"\n  Report saved to: {report_path}")
    print(f"{'─' * 50}")

    if not args.no_server:
        print(f"\nStarting visualization server on port {args.port}...")
        print(f"Open http://localhost:{args.port} in your browser\n")
        launch_server(report=report, report_dir=args.report_dir, port=args.port)
    else:
        print(f"\nTo visualize, run:")
        print(f"  python visualization/server.py --report {report_path}")


if __name__ == "__main__":
    main()
