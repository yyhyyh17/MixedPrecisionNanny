"""
数值统计分析与告警规则

设计原则：
  - 所有张量运算在原始设备（GPU）上完成，只传标量到 CPU
  - 对大张量随机采样后再计算分位数，避免 O(n log n) 排序开销
  - 统计量与告警规则解耦，方便独立扩展
"""
from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

# FP16 数值边界
FP16_MAX: float = 65504.0
FP16_MIN_NORMAL: float = 6.1e-5   # 最小正规化 FP16

# BF16 数值边界
BF16_MAX: float = 3.39e38
BF16_MIN_NORMAL: float = 1.18e-38

# 计算分位数时最多采样多少个元素（防止大 tensor 拖慢训练）
_QUANTILE_SAMPLE_CAP: int = 16_384


# ─── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class TensorStats:
    dtype: str
    shape: List[int]
    numel: int
    nan_count: int
    inf_count: int
    max_val: float          # 有限值中绝对值最大值
    min_nonzero: float      # 非零有限值中绝对值最小值
    mean_val: float         # 有限值均值
    std_val: float          # 有限值标准差
    p1: float               # 有限值 1% 分位数
    p99: float              # 有限值 99% 分位数
    fp16_saturation: float  # 有限值中 |x| > 0.9 * FP16_MAX 的比例
    fp16_underflow: float   # 非零有限值中 |x| < FP16_MIN_NORMAL 的比例


@dataclass
class Alert:
    alert_type: str   # NAN | INF | OVERFLOW | UNDERFLOW | GRAD_EXPLOSION | GRAD_VANISH
    severity: str     # ERROR | WARNING
    message: str
    value: float


@dataclass
class AlertConfig:
    # 精度检测目标："fp16" 或 "bf16"（饱和/下溢阈值据此选择）
    precision: Literal["fp16", "bf16"] = "fp16"

    # FP16/BF16 饱和（溢出）— 具体阈值由 precision 决定
    saturation_warn_threshold: float = 0.01    # > 1%  → WARNING
    saturation_error_threshold: float = 0.10   # > 10% → ERROR

    # FP16/BF16 下溢
    underflow_warn_threshold: float = 0.05     # > 5% 非零值下溢 → WARNING

    # 梯度范数（backward 阶段）
    grad_explosion_threshold: float = 1e4      # max(|grad|) 超此值 → ERROR
    grad_vanish_threshold: float = 1e-8        # max(|grad|) 低于此值（非零）→ WARNING

    # NaN / Inf 容忍数（通常 0，即任意一个都告警）
    nan_tolerance: int = 0
    inf_tolerance: int = 0


_DEFAULT_CONFIG = AlertConfig()


def _precision_bounds(precision: str) -> Tuple[float, float]:
    """返回 (max_val, min_normal) 用于饱和/下溢计算。"""
    if precision == "bf16":
        return BF16_MAX, BF16_MIN_NORMAL
    return FP16_MAX, FP16_MIN_NORMAL


# ─── 核心：计算统计量 ───────────────────────────────────────────────────────────

@torch.no_grad()
def compute_stats(tensor: torch.Tensor, precision: str = "fp16") -> Optional[TensorStats]:
    """
    计算张量的数值统计量。
    - 始终在张量所在设备上运算（GPU 友好）
    - 大张量分位数计算使用随机采样
    - precision: "fp16" 或 "bf16"，决定饱和/下溢使用的数值边界
    - 返回 None 表示张量为空
    """
    if tensor.numel() == 0:
        return None

    # 转 float32 保证统计精度，detach 避免影响计算图
    t = tensor.detach().float()

    nan_mask = torch.isnan(t)
    inf_mask = torch.isinf(t)
    nan_count = int(nan_mask.sum().item())
    inf_count = int(inf_mask.sum().item())

    # 只对有限值做统计
    finite_vals = t[~nan_mask & ~inf_mask]   # 1-D, on original device
    n_finite = finite_vals.numel()

    if n_finite == 0:
        return TensorStats(
            dtype=str(tensor.dtype),
            shape=list(tensor.shape),
            numel=tensor.numel(),
            nan_count=nan_count,
            inf_count=inf_count,
            max_val=float("nan"),
            min_nonzero=float("nan"),
            mean_val=float("nan"),
            std_val=float("nan"),
            p1=float("nan"),
            p99=float("nan"),
            fp16_saturation=0.0,
            fp16_underflow=0.0,
        )

    abs_finite = finite_vals.abs()
    prec_max, prec_min_normal = _precision_bounds(precision)

    max_val = float(abs_finite.max().item())
    mean_val = float(finite_vals.mean().item())
    std_val = float(finite_vals.std().item()) if n_finite > 1 else 0.0

    # 非零最小值
    nonzero_abs = abs_finite[abs_finite > 0]
    min_nonzero = float(nonzero_abs.min().item()) if nonzero_abs.numel() > 0 else 0.0

    # 分位数：随机采样后排序，避免全量 sort 开销
    sample = finite_vals
    if sample.numel() > _QUANTILE_SAMPLE_CAP:
        perm = torch.randperm(sample.numel(), device=sample.device)[:_QUANTILE_SAMPLE_CAP]
        sample = sample[perm]
    sorted_sample, _ = sample.sort()
    n = sorted_sample.numel()
    p1 = float(sorted_sample[max(0, int(n * 0.01))].item())
    p99 = float(sorted_sample[min(n - 1, int(n * 0.99))].item())

    # 饱和率：有限值中绝对值超过 0.9 * prec_max 的比例
    fp16_saturation = float((abs_finite > prec_max * 0.9).float().mean().item())

    # 下溢率：非零有限值中绝对值小于 prec_min_normal 的比例
    if nonzero_abs.numel() > 0:
        fp16_underflow = float((nonzero_abs < prec_min_normal).float().mean().item())
    else:
        fp16_underflow = 0.0

    return TensorStats(
        dtype=str(tensor.dtype),
        shape=list(tensor.shape),
        numel=tensor.numel(),
        nan_count=nan_count,
        inf_count=inf_count,
        max_val=max_val,
        min_nonzero=min_nonzero,
        mean_val=mean_val,
        std_val=std_val,
        p1=p1,
        p99=p99,
        fp16_saturation=fp16_saturation,
        fp16_underflow=fp16_underflow,
    )


# ─── 核心：生成告警 ────────────────────────────────────────────────────────────

def check_alerts(
    stats: TensorStats,
    layer_name: str,
    phase: str,
    config: AlertConfig = _DEFAULT_CONFIG,
) -> List[Alert]:
    """
    根据统计量和配置生成告警列表。

    Args:
        stats:      由 compute_stats() 返回的统计量
        layer_name: 层名称（用于消息拼接）
        phase:      "forward" 或 "backward"
        config:     告警阈值配置
    """
    alerts: List[Alert] = []

    # ── NaN ──────────────────────────────────────────────────────────────────
    if stats.nan_count > config.nan_tolerance:
        ratio = stats.nan_count / stats.numel
        alerts.append(Alert(
            alert_type="NAN",
            severity="ERROR",
            message=f"[{phase}] {layer_name}: {stats.nan_count} NaN ({ratio:.1%} of {stats.numel} values)",
            value=float(stats.nan_count),
        ))

    # ── Inf ──────────────────────────────────────────────────────────────────
    if stats.inf_count > config.inf_tolerance:
        ratio = stats.inf_count / stats.numel
        alerts.append(Alert(
            alert_type="INF",
            severity="ERROR",
            message=f"[{phase}] {layer_name}: {stats.inf_count} Inf ({ratio:.1%} of {stats.numel} values)",
            value=float(stats.inf_count),
        ))

    # ── FP16/BF16 饱和 / 溢出 ──────────────────────────────────────────────────────
    prec_label = config.precision.upper()
    sat = stats.fp16_saturation
    if sat >= config.saturation_error_threshold:
        alerts.append(Alert(
            alert_type="OVERFLOW",
            severity="ERROR",
            message=(
                f"[{phase}] {layer_name}: {prec_label} saturation {sat:.1%} "
                f"(threshold ERROR={config.saturation_error_threshold:.0%})"
            ),
            value=sat,
        ))
    elif sat >= config.saturation_warn_threshold:
        alerts.append(Alert(
            alert_type="OVERFLOW",
            severity="WARNING",
            message=(
                f"[{phase}] {layer_name}: {prec_label} saturation {sat:.1%} "
                f"(threshold WARN={config.saturation_warn_threshold:.0%})"
            ),
            value=sat,
        ))

    # ── FP16/BF16 下溢 ─────────────────────────────────────────────────────────────
    udf = stats.fp16_underflow
    if udf >= config.underflow_warn_threshold:
        alerts.append(Alert(
            alert_type="UNDERFLOW",
            severity="WARNING",
            message=(
                f"[{phase}] {layer_name}: {prec_label} underflow {udf:.1%} of nonzero values "
                f"(threshold={config.underflow_warn_threshold:.0%})"
            ),
            value=udf,
        ))

    # ── 梯度爆炸 / 消失（仅 backward 阶段） ────────────────────────────────────
    if phase == "backward":
        max_v = stats.max_val
        if max_v > config.grad_explosion_threshold:
            alerts.append(Alert(
                alert_type="GRAD_EXPLOSION",
                severity="ERROR",
                message=(
                    f"[backward] {layer_name}: grad max={max_v:.3e} "
                    f"(threshold={config.grad_explosion_threshold:.0e})"
                ),
                value=max_v,
            ))
        elif 0.0 < max_v < config.grad_vanish_threshold:
            alerts.append(Alert(
                alert_type="GRAD_VANISH",
                severity="WARNING",
                message=(
                    f"[backward] {layer_name}: grad max={max_v:.3e} "
                    f"(threshold={config.grad_vanish_threshold:.0e})"
                ),
                value=max_v,
            ))

    return alerts
