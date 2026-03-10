"""
数值统计分析与告警规则

设计原则：
  - 所有张量运算在原始设备（GPU）上完成，只传标量到 CPU
  - 对大张量随机采样后再计算分位数，避免 O(n log n) 排序开销
  - 统计量与告警规则解耦，方便独立扩展

工作模式说明：
  - 训练精度由用户决定，Nanny 不强制 FP32。用户用 autocast(dtype=torch.bfloat16) 时，
    hook 收到的就是 BF16 张量。
  - FP32 训练 + 模拟：当张量为 FP32 时，fp16_underflow/fp16_saturation 表示「若转为
    FP16/BF16 会有多少比例下溢/接近上溢」。
  - BF16/FP16 直接训练：张量已是低精度时，发生下溢的值会变成 0，fp16_underflow 恒为 0。
    上溢通过 Inf 计数 + saturation 可直接检测；下溢通过 exact_zero_ratio 超过阈值间接检测
    （需设置 AlertConfig.underflow_zero_ratio_threshold，如 0.5）。
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
    exact_zero_ratio: float  # 全量元素中精确为 0 的比例（用于 BF16/FP16 直接下溢检测）


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

    # BF16/FP16 直接训练模式下的下溢检测：张量已是低精度时，下溢后的值会变成 0
    # 当 exact_zero_ratio 超过此阈值时告警（潜在下溢）。None 表示不启用（默认）
    underflow_zero_ratio_threshold: Optional[float] = 0.5

    # 梯度范数（backward 阶段）
    grad_explosion_threshold: float = 1e4      # max(|grad|) 超此值 → ERROR
    grad_vanish_threshold: float = 1e-8        # max(|grad|) 低于此值（非零）→ WARNING

    # NaN / Inf 容忍数（通常 0，即任意一个都告警）
    nan_tolerance: int = 0
    inf_tolerance: int = 0

    # 是否使用快速统计（跳过 std、分位数，降低 trace 开销）
    fast_stats: bool = False


_DEFAULT_CONFIG = AlertConfig()


def _precision_bounds(precision: str) -> Tuple[float, float]:
    """返回 (max_val, min_normal) 用于饱和/下溢计算。"""
    if precision == "bf16":
        return BF16_MAX, BF16_MIN_NORMAL
    return FP16_MAX, FP16_MIN_NORMAL


# ─── 快速 NaN/Inf 检测 ──────────────────────────────────────────────────────────

@torch.no_grad()
def detect_nan_inf_fast(tensor: torch.Tensor) -> Tuple[int, int]:
    """
    快速 NaN/Inf 检测，返回 (nan_count, inf_count)。

    先用 isfinite 做单次全量扫描（GPU 上一次 kernel launch）；
    若全部有限则直接返回 (0, 0)（仅 1 次 GPU→CPU 同步），
    仅当发现非有限值时才分别计算 NaN 和 Inf 数量。

    适用于非 trace 步的轻量级始终在线检测。
    """
    if tensor.numel() == 0:
        return 0, 0
    if torch.all(torch.isfinite(tensor)).item():
        return 0, 0
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    return nan_count, inf_count


# ─── 核心：计算统计量 ───────────────────────────────────────────────────────────

@torch.no_grad()
def compute_stats(
    tensor: torch.Tensor,
    precision: str = "fp16",
    fast_stats: bool = False,
) -> Optional[TensorStats]:
    """
    计算张量的数值统计量。
    - 始终在张量所在设备上运算（GPU 友好）
    - 大张量分位数计算使用随机采样（fast_stats=True 时跳过分位数与 std）
    - precision: "fp16" 或 "bf16"，决定饱和/下溢使用的数值边界
    - fast_stats: 为 True 时跳过 std、p1、p99，减少算力开销
    - 返回 None 表示张量为空

    优化：使用 torch.stack + tolist() 批量传输 GPU 标量，
    将 GPU→CPU 同步点从 ~11 次减少到 1-2 次。
    """
    if tensor.numel() == 0:
        return None

    t = tensor.detach().float()
    numel = tensor.numel()

    # 融合 NaN/Inf 检测：isfinite 一次覆盖两者
    finite_mask = torch.isfinite(t)
    nan_mask = torch.isnan(t)

    nan_sum = nan_mask.sum()
    inf_sum = (~finite_mask).sum() - nan_sum
    zero_sum = (t == 0).sum()

    finite_vals = t[finite_mask]
    n_finite = finite_vals.numel()

    if n_finite == 0:
        batch = torch.stack([nan_sum.float(), inf_sum.float(), zero_sum.float()]).tolist()
        return TensorStats(
            dtype=str(tensor.dtype),
            shape=list(tensor.shape),
            numel=numel,
            nan_count=int(batch[0]),
            inf_count=int(batch[1]),
            max_val=float("nan"),
            min_nonzero=float("nan"),
            mean_val=float("nan"),
            std_val=float("nan"),
            p1=float("nan"),
            p99=float("nan"),
            fp16_saturation=0.0,
            fp16_underflow=0.0,
            exact_zero_ratio=batch[2] / numel,
        )

    abs_finite = finite_vals.abs()
    prec_max, prec_min_normal = _precision_bounds(precision)

    nonzero_abs = abs_finite[abs_finite > 0]
    has_nonzero = nonzero_abs.numel() > 0

    # 在 GPU 上累积所有标量，最后一次性传输到 CPU
    gpu_scalars: list = [
        nan_sum.float(),             # [0] nan_count
        inf_sum.float(),             # [1] inf_count
        zero_sum.float(),            # [2] zero_count → exact_zero_ratio
        abs_finite.max(),            # [3] max_val
        finite_vals.mean(),          # [4] mean_val
    ]

    std_idx = -1
    if n_finite > 1 and not fast_stats:
        std_idx = len(gpu_scalars)
        gpu_scalars.append(finite_vals.std())

    min_nz_idx = -1
    udf_idx = -1
    if has_nonzero:
        min_nz_idx = len(gpu_scalars)
        gpu_scalars.append(nonzero_abs.min())
        udf_idx = len(gpu_scalars)
        gpu_scalars.append((nonzero_abs < prec_min_normal).float().mean())

    sat_idx = len(gpu_scalars)
    gpu_scalars.append((abs_finite > prec_max * 0.9).float().mean())

    # 分位数：快速模式下跳过（节省采样+排序）
    p1_idx = -1
    p99_idx = -1
    if not fast_stats:
        sample = finite_vals
        if sample.numel() > _QUANTILE_SAMPLE_CAP:
            perm = torch.randperm(sample.numel(), device=sample.device)[:_QUANTILE_SAMPLE_CAP]
            sample = sample[perm]
        sorted_sample, _ = sample.sort()
        n = sorted_sample.numel()
        p1_idx = len(gpu_scalars)
        gpu_scalars.append(sorted_sample[max(0, int(n * 0.01))])
        p99_idx = len(gpu_scalars)
        gpu_scalars.append(sorted_sample[min(n - 1, int(n * 0.99))])

    # === 单次 GPU→CPU 传输（从 ~11 次同步减少到 1 次） ===
    vals = torch.stack(gpu_scalars).tolist()

    return TensorStats(
        dtype=str(tensor.dtype),
        shape=list(tensor.shape),
        numel=numel,
        nan_count=int(vals[0]),
        inf_count=int(vals[1]),
        max_val=vals[3],
        min_nonzero=vals[min_nz_idx] if min_nz_idx >= 0 else 0.0,
        mean_val=vals[4],
        std_val=vals[std_idx] if std_idx >= 0 else 0.0,
        p1=vals[p1_idx] if p1_idx >= 0 else 0.0,
        p99=vals[p99_idx] if p99_idx >= 0 else 0.0,
        fp16_saturation=vals[sat_idx],
        fp16_underflow=vals[udf_idx] if udf_idx >= 0 else 0.0,
        exact_zero_ratio=vals[2] / numel,
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
        phase:      "forward" | "backward" | "optimizer_grad" | "optimizer_param"
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

    # ── BF16/FP16 直接训练模式：高零比例 → 潜在下溢 ─────────────────────────────────
    # 当张量已是 bfloat16/float16 时，发生下溢的值会变成 0，fp16_underflow 恒为 0
    # 通过 exact_zero_ratio 超过阈值来间接检测“大量下溢”
    thr = getattr(config, "underflow_zero_ratio_threshold", None)
    if thr is not None and hasattr(stats, "exact_zero_ratio"):
        dtype_str = stats.dtype.lower()
        if ("bfloat16" in dtype_str or "float16" in dtype_str) and stats.exact_zero_ratio >= thr:
            alerts.append(Alert(
                alert_type="UNDERFLOW",
                severity="WARNING",
                message=(
                    f"[{phase}] {layer_name}: {prec_label} direct underflow? "
                    f"exact_zero_ratio={stats.exact_zero_ratio:.1%} (threshold={thr:.0%})"
                ),
                value=stats.exact_zero_ratio,
            ))

    # ── 梯度爆炸 / 消失（backward 或 optimizer_grad 阶段） ─────────────────────────
    if phase in ("backward", "optimizer_grad"):
        max_v = stats.max_val
        phase_label = "optimizer_grad" if phase == "optimizer_grad" else "backward"
        if max_v > config.grad_explosion_threshold:
            alerts.append(Alert(
                alert_type="GRAD_EXPLOSION",
                severity="ERROR",
                message=(
                    f"[{phase_label}] {layer_name}: grad max={max_v:.3e} "
                    f"(threshold={config.grad_explosion_threshold:.0e})"
                ),
                value=max_v,
            ))
        elif 0.0 < max_v < config.grad_vanish_threshold:
            alerts.append(Alert(
                alert_type="GRAD_VANISH",
                severity="WARNING",
                message=(
                    f"[{phase_label}] {layer_name}: grad max={max_v:.3e} "
                    f"(threshold={config.grad_vanish_threshold:.0e})"
                ),
                value=max_v,
            ))

    return alerts
