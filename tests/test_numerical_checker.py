"""
tests/test_numerical_checker.py — 数值统计与告警规则单元测试

覆盖：
  compute_stats()
    - 空 tensor → None
    - 正常 tensor 的统计量正确性（shape / dtype / nan / inf / max / mean / std / p1 / p99）
    - 全 NaN / 全 Inf / 混合情况
    - FP16 saturation 检测（绝对值 > 0.9 * 65504）
    - FP16 underflow 检测（非零绝对值 < 6.1e-5）
    - 大 tensor 分位数采样（不超时）
    - 负数、全零、单元素 tensor

  check_alerts()
    - 正常 tensor 无告警
    - NaN → ERROR
    - Inf → ERROR
    - FP16 saturation 1.5% → WARNING；15% → ERROR
    - FP16 underflow 6% → WARNING
    - backward 梯度爆炸 → ERROR；梯度消失 → WARNING
    - forward 不产生梯度相关告警
    - 自定义 AlertConfig 阈值生效
"""
import math
import pytest

torch = pytest.importorskip("torch")

from analyzer.numerical_checker import (
    FP16_MAX,
    FP16_MIN_NORMAL,
    Alert,
    AlertConfig,
    TensorStats,
    check_alerts,
    compute_stats,
)


# ════════════════════════════════════════════════════════════════════════════════
# compute_stats
# ════════════════════════════════════════════════════════════════════════════════

class TestComputeStats:

    # ── 基本元信息 ──────────────────────────────────────────────────────────────

    def test_empty_tensor_returns_none(self):
        t = torch.tensor([])
        assert compute_stats(t) is None

    def test_empty_2d_tensor_returns_none(self):
        t = torch.zeros(0, 128)
        assert compute_stats(t) is None

    def test_returns_tensor_stats_instance(self):
        t = torch.randn(10)
        stats = compute_stats(t)
        assert isinstance(stats, TensorStats)

    def test_dtype_preserved_float32(self):
        t = torch.randn(10, dtype=torch.float32)
        stats = compute_stats(t)
        assert "float32" in stats.dtype

    def test_dtype_preserved_float16(self):
        t = torch.randn(10, dtype=torch.float16)
        stats = compute_stats(t)
        assert "float16" in stats.dtype

    def test_shape_preserved(self):
        t = torch.randn(3, 4, 5)
        stats = compute_stats(t)
        assert stats.shape == [3, 4, 5]

    def test_numel_correct(self):
        t = torch.randn(4, 8)
        stats = compute_stats(t)
        assert stats.numel == 32

    # ── NaN / Inf 计数 ──────────────────────────────────────────────────────────

    def test_no_nan_no_inf_for_normal_tensor(self):
        t = torch.randn(100)
        stats = compute_stats(t)
        assert stats.nan_count == 0
        assert stats.inf_count == 0

    def test_nan_count_exact(self):
        t = torch.tensor([1.0, float("nan"), 2.0, float("nan")])
        stats = compute_stats(t)
        assert stats.nan_count == 2
        assert stats.inf_count == 0

    def test_inf_count_exact(self):
        t = torch.tensor([1.0, float("inf"), -float("inf"), 0.0])
        stats = compute_stats(t)
        assert stats.nan_count == 0
        assert stats.inf_count == 2

    def test_all_nan_tensor(self):
        t = torch.full((10,), float("nan"))
        stats = compute_stats(t)
        assert stats is not None
        assert stats.nan_count == 10
        assert math.isnan(stats.max_val)
        assert math.isnan(stats.mean_val)

    def test_all_inf_tensor(self):
        t = torch.full((5,), float("inf"))
        stats = compute_stats(t)
        assert stats.inf_count == 5
        assert math.isnan(stats.max_val)  # all inf → no finite values

    def test_mixed_nan_inf_and_finite(self):
        t = torch.tensor([1.0, float("nan"), float("inf"), -float("inf"), 3.0])
        stats = compute_stats(t)
        assert stats.nan_count == 1
        assert stats.inf_count == 2
        # 有限值只有 1.0 和 3.0
        assert abs(stats.max_val - 3.0) < 1e-5
        assert abs(stats.min_nonzero - 1.0) < 1e-5

    # ── 统计量数值正确性 ─────────────────────────────────────────────────────────

    def test_max_val_is_abs_max(self):
        """max_val 是有限值绝对值的最大值（包含负数情况）。"""
        t = torch.tensor([-5.0, 2.0, 3.0, -1.0])
        stats = compute_stats(t)
        assert abs(stats.max_val - 5.0) < 1e-5

    def test_min_nonzero_ignores_zeros(self):
        t = torch.tensor([0.0, 0.0, 0.1, 0.5, 1.0])
        stats = compute_stats(t)
        assert abs(stats.min_nonzero - 0.1) < 1e-5

    def test_min_nonzero_is_zero_for_all_zeros(self):
        t = torch.zeros(10)
        stats = compute_stats(t)
        assert stats.min_nonzero == 0.0

    def test_mean_val_correct(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        stats = compute_stats(t)
        assert abs(stats.mean_val - 2.5) < 1e-4

    def test_std_val_correct(self):
        t = torch.tensor([1.0, 1.0, 1.0, 1.0])
        stats = compute_stats(t)
        assert abs(stats.std_val) < 1e-6

    def test_single_element_no_crash(self):
        t = torch.tensor([42.0])
        stats = compute_stats(t)
        assert stats is not None
        assert abs(stats.max_val - 42.0) < 1e-5
        assert stats.std_val == 0.0

    def test_p1_less_than_p99(self):
        t = torch.randn(1000)
        stats = compute_stats(t)
        assert stats.p1 < stats.p99

    def test_p1_p99_within_range(self):
        """p1 和 p99 应在 tensor 的 min/max 范围内。"""
        t = torch.linspace(-10.0, 10.0, steps=500)
        stats = compute_stats(t)
        assert stats.p1 >= -10.0 - 0.1
        assert stats.p99 <= 10.0 + 0.1

    # ── FP16 Saturation ─────────────────────────────────────────────────────────

    def test_fp16_saturation_zero_for_normal_values(self):
        t = torch.randn(1000)  # 标准正态，不会超过 FP16_MAX
        stats = compute_stats(t)
        assert stats.fp16_saturation == 0.0

    def test_fp16_saturation_detected(self):
        """全部值超过 0.9 * FP16_MAX → saturation = 1.0。"""
        sat_val = FP16_MAX * 0.95
        t = torch.full((100,), sat_val)
        stats = compute_stats(t)
        assert stats.fp16_saturation == 1.0

    def test_fp16_saturation_partial(self):
        """一半值饱和 → saturation ≈ 0.5。"""
        sat = torch.full((50,), FP16_MAX * 0.95)
        normal = torch.ones(50)
        t = torch.cat([sat, normal])
        stats = compute_stats(t)
        assert abs(stats.fp16_saturation - 0.5) < 0.01

    def test_fp16_saturation_excludes_nan_from_denominator(self):
        """NaN 不应算在分母里；若有限值全饱和则 saturation=1.0。"""
        nan_part = torch.full((10,), float("nan"))
        sat_part = torch.full((10,), FP16_MAX * 0.95)
        t = torch.cat([nan_part, sat_part])
        stats = compute_stats(t)
        assert stats.fp16_saturation == 1.0

    # ── FP16 Underflow ──────────────────────────────────────────────────────────

    def test_fp16_underflow_zero_for_normal_values(self):
        t = torch.ones(100)
        stats = compute_stats(t)
        assert stats.fp16_underflow == 0.0

    def test_fp16_underflow_zero_for_all_zeros(self):
        """全零 tensor：非零值为 0 个，underflow 率应为 0。"""
        t = torch.zeros(100)
        stats = compute_stats(t)
        assert stats.fp16_underflow == 0.0

    def test_fp16_underflow_detected(self):
        """全部非零值小于 FP16_MIN_NORMAL → underflow = 1.0。"""
        tiny = FP16_MIN_NORMAL * 0.01
        t = torch.full((100,), tiny)
        stats = compute_stats(t)
        assert stats.fp16_underflow == 1.0

    def test_fp16_underflow_partial(self):
        """一半非零值下溢 → underflow ≈ 0.5。"""
        tiny = torch.full((50,), FP16_MIN_NORMAL * 0.01)
        normal = torch.ones(50)
        t = torch.cat([tiny, normal])
        stats = compute_stats(t)
        assert abs(stats.fp16_underflow - 0.5) < 0.01

    def test_exact_zero_ratio(self):
        """exact_zero_ratio 为全量元素中精确为 0 的比例。"""
        t = torch.tensor([0.0, 0.0, 1.0, 2.0, 0.0])  # 3/5 = 0.6
        stats = compute_stats(t)
        assert abs(stats.exact_zero_ratio - 0.6) < 1e-6

    def test_exact_zero_ratio_all_zeros(self):
        t = torch.zeros(10)
        stats = compute_stats(t)
        assert stats.exact_zero_ratio == 1.0

    # ── 性能：大 tensor ─────────────────────────────────────────────────────────

    def test_large_tensor_does_not_hang(self):
        """100 万元素 tensor 应在 1 秒内完成统计（采样保证）。"""
        import time
        t = torch.randn(1_000_000)
        start = time.time()
        stats = compute_stats(t)
        elapsed = time.time() - start
        assert elapsed < 1.0, f"compute_stats took {elapsed:.2f}s on 1M tensor"
        assert stats is not None

    # ── 不影响计算图 ─────────────────────────────────────────────────────────────

    def test_does_not_affect_grad(self):
        """compute_stats 不应影响 tensor 的梯度传播。"""
        t = torch.randn(10, requires_grad=True)
        y = (t * 2).sum()
        stats = compute_stats(t)
        y.backward()
        assert t.grad is not None


# ════════════════════════════════════════════════════════════════════════════════
# check_alerts
# ════════════════════════════════════════════════════════════════════════════════

def _alerts_of_type(alerts, alert_type: str):
    return [a for a in alerts if a.alert_type == alert_type]

def _alerts_of_severity(alerts, severity: str):
    return [a for a in alerts if a.severity == severity]


class TestCheckAlerts:

    def _normal_stats(self) -> TensorStats:
        """返回一个所有指标都健康的 TensorStats（直接从正常 tensor 计算）。"""
        t = torch.randn(100)
        return compute_stats(t)

    # ── 无告警 ──────────────────────────────────────────────────────────────────

    def test_no_alerts_for_normal_forward_tensor(self):
        stats = self._normal_stats()
        alerts = check_alerts(stats, "layer", "forward")
        assert alerts == []

    def test_no_alerts_for_normal_backward_tensor(self):
        """正常梯度（max_val 在合理范围）不产生告警。"""
        t = torch.randn(100) * 10  # max ~40，远低于 1e4
        stats = compute_stats(t)
        alerts = check_alerts(stats, "layer", "backward")
        assert alerts == []

    # ── NaN ─────────────────────────────────────────────────────────────────────

    def test_nan_produces_error_alert(self):
        t = torch.tensor([1.0, float("nan"), 2.0])
        stats = compute_stats(t)
        alerts = check_alerts(stats, "enc.0", "forward")
        assert len(_alerts_of_type(alerts, "NAN")) == 1
        assert _alerts_of_type(alerts, "NAN")[0].severity == "ERROR"

    def test_nan_alert_message_contains_layer_name(self):
        t = torch.tensor([float("nan")])
        stats = compute_stats(t)
        alerts = check_alerts(stats, "my_special_layer", "forward")
        assert "my_special_layer" in alerts[0].message

    def test_nan_tolerance_suppresses_alert(self):
        """nan_tolerance=5：5 个 NaN 不告警，6 个才告警。"""
        cfg = AlertConfig(nan_tolerance=5)
        t5 = torch.tensor([float("nan")] * 5 + [1.0] * 95)
        t6 = torch.tensor([float("nan")] * 6 + [1.0] * 94)
        assert check_alerts(compute_stats(t5), "l", "forward", cfg) == []
        assert len(check_alerts(compute_stats(t6), "l", "forward", cfg)) > 0

    # ── Inf ─────────────────────────────────────────────────────────────────────

    def test_inf_produces_error_alert(self):
        t = torch.tensor([1.0, float("inf"), -float("inf")])
        stats = compute_stats(t)
        alerts = check_alerts(stats, "layer", "forward")
        assert len(_alerts_of_type(alerts, "INF")) == 1
        assert _alerts_of_type(alerts, "INF")[0].severity == "ERROR"

    # ── FP16 Saturation ─────────────────────────────────────────────────────────

    def test_saturation_below_warn_threshold_no_alert(self):
        """saturation = 0.5%，低于 WARN 阈值 1%，不告警。"""
        sat_count = 5    # 5/1000 = 0.5%
        sat = torch.full((sat_count,), FP16_MAX * 0.95)
        normal = torch.ones(1000 - sat_count)
        stats = compute_stats(torch.cat([sat, normal]))
        assert _alerts_of_type(check_alerts(stats, "l", "forward"), "OVERFLOW") == []

    def test_saturation_above_warn_threshold_warning(self):
        """saturation ≈ 2%，高于 WARN(1%) 但低于 ERROR(10%)→ WARNING。"""
        sat = torch.full((20,), FP16_MAX * 0.95)
        normal = torch.ones(980)
        stats = compute_stats(torch.cat([sat, normal]))
        alerts = _alerts_of_type(check_alerts(stats, "l", "forward"), "OVERFLOW")
        assert len(alerts) == 1
        assert alerts[0].severity == "WARNING"

    def test_saturation_above_error_threshold_error(self):
        """saturation ≈ 15% → ERROR。"""
        sat = torch.full((150,), FP16_MAX * 0.95)
        normal = torch.ones(850)
        stats = compute_stats(torch.cat([sat, normal]))
        alerts = _alerts_of_type(check_alerts(stats, "l", "forward"), "OVERFLOW")
        assert len(alerts) == 1
        assert alerts[0].severity == "ERROR"

    def test_saturation_produces_at_most_one_alert(self):
        """WARNING 和 ERROR 互斥，只产生一条 OVERFLOW 告警。"""
        stats = compute_stats(torch.full((100,), FP16_MAX * 0.95))
        overflow_alerts = _alerts_of_type(check_alerts(stats, "l", "forward"), "OVERFLOW")
        assert len(overflow_alerts) == 1

    # ── FP16 Underflow ──────────────────────────────────────────────────────────

    def test_underflow_above_threshold_warning(self):
        """underflow ≈ 10% → WARNING。"""
        tiny = torch.full((100,), FP16_MIN_NORMAL * 0.01)
        normal = torch.ones(900)
        stats = compute_stats(torch.cat([tiny, normal]))
        alerts = _alerts_of_type(check_alerts(stats, "l", "forward"), "UNDERFLOW")
        assert len(alerts) == 1
        assert alerts[0].severity == "WARNING"

    def test_underflow_below_threshold_no_alert(self):
        """underflow = 1%，低于阈值 5%，不告警。"""
        tiny = torch.full((10,), FP16_MIN_NORMAL * 0.01)
        normal = torch.ones(990)
        stats = compute_stats(torch.cat([tiny, normal]))
        alerts = _alerts_of_type(check_alerts(stats, "l", "forward"), "UNDERFLOW")
        assert alerts == []

    def test_bf16_direct_underflow_alert_when_zero_ratio_high(self):
        """BF16 张量 + underflow_zero_ratio_threshold=0.5，零比例≥0.5 时产生 UNDERFLOW 告警。"""
        cfg = AlertConfig(precision="bf16", underflow_zero_ratio_threshold=0.5)
        # 构造一个 bfloat16 张量，其中 60% 为 0（模拟下溢后变零）
        t = torch.zeros(60, dtype=torch.bfloat16)
        t = torch.cat([t, torch.ones(40, dtype=torch.bfloat16)])
        stats = compute_stats(t, precision="bf16")
        assert stats.dtype == "torch.bfloat16"
        assert abs(stats.exact_zero_ratio - 0.6) < 0.01
        alerts = _alerts_of_type(check_alerts(stats, "layer", "backward", cfg), "UNDERFLOW")
        assert len(alerts) >= 1
        direct = [a for a in alerts if "direct" in a.message or "exact_zero_ratio" in a.message]
        assert len(direct) == 1
        assert direct[0].value >= 0.5

    def test_bf16_direct_underflow_disabled_when_threshold_none(self):
        """underflow_zero_ratio_threshold=None 时不产生高零比例告警。"""
        cfg = AlertConfig(precision="bf16", underflow_zero_ratio_threshold=None)
        t = torch.zeros(80, dtype=torch.bfloat16)
        t = torch.cat([t, torch.ones(20, dtype=torch.bfloat16)])
        stats = compute_stats(t, precision="bf16")
        alerts = check_alerts(stats, "layer", "backward", cfg)
        direct = [a for a in alerts if "exact_zero_ratio" in a.message]
        assert len(direct) == 0

    # ── 梯度爆炸 / 消失 ──────────────────────────────────────────────────────────

    def test_grad_explosion_in_backward(self):
        """max(|grad|) > 1e4 → GRAD_EXPLOSION ERROR（仅 backward）。"""
        t = torch.tensor([2e4, 1.0, 0.5])
        stats = compute_stats(t)
        alerts = check_alerts(stats, "layer", "backward")
        explosion = _alerts_of_type(alerts, "GRAD_EXPLOSION")
        assert len(explosion) == 1
        assert explosion[0].severity == "ERROR"

    def test_grad_explosion_not_in_forward(self):
        """forward 阶段即使数值很大，也不产生 GRAD_EXPLOSION。"""
        t = torch.tensor([2e4])
        stats = compute_stats(t)
        alerts = check_alerts(stats, "layer", "forward")
        assert _alerts_of_type(alerts, "GRAD_EXPLOSION") == []

    def test_grad_vanish_in_backward(self):
        """max(|grad|) < 1e-8 且 > 0 → GRAD_VANISH WARNING（仅 backward）。"""
        t = torch.tensor([1e-10, 5e-11])
        stats = compute_stats(t)
        alerts = check_alerts(stats, "layer", "backward")
        vanish = _alerts_of_type(alerts, "GRAD_VANISH")
        assert len(vanish) == 1
        assert vanish[0].severity == "WARNING"

    def test_grad_vanish_not_triggered_for_all_zeros(self):
        """全零梯度：max=0，0 < 0 = False → 不触发消失告警。"""
        t = torch.zeros(10)
        stats = compute_stats(t)
        alerts = check_alerts(stats, "layer", "backward")
        assert _alerts_of_type(alerts, "GRAD_VANISH") == []

    def test_grad_vanish_not_in_forward(self):
        t = torch.tensor([1e-10])
        stats = compute_stats(t)
        alerts = check_alerts(stats, "layer", "forward")
        assert _alerts_of_type(alerts, "GRAD_VANISH") == []

    # ── 自定义 AlertConfig ────────────────────────────────────────────────────────

    def test_custom_saturation_threshold(self):
        """调低 WARN 阈值到 0.1%，则 0.5% 饱和也会告警。"""
        cfg = AlertConfig(saturation_warn_threshold=0.001)
        sat = torch.full((5,), FP16_MAX * 0.95)
        normal = torch.ones(995)
        stats = compute_stats(torch.cat([sat, normal]))
        alerts = _alerts_of_type(check_alerts(stats, "l", "f", cfg), "OVERFLOW")
        assert len(alerts) == 1

    def test_custom_grad_explosion_threshold(self):
        """调高 grad explosion 阈值到 1e6，1e4 的值不再告警。"""
        cfg = AlertConfig(grad_explosion_threshold=1e6)
        t = torch.tensor([2e4])
        stats = compute_stats(t)
        assert _alerts_of_type(check_alerts(stats, "l", "backward", cfg), "GRAD_EXPLOSION") == []

    def test_alert_value_field_set_correctly(self):
        """Alert.value 应存储实际触发值（nan_count / saturation_ratio 等）。"""
        t = torch.tensor([float("nan"), float("nan"), 1.0])
        stats = compute_stats(t)
        alerts = check_alerts(stats, "l", "forward")
        nan_alert = _alerts_of_type(alerts, "NAN")[0]
        assert nan_alert.value == 2.0  # nan_count

    def test_multiple_issues_produce_multiple_alerts(self):
        """同时存在 NaN 和 FP16 saturation → 多条告警。"""
        nan_part = torch.tensor([float("nan")] * 5)
        sat_part = torch.full((200,), FP16_MAX * 0.95)
        normal = torch.ones(795)
        t = torch.cat([nan_part, sat_part, normal])
        stats = compute_stats(t)
        alerts = check_alerts(stats, "l", "forward")
        types = {a.alert_type for a in alerts}
        assert "NAN" in types
        assert "OVERFLOW" in types
