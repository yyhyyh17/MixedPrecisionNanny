"""
tests/test_hook_manager.py — HookManager 单元测试

覆盖：
  - attach() 返回正确 module 数量
  - forward hook 在 trace step 捕获数据、非 trace step 跳过
  - backward hook 捕获梯度
  - on_alert 回调在异常时被调用
  - detach() 干净移除 hook（之后不再写入）
  - exclude_prefixes 过滤层
  - max_depth 深度过滤
  - _extract_first_tensor 对各种 output 类型的处理
  - hook 内部异常不中断训练
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from analyzer.numerical_checker import Alert, AlertConfig
from storage.sqlite_writer import SQLiteWriter
from tracer.hook_manager import HookManager, _extract_first_tensor, _is_inplace_module
from tracer.sampler import Sampler
from tests.conftest import db_count, db_query


# ─── 工具：构造一个简单嵌套模型 ──────────────────────────────────────────────────

class TwoLayerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class TwoLayerMLPInplaceReLU(nn.Module):
    """使用 ReLU(inplace=True)，用于测试 inplace 模块仅挂 forward hook。"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class DeeplyNested(nn.Module):
    """encoder.layer.0.linear — depth=3"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleDict({
            "layer": nn.ModuleList([
                nn.Sequential(nn.Linear(4, 4))
            ])
        })

    def forward(self, x):
        return self.encoder["layer"][0](x)


# ─── 工具：构造每次 forward 输出包含 NaN 的模型 ──────────────────────────────────

class _NaNOp(nn.Module):
    """将输入乘以 NaN，使输出全为 NaN。封装为 module 以便 hook 捕获。"""
    def forward(self, x):
        return x * float("nan")


class NaNOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.nan_op = _NaNOp()

    def forward(self, x):
        return self.nan_op(x)


# ─── fixture：组装好的 HookManager + SQLiteWriter ────────────────────────────────

def make_hook_manager(model, tmp_path, sampler=None, alert_config=None,
                      on_alert=None, exclude_prefixes=None, max_depth=None):
    db_path = str(tmp_path / "metrics.db")
    writer = SQLiteWriter(db_path)
    sampler = sampler or Sampler(trace_interval=1)  # 默认每步都 trace
    hm = HookManager(
        model=model,
        sampler=sampler,
        writer=writer,
        alert_config=alert_config,
        on_alert=on_alert,
        exclude_prefixes=exclude_prefixes,
        max_depth=max_depth,
    )
    return hm, writer, db_path


# ════════════════════════════════════════════════════════════════════════════════
# attach / detach
# ════════════════════════════════════════════════════════════════════════════════

class TestAttachDetach:

    def test_attach_returns_module_count(self, tmp_path):
        model = TwoLayerMLP()
        hm, writer, _ = make_hook_manager(model, tmp_path)
        n = hm.attach()
        # TwoLayerMLP 有 fc1, relu, fc2 三个子模块
        assert n == 3
        writer.close()

    def test_attach_registers_handles(self, tmp_path):
        model = TwoLayerMLP()
        hm, writer, _ = make_hook_manager(model, tmp_path)
        hm.attach()
        # 每个 module 注册 2 个 handle（fwd + bwd）
        assert len(hm._handles) == 6
        writer.close()

    def test_inplace_module_only_forward_hook(self, tmp_path):
        """ReLU(inplace=True) 只注册 forward hook，backward 不注册，训练可正常 backward。"""
        model = TwoLayerMLPInplaceReLU()
        hm, writer, db_path = make_hook_manager(model, tmp_path, sampler=Sampler(trace_interval=1))
        hm.attach()
        # 3 层：fc1(fwd+bwd), relu(仅 fwd), fc2(fwd+bwd) → 5 handles
        assert len(hm._handles) == 5
        hm.set_step(0)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()  # 不应报错
        writer.flush()
        # forward 应有 3 条，backward 只有 2 条（relu inplace 无 backward hook）
        assert db_count(db_path, "layer_stats", "phase='forward'") == 3
        assert db_count(db_path, "layer_stats", "phase='backward'") == 2
        writer.close()

    def test_detach_clears_handles(self, tmp_path):
        model = TwoLayerMLP()
        hm, writer, _ = make_hook_manager(model, tmp_path)
        hm.attach()
        hm.detach()
        assert len(hm._handles) == 0
        writer.close()

    def test_detach_stops_hook_from_firing(self, tmp_path):
        """detach 后 forward 不再产生 stats 记录。"""
        model = TwoLayerMLP()
        sampler = Sampler(trace_interval=1)
        hm, writer, db_path = make_hook_manager(model, tmp_path, sampler=sampler)
        hm.attach()
        hm.detach()

        sampler.advance(0)
        hm.set_step(0)
        model(torch.randn(2, 4))
        writer.flush()

        assert db_count(db_path, "layer_stats") == 0
        writer.close()

    def test_attach_idempotent_warning(self, tmp_path):
        """重复 attach 会追加 hook（不抛异常，但 handles 数量翻倍）。"""
        model = TwoLayerMLP()
        hm, writer, _ = make_hook_manager(model, tmp_path)
        hm.attach()
        n1 = len(hm._handles)
        hm.attach()
        n2 = len(hm._handles)
        assert n2 == n1 * 2  # 双倍，符合文档注释
        writer.close()


# ════════════════════════════════════════════════════════════════════════════════
# 前向 hook
# ════════════════════════════════════════════════════════════════════════════════

class TestForwardHook:

    def test_forward_hook_records_stats_on_trace_step(self, tmp_path):
        model = TwoLayerMLP()
        sampler = Sampler(trace_interval=1)
        hm, writer, db_path = make_hook_manager(model, tmp_path, sampler=sampler)
        hm.attach()

        sampler.advance(0)
        hm.set_step(0)
        model(torch.randn(2, 4))
        writer.flush()

        # 3 个子模块，每个都应写入 forward stats
        assert db_count(db_path, "layer_stats", "phase='forward'") == 3
        writer.close()

    def test_forward_hook_skips_on_non_trace_step(self, tmp_path):
        """trace_interval=100，step=1 不是 trace step，不写入。"""
        model = TwoLayerMLP()
        sampler = Sampler(trace_interval=100)
        hm, writer, db_path = make_hook_manager(model, tmp_path, sampler=sampler)
        hm.attach()

        sampler.advance(1)   # step=1，不是 100 的倍数
        hm.set_step(1)
        model(torch.randn(2, 4))
        writer.flush()

        assert db_count(db_path, "layer_stats", "phase='forward'") == 0
        writer.close()

    def test_forward_stats_contain_layer_name(self, tmp_path):
        model = TwoLayerMLP()
        sampler = Sampler(trace_interval=1)
        hm, writer, db_path = make_hook_manager(model, tmp_path, sampler=sampler)
        hm.attach()

        sampler.advance(0)
        hm.set_step(0)
        model(torch.randn(1, 4))
        writer.flush()

        rows = db_query(db_path, "SELECT layer_name FROM layer_stats")
        layer_names = {r[0] for r in rows}
        assert "fc1" in layer_names
        assert "fc2" in layer_names
        writer.close()

    def test_forward_hook_does_not_affect_output(self, tmp_path):
        """hook 不应改变模型输出值。"""
        model = TwoLayerMLP()
        hm, writer, _ = make_hook_manager(model, tmp_path)
        x = torch.randn(2, 4)
        out_before = model(x).detach().clone()
        hm.attach()
        Sampler(trace_interval=1).advance(0)
        out_after = model(x).detach()
        assert torch.allclose(out_before, out_after)
        writer.close()


# ════════════════════════════════════════════════════════════════════════════════
# 反向 hook
# ════════════════════════════════════════════════════════════════════════════════

class TestBackwardHook:

    def test_backward_hook_records_grad_stats(self, tmp_path):
        model = TwoLayerMLP()
        sampler = Sampler(trace_interval=1)
        hm, writer, db_path = make_hook_manager(model, tmp_path, sampler=sampler)
        hm.attach()

        sampler.advance(0)
        hm.set_step(0)
        loss = model(torch.randn(2, 4)).sum()
        loss.backward()
        writer.flush()

        assert db_count(db_path, "layer_stats", "phase='backward'") > 0
        writer.close()

    def test_backward_grad_dtype_recorded(self, tmp_path):
        model = TwoLayerMLP()
        sampler = Sampler(trace_interval=1)
        hm, writer, db_path = make_hook_manager(model, tmp_path, sampler=sampler)
        hm.attach()

        sampler.advance(0)
        hm.set_step(0)
        model(torch.randn(2, 4)).sum().backward()
        writer.flush()

        rows = db_query(db_path, "SELECT dtype FROM layer_stats WHERE phase='backward'")
        assert len(rows) > 0
        writer.close()


# ════════════════════════════════════════════════════════════════════════════════
# 告警 & on_alert 回调
# ════════════════════════════════════════════════════════════════════════════════

class TestAlertCallback:

    def test_on_alert_called_when_nan_detected(self, tmp_path):
        model = NaNOutputModel()
        sampler = Sampler(trace_interval=1)

        received_alerts = []
        def on_alert(alert, step):
            received_alerts.append(alert)

        hm, writer, db_path = make_hook_manager(
            model, tmp_path, sampler=sampler, on_alert=on_alert
        )
        hm.attach()
        sampler.advance(0)
        hm.set_step(0)
        model(torch.randn(4))
        writer.flush()

        assert any(a.alert_type == "NAN" for a in received_alerts)
        writer.close()

    def test_alert_written_to_db(self, tmp_path):
        model = NaNOutputModel()
        sampler = Sampler(trace_interval=1)
        hm, writer, db_path = make_hook_manager(model, tmp_path, sampler=sampler)
        hm.attach()
        sampler.advance(0)
        hm.set_step(0)
        model(torch.randn(4))
        writer.flush()

        assert db_count(db_path, "alerts", "alert_type='NAN'") > 0
        writer.close()

    def test_on_alert_receives_correct_step(self, tmp_path):
        model = NaNOutputModel()
        sampler = Sampler(trace_interval=1)

        received_steps = []
        def on_alert(alert, step):
            received_steps.append(step)

        hm, writer, _ = make_hook_manager(
            model, tmp_path, sampler=sampler, on_alert=on_alert
        )
        hm.attach()
        sampler.advance(42)
        hm.set_step(42)
        model(torch.randn(4))
        writer.close()

        assert all(s == 42 for s in received_steps)

    def test_on_alert_exception_does_not_crash(self, tmp_path):
        """on_alert 抛异常不应中断训练（hook 内部 catch 了）。"""
        model = NaNOutputModel()
        sampler = Sampler(trace_interval=1)

        def bad_callback(alert, step):
            raise RuntimeError("callback error")

        hm, writer, _ = make_hook_manager(
            model, tmp_path, sampler=sampler, on_alert=bad_callback
        )
        hm.attach()
        sampler.advance(0)
        hm.set_step(0)
        # 不应抛异常
        out = model(torch.randn(4))
        assert out is not None
        writer.close()


# ════════════════════════════════════════════════════════════════════════════════
# 过滤：exclude_prefixes / max_depth
# ════════════════════════════════════════════════════════════════════════════════

class TestFiltering:

    def test_exclude_prefixes_skips_matching_layers(self, tmp_path):
        model = TwoLayerMLP()
        sampler = Sampler(trace_interval=1)
        hm, writer, db_path = make_hook_manager(
            model, tmp_path, sampler=sampler, exclude_prefixes=["fc2"]
        )
        n = hm.attach()
        assert n == 2   # fc1, relu（fc2 被排除）
        writer.close()

    def test_exclude_prefixes_no_data_for_excluded_layer(self, tmp_path):
        model = TwoLayerMLP()
        sampler = Sampler(trace_interval=1)
        hm, writer, db_path = make_hook_manager(
            model, tmp_path, sampler=sampler, exclude_prefixes=["fc2"]
        )
        hm.attach()
        sampler.advance(0)
        hm.set_step(0)
        model(torch.randn(2, 4))
        writer.flush()

        layer_names = {r[0] for r in db_query(db_path, "SELECT layer_name FROM layer_stats")}
        assert "fc2" not in layer_names
        writer.close()

    def test_max_depth_limits_registered_modules(self, tmp_path):
        model = DeeplyNested()
        sampler = Sampler(trace_interval=1)
        # encoder 深度=0，encoder.layer 深度=1，encoder.layer.0 深度=2，
        # encoder.layer.0.0 深度=3（Sequential 内的 Linear）
        hm, writer, _ = make_hook_manager(
            model, tmp_path, sampler=sampler, max_depth=1
        )
        n = hm.attach()
        # 只有深度 ≤ 1 的层被注册：encoder(0), encoder.layer(1)
        # encoder.layer.0 深度=2 被排除
        assert n == 2
        writer.close()

    def test_exclude_all_layers_attach_returns_zero(self, tmp_path):
        model = TwoLayerMLP()
        hm, writer, _ = make_hook_manager(
            model, tmp_path, exclude_prefixes=["fc", "relu"]
        )
        n = hm.attach()
        assert n == 0
        writer.close()


# ════════════════════════════════════════════════════════════════════════════════
# _extract_first_tensor / _is_inplace_module（工具函数）
# ════════════════════════════════════════════════════════════════════════════════

class TestIsInplaceModule:

    def test_relu_inplace_true(self):
        assert _is_inplace_module(nn.ReLU(inplace=True)) is True

    def test_relu_inplace_false(self):
        assert _is_inplace_module(nn.ReLU(inplace=False)) is False

    def test_linear_has_no_inplace(self):
        assert _is_inplace_module(nn.Linear(4, 8)) is False


class TestExtractFirstTensor:

    def test_returns_tensor_directly(self):
        t = torch.randn(3)
        assert _extract_first_tensor(t) is t

    def test_returns_first_tensor_in_tuple(self):
        t1 = torch.randn(3)
        t2 = torch.randn(4)
        result = _extract_first_tensor((t1, t2))
        assert result is t1

    def test_handles_nested_tuple(self):
        t = torch.randn(3)
        result = _extract_first_tensor(((None, t), "ignored"))
        assert result is t

    def test_handles_list(self):
        t = torch.randn(2)
        result = _extract_first_tensor([None, t])
        assert result is t

    def test_handles_dict(self):
        t = torch.randn(5)
        result = _extract_first_tensor({"a": 1, "b": t})
        assert result is t

    def test_returns_none_for_none(self):
        assert _extract_first_tensor(None) is None

    def test_returns_none_for_scalar(self):
        assert _extract_first_tensor(42) is None

    def test_returns_none_for_empty_tuple(self):
        assert _extract_first_tensor(()) is None

    def test_handles_tuple_with_none_first(self):
        """第一个元素是 None，应跳过并返回第二个 tensor。"""
        t = torch.randn(3)
        result = _extract_first_tensor((None, t))
        assert result is t

    def test_mixed_types_in_list(self):
        t = torch.randn(2)
        result = _extract_first_tensor(["str", 123, t, torch.randn(4)])
        assert result is t
