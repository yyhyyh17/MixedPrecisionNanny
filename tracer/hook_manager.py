"""
Hook 注册与管理

职责：
  - 遍历 model.named_modules()，为每个 module 注册 forward hook + full_backward_hook
  - hook 内部：先问 Sampler 是否需要 trace；若是，调用 compute_stats → check_alerts → 写库
  - 异常发生时触发 Sampler 切换到密集采样模式
  - detach() 干净移除所有 hook，不留内存泄漏

注意：
  - register_full_backward_hook 是 PyTorch ≥ 1.8 的 API，能正确处理多输入 module
  - 对 inplace 模块（如 ReLU(inplace=True)）只注册 forward hook，不注册 backward hook，
    否则 backward 时会与 autograd 冲突导致报错；该层仍可采集前向统计。
  - 为避免影响计算图，所有张量操作都用 detach()
  - hook 内任何异常都被 catch 并打印警告，不会中断训练
"""
from __future__ import annotations

import time
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn

from analyzer.numerical_checker import (
    Alert,
    AlertConfig,
    TensorStats,
    check_alerts,
    compute_stats,
)
from storage.sqlite_writer import SQLiteWriter
from tracer.sampler import Sampler


class HookManager:
    """
    在 model 的每个子 module 上注册 forward / backward hook。

    Args:
        model:          被监控的 nn.Module
        sampler:        Sampler 实例（控制哪些 step 需要 trace）
        writer:         SQLiteWriter 实例（负责异步写盘）
        alert_config:   告警阈值配置
        on_alert:       alert 回调，签名 (alert: Alert, step: int) -> None
                        通常用于触发 sampler.trigger_dense()
        exclude_prefixes: 跳过名称以这些前缀开头的层（如 ["loss", "criterion"]）
        max_depth:      只监控层级深度 ≤ max_depth 的 module（None = 不限）
                        深度定义：名称中 "." 的数量，例如 "encoder.layer.0" 深度为 2
        layer_sample_n: 采集时只统计 1/n 的层（按 layer_name 哈希取模）；1 表示全部层，2 表示约一半
    """

    def __init__(
        self,
        model: nn.Module,
        sampler: Sampler,
        writer: SQLiteWriter,
        alert_config: Optional[AlertConfig] = None,
        on_alert: Optional[Callable[[Alert, int], None]] = None,
        exclude_prefixes: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        layer_sample_n: int = 1,
    ):
        self._model = model
        self._sampler = sampler
        self._writer = writer
        self._alert_config = alert_config or AlertConfig()
        self._on_alert = on_alert
        self._exclude_prefixes: List[str] = exclude_prefixes or []
        self._max_depth = max_depth
        self._layer_sample_n = max(1, int(layer_sample_n))

        self._handles: List[torch.utils.hooks.RemovableHook] = []
        self._current_step: int = 0
        # 本 step 是否采集：在 set_step 时算一次，避免每个 hook 都调 should_trace()
        self._trace_this_step: bool = False

    # ─── 公共 API ────────────────────────────────────────────────────────────

    def attach(self) -> int:
        """
        注册所有 hook。返回注册的 module 数量。
        重复调用会追加 hook（先调用 detach() 再 attach()）。
        对 inplace 模块（如 ReLU(inplace=True)）仅注册 forward hook，避免 backward 报错。
        """
        count = 0
        for name, module in self._model.named_modules():
            # 根模块（name == ""）跳过，它的 output 和子模块重复
            if not name:
                continue
            if self._should_skip(name):
                continue

            h_fwd = module.register_forward_hook(
                self._make_forward_hook(name, type(module).__name__)
            )
            self._handles.append(h_fwd)

            # inplace 模块不注册 backward hook，否则 backward 时与 autograd 冲突会报错
            skip_bwd = _is_inplace_module(module)
            if not skip_bwd:
                try:
                    h_bwd = module.register_full_backward_hook(
                        self._make_backward_hook(name, type(module).__name__)
                    )
                    self._handles.append(h_bwd)
                except Exception as exc:
                    # 某些模块或 PyTorch 版本下 backward hook 注册/执行会失败，仅保留 forward
                    print(f"[Nanny][WARN] skip backward hook @ {name} ({type(module).__name__}): {exc}")
            count += 1
        return count

    def detach(self) -> None:
        """移除所有已注册的 hook。"""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def set_step(self, step: int) -> None:
        """由 Nanny 在每个 step 开始前调用，同步当前 step 编号并缓存本 step 是否采集。"""
        self._current_step = step
        self._trace_this_step = self._sampler.should_trace()

    # ─── Hook 工厂 ───────────────────────────────────────────────────────────

    def _make_forward_hook(self, layer_name: str, layer_type: str):
        def hook(module: nn.Module, inputs: tuple, output: Any) -> None:
            if not self._trace_this_step:
                return
            tensor = _extract_first_tensor(output)
            if tensor is None:
                return
            try:
                self._process(tensor, layer_name, layer_type, "forward")
            except Exception as exc:
                print(f"[Nanny][WARN] forward hook error @ {layer_name}: {exc}")
        return hook

    def _make_backward_hook(self, layer_name: str, layer_type: str):
        def hook(module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
            if not self._trace_this_step:
                return
            # grad_output：本 module 输出端流入的梯度（来自上游）
            # 取第一个非 None 的梯度张量
            for grad in grad_output:
                if grad is None:
                    continue
                tensor = _extract_first_tensor(grad)
                if tensor is None:
                    continue
                try:
                    self._process(tensor, layer_name, layer_type, "backward")
                except Exception as exc:
                    print(f"[Nanny][WARN] backward hook error @ {layer_name}: {exc}")
                break  # 每层只处理第一个 grad_output，避免重复统计
        return hook

    # ─── 核心处理 ────────────────────────────────────────────────────────────

    def _process(
        self,
        tensor: torch.Tensor,
        layer_name: str,
        layer_type: str,
        phase: str,
    ) -> None:
        # 层级采样：只统计 1/layer_sample_n 的层，降低 trace 时的 CPU/IO
        if self._layer_sample_n > 1 and (hash(layer_name) % self._layer_sample_n) != 0:
            return
        stats = compute_stats(
            tensor,
            precision=self._alert_config.precision,
            fast_stats=getattr(self._alert_config, "fast_stats", False),
        )
        if stats is None:
            return

        ts = time.time()
        self._writer.write_stats(
            step=self._current_step,
            phase=phase,
            layer_name=layer_name,
            layer_type=layer_type,
            stats=stats,
            ts=ts,
        )

        alerts = check_alerts(stats, layer_name, phase, self._alert_config)
        for alert in alerts:
            self._writer.write_alert(
                step=self._current_step,
                phase=phase,
                layer_name=layer_name,
                alert_type=alert.alert_type,
                severity=alert.severity,
                message=alert.message,
                value=alert.value,
                ts=ts,
            )
            if self._on_alert is not None:
                try:
                    self._on_alert(alert, self._current_step)
                except Exception:
                    pass

    # ─── 过滤逻辑 ────────────────────────────────────────────────────────────

    def _should_skip(self, name: str) -> bool:
        # 用户指定的前缀排除
        for prefix in self._exclude_prefixes:
            if name.startswith(prefix):
                return True
        # 深度过滤：name 中 "." 的数量等于深度
        if self._max_depth is not None:
            depth = name.count(".")
            if depth > self._max_depth:
                return True
        return False


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def _is_inplace_module(module: nn.Module) -> bool:
    """
    判断是否为 inplace 模块（如 ReLU(inplace=True)）。
    此类模块注册 full_backward_hook 会在 backward 时与 autograd 冲突报错，应只挂 forward hook。
    """
    return getattr(module, "inplace", False) is True


def _extract_first_tensor(obj: Any) -> Optional[torch.Tensor]:
    """
    从任意 module output 中提取第一个 Tensor。
    处理常见情形：Tensor、(Tensor, ...)、[(Tensor, ...), ...]、dict 等。
    """
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            t = _extract_first_tensor(item)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for v in obj.values():
            t = _extract_first_tensor(v)
            if t is not None:
                return t
    return None
