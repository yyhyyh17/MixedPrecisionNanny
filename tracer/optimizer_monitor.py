"""
Optimizer 层面监控：在 optimizer.step() 前对梯度和参数做数值统计与告警

职责：
  - 包装 optimizer.step()，在真正执行 step 前遍历所有 param_groups
  - 对每个 param：若有 param.grad，对 grad 做 compute_stats + check_alerts(phase=optimizer_grad)
  - 对每个 param：对 param.data 做 compute_stats + check_alerts(phase=optimizer_param)
  - 与 HookManager 共用同一套 TensorStats / AlertConfig / SQLiteWriter，phase 区分来源
  - 使用 model.named_parameters() 得到 param -> name 映射，便于日志和 DB 中的 layer_name

适用场景：
  - 混合精度下关注「更新前」的梯度与参数数值（饱和、下溢、NaN/Inf、梯度爆炸/消失）
  - 不需要逐层 hook 时，可仅用本监控降低开销
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from analyzer.numerical_checker import (
    Alert,
    AlertConfig,
    check_alerts,
    compute_stats,
)
from storage.sqlite_writer import SQLiteWriter
from tracer.sampler import Sampler


def _param_name_map(model: nn.Module) -> Dict[int, str]:
    """从 model.named_parameters() 得到 id(param) -> name 的映射。"""
    return {id(p): name for name, p in model.named_parameters()}


class OptimizerMonitor:
    """
    在 optimizer.step() 执行前，对当前梯度和参数做数值统计与告警。

    Args:
        model:           与 optimizer 对应的 nn.Module（用于解析参数名）
        optimizer:       被包装的 torch.optim.Optimizer
        sampler:         Sampler 实例
        writer:          SQLiteWriter 实例
        alert_config:    告警阈值配置
        on_alert:        alert 回调 (alert, step) -> None
        param_sample_n:  采集时只统计 1/n 的参数（按 name 哈希）；1=全部
        check_post_step: 是否在 optimizer.step() 之后检查参数（检测更新引入的数值问题）
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        sampler: Sampler,
        writer: SQLiteWriter,
        alert_config: Optional[AlertConfig] = None,
        on_alert: Optional[Callable[[Alert, int], None]] = None,
        param_sample_n: int = 1,
        check_post_step: bool = False,
    ):
        self._model = model
        self._optimizer = optimizer
        self._sampler = sampler
        self._writer = writer
        self._alert_config = alert_config or AlertConfig()
        self._on_alert = on_alert
        self._param_sample_n = max(1, int(param_sample_n))
        self._check_post_step = check_post_step

        self._param_names: Dict[int, str] = _param_name_map(model)
        self._current_step: int = 0
        self._trace_this_step: bool = False

        self._original_step: Optional[Callable[..., Any]] = None
        self._attached: bool = False
        # PyTorch >= 2.1 原生 optimizer hook
        self._use_native_hook: bool = False
        self._pre_hook_handle: Optional[Any] = None
        self._post_hook_handle: Optional[Any] = None

    def attach(self) -> None:
        """
        包装 optimizer.step()，在调用前执行本 step 的梯度和参数检查。

        优先使用 PyTorch >= 2.1 的 register_step_pre_hook（更稳定，兼容 GradScaler），
        若不可用则 fallback 到 monkey-patch optimizer.step()。
        """
        if self._attached:
            return

        # 尝试使用原生 optimizer hook API（PyTorch >= 2.1）
        try:
            if hasattr(self._optimizer, "register_step_pre_hook"):
                self._pre_hook_handle = self._optimizer.register_step_pre_hook(
                    self._native_pre_hook
                )
                if self._check_post_step and hasattr(self._optimizer, "register_step_post_hook"):
                    self._post_hook_handle = self._optimizer.register_step_post_hook(
                        self._native_post_hook
                    )
                self._use_native_hook = True
                self._attached = True
                return
        except Exception:
            pass

        # Fallback: monkey-patch optimizer.step()
        self._original_step = self._optimizer.step
        self._optimizer.step = self._wrapped_step
        self._use_native_hook = False
        self._attached = True

    def detach(self) -> None:
        """恢复原始 optimizer.step() 或移除原生 hook。"""
        if not self._attached:
            return
        if self._use_native_hook:
            if self._pre_hook_handle is not None:
                self._pre_hook_handle.remove()
                self._pre_hook_handle = None
            if self._post_hook_handle is not None:
                self._post_hook_handle.remove()
                self._post_hook_handle = None
        else:
            if self._original_step is not None:
                self._optimizer.step = self._original_step
                self._original_step = None
        self._attached = False

    def set_step(self, step: int) -> None:
        """由 Nanny 在每个 step 开始前调用。"""
        self._current_step = step
        self._trace_this_step = self._sampler.should_trace()

    # ─── 原生 optimizer hook (PyTorch >= 2.1) ──────────────────────────────

    def _native_pre_hook(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        if self._trace_this_step:
            try:
                self._check_grads_and_params()
            except Exception as exc:
                print(f"[Nanny][WARN] optimizer monitor error: {exc}")

    def _native_post_hook(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        if self._trace_this_step:
            try:
                self._check_params_post_step()
            except Exception as exc:
                print(f"[Nanny][WARN] optimizer post-step monitor error: {exc}")

    # ─── Monkey-patch fallback ─────────────────────────────────────────────

    def _wrapped_step(self, *args: Any, **kwargs: Any) -> Optional[Any]:
        """先执行梯度和参数的数值检查（若本 step 需要 trace），再调用原始 step。"""
        if self._trace_this_step:
            try:
                self._check_grads_and_params()
            except Exception as exc:
                print(f"[Nanny][WARN] optimizer monitor error: {exc}")
        result = self._original_step(*args, **kwargs)
        if self._trace_this_step and self._check_post_step:
            try:
                self._check_params_post_step()
            except Exception as exc:
                print(f"[Nanny][WARN] optimizer post-step monitor error: {exc}")
        return result

    # ─── 参数名解析（惰性刷新） ────────────────────────────────────────────

    def _resolve_param_name(self, param: torch.Tensor) -> str:
        """
        解析参数名称，支持惰性刷新以处理动态模型。
        当 id(param) 不在映射中时自动重建映射表。
        """
        pid = id(param)
        name = self._param_names.get(pid)
        if name is None:
            self._param_names = _param_name_map(self._model)
            name = self._param_names.get(pid, f"param_{pid}")
        return name

    # ─── 核心检查逻辑 ──────────────────────────────────────────────────────

    def _check_grads_and_params(self) -> None:
        """遍历所有参数：对 grad 做 optimizer_grad 统计/告警，对 data 做 optimizer_param。"""
        cfg = self._alert_config
        fast = getattr(cfg, "fast_stats", False)
        ts = time.time()

        for group in self._optimizer.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                name = self._resolve_param_name(param)
                if self._param_sample_n > 1 and (hash(name) % self._param_sample_n) != 0:
                    continue

                if param.grad is not None:
                    grad = param.grad
                    stats = compute_stats(grad, precision=cfg.precision, fast_stats=fast)
                    if stats is not None:
                        self._writer.write_stats(
                            step=self._current_step,
                            phase="optimizer_grad",
                            layer_name=name,
                            layer_type="grad",
                            stats=stats,
                            ts=ts,
                        )
                        for alert in check_alerts(stats, name, "optimizer_grad", cfg):
                            self._emit_alert(alert, ts, "optimizer_grad", name)

                stats = compute_stats(param.data, precision=cfg.precision, fast_stats=fast)
                if stats is not None:
                    self._writer.write_stats(
                        step=self._current_step,
                        phase="optimizer_param",
                        layer_name=name,
                        layer_type="param",
                        stats=stats,
                        ts=ts,
                    )
                    for alert in check_alerts(stats, name, "optimizer_param", cfg):
                        self._emit_alert(alert, ts, "optimizer_param", name)

    def _check_params_post_step(self) -> None:
        """optimizer.step() 之后检查参数，检测更新本身引入的 NaN/Inf/饱和等数值问题。"""
        cfg = self._alert_config
        fast = getattr(cfg, "fast_stats", False)
        ts = time.time()

        for group in self._optimizer.param_groups:
            for param in group["params"]:
                if not param.requires_grad:
                    continue
                name = self._resolve_param_name(param)
                if self._param_sample_n > 1 and (hash(name) % self._param_sample_n) != 0:
                    continue

                stats = compute_stats(param.data, precision=cfg.precision, fast_stats=fast)
                if stats is not None:
                    self._writer.write_stats(
                        step=self._current_step,
                        phase="optimizer_post_param",
                        layer_name=name,
                        layer_type="param",
                        stats=stats,
                        ts=ts,
                    )
                    for alert in check_alerts(stats, name, "optimizer_post_param", cfg):
                        self._emit_alert(alert, ts, "optimizer_post_param", name)

    def _emit_alert(self, alert: Alert, ts: float, phase: str, layer_name: str) -> None:
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
