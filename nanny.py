"""
MixedPrecisionNanny — 混合精度训练实时监控主入口

用法（context manager，推荐）：

    from nanny import MixedPrecisionNanny

    nanny = MixedPrecisionNanny(model, scaler=scaler, trace_interval=100)

    for step, batch in enumerate(dataloader):
        with nanny.step(step):          # ← 唯一侵入点
            with autocast():
                loss = model(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    nanny.close()   # 或用 with MixedPrecisionNanny(...) as nanny:

用法（手动 begin/end）：

    nanny.begin_step(step)
    ... training code ...
    nanny.end_step()
"""
from __future__ import annotations

import contextlib
import os
from typing import List, Optional

import torch.nn as nn

from analyzer.numerical_checker import Alert, AlertConfig
from storage.sqlite_writer import SQLiteWriter
from tracer.hook_manager import HookManager
from tracer.sampler import Sampler


class MixedPrecisionNanny:
    """
    混合精度训练监控器。

    Args:
        model:            被监控的 nn.Module
        scaler:           torch.cuda.amp.GradScaler 实例（可选）
                          传入后会自动记录 Loss Scale 变化历史
        trace_interval:   每隔多少 step trace 一次（默认 100）
        output_dir:       数据存储目录（默认 ./nanny_logs）
        alert_config:     告警阈值配置，不传则使用默认值
        precision:        精度检测目标 "fp16" 或 "bf16"（默认 "fp16"）
                          仅当未传 alert_config 时生效，否则用 alert_config.precision
        exclude_prefixes: 不监控的层名前缀列表
        max_depth:        只监控层级深度 ≤ max_depth 的 module（None = 不限）
        verbose:          是否打印实时告警和摘要（默认 True）
    """

    def __init__(
        self,
        model: nn.Module,
        scaler=None,
        trace_interval: int = 100,
        output_dir: str = "./nanny_logs",
        alert_config: Optional[AlertConfig] = None,
        precision: str = "fp16",
        exclude_prefixes: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        verbose: bool = True,
    ):
        self._model = model
        self._scaler = scaler
        self._verbose = verbose
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        db_path = os.path.join(output_dir, "metrics.db")
        self._writer = SQLiteWriter(db_path)

        self._sampler = Sampler(trace_interval=trace_interval)

        if alert_config is None:
            alert_config = AlertConfig(precision=precision)
        self._hook_manager = HookManager(
            model=model,
            sampler=self._sampler,
            writer=self._writer,
            alert_config=alert_config,
            on_alert=self._on_alert,
            exclude_prefixes=exclude_prefixes,
            max_depth=max_depth,
        )
        n_layers = self._hook_manager.attach()

        # Loss Scale 跟踪：记录上一步的 scale，判断是否发生 overflow
        self._prev_scale: Optional[float] = None

        # 当前 step 内的告警计数（用于 verbose 摘要）
        self._step_error_count: int = 0
        self._step_warn_count: int = 0

        self._current_step: int = 0

        if verbose:
            print(
                f"[Nanny] Attached to {n_layers} layers. "
                f"trace_interval={trace_interval}. "
                f"DB → {db_path}"
            )

    # ─── 核心接口：context manager ─────────────────────────────────────────────

    @contextlib.contextmanager
    def step(self, step: int):
        """
        包裹一个完整训练 step 的 context manager。

        with nanny.step(step):
            forward / backward / optimizer.step()
        """
        self.begin_step(step)
        try:
            yield self
        finally:
            self.end_step()

    # ─── 核心接口：手动模式 ────────────────────────────────────────────────────

    def begin_step(self, step: int) -> None:
        """在 forward 之前调用。"""
        self._current_step = step
        self._step_error_count = 0
        self._step_warn_count = 0
        self._sampler.advance(step)
        self._hook_manager.set_step(step)

    def end_step(self) -> None:
        """在 optimizer.step() 之后调用。记录 scaler 状态并打印摘要。"""
        if self._scaler is not None:
            self._record_loss_scale(self._current_step)

        if self._verbose and self._sampler.should_trace():
            self._print_step_summary()

        # 必须在 should_trace() 和 _on_alert() 都执行完之后再消耗触发计数
        self._sampler.consume()

    # ─── 辅助接口 ─────────────────────────────────────────────────────────────

    def flush(self) -> None:
        """阻塞直到所有数据写入磁盘。"""
        self._writer.flush()

    def close(self) -> None:
        """移除 hook，flush，关闭 DB 写入器。"""
        self._hook_manager.detach()
        self._writer.flush()
        self._writer.close()
        if self._verbose:
            print(f"[Nanny] Closed. Metrics saved to: {self._output_dir}/metrics.db")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ─── 内部 ─────────────────────────────────────────────────────────────────

    def _on_alert(self, alert: Alert, step: int) -> None:
        """HookManager 每产生一个 alert 就回调此函数。"""
        if alert.severity == "ERROR":
            self._step_error_count += 1
            # 切换到密集采样模式，接下来 50 步全部 trace
            self._sampler.trigger_dense(duration=50)
            if self._verbose:
                print(f"[Nanny][ERROR] step={step} | {alert.message}")
        elif alert.severity == "WARNING":
            self._step_warn_count += 1
            if self._verbose:
                print(f"[Nanny][WARN]  step={step} | {alert.message}")

    def _record_loss_scale(self, step: int) -> None:
        """记录 GradScaler 当前 scale 值，判断是否发生了 overflow。"""
        try:
            scale = self._scaler.get_scale()
            overflow = (
                self._prev_scale is not None and scale < self._prev_scale
            )
            self._writer.write_loss_scale(
                step=step,
                scale=scale,
                overflow=overflow,
            )
            if self._verbose and self._sampler.should_trace():
                status = "OVERFLOW" if overflow else "ok"
                if scale < 128 or overflow:
                    print(
                        f"[Nanny][WARN]  step={step} | "
                        f"LossScale={scale:.1f} [{status}]"
                    )
            self._prev_scale = scale
        except Exception:
            pass  # scaler 监控出错不影响训练

    def _print_step_summary(self) -> None:
        mode = "TRIGGERED" if self._sampler.is_triggered else "PERIODIC"
        msg = (
            f"[Nanny] step={self._current_step} [{mode}] "
            f"errors={self._step_error_count} warnings={self._step_warn_count}"
        )
        if self._step_error_count > 0 or self._step_warn_count > 0:
            print(msg)
