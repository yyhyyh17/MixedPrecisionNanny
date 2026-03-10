"""
混合精度 vs FP32 前向传播精度对比分析

核心功能：
  - 对同一模型和输入，分别以 FP32 和混合精度（FP16/BF16）执行 forward
  - 通过 forward hook 捕获每一层的中间输出
  - 逐层计算两种精度下输出的差异指标（绝对误差、相对误差、余弦相似度等）
  - 输出结构化报告，供可视化前端展示

设计原则：
  - 使用 model.eval() + torch.no_grad() 确保两次 forward 结果可比
  - hook 只捕获第一个 Tensor 输出（与 HookManager 一致）
  - 所有 diff 计算在 FP32 下进行，避免低精度比较引入额外误差
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LayerDiffResult:
    """单层的精度对比结果。"""
    layer_name: str
    layer_type: str
    shape: List[int]
    numel: int

    # FP32 输出统计
    fp32_max: float
    fp32_min: float
    fp32_mean: float
    fp32_std: float

    # 混合精度输出统计
    mp_max: float
    mp_min: float
    mp_mean: float
    mp_std: float
    mp_dtype: str

    # diff 指标
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    mean_rel_diff: float
    cosine_similarity: float
    rmse: float

    # 异常计数
    mp_nan_count: int = 0
    mp_inf_count: int = 0


@dataclass
class DiffReport:
    """完整的精度对比报告。"""
    model_name: str
    precision: str
    device: str
    timestamp: float
    total_layers: int
    input_shape: List[int]

    # 最终输出 diff
    final_output_max_abs_diff: float
    final_output_mean_abs_diff: float
    final_output_cosine_similarity: float

    # 各层详情
    layers: List[LayerDiffResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "DiffReport":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        layers = [LayerDiffResult(**ld) for ld in data.pop("layers")]
        return cls(**data, layers=layers)


def _extract_first_tensor(obj: Any) -> Optional[torch.Tensor]:
    """从 module output 中提取第一个 Tensor（与 hook_manager 一致）。"""
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


@torch.no_grad()
def _compute_layer_diff(
    fp32_tensor: torch.Tensor,
    mp_tensor: torch.Tensor,
    layer_name: str,
    layer_type: str,
) -> Optional[LayerDiffResult]:
    """计算单层 FP32 vs 混合精度输出的 diff 指标。"""
    if fp32_tensor.shape != mp_tensor.shape:
        return None

    a = fp32_tensor.float().flatten()
    b = mp_tensor.float().flatten()

    numel = a.numel()
    if numel == 0:
        return None

    diff = (a - b).abs()
    denom = a.abs().clamp(min=1e-12)
    rel_diff = diff / denom

    cos_sim = torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()

    rmse = (diff ** 2).mean().sqrt().item()

    return LayerDiffResult(
        layer_name=layer_name,
        layer_type=layer_type,
        shape=list(fp32_tensor.shape),
        numel=numel,
        fp32_max=a.max().item(),
        fp32_min=a.min().item(),
        fp32_mean=a.mean().item(),
        fp32_std=a.std().item() if numel > 1 else 0.0,
        mp_max=b.max().item(),
        mp_min=b.min().item(),
        mp_mean=b.mean().item(),
        mp_std=b.std().item() if numel > 1 else 0.0,
        mp_dtype=str(mp_tensor.dtype),
        max_abs_diff=diff.max().item(),
        mean_abs_diff=diff.mean().item(),
        max_rel_diff=rel_diff.max().item(),
        mean_rel_diff=rel_diff.mean().item(),
        cosine_similarity=cos_sim,
        rmse=rmse,
        mp_nan_count=int(torch.isnan(b).sum().item()),
        mp_inf_count=int(torch.isinf(b).sum().item()),
    )


class PrecisionDiffAnalyzer:
    """
    混合精度前向传播精度对比分析器。

    分别以 FP32 和混合精度运行 model.forward()，捕获每层中间输出并计算 diff。

    Args:
        model:      待分析的 nn.Module（不会修改模型权重）
        precision:  混合精度类型 "fp16" 或 "bf16"
        device:     计算设备（None 则自动检测）
        exclude_prefixes:  排除的层名前缀
        max_depth:  最大层级深度
    """

    _AUTOCAST_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}

    def __init__(
        self,
        model: nn.Module,
        precision: Literal["fp16", "bf16"] = "fp16",
        device: Optional[torch.device] = None,
        exclude_prefixes: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
    ):
        self._model = model
        self._precision = precision
        self._autocast_dtype = self._AUTOCAST_DTYPE.get(precision, torch.float16)
        self._exclude_prefixes = exclude_prefixes or []
        self._max_depth = max_depth

        if device is None:
            p = next(model.parameters(), None)
            self._device = p.device if p is not None else torch.device("cpu")
        else:
            self._device = device

    def _should_skip(self, name: str) -> bool:
        if not name:
            return True
        for prefix in self._exclude_prefixes:
            if name.startswith(prefix):
                return True
        if self._max_depth is not None and name.count(".") > self._max_depth:
            return True
        return False

    def analyze(
        self,
        sample_input: Any,
        **forward_kwargs: Any,
    ) -> DiffReport:
        """
        执行精度对比分析。

        Args:
            sample_input: 模型输入（Tensor 或 tuple/dict）
            **forward_kwargs: 传递给 model.forward() 的额外参数

        Returns:
            DiffReport 包含逐层和最终输出的 diff 指标
        """
        self._model.eval()

        module_list = [
            (name, module)
            for name, module in self._model.named_modules()
            if not self._should_skip(name)
        ]

        fp32_outputs: Dict[str, torch.Tensor] = {}
        mp_outputs: Dict[str, torch.Tensor] = {}
        layer_types: Dict[str, str] = {}

        def _make_capture_hook(store: Dict[str, torch.Tensor], name: str):
            def hook(module: nn.Module, inputs: tuple, output: Any) -> None:
                t = _extract_first_tensor(output)
                if t is not None:
                    store[name] = t.detach().clone().float()
            return hook

        # --- Pass 1: FP32 forward ---
        handles = []
        for name, module in module_list:
            layer_types[name] = type(module).__name__
            h = module.register_forward_hook(_make_capture_hook(fp32_outputs, name))
            handles.append(h)

        with torch.no_grad():
            fp32_final = self._model(sample_input, **forward_kwargs)
        for h in handles:
            h.remove()

        fp32_final_tensor = _extract_first_tensor(fp32_final)
        fp32_final_flat = fp32_final_tensor.detach().clone().float() if fp32_final_tensor is not None else None

        # --- Pass 2: Mixed precision forward ---
        handles = []
        for name, module in module_list:
            h = module.register_forward_hook(_make_capture_hook(mp_outputs, name))
            handles.append(h)

        autocast_device = "cuda" if self._device.type == "cuda" else "cpu"
        with torch.no_grad(), torch.autocast(
            device_type=autocast_device,
            dtype=self._autocast_dtype,
        ):
            mp_final = self._model(sample_input, **forward_kwargs)
        for h in handles:
            h.remove()

        mp_final_tensor = _extract_first_tensor(mp_final)
        mp_final_flat = mp_final_tensor.detach().clone().float() if mp_final_tensor is not None else None

        # --- Compute per-layer diffs ---
        layer_results: List[LayerDiffResult] = []
        for name, _ in module_list:
            if name not in fp32_outputs or name not in mp_outputs:
                continue
            result = _compute_layer_diff(
                fp32_outputs[name],
                mp_outputs[name],
                layer_name=name,
                layer_type=layer_types.get(name, ""),
            )
            if result is not None:
                layer_results.append(result)

        # --- Final output diff ---
        final_max_abs = 0.0
        final_mean_abs = 0.0
        final_cos_sim = 1.0
        if fp32_final_flat is not None and mp_final_flat is not None:
            diff = (fp32_final_flat.flatten() - mp_final_flat.flatten()).abs()
            final_max_abs = diff.max().item()
            final_mean_abs = diff.mean().item()
            final_cos_sim = torch.nn.functional.cosine_similarity(
                fp32_final_flat.flatten().unsqueeze(0),
                mp_final_flat.flatten().unsqueeze(0),
            ).item()

        input_shape = []
        if isinstance(sample_input, torch.Tensor):
            input_shape = list(sample_input.shape)

        return DiffReport(
            model_name=type(self._model).__name__,
            precision=self._precision,
            device=str(self._device),
            timestamp=time.time(),
            total_layers=len(layer_results),
            input_shape=input_shape,
            final_output_max_abs_diff=final_max_abs,
            final_output_mean_abs_diff=final_mean_abs,
            final_output_cosine_similarity=final_cos_sim,
            layers=layer_results,
        )
