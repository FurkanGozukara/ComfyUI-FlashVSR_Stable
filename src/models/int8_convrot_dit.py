"""
INT8 ConvRot for the FlashVSR Wan DiT.

The DiT's Linear weights are Hadamard-rotated and stored as row-wise INT8 in one
safetensors cache ("<model>_int8_convrot.safetensors", the same file the SECourses
model downloader ships). Matmuls run as INT8 GEMMs (Triton / torch._int_mm), so the
DiT needs about 1.75 GB of VRAM instead of 2.8 GB in bf16. When no cache exists it
is built once from the source checkpoint with the shared conversion engine.

INT8 layers are their own module type (not nn.Linear), so the pipeline's VRAM
manager does not wrap and dtype-cast them; they move between CPU and GPU by device
only and keep their float32 scales.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .int8_convrot import int8_convrot_linear

INT8_CACHE_SUFFIX = "_int8_convrot.safetensors"
INT8_CACHE_FORMAT = "flashvsr-dit-int8-convrot-v2"
INT8_GROUPSIZE_KEY_SUFFIX = ".int8_convrot_groupsize"
INT8_ARA_DOWN_SUFFIX = ".int8_ara_down"
INT8_ARA_UP_SUFFIX = ".int8_ara_up"
_MODEL_CONFIG_KEYS = ("has_image_input", "patch_size", "in_dim", "dim", "ffn_dim", "freq_dim",
                      "text_dim", "out_dim", "num_heads", "num_layers", "eps")


def int8_cache_path(cache_dir: str | Path, model_name: str) -> Path:
    return Path(cache_dir) / f"{model_name}{INT8_CACHE_SUFFIX}"


def has_valid_int8_cache(cache_path: str | Path, source_checkpoint: str | Path | None = None) -> bool:
    """Format marker + group-size keys; the recorded source size must match when the source exists."""
    cache = Path(cache_path)
    if not cache.is_file():
        return False
    try:
        with safe_open(str(cache), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            has_groups = any(key.endswith(INT8_GROUPSIZE_KEY_SUFFIX) for key in handle.keys())
        if (
            metadata.get("int8_convrot_format") != INT8_CACHE_FORMAT
            or metadata.get("flashvsr_int8_convrot") != "true"
            or not metadata.get("model_config")
            or not has_groups
        ):
            return False
        source = Path(source_checkpoint) if source_checkpoint else None
        if source is None or not source.is_file():
            return True
        recorded = metadata.get("source_size")
        return recorded is None or recorded == str(source.stat().st_size)
    except Exception:
        return False


class Int8ConvRotLinear(nn.Module):
    """Drop-in replacement for an nn.Linear whose weight is INT8 ConvRot."""

    def __init__(self, in_features: int, out_features: int, q_weight: torch.Tensor, scale: torch.Tensor,
                 group_size: int, bias: Optional[torch.Tensor], ara_down: Optional[torch.Tensor] = None,
                 ara_up: Optional[torch.Tensor] = None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.group_size = int(group_size)
        self.weight = nn.Parameter(q_weight, requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.register_buffer("scale_weight", scale.float())
        self.register_buffer("int8_ara_down", ara_down)
        self.register_buffer("int8_ara_up", ara_up)
        self.onload_device = None
        self.offload_device = torch.device("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight.device != x.device:
            self.to(x.device)
        return int8_convrot_linear(x, self.weight, self.scale_weight, self.group_size, self.bias,
                                   self.int8_ara_down, self.int8_ara_up)

    def _apply(self, fn, recurse=True):
        # Module.to(dtype=...) / .half() must not round the float32 scales; only their device follows.
        scale = self._buffers.pop("scale_weight")
        try:
            super()._apply(fn, recurse)
        finally:
            target = fn(torch.empty(0, dtype=torch.int8, device=scale.device)).device
            self._buffers["scale_weight"] = scale.to(target)
        return self

    # VRAM manager hooks (pipelines call these on every module that has them).
    def onload(self):
        if self.onload_device is not None and self.weight.device != self.onload_device:
            self.to(self.onload_device)

    def offload(self):
        if self.weight.device != self.offload_device:
            self.to(self.offload_device)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8_convrot gs={self.group_size}"


def _set_module(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, module)


def count_int8_linears(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, Int8ConvRotLinear))


def set_int8_onload_device(model: nn.Module, device) -> None:
    device = torch.device(device)
    for m in model.modules():
        if isinstance(m, Int8ConvRotLinear):
            m.onload_device = device


def load_int8_dit(cache_path: str | Path, dtype: torch.dtype) -> nn.Module:
    """Build the FlashVSR WanModel straight from the INT8 cache (the source checkpoint is not read)."""
    from .wan_video_dit import WanModel
    from .utils import init_weights_on_device

    path = Path(cache_path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    config = json.loads(metadata["model_config"])
    config = {k: config[k] for k in _MODEL_CONFIG_KEYS if k in config}
    if "patch_size" in config:
        config["patch_size"] = tuple(config["patch_size"])
    state = load_file(str(path), device="cpu")

    with init_weights_on_device(device=torch.device("meta")):
        model = WanModel(**config)
    layers = {k[: -len(INT8_GROUPSIZE_KEY_SUFFIX)] for k in state if k.endswith(INT8_GROUPSIZE_KEY_SUFFIX)}
    modules = dict(model.named_modules())
    unknown = [n for n in layers if not isinstance(modules.get(n), nn.Linear)]
    if unknown:
        raise RuntimeError(f"FlashVSR INT8 cache does not match the DiT; unknown layers: {sorted(unknown)[:6]}")
    for name in sorted(layers):
        lin = modules[name]
        q = state.pop(f"{name}.weight")
        bias = state.pop(f"{name}.bias", None)
        int8_layer = Int8ConvRotLinear(
            lin.in_features, lin.out_features, q, state.pop(f"{name}.scale_weight"),
            int(state.pop(f"{name}{INT8_GROUPSIZE_KEY_SUFFIX}").item()),
            None if bias is None else bias.to(dtype),
            state.pop(f"{name}{INT8_ARA_DOWN_SUFFIX}", None),
            state.pop(f"{name}{INT8_ARA_UP_SUFFIX}", None),
        )
        _set_module(model, name, int8_layer)
    state = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in state.items()}
    float_keys = {k for k, v in model.state_dict(keep_vars=True).items()
                  if not k.endswith(("scale_weight", "int8_ara_down", "int8_ara_up"))
                  and not any(k == f"{n}.weight" or k == f"{n}.bias" for n in layers)}
    missing = sorted(float_keys - set(state))
    unexpected = sorted(set(state) - float_keys)
    if missing or unexpected:
        raise RuntimeError(f"FlashVSR INT8 cache keys do not match the DiT (missing {missing[:6]}, "
                           f"unexpected {unexpected[:6]})")
    model.load_state_dict(state, strict=False, assign=True)
    model.eval().requires_grad_(False)
    print(f"[FlashVSR INT8] Loaded {len(layers)} INT8 ConvRot Linear layers from {path.name}", flush=True)
    return model


def quantize_dit_to_int8(dit: nn.Module, source_checkpoint: str | Path) -> int:
    """Convert the eligible Linear layers of a loaded DiT in place (one-time cache build)."""
    from .int8_convert_engine import Int8ConversionEngine

    engine = Int8ConversionEngine(source_checkpoint, log_prefix="[FlashVSR INT8]")
    candidates: Dict[str, nn.Linear] = {}
    for name, module in dit.named_modules():
        if isinstance(module, nn.Linear) and module.weight.ndim == 2 and module.weight.device.type != "meta":
            candidates[name] = module
    shapes = {n: (int(m.weight.shape[0]), int(m.weight.shape[1])) for n, m in candidates.items()}
    decisions = engine.plan(shapes)  # the layer policy keeps embeddings / head in bf16
    results = {}
    for name, module in candidates.items():
        decision = decisions.get(name)
        if decision is None or not decision.quantize:
            continue
        results[name] = engine.quantize_layer(name, module.weight, decision.group_size,
                                              has_bias=module.bias is not None,
                                              source_bytes_per_element=module.weight.element_size())
    rescued = engine.select_rescue(results)
    patched = 0
    for name, result in results.items():
        if name in rescued:
            continue
        lin = candidates[name]
        bias = None
        if lin.bias is not None:
            bias = lin.bias.detach()
            if result.bias_delta is not None:
                bias = (bias.float() - result.bias_delta.to(bias.device)).to(bias.dtype)
        _set_module(dit, name, Int8ConvRotLinear(
            lin.in_features, lin.out_features, result.q.cpu(), result.scale.cpu(), int(result.group_size),
            None if bias is None else bias.cpu(),
            None if result.ara_down is None else result.ara_down.cpu(),
            None if result.ara_up is None else result.ara_up.cpu(),
        ))
        patched += 1
    dit._int8_conversion_report = engine.report.metadata_json()
    print(engine.summary_line(results, rescued), flush=True)
    return patched


def save_int8_dit(dit: nn.Module, cache_path: str | Path, source_checkpoint: str | Path, model_config: dict) -> Path:
    """Save the whole DiT (LQ projector excluded) as one atomic safetensors file."""
    output = Path(cache_path)
    state: Dict[str, torch.Tensor] = {}
    for key, tensor in dit.state_dict().items():
        if key.startswith("LQ_proj_in."):
            continue
        state[key] = tensor.detach().cpu().contiguous()
    quantized = 0
    for name, module in dit.named_modules():
        if isinstance(module, Int8ConvRotLinear):
            state[f"{name}{INT8_GROUPSIZE_KEY_SUFFIX}"] = torch.tensor(module.group_size, dtype=torch.int32)
            quantized += 1
    if not quantized:
        raise RuntimeError("Refusing to save an empty FlashVSR INT8 ConvRot cache")
    source = Path(source_checkpoint)
    stat = source.stat()
    metadata = {
        "format": "pt",
        "int8_convrot_format": INT8_CACHE_FORMAT,
        "flashvsr_int8_convrot": "true",
        "component": "dit",
        "quantized_linear_layers": str(quantized),
        "scale_dtype": "float32",
        "rotation": "hadamard-regular",
        "mse_clip": "true",
        "model_config": json.dumps(model_config, separators=(",", ":")),
        "source": str(source.resolve()),
        "source_size": str(stat.st_size),
        "source_mtime_ns": str(stat.st_mtime_ns),
    }
    report = getattr(dit, "_int8_conversion_report", None)
    if report:
        metadata["conversion_report"] = str(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output.with_name(output.name + f".{os.getpid()}.tmp")
    try:
        save_file(state, str(temp_path), metadata=metadata)
        os.replace(temp_path, output)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return output


def dit_model_config(dit: nn.Module) -> dict:
    """Recover the WanModel constructor config from a loaded DiT (stored in the cache metadata)."""
    blocks = dit.blocks
    first = blocks[0]
    return {
        "has_image_input": bool(getattr(dit, "has_image_input", False)),
        "patch_size": list(dit.patch_size),
        "in_dim": int(dit.patch_embedding.in_channels),
        "dim": int(dit.dim),
        "ffn_dim": int(first.ffn[0].out_features),
        "freq_dim": int(dit.freq_dim),
        "text_dim": int(dit.text_embedding[0].in_features),
        "out_dim": int(dit.head.head.out_features // (dit.patch_size[0] * dit.patch_size[1] * dit.patch_size[2])),
        "num_heads": int(first.num_heads),
        "num_layers": len(blocks),
        "eps": float(first.norm1.eps),
    }
