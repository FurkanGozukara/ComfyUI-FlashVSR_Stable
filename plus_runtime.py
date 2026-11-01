# -*- coding: utf-8 -*-
"""
FlashVSR+ (official Turbo / Base presets, OpenImagingLab) inside the SECourses FlashVSR CLI.

The vendored engine (flashvsr_plus_engine) does the diffusion and TC decoding; this module keeps the
CLI's own behaviour around it:
  - the CLI's frame reader and ffmpeg writer (codec / CRF / pixel format, start / end frame);
  - input frames are upscaled with PIL bicubic (the engine's own resampler) and padded to multiples of
    128 by reflection, then the padding is cropped from the output (no crop of the picture);
  - the whole clip runs as ONE stream: the input is read and the output decoded and written in segments
    (--decode_every latents), so the temporal state never restarts and RAM stays bounded;
  - DiT tiling with a border-aware feather blend (tiles are diffused separately per frame chunk);
  - out-of-memory recovery: KV-cache offload to pinned RAM, then DiT tiles, then smaller tiles;
  - INT8 ConvRot DiT (the shared ConvRot engine; cache "<model>_int8_convrot.safetensors").
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLUS_MODELS = {
    "FlashVSR-Plus-Turbo": "turbo",
    "FlashVSR-Plus-Base": "base",
    "FlashVSR-v1.1-Plus-Runtime": "v1",
}
MULTIPLE = 128
INT8_PLUS_FORMAT = "flashvsr-plus-dit-int8-convrot-v1"
READ_CHUNK_FRAMES = 16


def is_plus_model(name: str) -> bool:
    return str(name) in PLUS_MODELS


_T0 = time.time()


def _log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------------------------
def model_paths(models_dir: str | Path, model: str) -> Dict[str, Path]:
    root = Path(models_dir)
    if model == "FlashVSR-v1.1-Plus-Runtime":
        folder = root / "FlashVSR-v1.1"
        dit = folder / "diffusion_pytorch_model_streaming_dmd.safetensors"
        lq = folder / "LQ_proj_in.ckpt"
    else:
        folder = root / model
        dit = folder / "dit.safetensors"
        lq = folder / "lq_proj_in.safetensors"
    tcd = folder / "TCDecoder.ckpt"
    if not tcd.is_file():
        tcd = root / "FlashVSR-v1.1" / "TCDecoder.ckpt"  # byte-identical in every FlashVSR release
    return {"folder": folder, "dit": dit, "lq": lq, "tcd": tcd, "prompt": ROOT / "posi_prompt.pth"}


def int8_cache_path(cache_dir: str | Path, model: str) -> Path:
    return Path(cache_dir) / f"{model}_int8_convrot.safetensors"


# ---------------------------------------------------------------------------------------------
# DiT construction (meta init + assign: no 5.6 GB random init, weights stay in their file dtype)
# ---------------------------------------------------------------------------------------------
def _model_config(config):
    from flashvsr_plus_engine.models import BlockReplacementConfig, FlashVSRModelConfig, LQProjectionConfig

    return FlashVSRModelConfig(
        attention=config.attention_config(),
        lq_projection=LQProjectionConfig(enabled=True, num_injection_layers=config.lq_projection_layers),
        replacement=BlockReplacementConfig(layers=config.replacement_layers),
    )


def _meta_model(config):
    from flashvsr_plus_engine.models import build_model

    with torch.device("meta"):
        return build_model(_model_config(config))


def _normalized_state(path: Path, *, prefix: Optional[str] = None) -> Dict[str, torch.Tensor]:
    from flashvsr_plus_engine.model_io import load_checkpoint_file, normalize_key

    out = {}
    for key, value in load_checkpoint_file(str(path)).items():
        k = normalize_key(key)
        if prefix and k.startswith(prefix):
            k = k[len(prefix):]
        out[k] = value
    return out


def _assign(module: nn.Module, state: Dict[str, torch.Tensor], dtype: torch.dtype, label: str,
            skip_prefixes: Tuple[str, ...] = ()) -> None:
    own = module.state_dict(keep_vars=True)
    accepted = {}
    for k, v in state.items():
        if k in own and tuple(own[k].shape) == tuple(v.shape):
            accepted[k] = v.to(dtype) if (v.is_floating_point() and own[k].dtype != torch.int8
                                          and not k.endswith("scale_weight")) else v
    expected = [k for k in own if not k.startswith(skip_prefixes)]
    missing = [k for k in expected if k not in accepted]
    if missing:
        raise RuntimeError(f"{label}: {len(missing)} weights missing (e.g. {missing[:4]})")
    module.load_state_dict(accepted, strict=False, assign=True)


def build_denoiser(config, paths: Dict[str, Path], dtype: torch.dtype, *, int8_cache: Optional[Path] = None):
    """FlashVSRDiT with its LQ projection; INT8 ConvRot Linear layers when int8_cache is given."""
    model = _meta_model(config)
    if int8_cache is not None:
        from safetensors.torch import load_file
        from src.models.int8_convrot_dit import (INT8_ARA_DOWN_SUFFIX, INT8_ARA_UP_SUFFIX,
                                                  INT8_GROUPSIZE_KEY_SUFFIX, Int8ConvRotLinear, _set_module)

        state = load_file(str(int8_cache), device="cpu")
        layers = sorted(k[: -len(INT8_GROUPSIZE_KEY_SUFFIX)] for k in state if k.endswith(INT8_GROUPSIZE_KEY_SUFFIX))
        modules = dict(model.named_modules())
        for name in layers:
            lin = modules.get(name)
            if not isinstance(lin, nn.Linear):
                raise RuntimeError(f"INT8 cache {int8_cache.name} does not match the FlashVSR+ DiT ({name})")
            bias = state.pop(f"{name}.bias", None)
            _set_module(model, name, Int8ConvRotLinear(
                lin.in_features, lin.out_features, state.pop(f"{name}.weight"), state.pop(f"{name}.scale_weight"),
                int(state.pop(f"{name}{INT8_GROUPSIZE_KEY_SUFFIX}").item()),
                None if bias is None else bias.to(dtype),
                state.pop(f"{name}{INT8_ARA_DOWN_SUFFIX}", None), state.pop(f"{name}{INT8_ARA_UP_SUFFIX}", None)))
        _assign(model, state, dtype, "FlashVSR+ INT8 DiT", skip_prefixes=("LQ_proj_in.",) + tuple(
            f"{n}." for n in layers))
        _log(f"[FlashVSR+ INT8] Loaded {len(layers)} INT8 ConvRot Linear layers from {int8_cache.name}")
    else:
        _assign(model, _normalized_state(paths["dit"]), dtype, "FlashVSR+ DiT", skip_prefixes=("LQ_proj_in.",))
    _assign(model.LQ_proj_in, _normalized_state(paths["lq"], prefix="LQ_proj_in."), dtype, "FlashVSR+ LQ projection")
    return model.eval().requires_grad_(False)


def build_int8_cache(config, paths: Dict[str, Path], cache: Path, preset: str) -> Path:
    """Quantize the bf16 DiT with the shared ConvRot engine once and save the cache atomically."""
    from safetensors.torch import save_file
    from src.models.int8_convrot_dit import INT8_GROUPSIZE_KEY_SUFFIX, Int8ConvRotLinear, quantize_dit_to_int8

    _log(f"[FlashVSR+ INT8] Building the INT8 ConvRot cache once: {cache}")
    model = build_denoiser(config, paths, torch.bfloat16)
    quantize_dit_to_int8(model, paths["dit"])
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if not k.startswith("LQ_proj_in.")}
    quantized = 0
    for name, module in model.named_modules():
        if isinstance(module, Int8ConvRotLinear):
            state[f"{name}{INT8_GROUPSIZE_KEY_SUFFIX}"] = torch.tensor(module.group_size, dtype=torch.int32)
            quantized += 1
    src = Path(paths["dit"])
    meta = {
        "format": "pt", "int8_convrot_format": INT8_PLUS_FORMAT, "flashvsr_int8_convrot": "true",
        "component": "dit", "preset": preset, "quantized_linear_layers": str(quantized),
        "scale_dtype": "float32", "rotation": "hadamard-regular", "source": src.name,
        "source_size": str(src.stat().st_size),
    }
    report = getattr(model, "_int8_conversion_report", None)
    if report:
        meta["conversion_report"] = str(report)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + f".{os.getpid()}.tmp")
    try:
        save_file(state, str(tmp), metadata=meta)
        os.replace(tmp, cache)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    del model, state
    gc.collect()
    _log(f"[FlashVSR+ INT8] Saved {quantized} INT8 layers: {cache}")
    return cache


def valid_int8_cache(cache: Path) -> bool:
    if not cache.is_file():
        return False
    try:
        from safetensors import safe_open

        with safe_open(str(cache), framework="pt", device="cpu") as h:
            meta = h.metadata() or {}
        return meta.get("int8_convrot_format") == INT8_PLUS_FORMAT
    except Exception:
        return False


# ---------------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------------
def padded_dims(w: int, h: int, scale: int) -> Tuple[int, int, int, int, int, int]:
    sw, sh = w * scale, h * scale
    tw, th = math.ceil(sw / MULTIPLE) * MULTIPLE, math.ceil(sh / MULTIPLE) * MULTIPLE
    return sw, sh, tw, th, (tw - sw) // 2, (th - sh) // 2


def upscale_pad(frames: torch.Tensor, scale: int, out_dtype=torch.bfloat16) -> Tuple[torch.Tensor, tuple]:
    """(N,H,W,C) floats in [0,1] -> [1,3,N,tH,tW] in [-1,1]: PIL bicubic xscale, reflect-padded to 128."""
    from PIL import Image

    n, h, w, _ = frames.shape
    sw, sh, tw, th, pl, pt = padded_dims(w, h, scale)
    u8 = frames.mul(255.0).round_().clamp_(0, 255).to(torch.uint8).numpy()
    up = np.stack([np.asarray(Image.fromarray(f, mode="RGB").resize((sw, sh), resample=Image.Resampling.BICUBIC))
                   for f in u8])
    t = torch.from_numpy(up).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    if (tw, th) != (sw, sh):
        pads = (pl, tw - sw - pl, pt, th - sh - pt)
        mode = "reflect" if min(sh, sw) > max(pads) else "replicate"
        t = F.pad(t, pads, mode=mode)
    return t.permute(1, 0, 2, 3).unsqueeze(0).to(out_dtype).contiguous(), (pt, pl, sh, sw)


def tile_coords(height: int, width: int, tile: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    stride = max(1, tile - overlap)
    rows = max(1, math.ceil(max(1, height - overlap) / stride))
    cols = max(1, math.ceil(max(1, width - overlap) / stride))
    coords = []
    for r in range(rows):
        for c in range(cols):
            y1, x1 = r * stride, c * stride
            y2, x2 = min(y1 + tile, height), min(x1 + tile, width)
            y1, x1 = max(0, y2 - tile), max(0, x2 - tile)
            coords.append((x1, y1, x2, y2))
    return coords


def feather_mask(h: int, w: int, overlap: int, edges: Tuple[bool, bool, bool, bool]) -> torch.Tensor:
    """Weights (h, w): linear ramps over `overlap` px on inner edges only (left, right, top, bottom).

    Edges on the image border keep weight 1, so no border pixel ends up with zero total weight."""
    mask = torch.ones(h, w)
    ov = max(0, min(int(overlap), min(h, w) - 1))
    if ov <= 0:
        return mask
    ramp = torch.linspace(0, 1, ov + 2)[1:-1]  # strictly positive
    left, right, top, bottom = edges
    if left:
        mask[:, :ov] = torch.minimum(mask[:, :ov], ramp.view(1, -1))
    if right:
        mask[:, -ov:] = torch.minimum(mask[:, -ov:], ramp.flip(0).view(1, -1))
    if top:
        mask[:ov, :] = torch.minimum(mask[:ov, :], ramp.view(-1, 1))
    if bottom:
        mask[-ov:, :] = torch.minimum(mask[-ov:, :], ramp.flip(0).view(-1, 1))
    return mask


# ---------------------------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------------------------
class _Sink:
    """segment_writer for the engine: crops the padding and hands [F,H,W,C] floats to the CLI writer."""

    def __init__(self, crop, write: Callable[[torch.Tensor], None]):
        self.crop = crop
        self.write = write
        self.frames = 0

    def append(self, segment: torch.Tensor) -> None:
        if segment is None or segment.shape[2] == 0:
            return
        pt, pl, sh, sw = self.crop
        v = segment[0, :, :, pt:pt + sh, pl:pl + sw].float().add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
        v = v.permute(1, 2, 3, 0).contiguous()
        self.write(v)
        self.frames += int(v.shape[0])


class PlusEngine:
    def __init__(self, args, device: str, dtype: torch.dtype):
        from flashvsr_plus_engine.inference import presets
        from flashvsr_plus_engine.inference.loading import ComponentFactories, build_model_bundle

        self.args = args
        self.preset = PLUS_MODELS[args.model]
        self.device = device
        self.dtype = dtype
        self.paths = model_paths(args.models_dir or (ROOT / "models"), args.model)
        for key in ("dit", "lq", "tcd"):
            if not self.paths[key].is_file() and not (key == "dit" and args.precision == "int8_convrot"):
                raise FileNotFoundError(f"FlashVSR+ model file missing: {self.paths[key]}")
        attn = "sdpa" if args.attention_mode == "sdpa" else "auto"
        self.config = presets.stream_config_for_mode(
            self.preset, device=device, seed=int(args.seed), decode_every=int(args.decode_every),
            tiny_decoder_mode=args.plus_tiny_decoder, tiny_decoder_channels_last=bool(args.plus_channels_last),
            cache_offload_mode=args.plus_kv_offload, temporal_compression=int(args.temporal_compression),
            attn_kernel=attn, hardware_profile="generic" if attn == "sdpa" else "auto",
            color_correction=bool(args.color_fix_effective),
        )
        self.config.dtype = {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(dtype, "bf16")
        int8_cache = None
        if args.precision == "int8_convrot":
            int8_cache = int8_cache_path(args.int8_cache_dir or self.paths["folder"], args.model)
            if not valid_int8_cache(int8_cache):
                if not self.paths["dit"].is_file():
                    raise FileNotFoundError(f"FlashVSR+ INT8 cache not found: {int8_cache} (and no source DiT to build it)")
                build_int8_cache(self.config, self.paths, int8_cache, self.preset)
        t0 = time.time()
        factories = ComponentFactories(denoiser=lambda cfg, _p: build_denoiser(cfg, self.paths, dtype, int8_cache=int8_cache))
        from flashvsr_plus_engine.inference.loading import ModelPaths

        self.bundle = build_model_bundle(self.config, ModelPaths(
            dit=str(self.paths["dit"]), lq_projection=str(self.paths["lq"]), context=str(self.paths["prompt"]),
            tiny_decoder=str(self.paths["tcd"])), factories)
        if args.precision == "int8_convrot":
            from src.models.int8_convrot_dit import set_int8_onload_device

            set_int8_onload_device(self.bundle.denoiser, device)
        if os.environ.get("SECOURSES_INT8_CALIBRATE", "").strip().lower() in {"1", "true", "yes", "on"} \
                and args.precision != "int8_convrot":
            from src.models.int8_calibration import maybe_start_calibration

            maybe_start_calibration(self.bundle.denoiser, self.paths["dit"])
        torch.cuda.synchronize()
        _log(f"[FlashVSR+] {args.model} ({self.preset}) loaded in {time.time() - t0:.1f}s: attention="
             f"{self.config.attn_kernel}, dtype={self.config.dtype}, precision={args.precision}, "
             f"temporal_compression={self.config.temporal_compression}, decode_every={self.config.decoder.decode_every}")
        self._patch_offload(not bool(args.keep_on_gpu))
        if os.environ.get("FLASHVSR_PLUS_CPU_COLOR_FIX", "").strip().lower() not in {"1", "true", "yes", "on"}:
            self._patch_color_correction(device)

    @staticmethod
    def _patch_color_correction(device: str) -> None:
        """Run the engine's wavelet color correction on the GPU.

        The engine applies it to decoded segments that already sit in RAM, so it ran as CPU convolutions
        (about 5 s per 64-frame 720p segment). Same math in float32; chunks sized to about 64 MP."""
        from flashvsr_plus_engine.inference.postprocessing import color_correction as cc_mod
        from flashvsr_plus_engine.inference.runners import stream as stream_mod

        base = cc_mod.WaveletColorCorrector

        class GpuWaveletColorCorrector(base):
            @torch.no_grad()
            def __call__(self, generated, reference):
                dev = torch.device(device)
                frames_per_chunk = max(1, int(64e6 // max(1, generated.shape[-1] * generated.shape[-2] * 3)))
                out = torch.empty_like(generated)
                inner = base(levels=self.levels, chunk_size=min(self.chunk_size, frames_per_chunk))
                for s in range(0, generated.shape[2], frames_per_chunk):
                    e = min(s + frames_per_chunk, generated.shape[2])
                    out[:, :, s:e] = inner(generated[:, :, s:e].to(dev), reference[:, :, s:e].to(dev)).to(out.device)
                return out

        stream_mod.WaveletColorCorrector = GpuWaveletColorCorrector
        PlusEngine._color_corrector_cls = GpuWaveletColorCorrector

    @staticmethod
    def _patch_offload(offload: bool) -> None:
        """Keep the DiT on the GPU while the TC decoder runs a segment unless offloading was asked for."""
        from flashvsr_plus_engine.inference.runners import stream as stream_mod

        if not hasattr(stream_mod, "_orig_offload_dit"):
            stream_mod._orig_offload_dit = stream_mod._offload_dit
            stream_mod._orig_restore_dit = stream_mod._restore_dit
        if offload:
            stream_mod._offload_dit = stream_mod._orig_offload_dit
            stream_mod._restore_dit = stream_mod._orig_restore_dit
        else:
            def _keep(bundle, state, device):
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            stream_mod._offload_dit = _keep
            stream_mod._restore_dit = lambda bundle, state, device: None

    def _runner(self, *, decode_every, kv_mode: str, ffn_chunk: Optional[int]):
        import copy
        from flashvsr_plus_engine.inference.runners import StreamRunner

        cfg = copy.deepcopy(self.config)
        cfg.decoder.decode_every = decode_every
        cfg.stream_cache_offload_mode = kv_mode
        cfg.stream_ffn_chunk_tokens = ffn_chunk
        cfg.validate()
        return StreamRunner(cfg, self.bundle), cfg

    def _auto_kv_mode(self, th: int, tw: int) -> str:
        """'never' or 'always' from measured RTX 5090 peaks and the real budget (the engine's own policy reads
        free device memory, ignores the app's VRAM cap and on Windows spills into shared memory instead)."""
        mp = th * tw / 1e6
        slope_never, slope_always = (2.63, 2.10) if self.preset == "turbo" else (4.86, 2.40)
        weights = 2.6 if self.args.precision == "int8_convrot" else 3.6
        limit = os.environ.get("SECOURSES_VRAM_LIMIT_GB")
        try:
            budget = float(limit) - 0.7 if limit else torch.cuda.mem_get_info()[0] / 1e9 + torch.cuda.memory_reserved() / 1e9
        except Exception:
            budget = torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.9
        need_never = (weights + slope_never * mp) * 1.2 + 0.6
        layers = 5 if self.preset == "turbo" else 30
        pinned_gb = layers * 2 * (mp * 1e6 / 256 * 6) * 1536 * 2 / 1e9
        if need_never <= budget:
            return "never"
        try:
            import psutil

            ram_gb = psutil.virtual_memory().total / 1e9
        except Exception:
            ram_gb = 64.0
        return "always" if pinned_gb <= 0.25 * ram_gb else "never"

    def _ffn_chunk(self, th: int, tw: int) -> Optional[int]:
        req = int(getattr(self.args, "ffn_chunk_tokens", 0) or 0)
        if req < 0:
            return None
        tokens = 4 * (th // 16) * (tw // 16)
        if req > 0:
            return req if tokens > req else None
        return 32768 if tokens > 65536 else None  # auto: bound the 8960-wide FFN slab at 4K

    def _count_steps(self, total: Optional[int]):
        """Print 'Processing: k/N' per DiT stream step (the app's progress parser reads it)."""
        den = self.bundle.denoiser
        if hasattr(den, "_plus_orig_forward_stream"):
            den.forward_stream = den._plus_orig_forward_stream
        den._plus_orig_forward_stream = den.forward_stream
        counter = {"k": 0}

        def wrapped(*a, **kw):
            out = den._plus_orig_forward_stream(*a, **kw)
            counter["k"] += 1
            print(f"Processing: {counter['k']}/{total if total else '?'} [{time.time() - _T0:.1f}s]", flush=True)
            return out

        den.forward_stream = wrapped
        return counter

    # -- untiled: one stream over the whole clip -------------------------------------------------
    def run_stream(self, reader_factory, write: Callable[[torch.Tensor], None], total_frames: int,
                   kv_mode: str) -> int:
        from flashvsr_plus_engine.inference.config import TEMPORAL_DOWNSAMPLE
        from flashvsr_plus_engine.inference.preprocessing import TemporalVideoStream

        geo = {}
        scale = int(self.args.scale)

        def chunks():
            for frames in reader_factory(READ_CHUNK_FRAMES):
                video, crop = upscale_pad(frames, scale)
                geo.setdefault("crop", crop)
                geo.setdefault("hw", video.shape[-2:])
                yield video

        it = chunks()
        first = next(it, None)
        if first is None:
            raise RuntimeError("No input frames")

        def all_chunks():
            yield first
            yield from it

        th, tw = geo["hw"]
        if kv_mode == "auto":
            kv_mode = self._auto_kv_mode(th, tw)
        runner, cfg = self._runner(decode_every=int(self.args.decode_every), kv_mode=kv_mode,
                                   ffn_chunk=self._ffn_chunk(th, tw))
        stream = TemporalVideoStream(all_chunks(), cfg.temporal_compression, TEMPORAL_DOWNSAMPLE)
        steps = None
        if total_frames and total_frames > 0:
            tc = cfg.temporal_compression
            eff = ((total_frames - 1 + tc - 1) // tc) * tc + 1
            expanded = 1 + (eff - 1) * (TEMPORAL_DOWNSAMPLE // tc)
            latents = (expanded - 1) // TEMPORAL_DOWNSAMPLE + 1
            steps = (latents + 3) // 4
        self._count_steps(steps)
        sink = _Sink(geo["crop"], write)
        _log(f"[FlashVSR+] Stream {tw}x{th} (output {geo['crop'][3]}x{geo['crop'][2]}), KV cache {kv_mode}, "
             f"FFN chunk {cfg.stream_ffn_chunk_tokens or 'off'}")
        try:
            out = runner.run(stream, None, progress=lambda m: _log(f"[FlashVSR+] {m}"), segment_writer=sink)
        finally:
            stream.close()
        _log(f"[FlashVSR+] KV policy: {out.metadata.get('cache_offload_decision_reason')}, "
             f"offloaded layers {len(out.metadata.get('attention_cache_offload_layers') or [])}")
        return sink.frames

    # -- tiled: per frame chunk, every tile is its own stream -----------------------------------
    def run_tiled(self, reader_factory, write, tile: int, overlap: int, chunk_frames: int, kv_mode: str) -> int:
        from flashvsr_plus_engine.inference.config import TEMPORAL_DOWNSAMPLE
        from flashvsr_plus_engine.inference.preprocessing import expand_for_temporal_compression, merge_temporal_output

        scale = int(self.args.scale)
        written = 0
        for frames in reader_factory(max(1, int(chunk_frames))):
            n, h, w, _ = frames.shape
            coords = tile_coords(h, w, tile, overlap)
            _log(f"[FlashVSR+] Tiled DiT: {len(coords)} tiles of {tile}px (overlap {overlap}) for {n} frames")
            canvas = torch.zeros((n, h * scale, w * scale, 3), dtype=torch.float16)
            weights = torch.zeros((1, h * scale, w * scale, 1), dtype=torch.float32)
            for i, (x1, y1, x2, y2) in enumerate(coords):
                print(f"Processing Tiles: {i + 1}/{len(coords)}", flush=True)
                video, crop = upscale_pad(frames[:, y1:y2, x1:x2, :], scale)
                th, tw = video.shape[-2:]
                runner, cfg = self._runner(decode_every="auto", kv_mode=kv_mode, ffn_chunk=self._ffn_chunk(th, tw))
                expansion = expand_for_temporal_compression(video, cfg.temporal_compression,
                                                            model_compression=TEMPORAL_DOWNSAMPLE)
                self._count_steps(((expansion.expanded_frames - 1) // TEMPORAL_DOWNSAMPLE + 1 + 3) // 4)
                out = runner.run(expansion.video, None)
                result = merge_temporal_output(out.video, expansion, mode=cfg.temporal_merge,
                                               gaussian_sigma=cfg.temporal_gaussian_sigma)[:, :, :expansion.source_frames]
                if cfg.color_correction.enabled:
                    from flashvsr_plus_engine.inference.postprocessing import WaveletColorCorrector

                    corrector_cls = getattr(PlusEngine, "_color_corrector_cls", None) or WaveletColorCorrector
                    result = corrector_cls(levels=cfg.color_correction.levels,
                                                   chunk_size=cfg.color_correction.chunk_size)(
                        result, video[:, :, :result.shape[2]].float())
                pt, pl, sh, sw = crop
                tile_rgb = result[0, :, :, pt:pt + sh, pl:pl + sw].float().add_(1).mul_(0.5).clamp_(0, 1)
                tile_rgb = tile_rgb.permute(1, 2, 3, 0).cpu()
                edges = (x1 > 0, x2 < w, y1 > 0, y2 < h)
                m = feather_mask(sh, sw, overlap * scale, edges).view(1, sh, sw, 1)
                oy, ox = y1 * scale, x1 * scale
                canvas[:, oy:oy + sh, ox:ox + sw] += (tile_rgb * m).to(torch.float16)
                weights[:, oy:oy + sh, ox:ox + sw] += m
                del video, out, result, tile_rgb, expansion
                torch.cuda.empty_cache()
            inv = 1.0 / weights.clamp_min(1e-6)
            for s in range(0, n, 16):
                write(canvas[s:s + 16].float().mul_(inv))
            written += n
            del canvas, weights
            gc.collect()
        return written


def _is_oom(error: BaseException) -> bool:
    """CUDA allocator OOM, or a failed pinned-RAM / driver allocation (AcceleratorError 'out of memory')."""
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def oom_ladder(tiled: bool, tile: int, overlap: int, kv_mode: str):
    """Next lower-VRAM setting after an out-of-memory error, or None."""
    if not tiled and kv_mode != "always":
        return False, tile, overlap, "always", "Offloading the KV cache of the full-attention layers to RAM"
    if not tiled:
        return True, 512, min(64, max(overlap, 32)), "auto", "Auto-enabling Tiled DiT (512 px)"
    if tile > 128:
        new = max(128, tile // 2)
        return True, new, min(overlap, max(8, new // 4)), "auto", f"Reducing DiT tile size to {new}"
    return None


def run_plus(args, VideoReader, make_writer, output_fps: float, total_frames: int) -> int:
    """Entry point from cli_main for the FlashVSR+ models. Returns the process exit code."""
    device = args.device
    if device == "auto":
        device = "cuda:0"
    if not torch.cuda.is_available():
        _log("FlashVSR+ needs an NVIDIA GPU (CUDA).")
        return 1
    torch.cuda.set_device(device)
    if args.precision in ("auto", "int8_convrot", "bf16"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float16
    engine = PlusEngine(args, device, dtype)

    state = {"writer": None}

    def write(frames: torch.Tensor) -> None:
        if state["writer"] is None:
            h, w = int(frames.shape[1]), int(frames.shape[2])
            print(f"Output dimensions: {w}x{h}")
            print(f"Saving output video to: {args.output}")
            state["writer"] = make_writer(w, h)
        state["writer"].write(frames)

    def reader_factory(chunk):
        return VideoReader(args.input, start_frame=args.start_frame, end_frame=args.end_frame, chunk_size=chunk)

    tiled = bool(args.tiled_dit)
    tile, overlap = int(args.tile_size), int(args.tile_overlap)
    kv_mode = args.plus_kv_offload
    chunk_frames = int(args.frame_chunk_size) if int(args.frame_chunk_size) > 0 else 241
    disable_recovery = os.environ.get("FLASHVSR_DISABLE_OOM_RECOVERY", "").strip().lower() in {"1", "true", "yes", "on"}
    t0 = time.time()
    produced = 0
    error = None
    while True:
        try:
            torch.cuda.reset_peak_memory_stats()
            if tiled:
                produced = engine.run_tiled(reader_factory, write, tile, overlap, chunk_frames, kv_mode)
            else:
                produced = engine.run_stream(reader_factory, write, total_frames, kv_mode)
            error = None
            break
        except Exception as e:  # noqa: BLE001 - out-of-memory is retried below, anything else is reported
            # Keep only (type, message, is_oom): holding the exception would keep its frames' tensors in VRAM.
            error = (type(e).__name__, (str(e).splitlines() or [""])[0], _is_oom(e))
            if not error[2]:
                import traceback

                traceback.print_exc()
                break
            e.__traceback__ = None
            del e
            if state["writer"] is not None:
                try:
                    state["writer"].release()
                except Exception:
                    pass
                state["writer"] = None
                try:
                    Path(args.output).unlink(missing_ok=True)
                except Exception:
                    pass
            gc.collect()
            torch.cuda.empty_cache()
            step = None if disable_recovery else oom_ladder(tiled, tile, overlap, kv_mode)
            if step is None:
                _log("Still out of memory at the smallest tiles. Cannot recover.")
                break
            tiled, tile, overlap, kv_mode, what = step
            _log(f"OOM detected ({error[0]}). {what} to stay inside the VRAM limit (no shared GPU memory), "
                 f"restarting the clip...")
    if state["writer"] is not None:
        print("[VideoWriter] finalize begin")
        state["writer"].release()
        print("[VideoWriter] finalize done")
    dur = time.time() - t0
    peak = torch.cuda.max_memory_reserved() / 1024 ** 3
    peak_alloc = torch.cuda.max_memory_allocated() / 1024 ** 3
    if error is not None:
        name, message, is_oom = error
        print(f"FlashVSR stopped after {produced} frame(s): " + (
            "CUDA out of memory (VRAM limit reached)." if is_oom else f"{name}: {message}"))
        return 3 if is_oom else 1
    print("\n" + "=" * 60)
    print("FlashVSR processing complete!")
    print(f"Total Frames Processed: {produced}" + (f"/{total_frames}" if total_frames > 0 else ""))
    print(f"Total Time: {dur:.1f}s ({produced / dur if dur > 0 else 0:.2f} FPS), peak VRAM reserved {peak:.2f} GB, "
          f"allocated {peak_alloc:.2f} GB")
    print("=" * 60)
    return 0
