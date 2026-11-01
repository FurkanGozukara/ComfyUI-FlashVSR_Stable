from __future__ import annotations

import math

import torch


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    if dim % 2:
        raise ValueError("Sinusoidal embedding dimension must be even")
    # FlashVSR 1.1/Wan computes this in float64 and casts back to the
    # timestep dtype. Keeping that detail matters for BF16 checkpoint parity.
    frequencies = torch.outer(
        position.to(torch.float64).flatten(),
        torch.pow(
            10000.0,
            -torch.arange(dim // 2, device=position.device, dtype=torch.float64)
            / (dim // 2),
        ),
    )
    return torch.cat([frequencies.cos(), frequencies.sin()], dim=-1).to(position.dtype)


def _axis_pair_counts(num_pairs: int) -> tuple[int, int, int]:
    base = num_pairs // 3
    return num_pairs - 2 * base, base, base


def rope_angles_3d(
    shape: tuple[int, int, int],
    head_dim: int,
    device: torch.device,
    theta: float = 10000.0,
    temporal_offset: int = 0,
) -> torch.Tensor:
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    frames, height, width = shape
    pair_counts = _axis_pair_counts(head_dim // 2)
    axes = (
        torch.arange(
            temporal_offset, temporal_offset + frames, device=device
        ).view(frames, 1, 1).expand(frames, height, width),
        torch.arange(height, device=device).view(1, height, 1).expand(frames, height, width),
        torch.arange(width, device=device).view(1, 1, width).expand(frames, height, width),
    )
    chunks = []
    for positions, pairs in zip(axes, pair_counts):
        inverse = theta ** (-torch.arange(pairs, device=device).float() / max(pairs, 1))
        chunks.append(positions.reshape(-1, 1).float() * inverse)
    return torch.cat(chunks, dim=-1)


try:
    import comfy_kitchen as _ck
except Exception:  # optional fused kernels
    _ck = None

_ROPE_CACHE: dict = {}


def _rope_freqs_cis_3d(shape, head_dim: int, device: torch.device, temporal_offset: int) -> torch.Tensor:
    """[1, 1, tokens, pairs, 2, 2] float32 rotation matrices, cached per window shape and offset.

    The angles are float32 (rope_angles_3d), so the old float64 rotation added no precision: it ran on
    GeForce's 1/64-rate FP64 units and rebuilt the table twice per layer (about 0.3 s per 1080p step)."""
    key = (tuple(shape), int(head_dim), device, int(temporal_offset))
    freqs = _ROPE_CACHE.get(key)
    if freqs is None:
        angles = rope_angles_3d(shape, head_dim, device, temporal_offset=temporal_offset)
        cos, sin = angles.cos(), angles.sin()
        freqs = torch.stack((cos, -sin, sin, cos), dim=-1).reshape(1, 1, angles.shape[0], angles.shape[1], 2, 2)
        if len(_ROPE_CACHE) >= 16:
            _ROPE_CACHE.clear()
        _ROPE_CACHE[key] = freqs
    return freqs


def apply_rope_3d(
    x: torch.Tensor,
    shape: tuple[int, int, int],
    *,
    temporal_offset: int = 0,
) -> torch.Tensor:
    freqs = _rope_freqs_cis_3d(shape, x.shape[-1], x.device, temporal_offset)
    if _ck is not None:
        # comfy_kitchen's fused CUDA RoPE (same interleaved pairs, float32 math): 24-38x faster than the
        # float64 version; differs from it only in the last BF16 bit of ~0.002 % of the values.
        try:
            return _ck.apply_rope1(x, freqs)
        except Exception:
            pass
    x_float = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    rotated = freqs[..., 0] * x_float[..., 0] + freqs[..., 1] * x_float[..., 1]
    return rotated.reshape(x.shape).to(dtype=x.dtype)
