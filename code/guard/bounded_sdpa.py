"""Bound CPU float32 attention workspace by splitting only the query rows.

Every chunk retains all original keys and values. No token, model weight, mask
entry, or attention scale is dropped. Each chunk uses PyTorch's math SDPA backend;
floating-point rounding may differ from an unchunked call. This adapter is for
inference with zero dropout and four-dimensional dense attention tensors.
"""

from __future__ import annotations

from functools import wraps
import os

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def _validate_mask(query, key, value, attention_mask, enable_gqa):
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("Bounded SDPA requires four-dimensional query, key, and value tensors")
    if query.shape[-1] != key.shape[-1] or key.shape[-2] != value.shape[-2]:
        raise ValueError("Bounded SDPA requires matching query/key dimensions and key/value lengths")
    if any(tensor.shape[-3] == 0 for tensor in (query, key, value)):
        raise ValueError("Bounded SDPA requires at least one attention head")
    if enable_gqa:
        if key.shape[-3] != value.shape[-3] or query.shape[-3] % key.shape[-3]:
            raise ValueError("GQA requires equal key/value head counts dividing the query head count")
        heads = query.shape[-3]
    else:
        try:
            heads = torch.broadcast_shapes(query.shape[-3:-2], key.shape[-3:-2], value.shape[-3:-2])[0]
        except RuntimeError as error:
            raise ValueError("Attention head counts do not broadcast") from error
    try:
        batch = torch.broadcast_shapes(query.shape[:1], key.shape[:1], value.shape[:1])[0]
    except RuntimeError as error:
        raise ValueError("Attention batch dimensions do not broadcast") from error
    if attention_mask is None:
        return
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("The attention mask must be a tensor or None")
    if attention_mask.ndim not in (2, 3, 4):
        raise ValueError("Bounded SDPA supports only 2D, 3D, or 4D broadcastable masks")
    if attention_mask.device != query.device:
        raise ValueError("The attention mask must be on the query device")
    if attention_mask.dtype not in (torch.bool, query.dtype):
        raise ValueError("The attention mask must be boolean or match the query dtype")
    target = (batch, heads, query.shape[-2], key.shape[-2])
    try:
        broadcast = torch.broadcast_shapes(attention_mask.shape, target)
    except RuntimeError as error:
        raise ValueError("The attention mask does not broadcast to the original attention matrix") from error
    if tuple(broadcast) != target:
        raise ValueError("The attention mask must not expand the original attention dimensions")


def make_bounded_sdpa(original, *, chunk_size=256, threshold=4096):
    """Create an additive SDPA adapter; threshold zero forces diagnostic chunking."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold < 0:
        raise ValueError("threshold must be a nonnegative integer")

    @wraps(original)
    def bounded(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False):
        if (
            any(tensor.device.type != "cpu" or tensor.dtype != torch.float32 for tensor in (query, key, value))
            or dropout_p != 0.0
            or query.ndim < 2
            or query.shape[-2] <= threshold
        ):
            return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                            is_causal=is_causal, scale=scale, enable_gqa=enable_gqa)
        _validate_mask(query, key, value, attn_mask, enable_gqa)
        if is_causal and attn_mask is not None:
            raise RuntimeError("_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True")
        if key.shape[-2] == 0:
            raise ValueError("Bounded SDPA requires a nonempty key/value sequence")
        outputs = []
        key_positions = torch.arange(key.shape[-2], device=query.device) if is_causal else None
        with sdpa_kernel(SDPBackend.MATH):
            for start in range(0, query.shape[-2], chunk_size):
                end = min(start + chunk_size, query.shape[-2])
                if is_causal:
                    query_positions = torch.arange(start, end, device=query.device)
                    chunk_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                elif attn_mask is None or attn_mask.shape[-2] == 1:
                    chunk_mask = attn_mask
                else:
                    chunk_mask = attn_mask[..., start:end, :]
                outputs.append(original(
                    query[..., start:end, :], key, value, attn_mask=chunk_mask,
                    dropout_p=dropout_p, is_causal=False, scale=scale, enable_gqa=enable_gqa,
                ))
        return torch.cat(outputs, dim=-2)

    bounded._jev_bounded_original = original
    bounded._jev_bounded_settings = {"chunk_size": chunk_size, "threshold": threshold}
    return bounded


def install_bounded_sdpa():
    """Install the adapter in this process using explicit environment settings."""
    chunk_size = int(os.environ.get("JEV_SDPA_CHUNK_SIZE", "256"))
    threshold = int(os.environ.get("JEV_SDPA_CHUNK_THRESHOLD", "4096"))
    original = getattr(F.scaled_dot_product_attention, "_jev_bounded_original", F.scaled_dot_product_attention)
    F.scaled_dot_product_attention = make_bounded_sdpa(original, chunk_size=chunk_size, threshold=threshold)
    return {"implementation": "query_chunked_cpu_float32_math_sdpa", "chunk_size": chunk_size,
            "threshold": threshold, "full_key_value_context": True, "truncation": False}
