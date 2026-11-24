"""Small fixed-seed numerical checks; no model loading, service, or API calls."""

import argparse
import ast
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import bounded_sdpa


def _peak_working_set():
    if os.name != "nt":
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("page_faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile",
            )
        ]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if ctypes.WinDLL("psapi").GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(counters), counters.cb):
        return counters.peak_working_set
    return None


def validate():
    torch.set_num_threads(4)
    torch.manual_seed(20261117)
    original = F.scaled_dot_product_attention
    bounded = bounded_sdpa.make_bounded_sdpa(original, chunk_size=7, threshold=0)
    q = torch.randn(2, 4, 37, 16)
    k = torch.randn(2, 2, 41, 16)
    v = torch.randn(2, 2, 41, 20)
    boolean = torch.rand(37, 41) > 0.25
    boolean[9] = False
    additive = torch.randn(37, 41) * 0.2
    additive[9] = -torch.inf
    finite_min = additive.clone()
    finite_min[9] = torch.finfo(torch.float32).min
    bool_broadcast = torch.rand(2, 1, 1, 41) > 0.25
    float_3d = torch.randn(4, 37, 41) * 0.2
    cases = [
        ("gqa_noncausal", q, k, v, {}),
        ("gqa_causal_rectangular", q, k, v, {"is_causal": True}),
        ("gqa_causal_square", q, k[..., :37, :], v[..., :37, :], {"is_causal": True}),
        ("gqa_causal_more_queries_than_keys", q, k[..., :29, :], v[..., :29, :], {"is_causal": True}),
        ("custom_scale", q, k, v, {"is_causal": True, "scale": 0.1875}),
        ("zero_scale", q, k, v, {"scale": 0.0}),
        ("boolean_mask_fully_masked_row", q, k, v, {"attn_mask": boolean}),
        ("additive_mask_fully_masked_row", q, k, v, {"attn_mask": additive}),
        ("finite_minimum_mask_row", q, k, v, {"attn_mask": finite_min}),
        ("boolean_4d_broadcast", q, k, v, {"attn_mask": bool_broadcast}),
        ("additive_3d_head_mask", q, k, v, {"attn_mask": float_3d}),
        ("boolean_2d_query_broadcast", q, k, v, {"attn_mask": boolean[:1]}),
        ("boolean_2d_key_broadcast", q, k, v, {"attn_mask": boolean[:, :1]}),
        ("gqa_batch_broadcast", q, k[:1], v[:1], {"attn_mask": boolean}),
        ("nongqa_equal_heads", q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), {}),
        ("nongqa_head_broadcast", q[:, :1], k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), {}),
        ("model_head_shape", torch.randn(1, 8, 73, 256), torch.randn(1, 2, 73, 256),
         torch.randn(1, 2, 73, 256), {"is_causal": True}),
    ]
    results = []
    with torch.inference_mode():
        for name, query, key, value, options in cases:
            options = dict(options, enable_gqa=query.shape[1] != key.shape[1] and query.shape[1] > key.shape[1])
            expected = original(query, key, value, **options)
            with sdpa_kernel(SDPBackend.MATH):
                expected_math = original(query, key, value, **options)
            actual = bounded(query, key, value, **options)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(actual, expected_math, rtol=2e-5, atol=2e-6)
            result = {
                "case": name, "query_shape": list(query.shape), "key_shape": list(key.shape),
                "max_absolute_error_vs_original_dispatch": float((actual - expected).abs().max()),
                "max_absolute_error_vs_original_math": float((actual - expected_math).abs().max()),
                "exact_vs_original_dispatch": bool(torch.equal(actual, expected)),
                "exact_vs_original_math": bool(torch.equal(actual, expected_math)),
                "all_finite": bool(actual.isfinite().all()),
            }
            if name in ("boolean_mask_fully_masked_row", "additive_mask_fully_masked_row"):
                assert torch.count_nonzero(actual[..., 9, :]).item() == 0
                result["fully_masked_row_is_exact_zero"] = True
            if name == "finite_minimum_mask_row":
                result["finite_minimum_row_max_abs"] = float(actual[..., 9, :].abs().max())
                assert result["finite_minimum_row_max_abs"] > 0
            results.append(result)

        long_q = torch.randn(1, 4, 4097, 8)
        short_kv = torch.randn(1, 2, 17, 8)
        default = bounded_sdpa.make_bounded_sdpa(original)
        expected = original(long_q, short_kv, short_kv, enable_gqa=True, is_causal=True)
        actual = default(long_q, short_kv, short_kv, enable_gqa=True, is_causal=True)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        results.append({"case": "default_threshold_and_nondivisible_256_chunk", "query_shape": list(long_q.shape),
                        "key_shape": list(short_kv.shape),
                        "max_absolute_error_vs_original_dispatch": float((actual - expected).abs().max()),
                        "exact_vs_original_dispatch": bool(torch.equal(actual, expected))})

    structural = []
    for name, query, key, value, options in [
        ("short", q, k, v, {}),
        ("nonfloat32", q.double(), k.double(), v.double(), {}),
        ("noncpu", torch.empty_like(q, device="meta"), torch.empty_like(k, device="meta"), torch.empty_like(v, device="meta"), {}),
        ("dropout", q, k, v, {"dropout_p": 0.1}),
    ]:
        spy = Mock(return_value=object())
        wrapper = bounded_sdpa.make_bounded_sdpa(spy, threshold=4096 if name == "short" else 0)
        assert wrapper(query, key, value, **options) is spy.return_value
        spy.assert_called_once()
        assert spy.call_args.args == (query, key, value)
        structural.append("original_delegation_" + name)

    calls = []
    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)
    wrapped = bounded_sdpa.make_bounded_sdpa(spy, chunk_size=7, threshold=0)
    with torch.inference_mode():
        wrapped(q, k, v, is_causal=True, enable_gqa=True, scale=0.1875)
    assert len(calls) == 6
    for index, (args, kwargs) in enumerate(calls):
        assert args[1] is k and args[2] is v
        assert kwargs["enable_gqa"] is True and kwargs["scale"] == 0.1875
        assert kwargs["is_causal"] is False
        start = index * 7
        allowed = torch.arange(41).unsqueeze(0) <= torch.arange(start, start + args[0].shape[-2]).unsqueeze(1)
        assert torch.equal(kwargs["attn_mask"], allowed)
    structural.append("full_key_value_identity_scale_gqa_and_global_causal_offsets")

    for mask in (torch.ones(41), torch.ones(36, 41), torch.ones(3, 4, 37, 41), torch.ones(37, 41, dtype=torch.int64)):
        spy = Mock()
        wrapper = bounded_sdpa.make_bounded_sdpa(spy, threshold=0)
        try:
            wrapper(q, k, v, attn_mask=mask, enable_gqa=True)
            raise AssertionError("Unsupported mask was accepted")
        except ValueError:
            pass
        spy.assert_not_called()
    structural.append("unsupported_masks_rejected_before_attention")

    for function in (original, bounded):
        try:
            function(q, k, v, attn_mask=boolean, is_causal=True, enable_gqa=True)
            raise AssertionError("Explicit mask plus causal flag was accepted")
        except RuntimeError as error:
            assert "Explicit attn_mask" in str(error)
    structural.append("original_explicit_mask_plus_causal_error_preserved")

    with patch.object(F, "scaled_dot_product_attention", original), patch.dict(os.environ, {"JEV_SDPA_CHUNK_SIZE": "7", "JEV_SDPA_CHUNK_THRESHOLD": "0"}):
        settings = bounded_sdpa.install_bounded_sdpa()
        assert settings["chunk_size"] == 7 and settings["threshold"] == 0
        bounded_sdpa.install_bounded_sdpa()
        assert F.scaled_dot_product_attention._jev_bounded_original is original
    assert F.scaled_dot_product_attention is original
    structural.append("environment_configuration_idempotent_install_and_test_restoration")

    for filename in ("bounded_sdpa.py", "serve_bounded.py"):
        ast.parse(Path(__file__).with_name(filename).read_text(encoding="utf-8"))
    structural.append("wrapper_and_launcher_syntax")
    peak = _peak_working_set()
    assert peak is None or peak < 500 * 1024 * 1024, peak
    return {
        "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__, "threads": torch.get_num_threads(), "seed": 20261117,
        "rtol": 2e-5, "atol": 2e-6, "numerical_cases": results, "structural_checks": structural,
        "max_absolute_error_vs_original_dispatch": max(x["max_absolute_error_vs_original_dispatch"] for x in results),
        "all_exact_vs_original_dispatch": all(x["exact_vs_original_dispatch"] for x in results),
        "peak_working_set_bytes": peak, "model_or_api_calls": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    report = validate()
    rendered = json.dumps(report, indent=2) + "\n"
    if arguments.report:
        with arguments.report.open("x", encoding="utf-8") as output:
            output.write(rendered)
    print(rendered)
