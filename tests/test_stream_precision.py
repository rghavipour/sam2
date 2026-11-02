# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Storing matmul weights in the autocast dtype once instead of casting them per call."""

import torch

from stream_helpers import (
    BOXES,
    copy_weights,
    run_stream,
    TINY_CONFIG,
    TINY_OVERRIDES,
    VIDEO_HW,
)


def _fresh(stream_predictor):
    from sam2.build_sam import build_sam2_stream_predictor

    p = build_sam2_stream_predictor(
        TINY_CONFIG, device="cpu", hydra_overrides_extra=TINY_OVERRIDES
    )
    return copy_weights(p, stream_predictor)


def test_casts_only_matmul_weights(stream_predictor):
    from sam2.stream.precision import cast_matmul_weights

    p = _fresh(stream_predictor)
    before = sum(t.numel() * t.element_size() for t in p.parameters())
    n = cast_matmul_weights(p, torch.bfloat16)
    after = sum(t.numel() * t.element_size() for t in p.parameters())
    assert n > 0
    assert after < 0.6 * before
    for m in p.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            assert m.weight.dtype == torch.bfloat16
        if isinstance(m, torch.nn.LayerNorm):
            assert m.weight.dtype == torch.float32
    # parameters used outside matmul modules stay as they are
    assert p.maskmem_tpos_enc.dtype == torch.float32
    assert p.no_mem_embed.dtype == torch.float32


def test_same_output_under_autocast(stream_predictor, upstream_images):
    from sam2.stream.precision import cast_matmul_weights

    cast = _fresh(stream_predictor)
    cast_matmul_weights(cast, torch.bfloat16)
    schedule = {0: [("add_prompt", {"obj_id": k, "box": BOXES[k]}) for k in (1, 2)]}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ref, _ = run_stream(
            stream_predictor, upstream_images, VIDEO_HW, schedule, num_frames=8
        )
        got, _ = run_stream(cast, upstream_images, VIDEO_HW, schedule, num_frames=8)
    for t in ref:
        torch.testing.assert_close(
            got[t].mask_logits, ref[t].mask_logits, atol=1e-5, rtol=0
        )


def _rope_inputs(dtype):
    from sam2.modeling.position_encoding import compute_axial_cis

    g = torch.Generator().manual_seed(0)
    freqs = compute_axial_cis(dim=16, end_x=4, end_y=4)  # [16, 8]
    q = torch.randn(2, 1, 16, 16, generator=g).to(dtype)
    k = torch.randn(2, 1, 48, 16, generator=g).to(dtype)  # 3 frames of 16 tokens
    return q, k, freqs


def test_low_precision_rotary_matches_the_fp32_path(monkeypatch):
    from sam2.modeling.position_encoding import apply_rotary_enc

    q, k, freqs = _rope_inputs(torch.bfloat16)
    monkeypatch.setenv("SAM2_LOWP_ATTENTION", "0")
    ref_q, ref_k = apply_rotary_enc(q, k, freqs, repeat_freqs_k=True)
    monkeypatch.setenv("SAM2_LOWP_ATTENTION", "1")
    got_q, got_k = apply_rotary_enc(q, k, freqs, repeat_freqs_k=True)
    assert got_q.dtype == got_k.dtype == torch.bfloat16
    assert got_k.shape == ref_k.shape
    # a few bf16 roundings apart (values are about unit size)
    assert (got_q.float() - ref_q.float()).abs().max() < 0.05
    assert (got_k.float() - ref_k.float()).abs().max() < 0.05
    # the repeated frames get the same rotation as the first
    torch.testing.assert_close(
        apply_rotary_enc(q, k[:, :, 16:32], freqs)[1], got_k[:, :, 16:32]
    )


def test_fp32_inputs_keep_the_fp32_path(monkeypatch):
    from sam2.modeling.position_encoding import apply_rotary_enc

    q, k, freqs = _rope_inputs(torch.float32)
    monkeypatch.setenv("SAM2_LOWP_ATTENTION", "1")
    on = apply_rotary_enc(q, k, freqs, repeat_freqs_k=True)
    monkeypatch.setenv("SAM2_LOWP_ATTENTION", "0")
    off = apply_rotary_enc(q, k, freqs, repeat_freqs_k=True)
    torch.testing.assert_close(on[0], off[0], rtol=0, atol=0)
    torch.testing.assert_close(on[1], off[1], rtol=0, atol=0)


def test_autocast_dtype():
    from sam2.stream.precision import autocast_dtype

    assert autocast_dtype("cpu") is None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert autocast_dtype("cpu") == torch.bfloat16


def test_low_precision_attention_keeps_the_masks(stream_predictor, upstream_images, monkeypatch):
    from sam2.stream.precision import cast_matmul_weights

    p = _fresh(stream_predictor)
    cast_matmul_weights(p, torch.bfloat16)
    schedule = {0: [("add_prompt", {"obj_id": k, "box": BOXES[k]}) for k in (1, 2)]}
    runs = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("SAM2_LOWP_ATTENTION", flag)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            runs[flag], _ = run_stream(p, upstream_images, VIDEO_HW, schedule, num_frames=8)
    agree = []
    for t in runs["0"]:
        for a, b in zip(runs["0"][t].mask_logits > 0, runs["1"][t].mask_logits > 0):
            union = (a | b).sum().item()
            agree.append((a & b).sum().item() / union if union else 1.0)
    assert min(agree) > 0.9, agree
