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
