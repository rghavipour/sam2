# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Non-overlap constraint across the objects of a session (memory and output)."""

import pytest
import torch

from stream_helpers import (
    BOXES,
    copy_weights,
    iou,
    run_stream,
    TINY_CONFIG,
    TINY_OVERRIDES,
    upstream_masks,
    VIDEO_HW,
)

NON_OVERLAP = ["++model.non_overlap_masks_for_mem_enc=true"]


@pytest.fixture(scope="module")
def stream_non_overlap(stream_predictor):
    from sam2.build_sam import build_sam2_stream_predictor

    p = build_sam2_stream_predictor(
        TINY_CONFIG, device="cpu", hydra_overrides_extra=TINY_OVERRIDES + NON_OVERLAP
    )
    return copy_weights(p, stream_predictor)


@pytest.fixture(scope="module")
def legacy_non_overlap(stream_predictor):
    from sam2.build_sam import build_sam2_video_predictor

    p = build_sam2_video_predictor(
        TINY_CONFIG,
        device="cpu",
        hydra_overrides_extra=TINY_OVERRIDES
        + NON_OVERLAP
        + ["++model._target_=sam2.sam2_video_predictor_legacy.SAM2VideoPredictor"],
    )
    return copy_weights(p, stream_predictor)


def _three_at_zero():
    return {0: [("add_prompt", {"obj_id": k, "box": BOXES[k]}) for k in (1, 2, 3)]}


def test_flag_is_applied_across_the_session(stream_non_overlap, stream_predictor):
    assert stream_non_overlap.non_overlap_across_objects is True
    # SAM2Base must not apply it on its own, only within one batch of rows
    assert stream_non_overlap.non_overlap_masks_for_mem_enc is False
    assert stream_predictor.non_overlap_across_objects is False


def test_rows_become_disjoint_against_each_other_and_competitors(stream_predictor):
    from sam2.sam2_stream_predictor import non_overlap_rows

    S = 16
    rows = torch.full((2, 1, S, S), 5.0)
    rows[1] = 3.0  # row 0 wins everywhere it competes with row 1
    others = torch.full((1, 1, S, S), -5.0)
    others[..., :, : S // 2] = 9.0  # a competitor wins the left half
    out = non_overlap_rows(stream_predictor, rows, others)
    assert out.shape == rows.shape
    positive = torch.cat([out > 0, others > 0], dim=0).sum(dim=0)
    assert int(positive.max()) == 1
    assert (out[0, 0, :, S // 2 :] > 0).all()
    assert (out[0, 0, :, : S // 2] <= 0).all()
    assert (out[1] <= 0).all()
    # nothing to compete with: unchanged
    torch.testing.assert_close(
        non_overlap_rows(stream_predictor, rows[:1], None), rows[:1]
    )


def test_matches_legacy_with_non_overlap(
    stream_non_overlap, legacy_non_overlap, frames_dir, upstream_images
):
    with torch.inference_mode():
        state = legacy_non_overlap.init_state(frames_dir)
        for k in (1, 2, 3):
            legacy_non_overlap.add_new_points_or_box(state, 0, k, box=BOXES[k])
        expected = upstream_masks(legacy_non_overlap, state)
    outs, _ = run_stream(
        stream_non_overlap, upstream_images, VIDEO_HW, _three_at_zero()
    )
    for t, masks in expected.items():
        torch.testing.assert_close(outs[t].mask_logits, masks, atol=1e-3, rtol=0)
        for row in range(3):
            assert iou(outs[t].mask_logits[row], masks[row]) >= 0.99


def test_constraint_changes_the_memory(
    stream_non_overlap, stream_predictor, upstream_images
):
    on, _ = run_stream(stream_non_overlap, upstream_images, VIDEO_HW, _three_at_zero())
    off, _ = run_stream(stream_predictor, upstream_images, VIDEO_HW, _three_at_zero())
    # frame 0 is the prompt output itself; later frames read the constrained memory
    torch.testing.assert_close(on[0].mask_logits, off[0].mask_logits)
    diff = max((on[t].mask_logits - off[t].mask_logits).abs().max().item() for t in on)
    assert diff > 1e-3
