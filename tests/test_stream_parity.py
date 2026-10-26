# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Parity of SAM2StreamPredictor with upstream SAM2VideoPredictor (and legacy)."""

import torch

from stream_helpers import BOXES, iou, run_stream, upstream_masks, VIDEO_HW

# object 3 moves 3 px/frame to the left; its box at frame 8
BOX_3_AT_8 = [96, 150, 126, 200]
# a positive click inside object 1 at frame 10, a negative click inside it at frame 12
CLICK_1_AT_10 = [[70, 35]]
CLICK_1_AT_12 = [[60, 25]]


def _box(obj_id, box=None):
    return ("add_prompt", {"obj_id": obj_id, "box": box or BOXES[obj_id]})


def _assert_close(stream_out, row, expected):
    torch.testing.assert_close(
        stream_out.mask_logits[row : row + 1], expected, atol=1e-4, rtol=0
    )


def _assert_iou(stream_out, row, expected, threshold=0.99):
    assert iou(stream_out.mask_logits[row], expected[0]) >= threshold


@torch.inference_mode()
def test_single_object_matches_upstream(
    stream_predictor_unbatched, video_predictor, frames_dir, upstream_images
):
    state = video_predictor.init_state(frames_dir)
    video_predictor.add_new_points_or_box(state, 0, 1, box=BOXES[1])
    expected = upstream_masks(video_predictor, state)

    outs, _ = run_stream(
        stream_predictor_unbatched, upstream_images, VIDEO_HW, {0: [_box(1)]}
    )
    assert sorted(expected) == sorted(outs)
    for t, masks in expected.items():
        _assert_close(outs[t], 0, masks)


@torch.inference_mode()
def test_three_objects_match_upstream(
    stream_predictor, video_predictor, frames_dir, upstream_images
):
    state = video_predictor.init_state(frames_dir)
    for k in (1, 2, 3):
        video_predictor.add_new_points_or_box(state, 0, k, box=BOXES[k])
    expected = upstream_masks(video_predictor, state)

    outs, _ = run_stream(
        stream_predictor, upstream_images, VIDEO_HW, {0: [_box(1), _box(2), _box(3)]}
    )
    for t, masks in expected.items():
        assert outs[t].obj_ids == [1, 2, 3]
        for row in range(3):
            _assert_iou(outs[t], row, masks[row : row + 1])


@torch.inference_mode()
def test_late_object_matches_single_upstream_session(
    stream_predictor,
    stream_predictor_unbatched,
    video_predictor,
    frames_dir,
    upstream_images,
):
    state = video_predictor.init_state(frames_dir)
    video_predictor.add_new_points_or_box(state, 8, 3, box=BOX_3_AT_8)
    expected = upstream_masks(video_predictor, state, start_frame_idx=8)
    assert sorted(expected) == list(range(8, 24))

    schedule = {0: [_box(1), _box(2)], 8: [_box(3, BOX_3_AT_8)]}
    outs_u, _ = run_stream(
        stream_predictor_unbatched, upstream_images, VIDEO_HW, schedule
    )
    outs_b, _ = run_stream(stream_predictor, upstream_images, VIDEO_HW, schedule)
    for t, masks in expected.items():
        assert outs_u[t].obj_ids == [1, 2, 3]
        _assert_close(outs_u[t], 2, masks)
        _assert_iou(outs_b[t], 2, masks)


@torch.inference_mode()
def test_reprompt_on_new_frame_matches_upstream(
    stream_predictor_unbatched, video_predictor, frames_dir, upstream_images
):
    state = video_predictor.init_state(frames_dir)
    video_predictor.add_new_points_or_box(state, 0, 1, box=BOXES[1])
    first = upstream_masks_limited(video_predictor, state, 9)
    assert sorted(first) == list(range(10))
    video_predictor.add_new_points_or_box(
        state, 10, 1, points=CLICK_1_AT_10, labels=[1]
    )
    expected = upstream_masks(video_predictor, state, start_frame_idx=10)

    schedule = {
        0: [_box(1)],
        10: [("add_prompt", {"obj_id": 1, "points": CLICK_1_AT_10, "labels": [1]})],
    }
    outs, session = run_stream(
        stream_predictor_unbatched, upstream_images, VIDEO_HW, schedule
    )
    assert sorted(session.objects[1].cond_outputs) == [0, 10]
    for t in range(10):
        _assert_close(outs[t], 0, first[t])
    for t in range(10, 24):
        _assert_close(outs[t], 0, expected[t])


@torch.inference_mode()
def test_correction_matches_legacy(
    stream_predictor_unbatched, legacy_predictor, frames_dir, upstream_images
):
    state = legacy_predictor.init_state(frames_dir)
    legacy_predictor.add_new_points_or_box(state, 0, 1, box=BOXES[1])
    upstream_masks(legacy_predictor, state)
    legacy_predictor.add_new_points_or_box(
        state, 12, 1, points=CLICK_1_AT_12, labels=[0]
    )
    expected = upstream_masks(legacy_predictor, state, start_frame_idx=12)

    p = stream_predictor_unbatched
    session = p.create_session()
    outs = {}
    for t in range(24):
        p.set_preprocessed_frame(session, upstream_images[t], VIDEO_HW)
        if t == 0:
            p.add_prompt(session, 1, box=BOXES[1])
        outs[t] = p.track(session)
        if t == 12:
            # the correction comes after the frame was tracked
            outs[t] = p.add_prompt(session, 1, points=CLICK_1_AT_12, labels=[0])
    assert sorted(session.objects[1].cond_outputs) == [0]
    for t in range(12, 24):
        _assert_close(outs[t], 0, expected[t])


def upstream_masks_limited(video_predictor, state, max_frame_num_to_track):
    out = {}
    for f, _, masks in video_predictor.propagate_in_video(
        state, max_frame_num_to_track=max_frame_num_to_track
    ):
        out[f] = masks.clone()
    return out
