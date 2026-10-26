# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from stream_helpers import BOXES, iou, run_stream, VIDEO_HW


def _box(obj_id, box=None):
    return ("add_prompt", {"obj_id": obj_id, "box": box or BOXES[obj_id]})


def _three_at_zero():
    return {0: [_box(1), _box(2), _box(3)]}


def test_three_objects_shapes(stream_predictor, upstream_images):
    outs, _ = run_stream(
        stream_predictor, upstream_images, VIDEO_HW, _three_at_zero(), num_frames=10
    )
    out = outs[9]
    assert out.frame_idx == 9
    assert out.obj_ids == [1, 2, 3]
    assert out.mask_logits.shape == (3, 1, 240, 320)
    assert out.low_res_mask_logits.shape == (3, 1, 64, 64)
    assert out.object_score_logits.shape == (3, 1)
    assert out.mask_logits.dtype == torch.float32


def test_late_object_does_not_change_others(stream_predictor, upstream_images):
    two = {0: [_box(1), _box(2)]}
    late = {0: [_box(1), _box(2)], 5: [_box(3, [105, 150, 135, 200])]}
    outs_a, _ = run_stream(stream_predictor, upstream_images, VIDEO_HW, two)
    outs_b, _ = run_stream(stream_predictor, upstream_images, VIDEO_HW, late)
    for t in outs_a:
        assert outs_b[t].obj_ids == ([1, 2, 3] if t >= 5 else [1, 2])
        torch.testing.assert_close(
            outs_b[t].mask_logits[:2], outs_a[t].mask_logits, atol=1e-4, rtol=0
        )


def test_requires_frame_first(stream_predictor):
    session = stream_predictor.create_session()
    msg = r"call set_frame\(\) before add_prompt\(\) or track\(\)"
    with pytest.raises(RuntimeError, match=msg):
        stream_predictor.add_prompt(session, 1, box=BOXES[1])
    with pytest.raises(RuntimeError, match=msg):
        stream_predictor.track(session)


def test_track_no_objects(stream_predictor, upstream_images):
    session = stream_predictor.create_session()
    stream_predictor.set_preprocessed_frame(session, upstream_images[0], VIDEO_HW)
    out = stream_predictor.track(session)
    assert out.obj_ids == []
    assert out.mask_logits.shape == (0, 1, 240, 320)
    assert out.low_res_mask_logits.shape == (0, 1, 64, 64)
    assert out.object_score_logits.shape == (0, 1)


def test_track_twice_same_frame(stream_predictor, upstream_images):
    p = stream_predictor
    session = p.create_session()
    for t in range(3):
        p.set_preprocessed_frame(session, upstream_images[t], VIDEO_HW)
        if t == 0:
            p.add_prompt(session, 1, box=BOXES[1])
        first = p.track(session)
    stored = session.objects[1].non_cond_outputs[2]
    second = p.track(session)
    assert session.objects[1].non_cond_outputs[2] is stored
    assert torch.equal(first.mask_logits, second.mask_logits)
    assert torch.equal(first.object_score_logits, second.object_score_logits)


def test_frame_size_change(stream_predictor, upstream_images):
    p = stream_predictor
    session = p.create_session()
    for t in range(12):
        hw = (120, 160) if t == 10 else VIDEO_HW
        p.set_preprocessed_frame(session, upstream_images[t], hw)
        if t == 0:
            for k in (1, 2, 3):
                p.add_prompt(session, k, box=BOXES[k])
        out = p.track(session)
        assert out.mask_logits.shape == (3, 1, *hw)
    assert out.frame_idx == 11


def test_prompt_on_new_frame_is_cond(stream_predictor, upstream_images):
    p = stream_predictor
    outs, session = run_stream(
        p,
        upstream_images,
        VIDEO_HW,
        {
            0: [_box(1)],
            10: [("add_prompt", {"obj_id": 1, "points": [[45, 35]], "labels": [1]})],
        },
        num_frames=12,
    )
    obj = session.objects[1]
    assert sorted(obj.cond_outputs) == [0, 10]
    assert 10 not in obj.non_cond_outputs
    assert obj.cond_outputs[10]["maskmem_features"] is not None


def test_correction_after_track_is_non_cond(stream_predictor, upstream_images):
    p = stream_predictor
    session = p.create_session()
    for t in range(6):
        p.set_preprocessed_frame(session, upstream_images[t], VIDEO_HW)
        if t == 0:
            p.add_prompt(session, 1, box=BOXES[1])
        p.track(session)
    tracked = session.objects[1].non_cond_outputs[5]
    out = p.add_prompt(session, 1, points=[[40, 30]], labels=[0])
    obj = session.objects[1]
    assert obj.pending is not None and not obj.pending_is_cond
    assert torch.equal(out.low_res_mask_logits[0:1], obj.pending["pred_masks"])
    p.set_preprocessed_frame(session, upstream_images[6], VIDEO_HW)
    assert sorted(obj.cond_outputs) == [0]
    assert obj.non_cond_outputs[5] is not tracked
    assert obj.non_cond_outputs[5]["maskmem_features"] is not None
    assert obj.pending is None


def test_prompt_on_tracked_cond_frame_replaces_cond(stream_predictor, upstream_images):
    p = stream_predictor
    session = p.create_session()
    p.set_preprocessed_frame(session, upstream_images[0], VIDEO_HW)
    p.add_prompt(session, 1, box=BOXES[1])
    p.track(session)
    p.add_prompt(session, 1, points=[[30, 25]], labels=[1], clear_old_points=False)
    obj = session.objects[1]
    assert obj.pending_is_cond
    p.set_preprocessed_frame(session, upstream_images[1], VIDEO_HW)
    assert sorted(obj.cond_outputs) == [0]
    assert 0 not in obj.non_cond_outputs
    assert obj.point_inputs["point_labels"].tolist() == [[2, 3, 1]]


def test_mask_prompt(stream_predictor, upstream_images):
    p = stream_predictor
    mask = np.zeros(VIDEO_HW, dtype=bool)
    x1, y1, x2, y2 = BOXES[2]
    mask[y1:y2, x1:x2] = True
    session = p.create_session()
    p.set_preprocessed_frame(session, upstream_images[0], VIDEO_HW)
    out = p.add_prompt(session, 2, mask=mask)
    assert iou(out.mask_logits[0, 0], torch.from_numpy(mask).float() * 2 - 1) > 0.9
    assert session.objects[2].point_inputs is None
    p.track(session)
    p.set_preprocessed_frame(session, upstream_images[1], VIDEO_HW)
    assert p.track(session).mask_logits.shape == (1, 1, 240, 320)
    with pytest.raises(ValueError):
        p.add_prompt(session, 2, mask=mask, box=BOXES[2])


def test_points_accumulate_with_clear_old_points_false(
    stream_predictor, upstream_images
):
    p = stream_predictor
    session = p.create_session()
    p.set_preprocessed_frame(session, upstream_images[0], VIDEO_HW)
    p.add_prompt(session, 1, points=[[40, 35]], labels=[1])
    p.add_prompt(session, 1, points=[[10, 10]], labels=[0], clear_old_points=False)
    assert session.objects[1].point_inputs["point_labels"].tolist() == [[1, 0]]
    p.add_prompt(session, 1, points=[[40, 35]], labels=[1])
    assert session.objects[1].point_inputs["point_labels"].tolist() == [[1]]


def test_box_with_clear_old_points_false_raises(stream_predictor, upstream_images):
    p = stream_predictor
    session = p.create_session()
    p.set_preprocessed_frame(session, upstream_images[0], VIDEO_HW)
    with pytest.raises(ValueError):
        p.add_prompt(session, 1, box=BOXES[1], clear_old_points=False)
    with pytest.raises(ValueError):
        p.add_prompt(session, 1)
    with pytest.raises(ValueError):
        p.add_prompt(session, 1, points=[[1, 2]])


def test_max_objects(stream_predictor, upstream_images):
    p = stream_predictor
    session = p.create_session(max_objects=2)
    p.set_preprocessed_frame(session, upstream_images[0], VIDEO_HW)
    p.add_prompt(session, 1, box=BOXES[1])
    p.add_prompt(session, 2, box=BOXES[2])
    with pytest.raises(
        RuntimeError, match="session already tracks max_objects=2 objects"
    ):
        p.add_prompt(session, 3, box=BOXES[3])
    p.add_prompt(session, 2, box=BOXES[2])  # existing ids still accept prompts
    with pytest.raises(ValueError):
        p.create_session(max_objects=0)


def test_batched_equals_unbatched(
    stream_predictor, stream_predictor_unbatched, upstream_images
):
    schedule = {0: [_box(1), _box(2)], 4: [_box(3, [108, 150, 138, 200])]}
    outs_b, _ = run_stream(stream_predictor, upstream_images, VIDEO_HW, schedule)
    outs_u, _ = run_stream(
        stream_predictor_unbatched, upstream_images, VIDEO_HW, schedule
    )
    for t in outs_b:
        assert outs_b[t].obj_ids == outs_u[t].obj_ids
        torch.testing.assert_close(
            outs_b[t].mask_logits, outs_u[t].mask_logits, atol=1e-3, rtol=0
        )
        for i in range(len(outs_b[t].obj_ids)):
            assert iou(outs_b[t].mask_logits[i], outs_u[t].mask_logits[i]) >= 0.99
