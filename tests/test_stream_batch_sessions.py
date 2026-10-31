# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Several sessions in one step (`set_frames`, `track_many`, `step_many`) give the
same masks as each session on its own (`set_frame` + `track`). On CPU in fp32 they
agree to float rounding (about 3e-7; the image encoder, memory attention and heads
round differently at a different batch size) and the binary masks agree. A logic
error, such as rows taking another session's features, shows as a large
difference instead.
"""

import pytest
import torch

from stream_helpers import BOXES, copy_weights, iou, make_frame, TINY_CONFIG, TINY_OVERRIDES

OFFSETS = (0, 2, 4)  # each session sees the synthetic clip from its own start
TOL = 1e-4  # float rounding; masks must agree too (MIN_IOU)
MIN_IOU = 0.999


def _frame(t, i):
    return make_frame(t + OFFSETS[i % len(OFFSETS)])


def _run(pred, n, steps, schedule, batched):
    """schedule: {t: [(i, op, kwargs)]}; op "add" runs after the frame is set,
    "remove" before. Returns per step the list of per-session outputs."""
    sessions = [pred.create_session() for _ in range(n)]
    outs = []
    for t in range(steps):
        for i, op, kw in schedule.get(t, []):
            if op == "remove":
                pred.remove_object(sessions[i], kw["obj_id"])
        frames = [_frame(t, i) for i in range(n)]
        if batched:
            pred.set_frames(sessions, frames)
        else:
            for s, f in zip(sessions, frames):
                pred.set_frame(s, f)
        for i, op, kw in schedule.get(t, []):
            if op == "add":
                pred.add_prompt(sessions[i], kw["obj_id"], box=kw["box"])
        if batched:
            outs.append(pred.track_many(sessions))
        else:
            outs.append([pred.track(s) for s in sessions])
    return outs


def _assert_same(a, b, tol=None):
    tol = TOL if tol is None else tol
    assert len(a) == len(b)
    for t, (sa, sb) in enumerate(zip(a, b)):
        for i, (oa, ob) in enumerate(zip(sa, sb)):
            assert list(oa.obj_ids) == list(ob.obj_ids), (t, i)
            torch.testing.assert_close(oa.mask_logits, ob.mask_logits, atol=tol, rtol=0)
            for k in range(oa.mask_logits.shape[0]):
                assert iou(oa.mask_logits[k], ob.mask_logits[k]) >= MIN_IOU, (t, i, k)
            torch.testing.assert_close(
                oa.object_score_logits, ob.object_score_logits, atol=tol, rtol=0
            )


def _prompt_all(n, t=0):
    return {t: [(i, "add", {"obj_id": k, "box": BOXES[k]}) for i in range(n) for k in (1, 2, 3)]}


def test_sessions_in_one_step_match_one_by_one(stream_predictor):
    sched = _prompt_all(2)
    one = _run(stream_predictor, 2, 8, sched, batched=False)
    many = _run(stream_predictor, 2, 8, sched, batched=True)
    _assert_same(one, many)


def test_different_memory_layouts_and_removals(stream_predictor):
    # session 0 from frame 0, session 1 from frame 3 (shorter memory), session 2
    # adds an object late; session 0 drops one object mid-run
    sched = {
        0: [(0, "add", {"obj_id": k, "box": BOXES[k]}) for k in (1, 2, 3)],
        3: [(1, "add", {"obj_id": k, "box": BOXES[k]}) for k in (1, 2)]
        + [(2, "add", {"obj_id": 1, "box": BOXES[1]})],
        6: [(2, "add", {"obj_id": 3, "box": BOXES[3]})],
        8: [(0, "remove", {"obj_id": 2})],
    }
    one = _run(stream_predictor, 3, 12, sched, batched=False)
    many = _run(stream_predictor, 3, 12, sched, batched=True)
    _assert_same(one, many)


def test_a_session_without_objects_in_the_step(stream_predictor):
    sched = {0: [(0, "add", {"obj_id": 1, "box": BOXES[1]})]}
    one = _run(stream_predictor, 2, 5, sched, batched=False)
    many = _run(stream_predictor, 2, 5, sched, batched=True)
    _assert_same(one, many)
    assert list(many[-1][1].obj_ids) == []


def test_one_session_step_many_is_step(stream_predictor):
    sa, sb = stream_predictor.create_session(), stream_predictor.create_session()
    for s in (sa, sb):
        stream_predictor.set_frame(s, _frame(0, 0))
        stream_predictor.add_prompt(s, 1, box=BOXES[1])
    for t in range(1, 5):
        a = stream_predictor.step(sa, _frame(t, 0))
        (b,) = stream_predictor.step_many([sb], [_frame(t, 0)])
        assert torch.equal(a.mask_logits, b.mask_logits)  # same batch size: exact


def _variant(stream_predictor, overrides):
    from sam2.build_sam import build_sam2_stream_predictor

    p = build_sam2_stream_predictor(
        TINY_CONFIG, device="cpu", hydra_overrides_extra=TINY_OVERRIDES + overrides
    )
    return copy_weights(p, stream_predictor)


def test_unbatched_predictor(stream_predictor_unbatched):
    sched = _prompt_all(2)
    one = _run(stream_predictor_unbatched, 2, 6, sched, batched=False)
    many = _run(stream_predictor_unbatched, 2, 6, sched, batched=True)
    _assert_same(one, many)


def test_non_overlap_stays_within_each_session(stream_predictor):
    p = _variant(stream_predictor, ["++model.non_overlap_masks_for_mem_enc=true"])
    assert p.non_overlap_across_objects
    # the constraint picks, per pixel, the highest-scoring object; the tiny random
    # model's masks overlap almost everywhere, so rounding at near-ties grows through
    # memory (to about 1e-3 by frame 8) while the masks still agree
    sched = _prompt_all(2)
    one = _run(p, 2, 6, sched, batched=False)
    many = _run(p, 2, 6, sched, batched=True)
    _assert_same(one, many, tol=1e-3)


def test_bad_input(stream_predictor):
    s = stream_predictor.create_session()
    with pytest.raises(ValueError):
        stream_predictor.set_frames([s, stream_predictor.create_session()], [_frame(0, 0)])
    with pytest.raises(ValueError):
        stream_predictor.set_frames([s, s], [_frame(0, 0), _frame(0, 1)])
    assert stream_predictor.set_frames([], []) == []
    assert stream_predictor.track_many([]) == []
    with pytest.raises(RuntimeError):
        stream_predictor.track_many([stream_predictor.create_session()])  # no frame set
