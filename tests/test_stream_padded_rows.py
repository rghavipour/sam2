# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Rows with different memory layouts (a frame dropped from one object's memory, a
view promoted to a conditioning frame of another) run in one memory-attention
call: each row's memory is padded to one shape and the padding is masked out.
Masked keys get no attention weight, so on CPU in fp32 the result equals one call
per layout to float rounding, with matching masks.
"""

import torch

from stream_helpers import BOXES, iou, make_frame

from sam2.stream.memory import RowMemory, stack_rows_padded

OFFSETS = (0, 3)
TOL = 1e-4
MIN_IOU = 0.999


def _diverge(sessions, t):
    """Memory edits that give rows different layouts (as a tracker makes them)."""
    if t == 4:
        sessions[0].objects[2].non_cond_outputs.pop(3, None)  # dropped bad frame
        obj = sessions[1].objects[1]
        obj.cond_outputs[3] = obj.non_cond_outputs.pop(3)  # promoted view
    if t == 6:
        sessions[1].objects[3].non_cond_outputs.pop(5, None)


def _run(pred, pad, steps=9):
    pred.pad_memory_layouts = pad
    sessions = [pred.create_session() for _ in OFFSETS]
    outs, groups = [], []
    for t in range(steps):
        _diverge(sessions, t)
        pred.set_frames(sessions, [make_frame(t + o) for o in OFFSETS])
        if t == 0:
            for s in sessions:
                for k in (1, 2, 3):
                    pred.add_prompt(s, k, box=BOXES[k])
        outs.append(pred.track_many(sessions))
        groups.append(pred.last_row_groups)
    return outs, groups


def test_padded_rows_match_one_call_per_layout(stream_predictor):
    plain, plain_groups = _run(stream_predictor, pad=False)
    padded, padded_groups = _run(stream_predictor, pad=True)
    stream_predictor.pad_memory_layouts = True
    assert max(plain_groups) > 1  # the edits did split the layouts
    assert max(padded_groups) == 1
    for t, (sa, sb) in enumerate(zip(plain, padded)):
        for i, (oa, ob) in enumerate(zip(sa, sb)):
            assert list(oa.obj_ids) == list(ob.obj_ids)
            torch.testing.assert_close(oa.mask_logits, ob.mask_logits, atol=TOL, rtol=0)
            torch.testing.assert_close(
                oa.object_score_logits, ob.object_score_logits, atol=TOL, rtol=0
            )
            for k in range(oa.mask_logits.shape[0]):
                assert iou(oa.mask_logits[k], ob.mask_logits[k]) >= MIN_IOU, (t, i, k)


def test_one_layout_is_not_padded(stream_predictor):
    stream_predictor.pad_memory_layouts = True
    sessions = [stream_predictor.create_session() for _ in range(2)]
    for t in range(3):
        stream_predictor.set_frames(sessions, [make_frame(t)] * 2)
        if t == 0:
            for s in sessions:
                stream_predictor.add_prompt(s, 1, box=BOXES[1])
        stream_predictor.track_many(sessions)
    assert stream_predictor.last_row_groups == 1
    assert stream_predictor.last_padded_rows == 0


def _row(frames, ptrs, hw=4, c=2, fill=1.0):
    m = frames * hw + ptrs
    mem = torch.full((m, 1, c), fill)
    pos = torch.full((m, 1, c), fill + 0.5)
    return RowMemory(mem, pos, ptrs, (frames, (), ptrs))


def test_stack_rows_padded_places_frames_and_pointers():
    rows = [_row(2, 3, fill=1.0), _row(1, 5, fill=2.0)]
    memory, memory_pos, num_ptr, mask = stack_rows_padded(rows, hw=4)
    # 2 frames x 4 tokens, then 5 pointer tokens
    assert memory.shape == (13, 2, 2) and num_ptr == 5 and mask.shape == (2, 13)
    assert mask[0].tolist() == [True] * 8 + [True] * 3 + [False] * 2
    assert mask[1].tolist() == [True] * 4 + [False] * 4 + [True] * 5
    assert torch.all(memory[:8, 0] == 1.0) and torch.all(memory[8:11, 0] == 1.0)
    assert torch.all(memory[11:, 0] == 0.0)
    assert torch.all(memory[:4, 1] == 2.0) and torch.all(memory[4:8, 1] == 0.0)
    assert torch.all(memory[8:, 1] == 2.0)
    assert torch.all(memory_pos[8:11, 0] == 1.5)
