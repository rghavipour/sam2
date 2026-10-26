# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import pytest


def _upstream_min_lookup(n, num_maskmem, stride, max_obj_ptrs, use_obj_ptrs):
    """Smallest non-cond frame index SAM2Base looks up when tracking frame `n` forward
    (a copy of the index arithmetic in `_prepare_memory_conditioned_features`)."""
    looked_up = []
    for t_pos in range(1, num_maskmem):
        t_rel = num_maskmem - t_pos
        if t_rel == 1:
            prev = n - t_rel
        else:
            prev = ((n - 2) // stride) * stride
            prev = prev - (t_rel - 2) * stride
        looked_up.append(prev)
    if use_obj_ptrs:
        for t_diff in range(1, max_obj_ptrs):
            t = n - t_diff
            if t < 0:
                break
            looked_up.append(t)
    return min(looked_up)


def test_min_needed_frame_defaults():
    from sam2.stream.state import min_needed_frame

    assert min_needed_frame(20, 7, 1, 16, True) == 5
    assert min_needed_frame(20, 7, 1, 16, False) == 14
    assert min_needed_frame(1, 7, 1, 16, True) <= 0


@pytest.mark.parametrize("stride", [1, 2, 5])
@pytest.mark.parametrize("use_obj_ptrs", [True, False])
def test_min_needed_frame_matches_upstream_loop(stride, use_obj_ptrs):
    from sam2.stream.state import min_needed_frame

    for n in range(0, 61):
        ours = min_needed_frame(n, 7, stride, 16, use_obj_ptrs)
        upstream = _upstream_min_lookup(n, 7, stride, 16, use_obj_ptrs)
        # frames before 0 do not exist, so only the clamped value matters
        assert max(0, ours) == max(0, upstream), n


def _obj(cond_keys, non_cond_keys):
    from sam2.stream.state import ObjectState

    obj = ObjectState(obj_id=1)
    obj.cond_outputs = {k: {"k": k} for k in cond_keys}
    obj.non_cond_outputs = OrderedDict((k, {"k": k}) for k in non_cond_keys)
    if cond_keys:
        obj.first_cond_frame = min(cond_keys)
    return obj


def test_prune_keeps_window_and_cond():
    from sam2.stream.state import prune_object

    obj = _obj([0, 3], range(1, 21))
    obj.pending = {"k": 20}
    prune_object(obj, 6)
    assert list(obj.non_cond_outputs) == list(range(6, 21))
    assert sorted(obj.cond_outputs) == [0, 3]
    assert obj.pending == {"k": 20}


def test_evict_keeps_first_and_newest():
    from sam2.stream.state import evict_cond_frames

    obj = _obj([0, 4, 9, 12, 30, 41], [])
    evict_cond_frames(obj, 4)
    assert sorted(obj.cond_outputs) == [0, 12, 30, 41]

    obj = _obj([0, 4, 9], [])
    evict_cond_frames(obj, 1)
    assert sorted(obj.cond_outputs) == [9]

    obj = _obj([2, 5], [])
    evict_cond_frames(obj, 4)
    assert sorted(obj.cond_outputs) == [2, 5]
