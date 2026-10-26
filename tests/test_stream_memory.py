# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import pytest
import torch

from stream_helpers import BOXES, VIDEO_HW

NUM_FRAMES_ARG = 1 << 30


def _features(model, image):
    backbone_out = model.forward_image(image[None])
    _, feats, pos, sizes = model._prepare_backbone_features(backbone_out)
    return feats, pos, sizes


def _box_points(model, box):
    h, w = VIDEO_HW
    pts = torch.tensor(box, dtype=torch.float32).reshape(1, 2, 2)
    pts = pts / torch.tensor([w, h]) * model.image_size
    return {
        "point_coords": pts,
        "point_labels": torch.tensor([[2, 3]], dtype=torch.int32),
    }


def _fill_object(model, images, box, num_frames):
    """An ObjectState whose outputs come from upstream `track_step` on frames 0..n-1."""
    from sam2.stream.state import ObjectState

    obj = ObjectState(obj_id=0)
    output_dict = {
        "cond_frame_outputs": obj.cond_outputs,
        "non_cond_frame_outputs": obj.non_cond_outputs,
    }
    for t in range(num_frames):
        feats, pos, sizes = _features(model, images[t])
        out = model.track_step(
            frame_idx=t,
            is_init_cond_frame=(t == 0),
            current_vision_feats=feats,
            current_vision_pos_embeds=pos,
            feat_sizes=sizes,
            point_inputs=_box_points(model, box) if t == 0 else None,
            mask_inputs=None,
            output_dict=output_dict,
            num_frames=NUM_FRAMES_ARG,
        )
        out["maskmem_features"] = out["maskmem_features"].to(torch.bfloat16)
        if t == 0:
            obj.cond_outputs[t] = out
            obj.first_cond_frame = 0
        else:
            obj.non_cond_outputs[t] = out
    return obj


@pytest.fixture(scope="module")
def filled(stream_predictor, upstream_images):
    with torch.inference_mode():
        return {
            k: _fill_object(stream_predictor, upstream_images, BOXES[k], 23)
            for k in (1, 2)
        }


@pytest.mark.parametrize("frame_idx", [1, 5, 20])
def test_row_memory_matches_upstream(
    stream_predictor, upstream_images, filled, frame_idx
):
    from sam2.stream.memory import build_row_memory

    model, obj = stream_predictor, filled[1]
    output_dict = {
        "cond_frame_outputs": obj.cond_outputs,
        "non_cond_frame_outputs": OrderedDict(
            (t, v) for t, v in obj.non_cond_outputs.items() if t < frame_idx
        ),
    }
    with torch.inference_mode():
        feats, pos, sizes = _features(model, upstream_images[frame_idx])
        expected = model._prepare_memory_conditioned_features(
            frame_idx=frame_idx,
            is_init_cond_frame=False,
            current_vision_feats=feats[-1:],
            current_vision_pos_embeds=pos[-1:],
            feat_sizes=sizes[-1:],
            output_dict=output_dict,
            num_frames=NUM_FRAMES_ARG,
        )
        row = build_row_memory(model, obj, frame_idx, NUM_FRAMES_ARG)
        got = model.memory_attention(
            curr=feats[-1:],
            curr_pos=pos[-1:],
            memory=row.memory,
            memory_pos=row.memory_pos,
            num_obj_ptr_tokens=row.num_obj_ptr_tokens,
        )
        H, W = sizes[-1]
        got = got.permute(1, 2, 0).view(1, model.hidden_dim, H, W)
    assert row.memory.shape[1] == 1
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=0)


def test_keys(stream_predictor, filled):
    from sam2.stream.memory import build_row_memory

    with torch.inference_mode():
        young = build_row_memory(stream_predictor, filled[1], 2, NUM_FRAMES_ARG)
        mature_1 = build_row_memory(stream_predictor, filled[1], 20, NUM_FRAMES_ARG)
        mature_2 = build_row_memory(stream_predictor, filled[2], 20, NUM_FRAMES_ARG)
    assert young.key != mature_1.key
    assert mature_1.key == mature_2.key
    assert mature_1.key[0] == 1  # one conditioning frame


def test_group_rows_and_stack(stream_predictor, filled):
    from sam2.stream.memory import build_row_memory, group_rows, stack_rows

    with torch.inference_mode():
        rows = [
            build_row_memory(stream_predictor, filled[1], 20, NUM_FRAMES_ARG),
            build_row_memory(stream_predictor, filled[1], 2, NUM_FRAMES_ARG),
            build_row_memory(stream_predictor, filled[2], 20, NUM_FRAMES_ARG),
        ]
        assert group_rows(rows) == [[0, 2], [1]]
        memory, memory_pos, n_ptr = stack_rows([rows[0], rows[2]])
    assert memory.shape == (rows[0].memory.shape[0], 2, rows[0].memory.shape[2])
    assert memory_pos.shape == memory.shape
    assert n_ptr == rows[0].num_obj_ptr_tokens
    torch.testing.assert_close(memory[:, 1:2], rows[2].memory, atol=0, rtol=0)


def test_stack_rows_rejects_mixed_keys(stream_predictor, filled):
    from sam2.stream.memory import build_row_memory, stack_rows

    with torch.inference_mode():
        rows = [
            build_row_memory(stream_predictor, filled[1], 20, NUM_FRAMES_ARG),
            build_row_memory(stream_predictor, filled[1], 2, NUM_FRAMES_ARG),
        ]
        with pytest.raises(AssertionError):
            stack_rows(rows)
