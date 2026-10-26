# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-stream state of `SAM2StreamPredictor`, kept outside the model."""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional

import torch


@dataclass
class ObjectState:
    """Memory and inputs of one tracked object."""

    obj_id: Hashable
    # {frame_idx: out} for frames that received prompts (normally just one)
    cond_outputs: Dict[int, dict] = field(default_factory=dict)
    # {frame_idx: out} for tracked frames, pruned to the window the model can read
    non_cond_outputs: "OrderedDict[int, dict]" = field(default_factory=OrderedDict)
    # output of `add_prompt` on the current frame, memory not encoded yet
    pending: Optional[dict] = None
    pending_is_cond: bool = False
    # point inputs on `point_inputs_frame` (to accumulate clicks)
    point_inputs: Optional[dict] = None
    point_inputs_frame: int = -1
    # the last frame `track` produced an output for this object
    tracked_frame: int = -1
    # the object's first conditioning frame, protected from eviction
    first_cond_frame: Optional[int] = None


@dataclass
class StreamSession:
    """Everything one stream needs; the predictor itself holds no stream state."""

    device: torch.device
    max_objects: Optional[int] = None
    frame_idx: int = -1
    video_h: int = 0
    video_w: int = 0
    # current frame only: batch-1 output of `forward_image` and the model input
    backbone_out: Optional[dict] = None
    image: Optional[torch.Tensor] = None
    # batch size -> (vision_feats, vision_pos_embeds, feat_sizes) for the current frame
    feature_cache: Dict[int, tuple] = field(default_factory=dict)
    objects: "OrderedDict[Hashable, ObjectState]" = field(default_factory=OrderedDict)
    # values shared by all frames and objects ("maskmem_pos_enc")
    constants: Dict[str, Any] = field(default_factory=dict)

    @property
    def obj_ids(self) -> List[Hashable]:
        return list(self.objects)


@dataclass
class StreamOutput:
    """Masks of every object of a session on one frame (in object order)."""

    frame_idx: int
    obj_ids: list
    mask_logits: torch.Tensor  # [N, 1, video_h, video_w]
    low_res_mask_logits: torch.Tensor  # [N, 1, image_size / 4, image_size / 4]
    object_score_logits: torch.Tensor  # [N, 1]


def min_needed_frame(next_frame_idx, num_maskmem, stride, max_obj_ptrs, use_obj_ptrs):
    """
    The smallest non-conditioning frame index `SAM2Base._prepare_memory_conditioned_features`
    can read when tracking `next_frame_idx` forward; older outputs can be dropped.
    """
    n, r = next_frame_idx, stride
    cands = [n - 1]
    for t_rel in range(2, num_maskmem):
        cands.append(((n - 2) // r) * r - (t_rel - 2) * r)
    if use_obj_ptrs:
        cands.append(n - (max_obj_ptrs - 1))
    return min(cands)


def prune_object(obj, keep_from):
    """Drop non-conditioning outputs older than `keep_from`."""
    for t in [t for t in obj.non_cond_outputs if t < keep_from]:
        del obj.non_cond_outputs[t]


def evict_cond_frames(obj, max_cond_frames):
    """
    Keep at most `max_cond_frames` conditioning outputs, dropping the oldest first but
    never the object's first conditioning frame (with a cap of 1, keep the newest).
    """
    if max_cond_frames == 1:
        if len(obj.cond_outputs) > 1:
            newest = max(obj.cond_outputs)
            obj.cond_outputs = {newest: obj.cond_outputs[newest]}
        return
    while len(obj.cond_outputs) > max_cond_frames:
        oldest = min(t for t in obj.cond_outputs if t != obj.first_cond_frame)
        del obj.cond_outputs[oldest]
