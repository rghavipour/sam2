# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared constants and helpers for the stream predictor tests."""

import numpy as np
import torch

TINY_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
TINY_OVERRIDES = [
    "++model.image_size=256",
    "++model.memory_attention.layer.self_attention.feat_sizes=[16,16]",
    "++model.memory_attention.layer.cross_attention.feat_sizes=[16,16]",
]

NUM_FRAMES = 24
FRAME_W, FRAME_H = 320, 240
VIDEO_HW = (FRAME_H, FRAME_W)

# (x, y, w, h) at frame 0, (dx, dy) per frame, RGB color
_RECTS = {
    1: ((20, 20, 40, 30), (3, 0), (220, 40, 40)),
    2: ((200, 40, 50, 40), (0, 3), (40, 200, 60)),
    3: ((120, 150, 30, 50), (-3, 0), (50, 70, 220)),
}

# frame-0 prompt boxes in xyxy pixels
BOXES = {k: [x, y, x + w, y + h] for k, ((x, y, w, h), _, _) in _RECTS.items()}


def make_frame(t):
    """RGB uint8 frame `t` of the synthetic clip."""
    img = np.full((FRAME_H, FRAME_W, 3), 128, dtype=np.uint8)
    for (x, y, w, h), (dx, dy), color in _RECTS.values():
        x0, y0 = x + dx * t, y + dy * t
        img[y0 : y0 + h, x0 : x0 + w] = color
    return img


def iou(a, b):
    """IoU of the `logits > 0` masks; 1.0 when both are empty."""
    ma, mb = a > 0, b > 0
    union = (ma | mb).sum().item()
    if union == 0:
        return 1.0
    return (ma & mb).sum().item() / union


def run_stream(predictor, images, video_hw, schedule, num_frames=None):
    """
    Feed preprocessed `images` through `predictor` one frame at a time.

    `schedule` maps a frame index to a list of `(method_name, kwargs)` calls
    made after `set_preprocessed_frame` and before `track`. Returns
    `{frame_idx: StreamOutput}` of the `track` calls.
    """
    session = predictor.create_session()
    outputs = {}
    n = len(images) if num_frames is None else num_frames
    for t in range(n):
        predictor.set_preprocessed_frame(session, images[t], video_hw)
        for method, kwargs in schedule.get(t, []):
            getattr(predictor, method)(session, **kwargs)
        outputs[t] = predictor.track(session)
    return outputs, session


def copy_weights(dst, src):
    dst.load_state_dict(src.state_dict(), strict=True)
    return dst


def upstream_masks(video_predictor, state, start_frame_idx=None):
    """{frame_idx: video_res_masks} from upstream propagate_in_video."""
    out = {}
    for f, _, masks in video_predictor.propagate_in_video(
        state, start_frame_idx=start_frame_idx
    ):
        out[f] = masks.clone()
    return out


def seed_all(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
