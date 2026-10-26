# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F


def preprocess_frame(
    frame,
    image_size,
    device,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    Resize and normalize one RGB uint8 HxWx3 frame on `device`.

    Returns the model input of [3, image_size, image_size] float32 and the original
    frame height and width.
    """
    if isinstance(frame, np.ndarray):
        t = torch.from_numpy(frame)
    elif isinstance(frame, torch.Tensor):
        t = frame
    else:
        t = None
    if t is None or t.dtype != torch.uint8 or t.dim() != 3 or t.shape[-1] != 3:
        raise ValueError(_bad_frame_message(frame))
    video_h, video_w = int(t.shape[0]), int(t.shape[1])
    if video_h < 1 or video_w < 1:
        raise ValueError(_bad_frame_message(frame))

    # convert to float only after moving to the device
    t = t.to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t = F.interpolate(
        t,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor(mean, dtype=torch.float32, device=t.device)[None, :, None, None]
    std = torch.tensor(std, dtype=torch.float32, device=t.device)[None, :, None, None]
    t = (t - mean) / std
    return t[0], video_h, video_w


def _bad_frame_message(frame):
    dtype = getattr(frame, "dtype", type(frame).__name__)
    shape = tuple(getattr(frame, "shape", ()))
    return f"frame must be a uint8 HxWx3 RGB array, got dtype={dtype} shape={shape}"
