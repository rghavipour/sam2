# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import torch

from stream_helpers import FRAME_H, FRAME_W

MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
CPU = torch.device("cpu")


def test_shape_dtype_device(rgb_frames):
    from sam2.stream.preprocess import preprocess_frame

    image, h, w = preprocess_frame(rgb_frames[0], 256, CPU)
    assert image.shape == (3, 256, 256)
    assert image.dtype == torch.float32
    assert image.device.type == "cpu"
    assert (h, w) == (FRAME_H, FRAME_W)


def test_numpy_and_tensor_equal(rgb_frames):
    from sam2.stream.preprocess import preprocess_frame

    a, _, _ = preprocess_frame(rgb_frames[3], 256, CPU)
    b, _, _ = preprocess_frame(torch.from_numpy(rgb_frames[3]), 256, CPU)
    assert torch.equal(a, b)


def test_close_to_upstream_loader(rgb_frames, upstream_images):
    from sam2.stream.preprocess import preprocess_frame

    ours, _, _ = preprocess_frame(rgb_frames[0], 256, CPU)
    # compare in [0, 1] intensity space; upstream resizes with PIL
    diff = (ours * STD + MEAN) - (upstream_images[0] * STD + MEAN)
    assert diff.abs().mean().item() < 0.02


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((240, 320), dtype=np.uint8),
        np.zeros((240, 320, 4), dtype=np.uint8),
        np.zeros((240, 320, 3), dtype=np.float32),
        np.zeros((0, 320, 3), dtype=np.uint8),
    ],
)
def test_rejects_bad_frames(frame):
    from sam2.stream.preprocess import preprocess_frame

    with pytest.raises(ValueError, match="frame must be a uint8 HxWx3 RGB array"):
        preprocess_frame(frame, 256, CPU)
