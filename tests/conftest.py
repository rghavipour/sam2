# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from PIL import Image

from stream_helpers import (
    copy_weights,
    make_frame,
    NUM_FRAMES,
    seed_all,
    TINY_CONFIG,
    TINY_OVERRIDES,
)


@pytest.fixture(scope="session")
def stream_predictor():
    from sam2.build_sam import build_sam2_stream_predictor

    seed_all(0)
    return build_sam2_stream_predictor(
        TINY_CONFIG, device="cpu", hydra_overrides_extra=TINY_OVERRIDES
    )


@pytest.fixture(scope="session")
def stream_predictor_unbatched(stream_predictor):
    from sam2.build_sam import build_sam2_stream_predictor

    p = build_sam2_stream_predictor(
        TINY_CONFIG,
        device="cpu",
        hydra_overrides_extra=TINY_OVERRIDES + ["++model.batch_objects=false"],
    )
    return copy_weights(p, stream_predictor)


@pytest.fixture(scope="session")
def video_predictor(stream_predictor):
    from sam2.build_sam import build_sam2_video_predictor

    p = build_sam2_video_predictor(
        TINY_CONFIG, device="cpu", hydra_overrides_extra=TINY_OVERRIDES
    )
    return copy_weights(p, stream_predictor)


@pytest.fixture(scope="session")
def legacy_predictor(stream_predictor):
    from sam2.build_sam import build_sam2_video_predictor

    p = build_sam2_video_predictor(
        TINY_CONFIG,
        device="cpu",
        hydra_overrides_extra=TINY_OVERRIDES
        + ["++model._target_=sam2.sam2_video_predictor_legacy.SAM2VideoPredictor"],
    )
    return copy_weights(p, stream_predictor)


@pytest.fixture(scope="session")
def rgb_frames():
    return [make_frame(t) for t in range(NUM_FRAMES)]


@pytest.fixture(scope="session")
def frames_dir(tmp_path_factory, rgb_frames):
    d = tmp_path_factory.mktemp("frames")
    for t, frame in enumerate(rgb_frames):
        Image.fromarray(frame).save(d / f"{t:05d}.jpg", quality=95)
    return str(d)


@pytest.fixture(scope="session")
def upstream_images(video_predictor, frames_dir):
    with torch.inference_mode():
        state = video_predictor.init_state(frames_dir)
    return state["images"]
