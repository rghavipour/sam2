# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from stream_helpers import TINY_CONFIG, TINY_OVERRIDES


def test_builder_returns_stream_predictor(stream_predictor):
    from sam2.sam2_stream_predictor import SAM2StreamPredictor

    p = stream_predictor
    assert isinstance(p, SAM2StreamPredictor)
    assert p.image_size == 256
    assert p.device.type == "cpu"
    assert p.fill_hole_area == 8
    assert p.batch_objects is True
    assert p.max_cond_frames_per_object == 4
    assert not p.training


def test_constructor_rejects_bad_cond_cap():
    from sam2.build_sam import build_sam2_stream_predictor

    with pytest.raises(Exception, match="max_cond_frames_per_object must be >= 1"):
        build_sam2_stream_predictor(
            TINY_CONFIG,
            device="cpu",
            hydra_overrides_extra=TINY_OVERRIDES
            + ["++model.max_cond_frames_per_object=0"],
        )
