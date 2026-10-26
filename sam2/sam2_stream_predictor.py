# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from sam2.modeling.sam2_base import SAM2Base

logger = logging.getLogger(__name__)

# passed as `num_frames` to SAM2Base, so the object pointer window and its temporal
# encoding are the same as upstream on any video of 16+ frames
STREAM_NUM_FRAMES = 1 << 30


class SAM2StreamPredictor(SAM2Base):
    """
    SAM 2 predictor for live streams: frames arrive one at a time, objects are
    independent and can be added, corrected or removed at any frame. All per-stream
    state lives in a `StreamSession`, so one predictor serves many streams.
    """

    def __init__(
        self,
        fill_hole_area=0,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to run the objects of a session that share a memory layout as one batch
        batch_objects=True,
        # the maximum number of conditioning frames kept per object (the first is protected)
        max_cond_frames_per_object=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if max_cond_frames_per_object < 1:
            raise ValueError("max_cond_frames_per_object must be >= 1")
        if self.num_maskmem == 0:
            raise ValueError("SAM2StreamPredictor needs num_maskmem >= 1")
        if self.non_overlap_masks_for_mem_enc:
            # upstream runs the memory encoder per object (batch 1), where this flag has
            # no effect; batching rows must not turn it on
            logger.warning(
                "non_overlap_masks_for_mem_enc is not supported by SAM2StreamPredictor; "
                "turning it off"
            )
            self.non_overlap_masks_for_mem_enc = False
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.batch_objects = batch_objects
        self.max_cond_frames_per_object = max_cond_frames_per_object
