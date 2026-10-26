# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Compare SAM2StreamPredictor with upstream SAM2VideoPredictor on a real video.

Variant A: three objects prompted on frame 0, against one upstream session.
Variant B: object 3 prompted on frame 30 instead, against an upstream session with
only that object started on frame 30.

Exits 1 when any per-object IoU falls below 0.99 (fp32) or 0.98 (bf16).
"""

import sys

import torch

from stream_common import (
    autocast,
    base_parser,
    cuda_or_skip,
    mask_iou,
    NOTEBOOK_PROMPTS,
    size_overrides,
)

LATE_FRAME = 30


def _upstream(video, state, start_frame_idx=None, num_frames=None):
    out = {}
    for f, obj_ids, masks in video.propagate_in_video(
        state, start_frame_idx=start_frame_idx
    ):
        if num_frames is not None and f >= num_frames:
            break
        out[f] = (list(obj_ids), masks.clone())
    return out


def _stream(stream, images, video_hw, schedule, num_frames):
    session = stream.create_session()
    out = {}
    for t in range(num_frames):
        stream.set_preprocessed_frame(session, images[t], video_hw)
        for obj_id, prompt in schedule.get(t, []):
            stream.add_prompt(session, obj_id, **prompt)
        out[t] = stream.track(session)
    return out


def _report(name, ious):
    worst = 1.0
    for obj_id, values in ious.items():
        lo, mean = min(values), sum(values) / len(values)
        worst = min(worst, lo)
        print(
            f"{name} object {obj_id}: min IoU {lo:.4f}  mean IoU {mean:.4f}  frames {len(values)}"
        )
    return worst


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    args = parser.parse_args()
    if not cuda_or_skip():
        return 0

    from sam2.build_sam import build_sam2_stream_predictor, build_sam2_video_predictor

    overrides = size_overrides(args.size)
    stream = build_sam2_stream_predictor(
        args.config, args.ckpt, hydra_overrides_extra=overrides
    )
    video = build_sam2_video_predictor(
        args.config, args.ckpt, hydra_overrides_extra=overrides
    )
    threshold = 0.99 if args.dtype == "fp32" else 0.98

    with torch.inference_mode(), autocast(args.dtype):
        state = video.init_state(args.video_dir)
        images = state["images"]
        n = min(args.frames, len(images))
        video_hw = (state["video_height"], state["video_width"])

        # variant A: all three objects on frame 0
        for obj_id, prompt in NOTEBOOK_PROMPTS.items():
            video.add_new_points_or_box(state, 0, obj_id, **prompt)
        ref = _upstream(video, state, num_frames=n)
        schedule = {0: list(NOTEBOOK_PROMPTS.items())}
        got = _stream(stream, images, video_hw, schedule, n)
        ious = {k: [] for k in NOTEBOOK_PROMPTS}
        for t, (obj_ids, masks) in ref.items():
            for row, obj_id in enumerate(obj_ids):
                ious[obj_id].append(mask_iou(got[t].mask_logits[row], masks[row]))
        worst = _report("A", ious)

        # variant B: object 3 added late
        state = video.init_state(args.video_dir)
        video.add_new_points_or_box(state, LATE_FRAME, 3, **NOTEBOOK_PROMPTS[3])
        ref = _upstream(video, state, start_frame_idx=LATE_FRAME, num_frames=n)
        schedule = {
            0: [(1, NOTEBOOK_PROMPTS[1]), (2, NOTEBOOK_PROMPTS[2])],
            LATE_FRAME: [(3, NOTEBOOK_PROMPTS[3])],
        }
        got = _stream(stream, images, video_hw, schedule, n)
        ious = {
            3: [
                mask_iou(got[t].mask_logits[2], masks[0])
                for t, (_, masks) in ref.items()
            ]
        }
        worst = min(worst, _report("B", ious))

    print(f"worst IoU {worst:.4f} (threshold {threshold})")
    return 0 if worst >= threshold else 1


if __name__ == "__main__":
    sys.exit(main())
