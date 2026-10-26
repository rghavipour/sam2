# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark SAM2StreamPredictor.step() on a real video.

Prints per-frame latency (mean, p50, p95) and FPS with CUDA-synchronized timing,
the same for upstream SAM2VideoPredictor.propagate_in_video as a reference, a
torch.profiler summary (CPU vs CUDA time per frame) and the number of host-device
syncs per frame.
"""

import sys
import time
import warnings

import torch

from stream_common import (
    autocast,
    base_parser,
    cuda_or_skip,
    load_rgb_frames,
    NOTEBOOK_PROMPTS,
    percentile,
    size_overrides,
)

PROFILE_FRAMES = 20
SYNC_DEBUG_FRAMES = 10


def _summary(name, ms):
    mean = sum(ms) / len(ms)
    print(
        f"{name}: mean {mean:.2f} ms  p50 {percentile(ms, 50):.2f} ms  "
        f"p95 {percentile(ms, 95):.2f} ms  fps {1000.0 / mean:.1f}  frames {len(ms)}"
    )


def _prompted_session(stream, frames, obj_ids):
    session = stream.create_session()
    stream.set_frame(session, frames[0])
    for obj_id in obj_ids:
        stream.add_prompt(session, obj_id, **NOTEBOOK_PROMPTS[obj_id])
    stream.track(session)
    return session


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--objects", type=int, choices=[1, 3], default=3)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    args = parser.parse_args()
    if not cuda_or_skip():
        return 0

    from sam2.build_sam import build_sam2_stream_predictor, build_sam2_video_predictor

    overrides = size_overrides(args.size)
    stream = build_sam2_stream_predictor(
        args.config, args.ckpt, hydra_overrides_extra=overrides
    )
    obj_ids = list(NOTEBOOK_PROMPTS)[: args.objects]
    total = args.warmup + args.frames
    # frames are decoded up front so JPEG decoding is not timed; longer runs loop
    frames = load_rgb_frames(args.video_dir, total + 1)
    print(
        f"device {torch.cuda.get_device_name()}  torch {torch.__version__}  "
        f"cuda {torch.version.cuda}  size {args.size}  objects {args.objects}  dtype {args.dtype}"
    )

    with torch.inference_mode(), autocast(args.dtype):
        # stream predictor
        session = _prompted_session(stream, frames, obj_ids)
        ms = []
        for i in range(1, total + 1):
            frame = frames[i % len(frames)]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            stream.step(session, frame)
            torch.cuda.synchronize()
            if i > args.warmup:
                ms.append((time.perf_counter() - t0) * 1000.0)
        _summary("stream step()", ms)

        # profiler
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.profiler.profile(activities=activities) as prof:
            for i in range(PROFILE_FRAMES):
                stream.step(session, frames[i % len(frames)])
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0 / PROFILE_FRAMES
        events = prof.key_averages()
        cuda_us = sum(getattr(e, "self_device_time_total", 0) for e in events)
        print(
            f"profiler: wall {wall_ms:.2f} ms/frame  "
            f"CUDA kernels {cuda_us / 1000.0 / PROFILE_FRAMES:.2f} ms/frame"
        )
        print(events.table(sort_by="self_cpu_time_total", row_limit=15))
        print(events.table(sort_by="self_device_time_total", row_limit=15))

        # host-device syncs
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for i in range(SYNC_DEBUG_FRAMES):
                stream.step(session, frames[i % len(frames)])
        torch.cuda.set_sync_debug_mode("default")
        syncs = sum("synchroniz" in str(w.message) for w in caught)
        print(f"host-device syncs: {syncs / SYNC_DEBUG_FRAMES:.1f} per frame")

    # upstream reference on the same video and prompts
    video = build_sam2_video_predictor(
        args.config, args.ckpt, hydra_overrides_extra=overrides
    )
    with torch.inference_mode(), autocast(args.dtype):
        state = video.init_state(args.video_dir)
        for obj_id in obj_ids:
            video.add_new_points_or_box(state, 0, obj_id, **NOTEBOOK_PROMPTS[obj_id])
        ms = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for f, _, _ in video.propagate_in_video(state):
            torch.cuda.synchronize()
            now = time.perf_counter()
            if f > args.warmup:
                ms.append((now - t0) * 1000.0)
            t0 = now
            if f >= total:
                break
        _summary("upstream propagate_in_video", ms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
