# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers of the stream predictor GPU tools."""

import argparse
import contextlib
import os

import numpy as np
import torch
from PIL import Image

# prompts from notebooks/video_predictor_example.ipynb (bedroom video, frame 0)
NOTEBOOK_PROMPTS = {
    1: {"points": [[210, 350], [250, 220]], "labels": [1, 1]},
    2: {"points": [[200, 300], [275, 175]], "labels": [1, 0]},
    3: {"points": [[400, 150]], "labels": [1]},
}


def base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", required=True, help="e.g. configs/sam2.1/sam2.1_hiera_s.yaml"
    )
    parser.add_argument(
        "--ckpt", required=True, help="e.g. checkpoints/sam2.1_hiera_small.pt"
    )
    parser.add_argument("--video-dir", default="notebooks/videos/bedroom")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--size", type=int, choices=[512, 1024], default=1024)
    return parser


def size_overrides(size):
    """hydra overrides for a non-default model input size."""
    if size == 1024:
        return []
    feat = size // 16
    return [
        f"++model.image_size={size}",
        f"++model.memory_attention.layer.self_attention.feat_sizes=[{feat},{feat}]",
        f"++model.memory_attention.layer.cross_attention.feat_sizes=[{feat},{feat}]",
    ]


def cuda_or_skip():
    """True when CUDA is usable; otherwise prints the skip message."""
    if torch.cuda.is_available():
        return True
    print("CUDA not available, skipping")
    return False


def autocast(dtype):
    if dtype == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def frame_names(video_dir):
    names = [
        n
        for n in os.listdir(video_dir)
        if os.path.splitext(n)[-1].lower() in [".jpg", ".jpeg"]
    ]
    return sorted(names, key=lambda n: int(os.path.splitext(n)[0]))


def load_rgb_frames(video_dir, num_frames):
    names = frame_names(video_dir)[:num_frames]
    return [
        np.array(Image.open(os.path.join(video_dir, n)).convert("RGB")) for n in names
    ]


def mask_iou(a, b):
    ma, mb = a > 0, b > 0
    union = (ma | mb).sum().item()
    return 1.0 if union == 0 else (ma & mb).sum().item() / union


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q))
