# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA path")
@pytest.mark.parametrize("script", ["stream_parity.py", "benchmark_stream.py"])
def test_gpu_tool_skips_without_cuda(script):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / script),
            "--config",
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            "--ckpt",
            "checkpoints/missing.pt",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "CUDA not available, skipping" in result.stdout


@pytest.mark.parametrize("script", ["stream_parity.py", "benchmark_stream.py"])
def test_gpu_tool_help(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / script), "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "--size" in result.stdout and "--video-dir" in result.stdout


def test_cuda_kernel_ms_counts_device_events_once():
    sys.path.insert(0, str(ROOT / "tools"))
    from stream_common import cuda_kernel_ms

    class Event:
        def __init__(self, device_type, us, annotation=False):
            self.device_type = device_type
            self.self_device_time_total = us
            self.is_user_annotation = annotation

    cpu, cuda = torch.autograd.DeviceType.CPU, torch.autograd.DeviceType.CUDA
    # a CPU op row carries the time of the kernel it launched; the kernel row again
    events = [
        Event(cpu, 3000),
        Event(cuda, 3000),
        Event(cuda, 1000),
        Event(cuda, 500, True),
    ]
    assert cuda_kernel_ms(events, frames=2) == 2.0
