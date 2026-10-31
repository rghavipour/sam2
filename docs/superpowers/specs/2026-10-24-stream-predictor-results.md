# SAM2StreamPredictor — GPU results

Run on 2026-10-26 on `afta2`: NVIDIA GeForce RTX 3080 Ti (12 GB), driver 595.84,
torch 2.5.1+cu124, CUDA 12.4, Python 3.11.10, commit `ef11f8d`.

Environment: throwaway container `sam2-stream-test` from
`registry.iftpdev.ir/back-end/dahi-live-ai:dino-sam2-small-v2`, the fork on
`PYTHONPATH` (not pip-installed), checkpoint `sam2.1_hiera_small.pt`, model input
512. The image has no CUDA compiler, so the `_C` hole-filling extension was not
built and hole filling was skipped in both upstream and the stream predictor.

The GPU was shared: about 9 GB of 12 GB was held by other services during every
run, so only about 3 GB was free and timings include contention.

## CPU test suite (in the container)

`python -m pytest tests -q`: **51 passed, 2 skipped** (the two skipped tests
check the no-CUDA path of the GPU tools).

## Parity with upstream `SAM2VideoPredictor` (bedroom video, notebook prompts)

The full 200-frame run did not fit in the free GPU memory (`parity_512_*.log`,
out of memory in the tool, which keeps all masks and both runs' frames on the
GPU). Runs below use the first 60 frames.

| Run | Object | min IoU | mean IoU | frames |
|---|---|---|---|---|
| A, batched, fp32 | 1 / 2 / 3 | 0.9991 / 0.9995 / 0.9998 | 0.9999 / 1.0000 / 1.0000 | 60 |
| A, batched, bf16 | 1 / 2 / 3 | 0.9672 / **0.9267** / 0.9576 | 0.9949 / 0.9954 / 0.9967 | 60 |
| A, unbatched, bf16 | 1 / 2 / 3 | 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 | 60 |
| B (object 3 added at frame 30), batched, fp32 | 3 | 1.0000 | 1.0000 | 30 |
| B, batched, bf16 | 3 | 0.9968 | 0.9993 | 30 |

A: three objects prompted on frame 0, against one upstream session.
B: object 3 prompted on frame 30 against an upstream session with only that
object, started on frame 30. A-batched comes from `tools/stream_parity.py
--frames 60`; the unbatched and B rows from an inline script with the same
prompts, `offload_video_to_cpu=True` for upstream to fit in memory
(`parity_512_extra_60f.log`).

Reading: the stream predictor reproduces upstream exactly when objects are not
batched, in fp32 and in bf16. Batching objects is exact in fp32; in bf16 it
drifts on some frames (worst frame IoU 0.927, means above 0.995), because bf16
matrix products give slightly different results at a different batch size and
the difference is fed back through memory. The spec's bf16 threshold of 0.98 is
missed by batched bf16 on its worst frames.

## Latency (`tools/benchmark_stream.py`, bf16, 150 timed frames after 30 warm-up)

| Objects | stream `step()` mean / p50 / p95 | stream FPS | upstream `propagate_in_video` mean | upstream FPS | CUDA kernels per frame |
|---|---|---|---|---|---|
| 1 | 27.88 / 27.75 / 29.47 ms | 35.9 | 33.73 ms | 29.6 | 9.06 ms |
| 3 | 30.20 / 29.84 / 32.45 ms | 33.1 | 77.11 ms | 13.0 | 11.99 ms |

- Three objects cost about 8 % more than one (upstream: 2.3x), because the image
  encoder runs once and the objects share one batched pass.
- The stream timing includes the frame upload and on-device resize; upstream's
  frames were preloaded on the GPU.
- CUDA kernel time is 9-12 ms of the 28-30 ms per frame; the rest is CPU-side.
  The profiler shows about 925 kernel launches per frame (`cudaLaunchKernel`
  is the largest self-CPU entry). This is the overhead CUDA graphs or
  `torch.compile` would remove.
- 3 host-device syncs per frame, from preprocessing (the mean/std tensors and
  the frame upload).

The profiler's own wall time (110-115 ms/frame) includes profiling overhead and
is not a latency figure.

## Not run

- Model input 1024 (not enough free GPU memory).
- Hole filling (needs the `_C` extension; the production image builds it).
- The full `dahi-live-ai` service with `TRACKER_IMPL=stream`.

Raw logs: `~/sam2-stream-test/results/` on `afta2`.
