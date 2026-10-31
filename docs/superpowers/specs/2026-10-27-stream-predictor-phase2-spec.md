# SAM2StreamPredictor phase 2: CUDA graphs and `torch.compile`

Status: draft for review. Version 0.1, 2026-10-27. Branch `feat/stream-predictor`
(fork), `feature/sam2-stream-predictor` (dahi-live-ai).
Builds on `2026-10-25-stream-predictor-spec.md` (phase 1, "the phase 1 spec");
everything there still holds unless this document changes it. "MUST", "MUST
NOT", "SHOULD" are used in their RFC 2119 sense.

---

## 1. Why

The live tracker is CPU-bound, not GPU-bound. Measured on afta2 (RTX 3080 Ti,
torch 2.5.1, bf16, model input 512, `sam2.1_hiera_small`, 3 objects):

| Measurement | Wall time per frame | CUDA kernel time per frame | Kernel launches per frame |
|---|---|---|---|
| `SAM2StreamPredictor.step()` alone (`tools/benchmark_stream.py`, commit `ef11f8d`) | 30 ms (p50) | 12 ms | ~925 |
| `StreamTracker.process_frame` replay (`benchmark/replay_stream_tracker.py`, dahi `58b5a4d`) | 56 ms (p50), 72 ms (p95) | 19 ms | ~1800 |

In the live service the GPU runs at about 15-20 % while one CPU core is busy.
About two thirds of every frame is Python and kernel-launch overhead under one
GIL. That is why adding GPUs does not add frames per second: the GPU waits
for the CPU.

Phase 2 removes most of that overhead. Phase 3 (multi-process workers per GPU
with NVIDIA MPS) then turns the freed GPU time into more rooms. Only both
together make the capacity statement for the organization true: more GPU
gives more rooms at the same FPS, or more FPS per room.

## 2. Goals and acceptance criteria

All numbers on afta2, same setup as §1, bedroom clip, after warm-up.

| ID | Criterion | Target |
|---|---|---|
| A1 | Predictor `step()` p50, 3 objects | ≤ 16 ms (from 30 ms) |
| A2 | Predictor kernel launches per frame, 3 objects, steady state | ≤ 150 (from ~925) |
| A3 | Tracker replay p50, 3 objects, ReID on | ≤ 32 ms (from 56 ms) |
| A4 | Tracker replay p95 | ≤ 42 ms (from 72 ms) |
| A5 | Parity, graphs vs eager, fp32 | per-frame mask IoU ≥ 0.999 on every frame, 200 frames |
| A6 | Parity, graphs vs eager, bf16 | per-frame mask IoU ≥ 0.99 on every frame, 200 frames |
| A7 | Parity, compiled (phase 2b) vs eager, bf16 | min IoU ≥ 0.98, mean ≥ 0.995 |
| A8 | VRAM reserved, whole service, increase over phase 1 | ≤ 300 MB |
| A9 | Service start, graphs only (2a) | ≤ 20 s more than today |
| A10 | Fallback to eager | correct output on every path; never an error to the room |

A1/A3 are the bar for "done". If 2a reaches them, 2b (compile) is optional and
decided by numbers, not by default.

## 3. Scope

### 3.1 In scope

- 2a. Manual CUDA graph capture (`torch.cuda.CUDAGraph`) of the fixed-shape
  parts of a frame in `SAM2StreamPredictor`: image encoder, memory attention
  with the SAM heads, memory encoder.
- 2a. Removing per-row launch overhead in memory assembly (precomputed
  pointer position table, one buffer write per row).
- 2a. dahi-live-ai: batching the per-object tracker work that today runs once
  per object (§9).
- 2b. `torch.compile` (inductor, no inductor CUDA graphs) of the same modules,
  captured by the same graph runner. Gated on 2a's numbers.
- Tests, parity and benchmark tooling for all of the above.

### 3.2 Out of scope

- Multi-process serving and NVIDIA MPS (phase 3).
- Batching rows from several sessions into one call (cross-room batching).
  This spec MUST NOT make it harder: buckets are keyed by batch size, so a
  later engine can fill a bucket with rows from several rooms.
- Model input 1024, the large model, TensorRT/ONNX export.
- Changes to `sam2/modeling/` (constraint C1 of the phase 1 spec still holds).

### 3.3 Hard constraints

- P1. Eager mode stays the default and stays bit-identical to phase 1. Graphs
  are opt-in: `SAM2StreamPredictor(..., cuda_graphs=False)` and
  `SAM2_CUDA_GRAPHS=0` in dahi-live-ai until A1-A10 pass.
- P2. Any shape, state or error the graph path does not handle runs eager for
  that call. A capture failure disables graphs for that key and logs once at
  WARNING; it never raises into the caller.
- P3. No tensor that a session keeps (memory features, pointers, masks) may
  alias a graph's static buffer. Outputs are copied out before the next replay
  can run.
- P4. Graph work is thread-safe: the dahi service calls `track()` from a
  thread pool, several rooms at once (§6).
- P5. No host-device synchronization inside a captured region.

## 4. Approach and the decision behind it

Three options were weighed:

1. `torch.compile(mode="reduce-overhead")`, as upstream `SAM2VideoPredictorVOS`
   does (with `max-autotune`). Inductor's CUDA graph trees are tied to the
   calling thread and re-record on new shapes; our calls come from a thread
   pool and the memory length changes while an object warms up. Startup
   compile with `max-autotune` takes minutes and needs a C compiler and a
   persistent cache in the image. Highest ceiling, most risk.
2. Manual CUDA graphs over eager kernels. Removes exactly what costs us (CPU
   launch overhead), keeps the same kernels so numerics match eager, captures
   in seconds, no compiler in the image, full control over threads and streams.
3. Padding the memory to one maximum length with an attention mask, so one
   graph per batch size covers everything. Rejected: a mask moves
   `scaled_dot_product_attention` off the flash kernel, and padding a typical
   7-frame memory to the 10-frame maximum costs about 40 % more attention work
   on every frame, to save captures that happen once.

Decision: option 2 first (phase 2a), with shape buckets (§5.2). Option 1's
kernel fusion comes back in phase 2b as `torch.compile` with
`mode="max-autotune-no-cudagraphs"`, captured by our own runner, only if 2a
leaves kernel time as the limit.

## 5. Design (fork, `sam2/`)

### 5.1 Files

| Path | Responsibility |
|---|---|
| `sam2/stream/graphs.py` (new) | `GraphRunner`: capture, bucket cache, replay, copy-out, locking |
| `sam2/stream/memory.py` | `write_row_memory` into a static buffer; pointer position table |
| `sam2/sam2_stream_predictor.py` | `cuda_graphs` flag; route image encoder, `_run_heads`, `_encode_rows` through the runner; `warmup_graphs(batch_sizes)` |
| `sam2/build_sam.py` | `build_sam2_stream_predictor(..., cuda_graphs=False, compile_mode=None)` |
| `tools/benchmark_stream.py`, `tools/stream_parity.py` | `--cuda-graphs`, `--compile` flags |
| `tests/test_stream_graphs.py` (new) | CPU tests (§10) |

### 5.2 Captured regions and their keys

| Region | Inputs (static buffers) | Key | Graphs |
|---|---|---|---|
| Image encoder (`forward_image` + `_prepare_backbone_features` for B=1) | image `[1, 3, S, S]` | `S` | 1 |
| Memory attention + SAM heads (`_run_heads` body) | features (shared with the image encoder graph's outputs, expanded to B inside), memory `[B, M, C_mem]`, memory pos `[B, M, C_mem]`, pointer positions `[B, P]` int | `(B, n_cond, non_cond_t_pos, P)` = phase 1 `RowMemory.key` plus B | one per key seen |
| Memory encoder (`_encode_new_memory`) | features, high-res masks `[B, 1, S, S]`, object scores `[B, 1]` | `B` | up to max objects |

`M = (n_cond + len(non_cond_t_pos)) * (S/16)^2 + P`. The temporal position
encoding added to each memory frame depends only on its slot `t_pos`, which is
part of the key, so it is a constant inside the graph, not a per-row kernel.

Steady state needs few keys. An object prompted once and tracked for more
than 16 frames has `n_cond = 1`, `non_cond_t_pos = (1..6)`, `P = 64`. A
re-anchor adds a conditioning frame (cap 4, phase 1 §6.6), so keys with
`n_cond` 2-4 also occur. With B in 1..3 that is about 12 keys in normal use.
The first ~16 frames of each object, and frames after a memory gap
(`_drop_memory_frame` in dahi), have other keys.

### 5.3 Bucket policy

- A key is captured on its second sighting (one-off keys from an object's
  first frames never cost a capture). Until then the call runs eager.
- Cache size limit: 24 graphs, least recently used evicted. All graphs share
  one memory pool (`torch.cuda.graph_pool_handle()`); safe because replays are
  serialized (§6).
- `warmup_graphs(batch_sizes=(1, 2, 3), cond_counts=(1, 2))` at service start
  captures the steady-state keys with a synthetic session run past 16 frames.
  The dahi warmup (`warmup_stream_predictor`) MUST call it and run at least
  20 frames per batch size when graphs are on.
- Counters, exposed via `predictor.graph_stats()`: captures, replays, eager
  calls by reason (`unseen_key`, `capture_failed`, `graphs_off`), cache
  evictions. A fallback rate above 10 % of track calls in steady state fails A2.

### 5.4 Memory assembly without per-row launch storms

Phase 1 `build_row_memory` runs about 30-50 small kernels per row per frame
(flatten, permute, add, `torch.tensor(pos_list).to(device)`, sine encoding,
`obj_ptr_tpos_proj`). Phase 2:

- Pointer positions: precompute `obj_ptr_tpos_proj(get_1d_sine_pe(t / t_max))`
  for every `t` in `0..max_obj_ptrs_in_encoder-1` once per model, as a table
  `[T, mem_dim]`. A row's pointer positions become one index into the table,
  done inside the graph from the int input buffer. MUST equal the phase 1
  values bit for bit in fp32 (test, §10).
- `write_row_memory(model, obj, frame_idx, out_memory, out_pos, out_ptr_pos)`
  writes one row straight into the static buffers: one `torch.cat(..., out=)`
  for memory, one for positions, one small host-to-device copy for pointer
  positions. The layout MUST equal `stack_rows(build_row_memory(...))`
  (test, §10).
- Eager mode keeps using `build_row_memory` unchanged (P1).

### 5.5 Outputs

After a replay the runner copies the outputs the session keeps
(`pred_masks`, `pred_masks_high_res` only while needed for the memory encoder
and non-overlap, `obj_ptr`, `object_score_logits`, `maskmem_features`,
`maskmem_pos_enc`) into fresh tensors with one `clone()` per output tensor per
group, never per row. Per-row views are then slices of those clones, as in
phase 1.

### 5.6 What stays eager

Preprocessing (resize, normalize), `_finalize`, `_compact`, hole filling,
`non_overlap_rows`, `_competitor_masks`, `_assemble_output`, and every
prompt-path call (`add_prompt` runs once per user click). Their launch count
is small and their shapes follow the video size; they are measured, not
captured. If the measurement shows one of them above 30 launches per frame it
gets its own item, not a graph.

## 6. Threads and streams

dahi-live-ai calls `track()` from `asyncio.to_thread` workers; each tracker
runs inside its own CUDA stream (`tracking_stream`). Static buffers are shared
by every room, so:

- One process-wide `threading.Lock` per predictor covers copy-in, replay and
  copy-out of every graph.
- All graph work runs on one dedicated predictor stream. On entry the
  predictor stream waits on an event recorded on the caller's stream; on exit
  the caller's stream waits on an event recorded after copy-out. Without this,
  room A's copy-out (queued on stream A) could still be pending when room B's
  replay (on stream B) overwrites the buffers.
- Captures use `capture_error_mode="thread_local"`, so YOLO, DINO and other
  rooms can keep using CUDA on other threads during a lazy capture.
- The lock is held for GPU enqueue only (well under a millisecond per replay),
  not for GPU execution, so it does not serialize rooms on the GPU.

## 7. Precision

- Graphs replay the same kernels as eager, so bf16 drift does not change. The
  phase 1 result stands: batched bf16 drifts (worst frame IoU 0.927 vs
  upstream); A6 compares graphs with eager at the same batch size, not with
  upstream.
- Capture happens under the same autocast dtype the caller uses. A replay
  under a different dtype MUST run eager (part of the key).
- `torch.backends.cudnn.benchmark` MUST be off during capture so the captured
  algorithms are the ones eager picks.

## 8. Phase 2b: `torch.compile` (gated)

Start only if 2a passes A5/A6 and kernel time is then more than half of the
predictor's frame time.

- Compile `image_encoder`, `memory_attention`, `sam_mask_decoder`,
  `sam_prompt_encoder` and `memory_encoder` forwards with
  `mode="max-autotune-no-cudagraphs"`, `dynamic=False`, then capture the
  compiled code with the same `GraphRunner`. Our runner keeps control of
  threads and buckets; inductor does fusion only.
- Each key compiles once; `torch._dynamo.config.cache_size_limit` raised to
  cover the bucket limit.
- Persistent caches (`TORCHINDUCTOR_CACHE_DIR`, FX graph cache, autotune
  cache) on a mounted volume, so a restart does not recompile.
- Image prerequisites to check first: a C compiler for Triton's launcher, and
  enough startup budget. A9 does not apply to 2b; its start-time budget is set
  after a first measurement.

## 9. dahi-live-ai side

The tracker adds about 870 launches and 26 ms per frame on top of the
predictor (§1). Graphs in the fork do not touch that. Work items, each with a
replay measurement before and after:

1. T0. Attribute launches and wall time per stage (`process_frame`,
   `_step_object`, `_extract_mask`, `_identity_sim`, `_publish_filter`, gallery
   updates, polygon) with `record_function` labels in
   `benchmark/replay_stream_tracker.py`. No code change outside the benchmark.
   The next items are ordered by what T0 finds.
2. Mask statistics for all objects in one pass: one `[N, H, W]` threshold,
   area and extent computation and one device-to-host copy per frame, instead
   of one per object.
3. DINO identity checks for all due objects in one batched forward per frame
   (the gallery update is already batched in phase 1).
4. Polygon extraction from one downloaded `[N, h, w]` mask block, not one
   download per object.
5. `SAM2_CUDA_GRAPHS` env flag (default `0`), passed to the builder through
   `StreamPredictorRegistry`; the startup warmup calls `warmup_graphs` when on.
6. `get_stats()` and the monitoring route expose `predictor.graph_stats()`.

## 10. Tests

CPU (`pytest`, fork; `unittest`, dahi), written first:

- Pointer position table equals phase 1 `obj_pos` for every `t`, fp32, exact.
- `write_row_memory` into a buffer equals `stack_rows(build_row_memory(...))`
  for: steady state, 2-4 conditioning frames, a gap in the non-conditioning
  frames, fewer than 16 frames tracked.
- `GraphRunner` with a CPU fake capture (calls the function, records keys):
  second-sighting policy, LRU eviction, eager fallback reasons, capture
  failure disables only its key, dtype in the key.
- Outputs kept by a session do not share storage with runner buffers
  (`untyped_storage().data_ptr()` check), P3.
- Predictor with `cuda_graphs=True` on CPU runs eager and matches
  `cuda_graphs=False` exactly.
- dahi: batched mask statistics and identity checks equal the per-object
  results; flag plumbing; warmup calls `warmup_graphs` only when on.

GPU (afta2, throwaway `sam2-stream-test` container, live service untouched):

- `tools/stream_parity.py --cuda-graphs`: A5, A6 (and A7 with `--compile`).
- Two-thread stress test: two sessions on two threads and two streams,
  1000 frames each, graphs on; masks equal to the same sessions run one after
  the other. This is the test for §6.
- `tools/benchmark_stream.py --cuda-graphs`: A1, A2, `graph_stats()`.
- `benchmark/replay_stream_tracker.py` with `SAM2_CUDA_GRAPHS=1`: A3, A4, A8.
- Service start time and VRAM after warmup: A8, A9.

## 11. Order of work

1. T0 measurement (§9.1) on the current code. Gives the real split between
   predictor and tracker before anything changes.
2. Pointer position table and `write_row_memory` (§5.4), eager users
   unchanged.
3. `GraphRunner` with the CPU fake; tests.
4. Image encoder graph; benchmark.
5. Memory attention + heads graphs with buckets; parity; benchmark.
6. Memory encoder graphs; parity; benchmark.
7. Threads and streams (§6); two-thread stress test.
8. dahi items 2-6 (§9), each measured.
9. Deploy to the test service with `SAM2_CUDA_GRAPHS=1`; live check.
10. Decide 2b from the numbers.

Each step is one commit with its own replay or benchmark numbers in the
results document.

## 12. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Hidden sync or dynamic allocation in a captured module | Capture fails | P2 fallback; capture errors list the op; fix outside `sam2/modeling/` |
| More keys than expected (re-anchors, memory gaps) | Many eager calls | `graph_stats()` fallback counter; raise cache size or add the key to warmup |
| Graph memory pool larger than expected | A8 missed | One shared pool; measure per region; drop memory encoder graphs first |
| Lock or stream bug between rooms | Wrong masks in one room | §6 design; two-thread stress test is a release gate |
| Tracker side dominates after 2a | A3 missed while A1 passes | §9 items are part of phase 2, not optional |
| bf16 batched drift | Already present in phase 1 | Unchanged by graphs; tracked separately |

## 13. Open questions for the reviewer

1. Is A3 (tracker p50 ≤ 32 ms, about 30 FPS for 3 objects in one room) the
   right bar, or should the bar be rooms per GPU at a fixed FPS? The second
   needs phase 3 to measure.
2. Phase 2b (compile) needs a C compiler in the production image and a cache
   volume. Is changing the image acceptable, or should 2b stay out until
   phase 3?
3. Should the dahi items (§9) land in the same branch as the fork changes, or
   in a separate branch so each can be deployed alone?
