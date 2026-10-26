# SAM2StreamPredictor — design

Date: 2026-10-24
Branch: `feat/stream-predictor`

## Why

`dahi-live-ai` tracks objects in live WebRTC streams with `SAM2CameraPredictor`
from the Gy920 `segment-anything-2-real-time` fork. That predictor has two
problems the service cannot work around:

1. **One object per session.** It uses the pre-12/2024 batched SAM 2 logic:
   objects can only be added before the first `track()`. Adding an object later
   means `reset` + `load_first_frame`, which wipes every object's memory. The
   service therefore tracks exactly one object per session.
2. **Does not scale with sessions.** Each session is a shallow copy of the
   predictor holding its state on `self`. Everything runs in one Python process
   in eager mode. The 2026-09-24 SLA run shows `SAM2 forward` at 106 ms for one
   room (SAM 2.1 small, 512 px, RTX 3080 Ti) with GPU SM at ~10 %, growing
   linearly with rooms while SM stays flat: the cost is CPU-side (Python,
   kernel launches, GIL), not GPU compute. A bigger GPU does not help.

It also carries bugs (undefined `add_all_frames_to_correct_as_cond`, tensor
`bbox` silently ignored, `vos_optimized` compiling nothing,
`add_new_prompt_during_track` broken).

Upstream SAM 2 (12/2024) already has the right semantics for objects
(`SAM2VideoPredictor`: every object independent, addable any time) but needs the
whole video up front. This work brings those semantics to live streams.

## Goals (this iteration)

- New `SAM2StreamPredictor` in `sam2/sam2_stream_predictor.py`, built by
  `build_sam2_stream_predictor(...)` in `sam2/build_sam.py`.
- Frames arrive one at a time; no video length is known; nothing grows with
  stream length (no frame list, bounded memory bank).
- Multiple independent objects per session. Starting target: **3 objects**.
  Objects can be added, re-prompted (corrected), re-anchored and removed at any
  frame without touching other objects.
- Session state lives outside the model. One predictor (one copy of weights)
  serves any number of sessions; no `copy.copy` of the module.
- Objects of a session are batched through memory attention, the SAM heads and
  the memory encoder; the image encoder runs once per frame.
- The internal per-frame core operates on "rows" (one row = one object with its
  own image-feature row), so a later cross-session engine can batch rows from
  different sessions without redesign.
- Frame preprocessing (resize + normalize) on the model device.
- Correctness proven against upstream `SAM2VideoPredictor`; tests are mandatory.

## Non-goals (this iteration)

- Cross-session batching engine, multi-process serving, `torch.compile` / CUDA
  graphs. The design must not block them; they are later phases.
- Changing the model or retraining. No edits to `sam2/modeling/*` unless a
  change is tiny and required; none are expected.
- Reverse tracking, CPU offload of session state, non-overlap constraints.
- Integration into `dahi-live-ai` (separate repo, separate change).

## Anti-goals

- Any behaviour change versus upstream SAM 2 for the same inputs. A single
  object tracked by the stream predictor must give the same masks as upstream
  `SAM2VideoPredictor` given the same preprocessed frames and prompts.
- Hidden global state, hardcoded `cuda`, or per-session copies of the model.
- Claims of speed-up without a benchmark number from the GPU server.

## Public API

```python
predictor = build_sam2_stream_predictor(config, ckpt, device="cuda")
session = predictor.create_session(max_objects=None)

predictor.set_frame(session, frame)                    # HxWx3 uint8 RGB ndarray or tensor
predictor.add_prompt(session, obj_id, box=..., points=..., labels=..., mask=...)
out = predictor.track(session)                         # StreamOutput for the current frame
out = predictor.step(session, frame)                   # set_frame + track
predictor.reset_object(session, obj_id)                # drop memory, keep id (re-anchor)
predictor.remove_object(session, obj_id)
predictor.reset_session(session)
```

`StreamOutput`: `frame_idx`, `obj_ids` (list), `mask_logits`
(`[N,1,H,W]` at video resolution, on device), `low_res_mask_logits`
(`[N,1,S/4,S/4]`), `object_score_logits` (`[N,1]`). Callers threshold at 0.

Frame lifecycle inside a session:

1. `set_frame` finalizes the previous frame (see below), drops its features,
   increments `frame_idx`, preprocesses the new frame on device and caches its
   backbone features.
2. `add_prompt` on the current frame:
   - unknown `obj_id` → new object; this frame becomes its conditioning frame
     (no memory used, like SAM);
   - known object → correction on this frame: re-run with the new points plus
     the previous mask logits, stored as non-conditioning (upstream default
     `add_all_frames_to_correct_as_cond=False`);
   - result is *pending*: memory is not encoded yet, so more clicks can follow.
   Returns the masks of all objects on this frame.
3. `track` returns masks for all objects on the current frame. Objects with a
   pending prompt output return it; the rest run memory-conditioned inference
   (batched) and their memory is encoded immediately.
4. Finalize (on the next `set_frame`): pending outputs get their memory encoded
   (batched, `is_mask_from_pts=True`) and are committed.

## Session and memory model

- `StreamSession`: device, video H/W, `frame_idx`, current frame image and
  backbone features, ordered `objects: {obj_id: ObjectState}`, constants cache
  (`maskmem_pos_enc`).
- `ObjectState`: `cond_outputs {frame_idx: out}` (normally one entry: the
  prompt frame), `non_cond_outputs` (`OrderedDict`, pruned), `pending` output
  for the current frame, `point_inputs` for the current frame (for corrections).
- Each output stores `maskmem_features` (bf16, as upstream), `maskmem_pos_enc`
  (shared constant), `pred_masks` (low-res), `obj_ptr`, `object_score_logits`.
- Pruning: after each frame, drop non-cond entries older than the window the
  model can read: memory frames (`num_maskmem - 1` with temporal stride) and
  object pointers (`max_obj_ptrs_in_encoder - 1`). With defaults this is the last
  15 frames. Conditioning outputs stay until `reset_object` / `remove_object`.
- `num_frames` passed to the model is a large constant so
  `min(num_frames, max_obj_ptrs_in_encoder)` equals upstream on any video of at
  least 16 frames.

## Batching rule

Memory attention needs the same memory layout for every row in a batch. Rows
are grouped by a key: (number of conditioning memories, the set of non-cond
`t_pos` slots present, number of pointer tokens). Each group is one batched
call; temporal positional encodings of object pointers are computed per row
(objects have different conditioning frames), not broadcast. In steady state
(all objects older than the window, one conditioning frame each) every object
of a session is in one group.

## Testing

- **CPU, local** (pytest, `tests/`): random-init tiny model at a small image
  size (e.g. 256) via hydra overrides. Covers API behaviour, object add/remove
  at any frame, re-anchor, memory bound over long streams, no growth of state,
  output shapes, device placement, and parity of single-object and batched
  paths against upstream `SAM2VideoPredictor` on the same preprocessed frames.
- **GPU, server**: real SAM 2.1 checkpoints on `notebooks/videos/bedroom`.
  Parity per object against upstream (objects added at different frames are
  compared to a single-object upstream session started at their prompt frame),
  and a benchmark script reporting ms/frame and FPS for 1 and 3 objects,
  with CUDA-synchronized timing and `torch.profiler` kernel-vs-CPU split.

Parity criterion: single-object path in fp32 on CPU matches upstream to
`atol=1e-4` on logits; batched path IoU ≥ 0.99 per object per frame; GPU bf16
runs IoU ≥ 0.98.

## Left to the builder

Exact internal function boundaries, test fixture layout, the grouping-key
implementation, and the benchmark script's output format.

## Calls made without the user (check these)

- New file and class instead of patching `SAM2CameraPredictor` or the fork.
- API shape (`set_frame` / `add_prompt` / `track`), names, and `StreamOutput`.
- Corrections are non-conditioning (upstream default).
- Long-stream pointer distance to the conditioning frame grows without bound,
  exactly as upstream on long videos; not changed here (anti-goal: no behaviour
  change). Re-anchoring via `reset_object` is the tool for very long streams.
