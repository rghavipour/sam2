# SAM2StreamPredictor — implementation specification

Status: normative. Version 1, 2026-10-25. Branch `feat/stream-predictor`.
Supersedes the API sketch in `2026-10-24-stream-predictor-design.md` wherever
the two differ. "MUST", "MUST NOT", "SHOULD" are used in their RFC 2119 sense.

Upstream references below are to this repository at commit `2b90b9f`:
`sam2/modeling/sam2_base.py` (`SAM2Base`), `sam2/sam2_video_predictor.py`
(`SAM2VideoPredictor`, "upstream"), `sam2/sam2_video_predictor_legacy.py`
("legacy").

---

## 1. Scope

### 1.1 In scope (phase 1)

- `SAM2StreamPredictor`: live-stream inference, one frame at a time, unknown
  stream length, any number of independent objects per session.
- Session state held outside the model (`StreamSession`).
- Batched inference of all objects of one session that share a memory layout.
- On-device frame preprocessing.
- CPU test suite, parity against upstream and legacy, GPU parity and benchmark
  tools.

### 1.2 Out of scope (phase 1)

Cross-session batching engine, multi-process serving, `torch.compile`, CUDA
graphs, reverse tracking, CPU offload of session state, Hugging Face loading,
integration into `dahi-live-ai`. Nothing in this spec may make these harder:
the per-frame core MUST operate on rows (§7) so rows from several sessions can
later share one call.

### 1.3 Hard constraints

- C1. No file under `sam2/modeling/` is modified.
- C2. No model copies (`copy.copy`, `copy.deepcopy`, re-instantiation) per
  session. One predictor instance serves every session.
- C3. No hardcoded device. Every tensor is created on `self.device` or on the
  session's device.
- C4. A session never stores past frames, past images or past backbone
  features. Only the current frame's features are cached.
- C5. For the same weights, preprocessed frames and prompts, a single object
  MUST produce the same mask logits as upstream (§12.2).
- C6. Public methods run under `torch.inference_mode()`. The predictor never
  enters `torch.autocast` itself; callers do.
- C7. No `print`. Use `logging.getLogger(__name__)`.

---

## 2. Files

| Path | Responsibility |
|---|---|
| `sam2/sam2_stream_predictor.py` | `SAM2StreamPredictor` class: public API, frame lifecycle, prompt path, track path, output assembly |
| `sam2/stream/__init__.py` | Re-exports `StreamSession`, `ObjectState`, `StreamOutput`, `preprocess_frame` |
| `sam2/stream/preprocess.py` | `preprocess_frame` |
| `sam2/stream/state.py` | Dataclasses, `min_needed_frame`, `prune_object`, `evict_cond_frames` |
| `sam2/stream/memory.py` | `RowMemory`, `build_row_memory`, `group_rows`, `stack_rows` |
| `sam2/build_sam.py` | Add `build_sam2_stream_predictor` (append only; existing functions unchanged) |
| `tests/conftest.py` | Fixtures (§12.1) |
| `tests/test_build.py`, `tests/test_stream_preprocess.py`, `tests/test_stream_state.py`, `tests/test_stream_memory.py`, `tests/test_stream_predictor.py`, `tests/test_stream_parity.py` | Test suites (§12) |
| `tools/stream_parity.py`, `tools/benchmark_stream.py` | GPU tools (§13) |
| `pyproject.toml` | `[tool.pytest.ini_options]`: `testpaths = ["tests"]`, filter the `Skipping the post-processing step` `UserWarning` |
| `setup.py` | `"dev"` extra gains `"pytest>=8.0"` |
| `.gitignore` | `.venv/` |

---

## 3. Constants

Defined in `sam2/sam2_stream_predictor.py`:

```python
STREAM_NUM_FRAMES = 1 << 30   # passed as `num_frames` to SAM2Base methods
```

Rationale: `SAM2Base._prepare_memory_conditioned_features` uses
`min(num_frames, max_obj_ptrs_in_encoder)` both as the pointer window and as
the temporal-encoding normalizer. Upstream on any video of ≥ 16 frames gets 16;
the stream must get the same.

`NO_OBJ_SCORE` is imported from `sam2.modeling.sam2_base` (value `-1024.0`).

---

## 4. Data model (`sam2/stream/state.py`)

All are `@dataclass` (not frozen). Tensors live on `session.device`.

### 4.1 Frame output record ("out")

A plain `dict` with exactly these keys, the same shape upstream's
`_run_single_frame_inference` produces for batch size 1:

| key | shape / type | notes |
|---|---|---|
| `maskmem_features` | `[1, C_mem, S/16, S/16]`, `bfloat16`, or `None` | `None` only while pending |
| `maskmem_pos_enc` | `list[Tensor[1, C_mem, S/16, S/16]]` or `None` | the session constant (§4.3) |
| `pred_masks` | `[1, 1, S/4, S/4]`, `float32` | low-res logits, holes filled if `fill_hole_area > 0` |
| `obj_ptr` | `[1, C]` | |
| `object_score_logits` | `[1, 1]` | |

`S = image_size`, `C = hidden_dim`, `C_mem = mem_dim`.

### 4.2 `ObjectState`

```python
@dataclass
class ObjectState:
    obj_id: Hashable
    cond_outputs: dict[int, dict] = field(default_factory=dict)
    non_cond_outputs: "OrderedDict[int, dict]" = field(default_factory=OrderedDict)
    pending: dict | None = None          # output from add_prompt on the current frame
    pending_is_cond: bool = False
    point_inputs: dict | None = None     # {"point_coords": [1,P,2], "point_labels": [1,P]}
    point_inputs_frame: int = -1         # frame the point_inputs belong to
    tracked_frame: int = -1              # last frame track() produced output for this object
    first_cond_frame: int | None = None  # protected from eviction (§6.6)
```

### 4.3 `StreamSession`

```python
@dataclass
class StreamSession:
    device: torch.device
    max_objects: int | None = None
    frame_idx: int = -1
    video_h: int = 0
    video_w: int = 0
    backbone_out: dict | None = None             # current frame, batch 1, from forward_image
    image: torch.Tensor | None = None            # current frame [1,3,S,S]
    feature_cache: dict[int, tuple] = field(default_factory=dict)  # batch size -> features (§7.1)
    objects: "OrderedDict[Hashable, ObjectState]" = field(default_factory=OrderedDict)
    constants: dict = field(default_factory=dict)  # "maskmem_pos_enc": list[Tensor[1,...]]

    @property
    def obj_ids(self) -> list: ...               # list(self.objects)
```

### 4.4 `StreamOutput`

```python
@dataclass
class StreamOutput:
    frame_idx: int
    obj_ids: list
    mask_logits: torch.Tensor          # [N,1,video_h,video_w] float32
    low_res_mask_logits: torch.Tensor  # [N,1,S/4,S/4] float32
    object_score_logits: torch.Tensor  # [N,1] float32
```

Object order is `session.objects` insertion order. `N == 0` gives tensors with
first dimension 0 and the shapes above.

---

## 5. Public API (`SAM2StreamPredictor`)

`class SAM2StreamPredictor(SAM2Base)`.

### 5.1 Constructor

```python
def __init__(self, fill_hole_area=0, non_overlap_masks=False,
             batch_objects=True, max_cond_frames_per_object=4, **kwargs)
```

- `**kwargs` go to `SAM2Base.__init__`.
- `ValueError("max_cond_frames_per_object must be >= 1")` if `< 1`.
- `ValueError("SAM2StreamPredictor needs num_maskmem >= 1")` if
  `self.num_maskmem == 0`.
- `non_overlap_masks_for_mem_enc` (config) becomes `self.non_overlap_across_objects`
  and SAM2Base's own flag is set to False. When on, every mask that enters the
  memory encoder (tracked rows in `track`, prompt outputs in finalize) is first
  constrained with `_apply_non_overlapping_constraints` against the other objects
  of the session on that frame (`non_overlap_rows`), so each pixel stays positive
  for at most one object. This is legacy SAM 2's multi-object behaviour; with the
  flag off (the default) objects stay independent, as in upstream.

### 5.2 Builder (`sam2/build_sam.py`)

```python
def build_sam2_stream_predictor(config_file, ckpt_path=None, device="cuda",
        mode="eval", hydra_overrides_extra=[], apply_postprocessing=True)
```

Identical to `build_sam2_video_predictor` except: target
`++model._target_=sam2.sam2_stream_predictor.SAM2StreamPredictor`, no
`vos_optimized`, no `**kwargs`. Same postprocessing overrides (dynamic
multimask, `binarize_mask_from_pts_for_mem_enc=true`, `fill_hole_area=8`).
Predictor options go through `hydra_overrides_extra`, e.g.
`"++model.batch_objects=false"`.

### 5.3 Methods

| Method | Returns | Behaviour |
|---|---|---|
| `create_session(max_objects=None)` | `StreamSession` | `device=self.device`. `max_objects` `None` or `>= 1`, else `ValueError` |
| `set_frame(session, frame)` | `int` frame index | §6.1; `frame` is validated and preprocessed (§9) |
| `set_preprocessed_frame(session, image, video_hw)` | `int` | §6.1 with an already-normalized `image` `[3,S,S]` float32 (any device; moved to session device). `ValueError` if shape is not `(3, S, S)`. `video_hw = (h, w)`, both `>= 1` |
| `add_prompt(session, obj_id, points=None, labels=None, box=None, mask=None, clear_old_points=True, normalize_coords=True)` | `StreamOutput` | §6.2 |
| `track(session)` | `StreamOutput` | §6.3 |
| `step(session, frame)` | `StreamOutput` | `set_frame` then `track` |
| `reset_object(session, obj_id)` | `None` | §6.5. `KeyError` if unknown |
| `remove_object(session, obj_id, strict=False)` | `None` | §6.5. Unknown id: no-op, or `KeyError` if `strict` |
| `reset_session(session)` | `None` | drops all objects, pending outputs, features and constants; `frame_idx` back to `-1` |

Error messages (exact):

- No frame yet: `RuntimeError("call set_frame() before add_prompt() or track()")`.
- Session full: `RuntimeError(f"session already tracks max_objects={n} objects")`.
- Bad frame: `ValueError(f"frame must be a uint8 HxWx3 RGB array, got dtype={d} shape={s}")`.

Thread safety: the predictor holds no per-session state, so different sessions
may be used from different threads. One session MUST NOT be used by two
threads at once (not enforced). All sessions on one predictor SHOULD use one
`image_size` (RoPE caches its table per input size on the module).

---

## 6. Behaviour

Notation: `f = session.frame_idx`; `obj = session.objects[obj_id]`.

### 6.1 `set_frame` / `set_preprocessed_frame`

In this order:

1. If `f >= 0`: **finalize** frame `f` (§6.4).
2. If `f >= 0`: prune every object with
   `keep_from = min_needed_frame(f + 1, ...)` (§6.6).
3. Drop `backbone_out`, `image`, `feature_cache`.
4. `f += 1`; store `video_h`, `video_w` of this frame (may differ from the
   previous frame; masks of later outputs use the new size).
5. `image` = preprocessed `[1,3,S,S]` on device;
   `backbone_out = self.forward_image(image)` (batch 1).
6. Return `f`.

### 6.2 `add_prompt`

Validation (raise `ValueError`):

- None of `points`, `box`, `mask` given.
- `points` given without `labels` or vice versa.
- `mask` given together with `points` or `box`.
- `box` given with `clear_old_points=False` (same rule and reason as upstream).
- `points` not shaped `[P,2]` / `[1,P,2]`; `labels` not `[P]` / `[1,P]`;
  `box` not 4 numbers (`[x1,y1,x2,y2]`, also accepted as `[2,2]`); `mask` not 2-D.

Object lookup: unknown `obj_id` → new `ObjectState` appended to
`session.objects` (check `max_objects` first). Existing ids keep their
position.

Mode selection:

| Condition | `is_init_cond_frame` | stored as |
|---|---|---|
| `obj.tracked_frame != f` (object has no output for this frame yet; includes new objects and frames not yet tracked) | `True` | pending cond |
| `obj.tracked_frame == f` and `f` is a cond frame of `obj` (`f in obj.cond_outputs`, or `obj.pending is not None and obj.pending_is_cond`) | `True` | pending cond (replaces that cond frame) |
| `obj.tracked_frame == f` and `f` is not a cond frame of `obj` | `False` (correction) | pending non-cond |

The first two rows follow upstream: a prompt on a frame the object has not been
tracked on is a conditioning frame, segmented without memory, like SAM. The
second row deliberately differs from upstream and legacy, which store the
correction as non-cond and then drop it in preflight because a cond output
exists on the same frame; here the cond frame is re-run instead.

Points path (points and/or box):

1. Convert to float32 / int32 tensors, add batch dim.
2. Box: prepend its two corners with labels `2, 3`.
3. `normalize_coords=True`: divide by `[video_w, video_h]` of the current
   frame. Then multiply by `image_size`. Move to device.
4. If `clear_old_points=False` and `obj.point_inputs_frame == f`: append to
   `obj.point_inputs` with `concat_points`; else start new.
5. Store `obj.point_inputs`, `obj.point_inputs_frame = f`.
6. `prev_sam_mask_logits`: first found of `obj.pending`, `obj.cond_outputs[f]`,
   `obj.non_cond_outputs[f]`; its `pred_masks` clamped to `[-32, 32]`; else
   `None`.

Mask path:

1. `mask` → bool → `[1,1,h,w]` float on device. If `(h, w) != (S, S)`:
   `F.interpolate(size=(S,S), mode="bilinear", align_corners=False, antialias=True)`
   then `>= 0.5`. (Same as upstream `add_new_mask`.)
2. `obj.point_inputs = None`; no `prev_sam_mask_logits`.

Run (both paths), batch 1, features from §7.1 with `B = 1`:

```python
current_out = self.track_step(
    frame_idx=f, is_init_cond_frame=is_init,
    current_vision_feats=feats, current_vision_pos_embeds=pos, feat_sizes=sizes,
    point_inputs=point_inputs_or_None, mask_inputs=mask_inputs_or_None,
    output_dict={"cond_frame_outputs": obj.cond_outputs,
                 "non_cond_frame_outputs": obj.non_cond_outputs},
    num_frames=STREAM_NUM_FRAMES, track_in_reverse=False,
    run_mem_encoder=False, prev_sam_mask_logits=prev)
obj.pending = self._compact(session, current_out)      # §7.4
obj.pending_is_cond = is_init
```

Return `self._assemble_output(session)` (§8).

### 6.3 `track`

1. Raise if no frame.
2. For each object in order, decide:
   - `obj.pending is not None` → output is `obj.pending`; set
     `obj.tracked_frame = f`.
   - `obj.tracked_frame == f` → already done; output is
     `obj.non_cond_outputs[f]` (pending outputs are only committed at
     finalize, so a tracked frame without pending is always non-cond).
   - `not obj.cond_outputs` (never prompted, or reset) → no inference; output
     is a placeholder (§8). `tracked_frame` unchanged.
   - otherwise → a **row** to run.
3. Rows: `build_row_memory` for each (§7.2). If `batch_objects` is True, group
   with `group_rows`; otherwise every row is its own group. Run each group
   through §7.3. Store each row's compacted output in
   `obj.non_cond_outputs[f]`; set `obj.tracked_frame = f`.
4. Return `_assemble_output(session)`.

`track` twice on one frame does no inference the second time and writes no
memory twice.

### 6.4 Finalize (inside `set_frame`)

For objects with `pending is not None`:

1. Collect them; for `batch_objects=True` encode all together, else one by one.
2. `high_res = F.interpolate(pending["pred_masks"], size=(S,S), mode="bilinear", align_corners=False)` (stack along batch for several).
3. `maskmem_features, maskmem_pos_enc = self._encode_new_memory(current_vision_feats=feats_B, feat_sizes=sizes, pred_masks_high_res=high_res, object_score_logits=stacked_scores, is_mask_from_pts=True)` with features from §7.1 for that batch size.
4. Per object: `pending["maskmem_features"] = maskmem_features[i:i+1].to(torch.bfloat16)`;
   `pending["maskmem_pos_enc"] = self._session_pos_enc(session, maskmem_pos_enc)` (§7.4).
5. Commit:
   - `pending_is_cond`: `obj.cond_outputs[f] = pending`;
     `obj.non_cond_outputs.pop(f, None)`; set `obj.first_cond_frame` if `None`;
     then `evict_cond_frames(obj, self.max_cond_frames_per_object)`.
   - else: `obj.non_cond_outputs[f] = pending` (replaces the tracked output).
6. `obj.pending = None`, `obj.pending_is_cond = False`.

Steps 2–3 equal upstream `propagate_in_video_preflight` for one object.

### 6.5 `reset_object` / `remove_object` / `reset_session`

- `reset_object`: clear `cond_outputs`, `non_cond_outputs`, `pending`,
  `point_inputs`; `point_inputs_frame = -1`; `tracked_frame = -1`;
  `first_cond_frame = None`. The id keeps its position. Until the next
  `add_prompt`, `track` returns a placeholder for it.
- `remove_object`: delete from `session.objects`. Re-adding the same id later
  creates a fresh `ObjectState` at the end of the order.
- `reset_session`: as in §5.3.

### 6.6 Memory window and eviction

`min_needed_frame(next_frame_idx, num_maskmem, stride, max_obj_ptrs, use_obj_ptrs) -> int`
returns the smallest non-cond frame index that
`_prepare_memory_conditioned_features` can read when tracking
`next_frame_idx` forward. With `n = next_frame_idx`, `r = stride`:

```
cands = [n - 1]
for t_rel in range(2, num_maskmem):                    # t_rel >= 2 slots
    cands.append(((n - 2) // r) * r - (t_rel - 2) * r)
if use_obj_ptrs:
    cands.append(n - (max_obj_ptrs - 1))               # max_obj_ptrs = min(STREAM_NUM_FRAMES, max_obj_ptrs_in_encoder)
return min(cands)
```

`prune_object(obj, keep_from)` deletes `non_cond_outputs` keys `< keep_from`.
It never touches `cond_outputs` or `pending`.

`evict_cond_frames(obj, k)`: while `len(obj.cond_outputs) > k`, delete the
smallest key that is not `obj.first_cond_frame`; if `k == 1`, keep only the
newest key. (`SAM2Base.max_cond_frames_in_attn` still applies on top, through
`select_closest_cond_frames`.)

With the shipped configs (`num_maskmem=7`, stride 1,
`max_obj_ptrs_in_encoder=16`) an object holds at most 16 non-cond outputs (the
15 the next frame can read, plus the current frame) and 4 cond outputs.

---

## 7. Row core

A **row** is one object to track on the current frame. Its image-feature row
is the session's current frame. The functions below take rows, not sessions,
so a later engine can mix rows of several sessions.

### 7.1 Features for batch size B

`_features(session, B) -> (vision_feats, vision_pos_embeds, feat_sizes)`,
cached in `session.feature_cache[B]` for the current frame. Built exactly like
upstream `SAM2VideoPredictor._get_image_feature(batch_size=B)`: expand every
`backbone_out["backbone_fpn"][i]` and `["vision_pos_enc"][i]` to batch `B`, then
`self._prepare_backbone_features(expanded)`; drop its first return value.

### 7.2 `build_row_memory(model, obj, frame_idx, num_frames) -> RowMemory`

```python
@dataclass
class RowMemory:
    memory: Tensor         # [M, 1, C_mem]
    memory_pos: Tensor     # [M, 1, C_mem]
    num_obj_ptr_tokens: int
    key: tuple             # (n_cond, tuple(non_cond_t_pos), num_obj_ptr_tokens)
```

It is the forward, non-init branch of
`SAM2Base._prepare_memory_conditioned_features` (lines 522–650 at `2b90b9f`)
for `B = 1` and `output_dict = {"cond_frame_outputs": obj.cond_outputs,
"non_cond_frame_outputs": obj.non_cond_outputs}`, with the final
`torch.cat`s of memory and positions, and without the `memory_attention` call.
Every rule there is kept: `select_closest_cond_frames` with
`max_cond_frames_in_attn`; cond memories first with `t_pos = 0`; non-cond slots
`t_pos = 1 .. num_maskmem-1` using `memory_temporal_stride_for_eval` and the
unselected-cond fallback; `maskmem_tpos_enc[num_maskmem - t_pos - 1]`;
pointer list with `only_obj_ptrs_in_the_past_for_eval`,
`use_signed_tpos_enc_to_obj_ptrs`, `add_tpos_enc_to_obj_ptrs`,
`proj_tpos_enc_in_obj_ptrs`, `obj_ptr_tpos_proj`, and the `mem_dim < C` split.
`key` lists `n_cond = len(selected_cond)`, the `t_pos` values whose output was
found, and the pointer token count. Tensors stay in the dtype upstream
produces (the `torch.cat` type promotion included).

`group_rows(rows) -> list[list[int]]`: row indices grouped by `key`, groups in
order of first appearance, indices ascending inside a group.

`stack_rows(rows) -> (memory [M,B,C_mem], memory_pos [M,B,C_mem], num_obj_ptr_tokens)`:
`torch.cat` along dim 1. `AssertionError` if keys differ.

### 7.3 Running one group of B rows

```python
feats, pos, sizes = self._features(session, B)
memory, memory_pos, n_ptr = stack_rows(group)
pix = self.memory_attention(curr=feats[-1:], curr_pos=pos[-1:], memory=memory,
                            memory_pos=memory_pos, num_obj_ptr_tokens=n_ptr)
H, W = sizes[-1]
pix = pix.permute(1, 2, 0).view(B, self.hidden_dim, H, W)
high_res = [x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(feats[:-1], sizes[:-1])] if len(feats) > 1 else None
(_, _, _, low_res_masks, high_res_masks, obj_ptr, score) = self._forward_sam_heads(
    backbone_features=pix, point_inputs=None, mask_inputs=None,
    high_res_features=high_res, multimask_output=self._use_multimask(False, None))
maskmem, maskmem_pos = self._encode_new_memory(
    current_vision_feats=feats, feat_sizes=sizes, pred_masks_high_res=high_res_masks,
    object_score_logits=score, is_mask_from_pts=False)
```

Then row `i` gets `self._compact` of the `i:i+1` slices. This is
`SAM2Base.track_step` with `run_mem_encoder=True`, minus the memory gathering,
for B rows at once. The memory encoder sees the masks before hole filling, as
upstream.

### 7.4 Compaction

`_compact(session, out) -> dict` (§4.1), for batch-1 outputs, mirroring
upstream `_run_single_frame_inference`:

- `maskmem_features`: `.to(torch.bfloat16)` if not `None`.
- `pred_masks`: `fill_holes_in_mask_scores(pred, self.fill_hole_area)` if
  `fill_hole_area > 0`.
- `maskmem_pos_enc`: `_session_pos_enc(session, out["maskmem_pos_enc"])`: on
  first use store `[x[0:1].clone() for x in ...]` in
  `session.constants["maskmem_pos_enc"]`; always return the stored list
  (batch 1). `None` stays `None`.
- `obj_ptr`, `object_score_logits` as is.

Slices of a batched result are stored as views; no `.clone()`.

---

## 8. Output assembly

`_assemble_output(session) -> StreamOutput`, for every object in order:

- low-res logits: `obj.pending["pred_masks"]` if pending; else the output at
  `f` if `obj.tracked_frame == f`; else a placeholder
  `torch.full((1,1,S/4,S/4), NO_OBJ_SCORE)`.
- score: same source's `object_score_logits`, placeholder `NO_OBJ_SCORE`.

Concatenate (`float32`). `mask_logits = F.interpolate(low_res, size=(video_h, video_w), mode="bilinear", align_corners=False)`;
if `non_overlap_masks`, apply `self._apply_non_overlapping_constraints` to
`mask_logits`. Callers threshold at `0`.

---

## 9. Preprocessing (`sam2/stream/preprocess.py`)

```python
def preprocess_frame(frame, image_size: int, device: torch.device,
                     mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                     ) -> tuple[torch.Tensor, int, int]
```

1. Accept `np.ndarray` or `torch.Tensor`, `dtype uint8`, shape `(H, W, 3)`,
   `H, W >= 1`. Anything else: the `ValueError` of §5.3.
2. `t = torch.as_tensor(frame).to(device)` (no CPU float conversion),
   `.permute(2, 0, 1).unsqueeze(0).float() / 255.0`.
3. `F.interpolate(t, size=(S,S), mode="bilinear", align_corners=False, antialias=True)`.
4. `(t - mean) / std` with mean/std as `[1,3,1,1]` on device.
5. Return `(t[0], H, W)`.

Frames are RGB. BGR input is not detected; it is the caller's job.

---

## 10. Performance rules (phase 1)

- P1. `set_frame`, `add_prompt` and `track` add no host-device sync beyond what
  the reused `SAM2Base` code already does (`.item()`, `.cpu()`, `.tolist()`,
  `nonzero`, printing tensors are not allowed in the new code).
- P2. Grouping keys are built from Python ints only.
- P3. Image encoder runs exactly once per `set_frame`.
- P4. Memory attention, SAM heads and memory encoder run once per group, not
  per object, when `batch_objects=True`.

---

## 11. Upstream conformance matrix

| Behaviour | Upstream | Stream predictor |
|---|---|---|
| Objects independent | yes | yes |
| Add object after tracking started | yes | yes |
| Prompt on untracked frame | new cond frame, no memory | same |
| Prompt on tracked non-cond frame | correction, stored non-cond, **then overwritten by propagate** | correction kept and encoded (as legacy) |
| Prompt on tracked cond frame | correction dropped in preflight | cond frame re-run and replaced |
| Memory encoder for prompt frames | in preflight, `is_mask_from_pts=True`, filled masks | in finalize, same inputs |
| Memory encoder for tracked frames | in `track_step`, unfilled high-res masks | same |
| `num_frames` for pointers | video length | `STREAM_NUM_FRAMES` (equal for videos ≥ 16 frames) |
| Cond frames per object | unbounded | `max_cond_frames_per_object` (default 4) |
| Non-cond history | whole video | window from `min_needed_frame` |
| `non_overlap_masks_for_mem_enc` | no effect (batch 1) | applied across all objects of the session (legacy behaviour; matches legacy in tests) |

---

## 12. Tests (CPU, `pytest`)

### 12.1 Fixtures (`tests/conftest.py`)

- `TINY_OVERRIDES = ["++model.image_size=256",
  "++model.memory_attention.layer.self_attention.feat_sizes=[16,16]",
  "++model.memory_attention.layer.cross_attention.feat_sizes=[16,16]"]`,
  config `configs/sam2.1/sam2.1_hiera_t.yaml`, no checkpoint, `device="cpu"`.
- `stream_predictor` (session scope): `torch.manual_seed(0)` then
  `build_sam2_stream_predictor`.
- `stream_predictor_unbatched`: same weights (`load_state_dict`), built with
  `"++model.batch_objects=false"`.
- `video_predictor`: `build_sam2_video_predictor(...)`, same overrides,
  `load_state_dict(stream_predictor.state_dict(), strict=True)`.
- `legacy_predictor`: `build_sam2_video_predictor` with extra override
  `++model._target_=sam2.sam2_video_predictor_legacy.SAM2VideoPredictor`,
  same weights.
- `frames_dir` (session scope): 24 JPEGs `00000.jpg`…`00023.jpg`, 320×240,
  quality 95, grey background with three filled rectangles (red, green, blue)
  moving 3 px/frame on different paths. Prompt boxes for frame 0 are exported
  as `BOXES = {1: [...], 2: [...], 3: [...]}` (xyxy pixels).
- `upstream_images`: `video_predictor.init_state(frames_dir)["images"]`
  (`[24,3,256,256]`), with `video_hw = (240, 320)`.
- Helper `run_stream(predictor, images, video_hw, schedule)`: `schedule` maps
  frame index → list of `(method, kwargs)` applied after
  `set_preprocessed_frame` and before `track`; returns `{frame: StreamOutput}`.
- Helper `iou(a, b)`: over `logits > 0` masks; `1.0` when both empty.

### 12.2 Required tests

`test_build.py`
- `test_builder_returns_stream_predictor`: instance type, `image_size == 256`,
  `device.type == "cpu"`, `fill_hole_area == 8`, `batch_objects is True`,
  `max_cond_frames_per_object == 4`, `not training`.
- `test_constructor_rejects_bad_cond_cap`.

`test_stream_preprocess.py`
- `test_shape_dtype_device`; `test_numpy_and_tensor_equal`;
  `test_close_to_upstream_loader` (de-normalized `[0,1]` mean abs diff
  `< 0.02` vs `upstream_images[0]`); `test_rejects_bad_frames` (`HxW`,
  `HxWx4`, `float32`).

`test_stream_state.py`
- `test_min_needed_frame_defaults`: `(20,7,1,16,True) == 5`,
  `(20,7,1,16,False) == 14`, `(1,7,1,16,True) <= 0`.
- `test_min_needed_frame_matches_upstream_loop`: for strides 1, 2, 5 and
  `n` in 0..60, equals the minimum index produced by a copy of the upstream
  loop.
- `test_prune_keeps_window_and_cond`; `test_evict_keeps_first_and_newest`.

`test_stream_memory.py`
- `test_row_memory_matches_upstream`: for an object with outputs filled by
  running the predictor, `memory_attention` over `build_row_memory` equals
  `_prepare_memory_conditioned_features` at frames 1, 5, 20 (`atol=1e-5`).
- `test_keys`: young object (2 frames history) vs mature (20) differ; two
  mature single-cond objects equal.
- `test_stack_rows_rejects_mixed_keys`.

`test_stream_predictor.py`
- `test_three_objects_shapes`: boxes for 1,2,3 at frame 0; 10 frames;
  `obj_ids == [1,2,3]`, `mask_logits.shape == (3,1,240,320)`,
  `low_res_mask_logits.shape == (3,1,64,64)`, `object_score_logits.shape == (3,1)`.
- `test_late_object_does_not_change_others`: run A (objects 1,2 from frame 0)
  vs run B (same plus object 3 at frame 5), frames 0..23: objects 1,2 logits
  equal (`atol=1e-4`; batch size of their group changes once object 3 joins
  it) on every frame.
- `test_requires_frame_first`, `test_track_no_objects`,
  `test_track_twice_same_frame` (second call returns equal tensors and the
  object's `non_cond_outputs[f]` is the same dict object),
  `test_frame_size_change` (frame 10 given `video_hw=(120,160)`: output
  `(3,1,120,160)`; frame 11 tracks normally).
- `test_prompt_on_new_frame_is_cond`, `test_correction_after_track_is_non_cond`,
  `test_prompt_on_tracked_cond_frame_replaces_cond`, `test_mask_prompt`,
  `test_points_accumulate_with_clear_old_points_false`,
  `test_box_with_clear_old_points_false_raises`, `test_max_objects`.
- `test_remove_then_readd_is_fresh`, `test_reset_object_placeholder_until_prompt`.
- `test_memory_bounded_long_stream`: 3 objects, 200 frames:
  `len(non_cond_outputs) <= 16`, `len(cond_outputs) <= 4` for every object;
  total bytes of tensors reachable from the session at frame 200 `<=` 1.01×
  frame 160.
- `test_batched_equals_unbatched`: same schedule through `stream_predictor`
  and `stream_predictor_unbatched`: IoU ≥ 0.99 per object per frame and
  `atol=1e-3` on logits.

`test_stream_parity.py`
- `test_single_object_matches_upstream`: object 1 box at frame 0, 24 frames,
  `stream_predictor_unbatched` vs `video_predictor.propagate_in_video`:
  `atol=1e-4` on `mask_logits` every frame.
- `test_three_objects_match_upstream`: all three at frame 0, batched, vs one
  upstream session with three objects: IoU ≥ 0.99.
- `test_late_object_matches_single_upstream_session`: object 3 at frame 8 vs
  an upstream session with only object 3, `propagate_in_video(start_frame_idx=8)`:
  `atol=1e-4` unbatched, IoU ≥ 0.99 batched.
- `test_reprompt_on_new_frame_matches_upstream`: object 1 at frame 0; extra
  points at frame 10 before `track` (new cond frame). Upstream: propagate
  0..9, add points at 10, `propagate_in_video(start_frame_idx=10)`.
  `atol=1e-4` frames 10..23.
- `test_correction_matches_legacy`: object 1 at frame 0, negative click at
  frame 12 after `track` (correction). Legacy: propagate 0..23, add the click
  at 12, `propagate_in_video(start_frame_idx=12)`. `atol=1e-4` frames 12..23.

Definition of done for phase 1: every test above passes with
`.venv/bin/pytest tests -q` on CPU, and `black --check` passes on new files.

---

## 13. GPU tools (run on the GPU server)

Both scripts exit `0` with `CUDA not available, skipping` when
`torch.cuda.is_available()` is False, and take `--config`, `--ckpt`,
`--video-dir` (default `notebooks/videos/bedroom`), `--frames`, `--size {512,1024}`.
`--size 512` adds the three 512 overrides analogous to §12.1 (`feat_sizes=[32,32]`).

`tools/stream_parity.py --dtype {fp32,bf16}`
- Prompts on frame 0, from `notebooks/video_predictor_example.ipynb`:
  object 1 points `[[210,350],[250,220]]` labels `[1,1]`; object 2
  `[[200,300],[275,175]]` labels `[1,0]`; object 3 `[[400,150]]` label `[1]`.
  Variant B adds object 3 at frame 30 instead.
- Compares against upstream as in §12.2; prints per-object min and mean IoU;
  exits `1` below 0.99 (fp32) or 0.98 (bf16).

`tools/benchmark_stream.py --objects {1,3} --warmup 30 --dtype bf16`
- Timing per `step()` with `torch.cuda.synchronize()` before and after; prints
  mean, p50, p95 ms and FPS.
- Same frames and objects through upstream `propagate_in_video`; prints its
  ms/frame as the reference.
- 20 frames under `torch.profiler` (CPU + CUDA): top 15 ops by self CPU time
  and by self CUDA time, and total CUDA kernel time per frame vs wall time.
- One pass under `torch.cuda.set_sync_debug_mode("warn")`, printing the count
  of sync warnings per frame.

Results go to `docs/superpowers/specs/2026-10-24-stream-predictor-results.md`
with GPU model, driver, torch and CUDA versions, and each command with its
output.

---

## 14. Decisions taken without the user

- §6.2 row 2: prompt on a tracked cond frame replaces the cond frame
  (upstream silently drops it).
- §6.2 row 3 / §11: corrections persist in memory, matching legacy, not the
  current upstream propagate.
- `max_cond_frames_per_object = 4`, first cond frame protected.
- Bilinear antialiased resize on device instead of PIL.
- Batched slices stored as views without cloning.
- `non_overlap_masks_for_mem_enc` forced off with a warning.
