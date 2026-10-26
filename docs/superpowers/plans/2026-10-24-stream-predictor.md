# SAM2StreamPredictor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A live-stream SAM 2 predictor with independent, add-any-time objects, bounded memory, external session state and batched per-object inference.

**Architecture:** `SAM2StreamPredictor(SAM2Base)` holds only weights; all per-stream state is a `StreamSession` passed to every call. Prompts reuse upstream `track_step` for one object; tracking builds each object's memory as a "row", groups rows with identical memory layout and runs memory attention, SAM heads and the memory encoder once per group.

**Tech Stack:** Python ≥3.10, PyTorch ≥2.5.1, hydra (existing configs), pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-stream-predictor-spec.md` (normative; wins over this plan and over the design doc `2026-10-24-stream-predictor-design.md` wherever they differ). Each task below implements the spec sections it names; test names and assertions are those of spec §12.2.

## Global Constraints

- No edits under `sam2/modeling/`. Reuse `SAM2Base` methods: `forward_image`, `_prepare_backbone_features`, `track_step`, `_forward_sam_heads`, `_encode_new_memory`, `_use_multimask`, `memory_attention`, `obj_ptr_tpos_proj`, `maskmem_tpos_enc`.
- No hardcoded `cuda`; use `self.device`. No `copy.copy` of the model. No frame list in the session.
- Memory features stored as `torch.bfloat16` (as upstream). Holes filled on stored/returned `pred_masks` with `fill_hole_area`, not on the masks fed to the memory encoder (as upstream).
- Model `num_frames` argument: constant `STREAM_NUM_FRAMES = 1 << 30`.
- Parity: single-object fp32 CPU logits `atol=1e-4` vs upstream; batched IoU ≥ 0.99 per object per frame; GPU bf16 IoU ≥ 0.98.
- Frames are RGB `uint8` `HxWx3` (`np.ndarray` or `torch.Tensor`).
- Tests run with `pytest tests/` on CPU; GPU scripts live in `tools/` and skip cleanly without CUDA/checkpoint.

## Review Focus

- `add_prompt` / `track` before any `set_frame` → `RuntimeError("call set_frame first")`, not a `KeyError`.
- `track` with zero objects → `StreamOutput` with `obj_ids == []` and `[0,1,H,W]` tensors.
- `track` called twice on one frame → same tensors, memory written once.
- Frame size changes mid-session → outputs at the new size; tracking continues.
- `remove_object(id)` then `add_prompt(id, ...)` → a fresh object with no old memory.

---

### Task 1: Dev environment, tiny-model fixture, builder and empty predictor

**Files:**
- Create: `sam2/sam2_stream_predictor.py`, `sam2/stream/__init__.py`, `tests/conftest.py`, `tests/test_build.py`
- Modify: `sam2/build_sam.py` (add builder after `build_sam2_video_predictor`), `setup.py` (`"dev"` extra: add `"pytest>=8.0"`), `.gitignore` (`.venv/`)

**Interfaces:**
- Produces: `build_sam2_stream_predictor(config_file, ckpt_path=None, device="cuda", mode="eval", hydra_overrides_extra=[], apply_postprocessing=True, **kwargs) -> SAM2StreamPredictor` — same postprocessing overrides as `build_sam2_video_predictor`, target `sam2.sam2_stream_predictor.SAM2StreamPredictor`.
- Produces: `SAM2StreamPredictor(fill_hole_area=0, non_overlap_masks=False, batch_objects=True, max_cond_frames_per_object=4, **kwargs)`.
- Produces every fixture and helper of spec §12.1 (`TINY_OVERRIDES`, `stream_predictor`, `stream_predictor_unbatched`, `video_predictor`, `legacy_predictor`, `frames_dir`, `BOXES`, `upstream_images`, `run_stream`, `iou`), plus `rgb_frames` (the frames as `uint8` arrays). Also `pyproject.toml` `[tool.pytest.ini_options]` per spec §2.

- [ ] **Step 1:** Local env: `python3 -m venv .venv && .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && SAM2_BUILD_CUDA=0 .venv/bin/pip install -e ".[dev]"`. Check: `.venv/bin/python -c "import sam2, torch; print(torch.__version__)"` prints a version ≥2.5.1.
- [ ] **Step 2: Failing test** `tests/test_build.py::test_builder_returns_stream_predictor`: `isinstance(p, SAM2StreamPredictor)`, `p.image_size == 256`, `p.device.type == "cpu"`, `p.fill_hole_area == 8`, `not p.training`.
- [ ] **Step 3:** Run `.venv/bin/pytest tests/test_build.py -v` → FAIL (import error).
- [ ] **Step 4:** Implement builder and class skeleton.
- [ ] **Step 5:** Run → PASS. Commit `feat(stream): add stream predictor builder and test fixtures`.

### Task 2: On-device frame preprocessing

**Files:** Create `sam2/stream/preprocess.py`, `tests/test_stream_preprocess.py`

**Interfaces:**
- Produces: `preprocess_frame(frame, image_size: int, device: torch.device, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> tuple[torch.Tensor, int, int]` returning `(image[3,S,S] float32 on device, video_h, video_w)`. Accepts `np.ndarray` or `torch.Tensor` `uint8` `HxWx3`; uses `F.interpolate(mode="bilinear", antialias=True, align_corners=False)` after moving to device.

- [ ] **Step 1: Failing tests:** `test_shape_dtype_device` (`[3,256,256]`, float32, cpu, returns `(240, 320)`); `test_numpy_and_tensor_equal` (`torch.equal`); `test_close_to_upstream_loader` (mean abs diff vs `sam2.utils.misc.load_video_frames` image for the same JPEG `< 0.05`); `test_rejects_bad_frames` (`ValueError` for `HxW` and `HxWx4`).
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS. **Step 5:** Commit `feat(stream): on-device frame preprocessing`.

### Task 3: Session state and memory window

**Files:** Create `sam2/stream/state.py`, `tests/test_stream_state.py`

**Interfaces:**
- Produces dataclasses:
  - `ObjectState(obj_id, cond_outputs: dict[int, dict], non_cond_outputs: OrderedDict[int, dict], pending: dict | None, pending_is_cond: bool, point_inputs: dict | None, tracked_frame: int)` (`tracked_frame=-1` until tracked).
  - `StreamSession(device, video_h=0, video_w=0, frame_idx=-1, image=None, features=None, objects: OrderedDict, constants: dict, max_objects: int | None = None)`.
  - `StreamOutput(frame_idx: int, obj_ids: list, mask_logits: Tensor, low_res_mask_logits: Tensor, object_score_logits: Tensor)`.
- Produces: `min_needed_frame(next_frame_idx: int, num_maskmem: int, stride: int, max_obj_ptrs: int, use_obj_ptrs: bool) -> int` — smallest non-cond frame index the model can read when tracking `next_frame_idx` (same index arithmetic as `SAM2Base._prepare_memory_conditioned_features`, forward direction).
- Produces: `prune_object(obj: ObjectState, keep_from: int) -> None` — drops `non_cond_outputs` keys `< keep_from`.

- [ ] **Step 1: Failing tests:** `test_min_needed_frame_defaults` (`min_needed_frame(20, 7, 1, 16, True) == 5`; `min_needed_frame(20, 7, 1, 16, False) == 14`; `min_needed_frame(3, 7, 1, 16, True) <= 0`); `test_min_needed_frame_matches_upstream_loop` (stride 5: equals the minimum `prev_frame_idx` produced by the upstream loop for `frame_idx=40`, computed in the test by copying that loop); `test_prune_keeps_window_and_cond` (prune leaves keys `>= keep_from`, never touches `cond_outputs`).
- [ ] **Steps 2–5:** run FAIL → implement → run PASS → commit `feat(stream): session state and memory window`.

### Task 4: Per-row memory builder and grouping

**Files:** Create `sam2/stream/memory.py`, `tests/test_stream_memory.py`

**Interfaces:**
- Consumes: `ObjectState` (Task 3), `SAM2Base` attributes listed in Global Constraints.
- Produces: `RowMemory(memory: Tensor[M,1,C_mem], memory_pos: Tensor[M,1,C_mem], num_obj_ptr_tokens: int, key: tuple)`.
- Produces: `build_row_memory(model: SAM2Base, obj: ObjectState, frame_idx: int, num_frames: int) -> RowMemory` — forward-direction, non-init-frame branch of `_prepare_memory_conditioned_features` for one object (cond selection via `select_closest_cond_frames`, non-cond slots with `memory_temporal_stride_for_eval`, object pointers with per-row temporal PE honoring `use_signed_tpos_enc_to_obj_ptrs`, `only_obj_ptrs_in_the_past_for_eval`, `proj_tpos_enc_in_obj_ptrs`, `mem_dim < C` pointer split). `key = (n_cond, tuple(non_cond_t_pos_present), num_obj_ptr_tokens)`.
- Produces: `group_rows(rows: list[RowMemory]) -> list[list[int]]` (row indices per key, first-seen order) and `stack_rows(rows: list[RowMemory]) -> tuple[Tensor[M,B,C], Tensor[M,B,C], int]` (asserts equal keys).

- [ ] **Step 1: Failing tests** (use `stream_predictor` fixture and real outputs made by running `track_step` on a few frames to fill an `ObjectState`):
  - `test_row_memory_matches_upstream`: `model.memory_attention(curr, curr_pos, memory, memory_pos, num_obj_ptr_tokens)` on `build_row_memory` output equals `model._prepare_memory_conditioned_features(...)` for the same object/frame, `atol=1e-5`, at frame indices 1, 5 and 20 (window full).
  - `test_keys`: object with 2 frames of history vs 20 frames → different keys; two objects both with 20 frames and one cond frame → equal keys.
  - `test_stack_rows_rejects_mixed_keys`: `AssertionError`.
- [ ] **Steps 2–5:** FAIL → implement → PASS → commit `feat(stream): per-row memory builder with grouping`.

### Task 5: Frame lifecycle, prompts and batched tracking

**Files:** Modify `sam2/sam2_stream_predictor.py`; create `tests/test_stream_predictor.py`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces (all `@torch.inference_mode()`):
  - `create_session(max_objects: int | None = None) -> StreamSession`
  - `set_frame(session, frame) -> int` and `set_preprocessed_frame(session, image: Tensor[3,S,S], video_hw: tuple[int,int]) -> int` — finalize previous frame (batched memory encoding of `pending`, `is_mask_from_pts=True`, commit to cond/non-cond, drop same-frame non-cond when cond), prune every object with `min_needed_frame(frame_idx+1, …)`, advance `frame_idx`, compute and cache backbone features once.
  - `add_prompt(session, obj_id, points=None, labels=None, box=None, mask=None, clear_old_points=True, normalize_coords=True) -> StreamOutput` — box prepended as labels 2/3; `mask` exclusive with points/box (`ValueError`); new id → new `ObjectState` (respect `max_objects`, `RuntimeError` when full); `is_init_cond_frame = obj.tracked_frame != frame_idx` (upstream rule: an untracked frame becomes a conditioning frame); otherwise correction with clamped previous logits (`[-32, 32]`); result stored as `pending` via `self.track_step(..., run_mem_encoder=False, num_frames=STREAM_NUM_FRAMES)`. Cap cond frames at `max_cond_frames_per_object` by evicting the oldest except the first.
  - `track(session) -> StreamOutput` — objects with `pending` return it; objects already tracked on this frame return their stored output; the rest are built with `build_row_memory`, grouped, and each group runs `memory_attention` → `_forward_sam_heads(point_inputs=None, mask_inputs=None, multimask_output=self._use_multimask(False, None))` → `_encode_new_memory(is_mask_from_pts=False)`; outputs compacted like upstream `_run_single_frame_inference`; `batch_objects=False` runs each row as its own group.
  - `step(session, frame) -> StreamOutput` = `set_frame` + `track`.
  - Output masks resized to the current frame's `(video_h, video_w)` with bilinear `align_corners=False`.

- Behaviour is spec §6.1–6.4, §7, §8 (prompt-mode table in §6.2 is normative; it replaces the `is_init_cond_frame` one-liner above).

- [ ] **Step 1: Failing tests** (spec §12.2 `test_stream_predictor.py`, all except the two removal tests and the long-stream test, which belong to Task 6): `test_three_objects_shapes` (3 boxes on frame 0, then 10 `step` calls: `obj_ids == [1,2,3]`, `mask_logits.shape == (3,1,240,320)`, `low_res_mask_logits.shape == (3,1,64,64)`); `test_late_object_does_not_change_others` (objects 1,2 from frame 0; object 3 added at frame 5; objects 1,2 outputs for frames 5–10 equal a run without object 3); `test_correction_after_track_is_non_cond`; `test_prompt_on_new_frame_is_cond`; `test_mask_prompt`; plus four Review Focus tests: `test_requires_frame_first`, `test_track_no_objects`, `test_track_twice_same_frame`, `test_frame_size_change` (the fifth, `test_remove_then_readd_is_fresh`, is in Task 6).
- [ ] **Steps 2–5:** FAIL → implement → PASS → commit `feat(stream): frame lifecycle, prompts and batched tracking`.

### Task 6: Object removal, re-anchor, reset and bounded memory

**Files:** Modify `sam2/sam2_stream_predictor.py`; extend `tests/test_stream_predictor.py`

**Interfaces:**
- Produces: `remove_object(session, obj_id, strict=False) -> None` (`KeyError` only when `strict`), `reset_object(session, obj_id) -> None` (clears cond/non-cond/pending, keeps id and order; next `add_prompt` is a conditioning frame), `reset_session(session) -> None`.

- [ ] **Step 1: Failing tests:** `test_remove_then_readd_is_fresh`; `test_reset_object_placeholder_until_prompt` (after `reset_object`, `track` returns `NO_OBJ_SCORE`-filled logits for that object until re-prompted, others unaffected); `test_memory_bounded_long_stream` (3 objects, 200 frames: every object `len(non_cond_outputs) <= 15`, `len(cond_outputs) <= 4`, session has no attribute holding past images, `torch` tensors reachable from the session do not grow between frame 50 and 200 — count via a helper that walks the dataclasses).
- [ ] **Steps 2–5:** FAIL → implement → PASS → commit `feat(stream): object removal, re-anchor and bounded memory`.

### Task 7: Parity against upstream SAM2VideoPredictor (CPU)

**Files:** Create `tests/test_stream_parity.py`

**Interfaces:** Consumes `video_predictor`, `stream_predictor`, `frames_dir` fixtures and the public API. Upstream frames come from `inference_state["images"]`; stream receives them via `set_preprocessed_frame(image, (video_h, video_w))`.

- [ ] **Step 1: Tests** exactly as spec §12.2 `test_stream_parity.py`: `test_single_object_matches_upstream`, `test_three_objects_match_upstream`, `test_late_object_matches_single_upstream_session`, `test_reprompt_on_new_frame_matches_upstream`, `test_correction_matches_legacy` (corrections are compared with the legacy predictor, because current upstream `propagate_in_video` overwrites a non-cond correction; spec §11).
- [ ] **Step 2:** Run `.venv/bin/pytest tests/test_stream_parity.py -v`. Any failure is a bug in Tasks 4–6: fix there, re-run the whole suite.
- [ ] **Step 3:** Commit `test(stream): parity with upstream SAM2VideoPredictor`.

### Task 8: GPU parity and benchmark tools (run on the GPU server)

**Files:** Create `tools/stream_parity.py`, `tools/benchmark_stream.py`, section "Stream predictor" in `tools/README.md`

**Interfaces:**
- `tools/stream_parity.py --config --ckpt --video-dir notebooks/videos/bedroom --frames 200 --dtype {fp32,bf16}`: prompts from the notebook (obj 1 points `[[210,350],[250,220]]` labels `[1,1]`; obj 2 `[[200,300],[275,175]]` labels `[1,0]`; obj 3 `[[400,150]]` label `[1]`, all frame 0; plus a late-added variant of obj 3 at frame 30), prints per-object min/mean IoU vs upstream and exits non-zero below the Global Constraints thresholds.
- `tools/benchmark_stream.py --config --ckpt --video-dir --objects {1,3} --frames 300 --warmup 30 --dtype bf16 --size {512,1024}`: per-frame `torch.cuda.synchronize()` timing, prints p50/p95/mean ms and FPS for `step()`, and a `torch.profiler` table (top 15 by CPU and CUDA time) for 20 frames. Also prints upstream `SAM2VideoPredictor.propagate_in_video` ms/frame on the same frames and objects as the reference number.

- [ ] **Step 1:** Write both scripts; they exit 0 with a "CUDA not available" message on CPU (check locally).
- [ ] **Step 2:** On the GPU server: clone branch, install, download `sam2.1_hiera_small.pt` and `sam2.1_hiera_tiny.pt` via `checkpoints/download_ckpts.sh`, run `pytest tests/`, then both tools for small @512 and @1024, 1 and 3 objects.
- [ ] **Step 3:** Record numbers in `docs/superpowers/specs/2026-10-24-stream-predictor-results.md` (GPU model, torch/CUDA version, every command and its output table). Commit `docs(stream): GPU parity and benchmark results`.
