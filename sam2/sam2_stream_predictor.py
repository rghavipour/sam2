# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn.functional as F

from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2.stream.memory import build_row_memory, group_rows, stack_rows
from sam2.stream.preprocess import preprocess_frame
from sam2.stream.state import (
    evict_cond_frames,
    min_needed_frame,
    ObjectState,
    prune_object,
    StreamOutput,
    StreamSession,
)
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores

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

    # ------------------------------------------------------------------------------
    # Sessions and frames
    # ------------------------------------------------------------------------------

    def create_session(self, max_objects=None):
        """Start the state of a new stream."""
        if max_objects is not None and max_objects < 1:
            raise ValueError("max_objects must be None or >= 1")
        return StreamSession(device=self.device, max_objects=max_objects)

    @torch.inference_mode()
    def set_frame(self, session, frame):
        """Make `frame` (RGB uint8 HxWx3) the session's current frame."""
        image, video_h, video_w = preprocess_frame(
            frame, self.image_size, session.device
        )
        return self._advance(session, image, video_h, video_w)

    @torch.inference_mode()
    def set_preprocessed_frame(self, session, image, video_hw):
        """Like `set_frame`, for an already normalized [3, S, S] model input."""
        S = self.image_size
        if tuple(image.shape) != (3, S, S):
            raise ValueError(
                f"image must have shape (3, {S}, {S}), got {tuple(image.shape)}"
            )
        video_h, video_w = int(video_hw[0]), int(video_hw[1])
        if video_h < 1 or video_w < 1:
            raise ValueError(f"video_hw must be positive, got {tuple(video_hw)}")
        image = image.to(session.device, dtype=torch.float32)
        return self._advance(session, image, video_h, video_w)

    @torch.inference_mode()
    def step(self, session, frame):
        """`set_frame` followed by `track`."""
        self.set_frame(session, frame)
        return self.track(session)

    def _advance(self, session, image, video_h, video_w):
        if session.frame_idx >= 0:
            self._finalize(session)
            keep_from = min_needed_frame(
                session.frame_idx + 1,
                self.num_maskmem,
                1 if self.training else self.memory_temporal_stride_for_eval,
                min(STREAM_NUM_FRAMES, self.max_obj_ptrs_in_encoder),
                self.use_obj_ptrs_in_encoder,
            )
            for obj in session.objects.values():
                prune_object(obj, keep_from)
        session.backbone_out = None
        session.image = None
        session.feature_cache = {}

        session.frame_idx += 1
        session.video_h, session.video_w = video_h, video_w
        session.image = image.unsqueeze(0)
        session.backbone_out = self.forward_image(session.image)
        return session.frame_idx

    def reset_object(self, session, obj_id):
        """Drop an object's memory but keep its id and position (e.g. to re-anchor)."""
        obj = session.objects[obj_id]
        obj.cond_outputs = {}
        obj.non_cond_outputs.clear()
        obj.pending = None
        obj.pending_is_cond = False
        obj.point_inputs = None
        obj.point_inputs_frame = -1
        obj.tracked_frame = -1
        obj.first_cond_frame = None

    def remove_object(self, session, obj_id, strict=False):
        """Stop tracking an object; unknown ids are ignored unless `strict`."""
        if obj_id not in session.objects:
            if strict:
                raise KeyError(f"unknown object id {obj_id!r}")
            return
        del session.objects[obj_id]

    def reset_session(self, session):
        """Drop every object and the current frame; the session starts over."""
        session.objects.clear()
        session.constants = {}
        session.backbone_out = None
        session.image = None
        session.feature_cache = {}
        session.frame_idx = -1
        session.video_h = session.video_w = 0

    def _require_frame(self, session):
        if session.frame_idx < 0 or session.backbone_out is None:
            raise RuntimeError("call set_frame() before add_prompt() or track()")

    def _features(self, session, batch_size):
        """Current frame features expanded to `batch_size` (as upstream `_get_image_feature`)."""
        cached = session.feature_cache.get(batch_size)
        if cached is not None:
            return cached
        backbone_out = session.backbone_out
        expanded = {
            "backbone_fpn": [
                x.expand(batch_size, -1, -1, -1) for x in backbone_out["backbone_fpn"]
            ],
            "vision_pos_enc": [
                x.expand(batch_size, -1, -1, -1) for x in backbone_out["vision_pos_enc"]
            ],
        }
        _, feats, pos, sizes = self._prepare_backbone_features(expanded)
        session.feature_cache[batch_size] = (feats, pos, sizes)
        return feats, pos, sizes

    # ------------------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------------------

    @torch.inference_mode()
    def add_prompt(
        self,
        session,
        obj_id,
        points=None,
        labels=None,
        box=None,
        mask=None,
        clear_old_points=True,
        normalize_coords=True,
    ):
        """
        Add points, a box or a mask for `obj_id` on the current frame. An unknown id
        starts a new object. Returns the masks of all objects on this frame.
        """
        self._require_frame(session)
        if points is None and box is None and mask is None:
            raise ValueError("at least one of points, box or mask must be provided")
        if (points is None) != (labels is None):
            raise ValueError("points and labels must be provided together")
        if mask is not None and (points is not None or box is not None):
            raise ValueError("mask cannot be combined with points or box")
        if box is not None and not clear_old_points:
            raise ValueError(
                "cannot add box without clearing old points, since "
                "box prompt must be provided before any point prompt "
                "(please use clear_old_points=True instead)"
            )
        if mask is None:
            new_points, new_labels = self._prompt_points(
                session, points, labels, box, normalize_coords
            )
        else:
            mask_inputs = self._prompt_mask(session, mask)

        obj = self._get_or_add_object(session, obj_id)
        f = session.frame_idx
        is_cond_frame = f in obj.cond_outputs or (
            obj.pending is not None and obj.pending_is_cond
        )
        is_init_cond_frame = obj.tracked_frame != f or is_cond_frame

        if mask is None:
            if not clear_old_points and obj.point_inputs_frame == f:
                point_inputs = concat_points(obj.point_inputs, new_points, new_labels)
            else:
                point_inputs = concat_points(None, new_points, new_labels)
            obj.point_inputs, obj.point_inputs_frame = point_inputs, f
            prev_sam_mask_logits = None
            prev_out = obj.pending
            if prev_out is None:
                prev_out = obj.cond_outputs.get(f)
            if prev_out is None:
                prev_out = obj.non_cond_outputs.get(f)
            if prev_out is not None and prev_out["pred_masks"] is not None:
                prev_sam_mask_logits = prev_out["pred_masks"].to(session.device)
                # clamp the scale of prev_sam_mask_logits to avoid rare numerical issues
                prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        else:
            point_inputs, obj.point_inputs, obj.point_inputs_frame = None, None, -1
            prev_sam_mask_logits = None

        feats, pos, sizes = self._features(session, 1)
        current_out = self.track_step(
            frame_idx=f,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=feats,
            current_vision_pos_embeds=pos,
            feat_sizes=sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs if mask is not None else None,
            output_dict={
                "cond_frame_outputs": obj.cond_outputs,
                "non_cond_frame_outputs": obj.non_cond_outputs,
            },
            num_frames=STREAM_NUM_FRAMES,
            track_in_reverse=False,
            # the memory is encoded when the frame is finalized, so more clicks can follow
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        obj.pending = self._compact(session, current_out)
        obj.pending_is_cond = is_init_cond_frame
        return self._assemble_output(session)

    def _get_or_add_object(self, session, obj_id):
        obj = session.objects.get(obj_id)
        if obj is not None:
            return obj
        n = session.max_objects
        if n is not None and len(session.objects) >= n:
            raise RuntimeError(f"session already tracks max_objects={n} objects")
        obj = ObjectState(obj_id=obj_id)
        session.objects[obj_id] = obj
        return obj

    def _prompt_points(self, session, points, labels, box, normalize_coords):
        """Points (and box corners) in model input coordinates, [1, P, 2] / [1, P]."""
        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int32)
        points = torch.as_tensor(points, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        if points.dim() != 3 or points.shape[0] != 1 or points.shape[-1] != 2:
            raise ValueError(f"points must be [P, 2], got {tuple(points.shape)}")
        if labels.dim() != 2 or labels.shape != points.shape[:2]:
            raise ValueError(
                f"labels must be [P] matching points, got {tuple(labels.shape)}"
            )
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float32)
            if box.numel() != 4:
                raise ValueError(
                    f"box must be [x1, y1, x2, y2], got {tuple(box.shape)}"
                )
            # the box goes first as two points with labels 2 and 3 (as SAM 2 is trained)
            box_coords = box.to(points.device).reshape(1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int32)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)
        if normalize_coords:
            points = points / torch.tensor([session.video_w, session.video_h])
        points = points * self.image_size
        return points.to(session.device), labels.to(session.device)

    def _prompt_mask(self, session, mask):
        """A binary mask resized to the model input, [1, 1, S, S] float."""
        mask = torch.as_tensor(mask)
        if mask.dim() != 2:
            raise ValueError(f"mask must be 2-D, got {tuple(mask.shape)}")
        mask = mask.bool()[None, None].float().to(session.device)
        S = self.image_size
        if mask.shape[-2:] != (S, S):
            mask = F.interpolate(
                mask,
                size=(S, S),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask = (mask >= 0.5).float()
        return mask

    # ------------------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------------------

    @torch.inference_mode()
    def track(self, session):
        """Masks of all objects on the current frame, tracking those that need it."""
        self._require_frame(session)
        f = session.frame_idx
        to_track = []
        for obj in session.objects.values():
            if obj.pending is not None:
                obj.tracked_frame = f
            elif obj.tracked_frame == f or not obj.cond_outputs:
                continue
            else:
                to_track.append(obj)

        if to_track:
            rows = [
                build_row_memory(self, obj, f, STREAM_NUM_FRAMES) for obj in to_track
            ]
            if self.batch_objects:
                groups = group_rows(rows)
            else:
                groups = [[i] for i in range(len(rows))]
            for group in groups:
                outs = self._run_rows(session, [rows[i] for i in group])
                for i, out in zip(group, outs):
                    to_track[i].non_cond_outputs[f] = out
                    to_track[i].tracked_frame = f
        return self._assemble_output(session)

    def _run_rows(self, session, rows):
        """`track_step` (with memory encoding) for rows that share one memory layout."""
        B = len(rows)
        feats, pos, sizes = self._features(session, B)
        memory, memory_pos, num_obj_ptr_tokens = stack_rows(rows)
        pix_feat = self.memory_attention(
            curr=feats[-1:],
            curr_pos=pos[-1:],
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        H, W = sizes[-1]
        pix_feat = pix_feat.permute(1, 2, 0).view(B, self.hidden_dim, H, W)
        if len(feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(feats[:-1], sizes[:-1])
            ]
        else:
            high_res_features = None
        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = self._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=high_res_features,
            multimask_output=self._use_multimask(False, None),
        )
        # as upstream, the memory encoder sees the masks before hole filling
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=feats,
            feat_sizes=sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=False,
        )
        outs = []
        for i in range(B):
            out = {
                "maskmem_features": maskmem_features[i : i + 1],
                "maskmem_pos_enc": [x[i : i + 1] for x in maskmem_pos_enc],
                "pred_masks": low_res_masks[i : i + 1],
                "obj_ptr": obj_ptr[i : i + 1],
                "object_score_logits": object_score_logits[i : i + 1],
            }
            outs.append(self._compact(session, out))
        return outs

    def _finalize(self, session):
        """Encode the memory of this frame's prompt outputs and commit them."""
        pending = [obj for obj in session.objects.values() if obj.pending is not None]
        if not pending:
            return
        f = session.frame_idx
        batches = [pending] if self.batch_objects else [[obj] for obj in pending]
        for batch in batches:
            feats, _, sizes = self._features(session, len(batch))
            pred_masks = torch.cat([obj.pending["pred_masks"] for obj in batch], dim=0)
            high_res_masks = F.interpolate(
                pred_masks.to(session.device),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            scores = torch.cat(
                [obj.pending["object_score_logits"] for obj in batch], dim=0
            )
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=feats,
                feat_sizes=sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=scores,
                # these frames are what the user interacted with
                is_mask_from_pts=True,
            )
            for i, obj in enumerate(batch):
                out = obj.pending
                out["maskmem_features"] = maskmem_features[i : i + 1].to(torch.bfloat16)
                out["maskmem_pos_enc"] = self._session_pos_enc(
                    session, [x[i : i + 1] for x in maskmem_pos_enc]
                )
                if obj.pending_is_cond:
                    obj.cond_outputs[f] = out
                    obj.non_cond_outputs.pop(f, None)
                    if obj.first_cond_frame is None:
                        obj.first_cond_frame = f
                    evict_cond_frames(obj, self.max_cond_frames_per_object)
                else:
                    obj.non_cond_outputs[f] = out
                obj.pending = None
                obj.pending_is_cond = False

    def _compact(self, session, out):
        """The stored form of a batch-1 frame output (as upstream `_run_single_frame_inference`)."""
        maskmem_features = out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
        pred_masks = out["pred_masks"]
        if self.fill_hole_area > 0:
            pred_masks = fill_holes_in_mask_scores(pred_masks, self.fill_hole_area)
        return {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": self._session_pos_enc(session, out["maskmem_pos_enc"]),
            "pred_masks": pred_masks,
            "obj_ptr": out["obj_ptr"],
            "object_score_logits": out["object_score_logits"],
        }

    def _session_pos_enc(self, session, maskmem_pos_enc):
        """`maskmem_pos_enc` is the same for all frames and objects; keep one copy."""
        if maskmem_pos_enc is None:
            return None
        if "maskmem_pos_enc" not in session.constants:
            session.constants["maskmem_pos_enc"] = [
                x[0:1].clone() for x in maskmem_pos_enc
            ]
        return session.constants["maskmem_pos_enc"]

    # ------------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------------

    def _assemble_output(self, session):
        f = session.frame_idx
        S4 = self.image_size // 4
        device = session.device
        low_res, scores = [], []
        for obj in session.objects.values():
            if obj.pending is not None:
                src = obj.pending
            elif obj.tracked_frame == f:
                src = obj.non_cond_outputs[f]
            else:
                src = None
            if src is None:
                low_res.append(torch.full((1, 1, S4, S4), NO_OBJ_SCORE, device=device))
                scores.append(torch.full((1, 1), NO_OBJ_SCORE, device=device))
            else:
                low_res.append(src["pred_masks"].to(device).float())
                scores.append(src["object_score_logits"].to(device).float())

        video_h, video_w = session.video_h, session.video_w
        if not low_res:
            low_res_masks = torch.zeros((0, 1, S4, S4), device=device)
            return StreamOutput(
                frame_idx=f,
                obj_ids=[],
                mask_logits=torch.zeros((0, 1, video_h, video_w), device=device),
                low_res_mask_logits=low_res_masks,
                object_score_logits=torch.zeros((0, 1), device=device),
            )
        low_res_masks = torch.cat(low_res, dim=0)
        mask_logits = F.interpolate(
            low_res_masks,
            size=(video_h, video_w),
            mode="bilinear",
            align_corners=False,
        )
        if self.non_overlap_masks:
            mask_logits = self._apply_non_overlapping_constraints(mask_logits)
        return StreamOutput(
            frame_idx=f,
            obj_ids=session.obj_ids,
            mask_logits=mask_logits,
            low_res_mask_logits=low_res_masks,
            object_score_logits=torch.cat(scores, dim=0),
        )
