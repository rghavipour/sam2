# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Per-row memory for `SAM2StreamPredictor`.

A row is one object to track on the current frame. `build_row_memory` gathers that
object's memory exactly like the forward, non-init branch of
`SAM2Base._prepare_memory_conditioned_features` (batch 1), but stops before the
memory attention call, so rows with the same memory layout can share one call.
"""

from dataclasses import dataclass

import torch

from sam2.modeling.sam2_utils import get_1d_sine_pe, select_closest_cond_frames


@dataclass
class RowMemory:
    memory: torch.Tensor  # [M, 1, C_mem]
    memory_pos: torch.Tensor  # [M, 1, C_mem]
    num_obj_ptr_tokens: int
    # (num cond memories, non-cond t_pos present, num pointer tokens)
    key: tuple


def build_row_memory(model, obj, frame_idx, num_frames):
    """Memory tokens and positions of `obj` for tracking `frame_idx` (forward)."""
    device = model.device
    C = model.hidden_dim
    num_maskmem = model.num_maskmem
    to_cat_memory, to_cat_memory_pos_embed = [], []

    # conditioning frames first (all with t_pos=0)
    cond_outputs = obj.cond_outputs
    assert len(cond_outputs) > 0
    selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
        frame_idx, cond_outputs, model.max_cond_frames_in_attn
    )
    t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
    # then the last (num_maskmem - 1) non-conditioning frames, with temporal stride
    stride = 1 if model.training else model.memory_temporal_stride_for_eval
    non_cond_t_pos = []
    for t_pos in range(1, num_maskmem):
        t_rel = num_maskmem - t_pos
        if t_rel == 1:
            prev_frame_idx = frame_idx - t_rel
        else:
            prev_frame_idx = ((frame_idx - 2) // stride) * stride
            prev_frame_idx = prev_frame_idx - (t_rel - 2) * stride
        out = obj.non_cond_outputs.get(prev_frame_idx, None)
        if out is None:
            out = unselected_cond_outputs.get(prev_frame_idx, None)
        if out is not None:
            non_cond_t_pos.append(t_pos)
        t_pos_and_prevs.append((t_pos, out))

    for t_pos, prev in t_pos_and_prevs:
        if prev is None:
            continue
        feats = prev["maskmem_features"].to(device, non_blocking=True)
        to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
        maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
        maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
        maskmem_enc = maskmem_enc + model.maskmem_tpos_enc[num_maskmem - t_pos - 1]
        to_cat_memory_pos_embed.append(maskmem_enc)

    num_obj_ptr_tokens = 0
    if model.use_obj_ptrs_in_encoder:
        max_obj_ptrs_in_encoder = min(num_frames, model.max_obj_ptrs_in_encoder)
        if not model.training and model.only_obj_ptrs_in_the_past_for_eval:
            ptr_cond_outputs = {
                t: out for t, out in selected_cond_outputs.items() if t <= frame_idx
            }
        else:
            ptr_cond_outputs = selected_cond_outputs
        pos_and_ptrs = [
            (
                (
                    (frame_idx - t)
                    if model.use_signed_tpos_enc_to_obj_ptrs
                    else abs(frame_idx - t)
                ),
                out["obj_ptr"],
            )
            for t, out in ptr_cond_outputs.items()
        ]
        for t_diff in range(1, max_obj_ptrs_in_encoder):
            t = frame_idx - t_diff
            if t < 0 or (num_frames is not None and t >= num_frames):
                break
            out = obj.non_cond_outputs.get(t, unselected_cond_outputs.get(t, None))
            if out is not None:
                pos_and_ptrs.append((t_diff, out["obj_ptr"]))
        if len(pos_and_ptrs) > 0:
            pos_list, ptrs_list = zip(*pos_and_ptrs)
            obj_ptrs = torch.stack(ptrs_list, dim=0)  # [L, 1, C]
            if model.add_tpos_enc_to_obj_ptrs:
                t_diff_max = max_obj_ptrs_in_encoder - 1
                tpos_dim = C if model.proj_tpos_enc_in_obj_ptrs else model.mem_dim
                obj_pos = torch.tensor(pos_list).to(device=device, non_blocking=True)
                obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                obj_pos = model.obj_ptr_tpos_proj(obj_pos)
                obj_pos = obj_pos.unsqueeze(1).expand(-1, 1, model.mem_dim)
            else:
                obj_pos = obj_ptrs.new_zeros(len(pos_list), 1, model.mem_dim)
            if model.mem_dim < C:
                # split a pointer into (C // mem_dim) tokens
                obj_ptrs = obj_ptrs.reshape(-1, 1, C // model.mem_dim, model.mem_dim)
                obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                obj_pos = obj_pos.repeat_interleave(C // model.mem_dim, dim=0)
            to_cat_memory.append(obj_ptrs)
            to_cat_memory_pos_embed.append(obj_pos)
            num_obj_ptr_tokens = obj_ptrs.shape[0]

    memory = torch.cat(to_cat_memory, dim=0)
    memory_pos = torch.cat(to_cat_memory_pos_embed, dim=0)
    key = (len(selected_cond_outputs), tuple(non_cond_t_pos), num_obj_ptr_tokens)
    return RowMemory(memory, memory_pos, num_obj_ptr_tokens, key)


def group_rows(rows):
    """Row indices grouped by memory layout, groups in order of first appearance."""
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row.key, []).append(i)
    return list(groups.values())


def stack_rows(rows):
    """Stack rows of one memory layout along the batch dimension."""
    key = rows[0].key
    assert all(row.key == key for row in rows), "rows must share one memory layout"
    memory = torch.cat([row.memory for row in rows], dim=1)
    memory_pos = torch.cat([row.memory_pos for row in rows], dim=1)
    return memory, memory_pos, rows[0].num_obj_ptr_tokens
