# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

# modules whose weights autocast casts to its lower-precision dtype on every call
_MATMUL_MODULES = (
    torch.nn.Linear,
    torch.nn.Conv1d,
    torch.nn.Conv2d,
    torch.nn.Conv3d,
    torch.nn.ConvTranspose1d,
    torch.nn.ConvTranspose2d,
    torch.nn.ConvTranspose3d,
)


def cast_matmul_weights(module, dtype=torch.bfloat16):
    """
    Store the weights of `module`'s Linear and Conv layers in `dtype`, in place.

    Under `torch.autocast(dtype=dtype)` these layers run in `dtype` anyway, so autocast
    casts their fp32 weights on every call (one extra kernel and one extra copy per
    layer per frame). Casting once gives the same results without that work and halves
    the memory those weights use. Everything else (norms, embeddings, buffers) keeps its
    dtype. The module must then only run under autocast with `dtype`.

    Returns the number of layers cast.
    """
    n = 0
    for m in module.modules():
        if isinstance(m, _MATMUL_MODULES):
            m.to(dtype)
            n += 1
    return n
