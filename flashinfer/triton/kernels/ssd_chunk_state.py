# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_state.py
# Copyright (c) 2024, Tri Dao, Albert Gu.
#
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Triton kernels for SSD (Structured State-Space Duality) chunk state operations.

This module provides the cumsum forward pass kernel used in Mamba2 SSD.
"""

import math

import torch
import triton
import triton.language as tl
from packaging import version

TRITON3 = version.parse(triton.__version__) >= version.parse("3.0.0")

if TRITON3:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
        return dt

else:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        return dt


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_H": 1}),
        triton.Config({"BLOCK_SIZE_H": 2}),
        triton.Config({"BLOCK_SIZE_H": 4}),
        triton.Config({"BLOCK_SIZE_H": 8}),
        triton.Config({"BLOCK_SIZE_H": 16}),
        triton.Config({"BLOCK_SIZE_H": 32}),
        triton.Config({"BLOCK_SIZE_H": 64}),
    ],
    key=["chunk_size", "nheads"],
)
@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to matrices
    dt_ptr,
    A_ptr,
    dt_bias_ptr,
    dt_out_ptr,
    dA_cumsum_ptr,
    # Matrix dimension
    batch,
    seqlen,
    nheads,
    chunk_size,
    dt_min,
    dt_max,
    # Strides
    stride_dt_batch,
    stride_dt_seqlen,
    stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_dt_out_batch,
    stride_dt_out_chunk,
    stride_dt_out_head,
    stride_dt_out_csize,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)

    # if dt is long, may cause problems, so use 64 bit
    # https://github.com/triton-lang/triton/issues/1058
    pid_c = tl.program_id(axis=1).to(tl.int64)
    pid_h = tl.program_id(axis=2)
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (
        offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen
    )
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (
        offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize
    )
    dA_cs_ptrs = dA_cumsum_ptr + (
        offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize
    )
    chunk_size_limit = tl.minimum(chunk_size, seqlen - pid_c * chunk_size)

    dt = tl.load(
        dt_ptrs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
        other=0.0,
    ).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(
            dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0
        ).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = softplus(dt)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where(
        (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0
    )
    tl.store(
        dt_out_ptrs,
        dt,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
    )
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(
        dA_cs_ptrs,
        dA_cs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
    )


def chunk_cumsum_fwd(
    dt: torch.Tensor,
    A: torch.Tensor,
    chunk_size: int,
    dt_bias: torch.Tensor = None,
    dt_softplus: bool = False,
    dt_limit: tuple = (0.0, float("inf")),
    dt_out_dtype: torch.dtype = torch.float32,
    valid_seqlen: int | None = None,
    cumsum_block_size_h: int | None = None,
):
    """
    Compute the cumulative sum of dA for SSD (Structured State-Space Duality).

    This is the first step in the SSD combined forward pass, computing:
    - dt_out: processed delta time values (after bias, softplus, and clamping)
    - dA_cumsum: cumulative sum of dt * A along the chunk dimension

    Parameters
    ----------
    dt : torch.Tensor
        Delta time tensor with shape (batch, seqlen, nheads)
    A : torch.Tensor
        A matrix with shape (nheads,) - should be negative for stability
    chunk_size : int
        Size of each chunk for processing
    dt_bias : torch.Tensor, optional
        Optional bias for dt with shape (nheads,)
    dt_softplus : bool
        Whether to apply softplus to dt
    dt_limit : tuple
        (min, max) limits for dt values after processing
    valid_seqlen : int, optional
        Number of logical tokens in a physically padded input. Values beyond
        this point are masked after bias/softplus, matching the reference SSD
        cumsum semantics. The same logical length is used for every batch row;
        partially padded serving inputs therefore require ``batch == 1``.
    cumsum_block_size_h : int, optional
        Fixed head tile for the Triton scan. By default the legacy autotuner
        selects from ``1, 2, 4, 8, 16, 32, 64``. Serving integrations that
        compare against another SSD implementation may provide the same tile
        explicitly to keep the floating-point reduction tree deterministic.

    Returns
    -------
    dA_cumsum : torch.Tensor
        Cumulative sum of dA with shape (batch, nheads, nchunks, chunk_size)
    dt_out : torch.Tensor
        Processed dt with shape (batch, nheads, nchunks, chunk_size)
    """
    batch, seqlen, nheads = dt.shape
    if valid_seqlen is None:
        valid_seqlen = seqlen
    if not isinstance(valid_seqlen, int):
        raise TypeError(f"valid_seqlen must be an int, got {type(valid_seqlen)}")
    if valid_seqlen <= 0 or valid_seqlen > seqlen:
        raise ValueError(
            f"valid_seqlen must be in [1, {seqlen}], got {valid_seqlen}"
        )
    if valid_seqlen != seqlen and batch != 1:
        raise ValueError("partially padded cumsum currently requires batch=1")
    supported_block_sizes = (1, 2, 4, 8, 16, 32, 64)
    if cumsum_block_size_h is not None and (
        type(cumsum_block_size_h) is not int
        or cumsum_block_size_h not in supported_block_sizes
    ):
        raise ValueError(
            "cumsum_block_size_h must be one of "
            f"{supported_block_sizes}, got {cumsum_block_size_h}"
        )
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = math.ceil(seqlen / chunk_size)
    dt_out = torch.empty(
        batch, nheads, nchunks, chunk_size, device=dt.device, dtype=dt_out_dtype
    )
    dA_cumsum = torch.empty(
        batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    if cumsum_block_size_h is None:
        kernel = _chunk_cumsum_fwd_kernel
        grid_chunk_cs = lambda META: (
            batch,
            nchunks,
            triton.cdiv(nheads, META["BLOCK_SIZE_H"]),
        )
        fixed_launch_options = {}
    else:
        # Bypass only the autotuner; use the exact same JIT kernel body.
        kernel = _chunk_cumsum_fwd_kernel.fn
        grid_chunk_cs = (
            batch,
            nchunks,
            triton.cdiv(nheads, cumsum_block_size_h),
        )
        fixed_launch_options = {
            "BLOCK_SIZE_H": cumsum_block_size_h,
            "num_warps": 4,
            "num_stages": 3,
            "num_ctas": 1,
        }
    with torch.cuda.device(dt.device.index):
        kernel[grid_chunk_cs](
            dt,
            A,
            dt_bias,
            dt_out,
            dA_cumsum,
            batch,
            valid_seqlen,
            nheads,
            chunk_size,
            dt_limit[0],
            dt_limit[1],
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_out.stride(0),
            dt_out.stride(2),
            dt_out.stride(1),
            dt_out.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
            **fixed_launch_options,
        )
    return dA_cumsum, dt_out
