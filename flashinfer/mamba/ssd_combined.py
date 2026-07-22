"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

SSD (Structured State-Space Duality) Combined Forward Pass for Mamba2
=====================================================================

This module provides the combined forward pass for Mamba2 SSD, combining:
1. Triton kernel for cumsum computation
2. CuTe DSL kernel for the main SSD computation (Blackwell optimized)
"""

import functools
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cuda.bindings.driver as cuda_drv
import torch
import tvm_ffi
from cutlass import Int32
from cutlass.base_dsl.compiler import GenerateLineInfo  # profiling
from ..jit.mamba.seq_chunk_cumsum import gen_seq_chunk_cumsum_module
from ..triton.kernels.ssd_chunk_state import chunk_cumsum_fwd
from .ssd_kernel import SSDKernel


# Feature-detection sentinel for serving-framework integrations.  Increment
# only when the packed-varlen state/output contract changes incompatibly.
SSD_COMBINED_VARLEN_STATE_API_VERSION = 2


@functools.cache
def _get_seq_chunk_cumsum_module():
    """Get cached seq_chunk_cumsum JIT module."""
    return gen_seq_chunk_cumsum_module().build_and_load()


@functools.cache
def _get_compiled_ssd_kernel(
    chunk_size: int,
    headdim: int,
    dstate: int,
    has_d: bool,
    d_has_hdim: bool,
    has_init_states: bool,
    has_varlen: bool,
    has_z: bool,
    has_intermediate_states: bool,
    output_token_major: bool,
    io_dtype,
    state_dtype,
    delta_dtype,
    seq_idx_dtype,
    cumsum_dtype,
    acc_dtype,
    max_active_clusters: int,
):
    """Compile SSD kernel with fake tensors for shape-polymorphic caching.

    Uses CuTe DSL fake tensors with symbolic dimensions so the compiled kernel
    can be reused across different batch sizes, sequence lengths, etc.
    The compiled kernel accepts torch tensors directly via TVM-FFI.
    """
    kernel_obj = SSDKernel(
        chunk_size,
        headdim,
        dstate,
        has_d,
        d_has_hdim,
        has_init_states,
        has_varlen,
        has_z,
        has_intermediate_states,
        output_token_major,
        io_dtype=io_dtype,
        state_dtype=state_dtype,
        delta_dtype=delta_dtype,
        seq_idx_dtype=seq_idx_dtype,
        cumsum_delta_dtype=cumsum_dtype,
        acc_dtype=acc_dtype,
    )

    L = chunk_size
    D = headdim
    N = dstate

    # Symbolic dimensions for all runtime-variable sizes
    sym_C = cute.sym_int()  # nchunks
    sym_EH = cute.sym_int()  # nheads
    sym_B = cute.sym_int()  # batch
    sym_G = cute.sym_int()  # ngroups

    # Stride order convention for make_fake_compact_tensor:
    #   stride_order[i] = rank of dim i, where rank 0 = innermost (stride 1).
    # This differs from mark_compact_shape_dynamic which uses inv(perm).
    # Formula: make_fake_stride_order[i] = N - 1 - permute_order[i]

    # x: (D, L, C, EH, B) — D stride 1 (from permute (4,2,1,3,0) of 5D)
    x_fake = cute.runtime.make_fake_compact_tensor(
        io_dtype,
        (D, L, sym_C, sym_EH, sym_B),
        stride_order=(0, 2, 3, 1, 4),
        assumed_align=16,
    )

    # cumsum_delta: (L, C, EH, B) — L stride 1 (from permute (3,2,1,0) of 4D)
    cumsum_delta_fake = cute.runtime.make_fake_compact_tensor(
        cumsum_dtype,
        (L, sym_C, sym_EH, sym_B),
        stride_order=(0, 1, 2, 3),
        assumed_align=16,
    )

    # delta: (L, C, EH, B) — L stride 1 (same layout as cumsum_delta)
    delta_fake = cute.runtime.make_fake_compact_tensor(
        delta_dtype,
        (L, sym_C, sym_EH, sym_B),
        stride_order=(0, 1, 2, 3),
        assumed_align=16,
    )

    # B: (L, N, C, G, B) — N stride 1 (from permute (2,4,1,3,0) of 5D)
    b_fake = cute.runtime.make_fake_compact_tensor(
        io_dtype,
        (L, N, sym_C, sym_G, sym_B),
        stride_order=(2, 0, 3, 1, 4),
        assumed_align=16,
    )

    # C: (L, N, C, G, B) — N stride 1 (same layout as B)
    c_fake = cute.runtime.make_fake_compact_tensor(
        io_dtype,
        (L, N, sym_C, sym_G, sym_B),
        stride_order=(2, 0, 3, 1, 4),
        assumed_align=16,
    )

    # y: (L, D, C, EH, B).  The legacy chunk-major output has L stride 1;
    # token-major output has D stride 1 and can be consumed directly by the
    # following projection/norm without a transpose-copy.
    y_fake = cute.runtime.make_fake_compact_tensor(
        io_dtype,
        (L, D, sym_C, sym_EH, sym_B),
        stride_order=(2, 0, 3, 1, 4) if output_token_major else (0, 2, 1, 3, 4),
        assumed_align=16,
    )

    # State tensors use dimensions independent of the packed input batch.  Keep
    # input and output symbols separate because final stores can be suppressed,
    # in which case fstate is only a one-row descriptor dummy while
    # initial_states still contains every sequence.
    sym_init_state_B = cute.sym_int()
    sym_final_state_B = cute.sym_int()

    # init_states: (N, D, EH, state_B) — N stride 1 (from permute (3,2,1,0) of 4D)
    init_states_fake = None
    if has_init_states:
        init_states_fake = cute.runtime.make_fake_compact_tensor(
            state_dtype,
            (N, D, sym_EH, sym_init_state_B),
            stride_order=(0, 1, 2, 3),
            assumed_align=16,
        )

    # fstate: (N, D, EH, state_B) — N stride 1 (same layout as init_states)
    fstate_fake = cute.runtime.make_fake_compact_tensor(
        state_dtype,
        (N, D, sym_EH, sym_final_state_B),
        stride_order=(0, 1, 2, 3),
        assumed_align=16,
    )

    intermediate_states_fake = None
    intermediate_state_indices_fake = None
    if has_intermediate_states:
        sym_num_state_slots = cute.sym_int()
        # Serving caches commonly place each SSM row inside a larger unified
        # page.  Inner N/D/head modes remain compact, while the cache-row
        # stride is runtime-dynamic (and can exceed the logical row size).
        intermediate_states_fake = cute.runtime.make_fake_tensor(
            state_dtype,
            (N, D, sym_EH, sym_num_state_slots),
            stride=(1, N, D * N, cute.sym_int()),
            assumed_align=16,
        )
        sym_nlc = cute.sym_int()
        intermediate_state_indices_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (sym_nlc,),
            assumed_align=4,
        )

    # D tensor: (D_dim, EH) — D_dim stride 1 (from permute (1,0) of 2D)
    d_fake = None
    if has_d:
        d_dim = D if d_has_hdim else 1
        d_fake = cute.runtime.make_fake_compact_tensor(
            io_dtype,
            (d_dim, sym_EH),
            stride_order=(0, 1),
            assumed_align=16,
        )

    # z: same layout as x — optional
    z_fake = None
    if has_z:
        z_fake = cute.runtime.make_fake_compact_tensor(
            io_dtype,
            (D, L, sym_C, sym_EH, sym_B),
            stride_order=(0, 2, 3, 1, 4),
            assumed_align=16,
        )

    # Varlen metadata — optional
    seq_idx_fake = None
    chunk_indices_fake = None
    chunk_offsets_fake = None
    seq_chunk_cumsum_fake = None
    if has_varlen:
        sym_batch_dim = cute.sym_int()
        sym_seqlen = cute.sym_int()
        seq_idx_fake = cute.runtime.make_fake_compact_tensor(
            seq_idx_dtype,
            (sym_batch_dim, sym_seqlen),
            stride_order=(1, 0),
            assumed_align=4,
        )
        sym_nlc = cute.sym_int()  # num_logical_chunks
        chunk_indices_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (sym_nlc,),
            assumed_align=4,
        )
        chunk_offsets_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (sym_nlc,),
            assumed_align=4,
        )
        sym_nsp1 = cute.sym_int()  # num_seqs + 1
        seq_chunk_cumsum_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (sym_nsp1,),
            assumed_align=4,
        )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled = cute.compile[GenerateLineInfo(True)](
        kernel_obj,
        x_fake,
        cumsum_delta_fake,
        delta_fake,
        b_fake,
        c_fake,
        y_fake,
        init_states_fake,
        fstate_fake,
        intermediate_states_fake,
        intermediate_state_indices_fake,
        d_fake,
        z_fake,
        seq_idx_fake,
        chunk_indices_fake,
        chunk_offsets_fake,
        seq_chunk_cumsum_fake,
        Int32(0),  # num_logical_chunks (dummy, runtime value)
        Int32(0),  # num_seqs (dummy, runtime value)
        Int32(0),  # valid_seqlen (dummy, runtime value)
        Int32(0),  # num_intermediate_state_slots (dummy, runtime value)
        Int32(1),  # write_final_states (dummy, runtime value)
        max_active_clusters,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled


class SSDCombined:
    """Mamba2 SSD combined forward pass with cached host-side state.

    Caches compiled kernel, hardware info, and pre-allocated buffers
    to minimize host-side overhead on repeated calls.

    Dtype expectations (no runtime conversions — assert on mismatch):
        io_dtype (bf16):  x, B, C, z, D, out
        state_dtype (bf16): initial_states, final_states
        fp32 always:      A, dA_cumsum
        any (bf16/fp32):  dt, dt_bias (consumed by Triton cumsum preprocessor)
        dt_processed_dtype (bf16/fp32): processed dt consumed by the SSD kernel

    Usage::

        ssd = SSDCombined(chunk_size=128, nheads=8, headdim=64,
                          dstate=128, ngroups=8)
        out, final_states = ssd.run(x, dt, A, B, C, D=D, dt_bias=dt_bias, ...)

    For packed serving workloads, ``has_intermediate_states=True`` enables
    sparse post-segment state stores into a caller-owned cache.  Pair it with
    ``output_token_major=True`` to write the output directly in contiguous
    ``(batch, seqlen, nheads, headdim)`` order.
    """

    varlen_state_api_version = SSD_COMBINED_VARLEN_STATE_API_VERSION
    supports_strided_intermediate_states = True
    supports_fp32_processed_dt = True

    def __init__(
        self,
        chunk_size: int,
        nheads: int,
        headdim: int,
        dstate: int,
        ngroups: int,
        io_dtype: torch.dtype = torch.bfloat16,
        state_dtype: torch.dtype = torch.bfloat16,
        has_d: bool = True,
        d_has_hdim: bool = False,
        has_initial_states: bool = False,
        has_varlen: bool = False,
        has_z: bool = False,
        seq_idx_dtype=torch.int64,
        has_intermediate_states: bool = False,
        output_token_major: bool = False,
        dt_processed_dtype: torch.dtype = torch.bfloat16,
        cumsum_block_size_h: Optional[int] = None,
    ):
        from ..utils import get_compute_capability

        if cumsum_block_size_h is not None and (
            type(cumsum_block_size_h) is not int
            or cumsum_block_size_h not in (1, 2, 4, 8, 16, 32, 64)
        ):
            raise ValueError(
                "cumsum_block_size_h must be one of "
                f"(1, 2, 4, 8, 16, 32, 64), got {cumsum_block_size_h}"
            )

        major, minor = get_compute_capability(torch.device("cuda"))
        # The SSD CuTe-DSL kernel uses tcgen05 MMA (MmaF16BF16Op), which is only
        # available on datacenter Blackwell (SM100/SM103/SM110). Consumer/workstation
        # Blackwell (SM120/SM121) lacks tcgen05, so reject it here with a clear message
        # instead of a cryptic cute-dsl "expects ... sm_100a ... got sm_120a" OpError.
        if major not in (10, 11):
            raise ValueError(
                f"SSDCombined requires SM100-SM110 (tcgen05): Blackwell datacenter "
                f"GPUs (SM100/SM103/SM110). Got SM{major}{minor}."
            )

        self.chunk_size = chunk_size
        self.nheads = nheads
        self.headdim = headdim
        self.dstate = dstate
        self.ngroups = ngroups
        self._has_d = has_d
        self._d_has_hdim = d_has_hdim
        self._has_init_states = has_initial_states
        self._has_varlen = has_varlen
        self._has_z = has_z
        self._has_intermediate_states = has_intermediate_states
        self._output_token_major = output_token_major
        self._state_torch_dtype = state_dtype
        self._dt_processed_torch_dtype = dt_processed_dtype
        self._cumsum_block_size_h = cumsum_block_size_h

        # Resolve dtypes
        assert io_dtype == torch.bfloat16, f"io_dtype must be bfloat16, got {io_dtype}"
        self._io_dtype = cutlass.BFloat16
        _state_dtype_map = {
            torch.bfloat16: cutlass.BFloat16,
            torch.float16: cutlass.Float16,
            # torch.float32: cutlass.Float32, # -- not supported yet
        }
        assert state_dtype in _state_dtype_map, (
            f"state_dtype must be one of {list(_state_dtype_map.keys())}, got {state_dtype}"
        )
        self._state_dtype = _state_dtype_map[state_dtype]
        _delta_dtype_map = {
            torch.bfloat16: cutlass.BFloat16,
            torch.float32: cutlass.Float32,
        }
        assert dt_processed_dtype in _delta_dtype_map, (
            "dt_processed_dtype must be torch.bfloat16 or torch.float32, "
            f"got {dt_processed_dtype}"
        )
        self._delta_dtype = _delta_dtype_map[dt_processed_dtype]
        self._cumsum_dtype = cutlass.Float32
        self._acc_dtype = cutlass.Float32
        self._io_torch_dtype = cutlass_torch.dtype(self._io_dtype)

        # Resolve seq_idx dtype
        _seq_idx_dtype_map = {
            torch.int32: cutlass.Int32,
            torch.int64: cutlass.Int64,
        }
        self._seq_idx_cutlass_dtype = _seq_idx_dtype_map.get(
            seq_idx_dtype, cutlass.Int32
        )

        # Compile kernel (cached across instances with same compile-time params).
        # HardwareInfo uses cuda.bindings.driver which requires a driver context.
        # torch.cuda.init() alone is insufficient; allocating on GPU forces
        # creation of the full driver context.
        _, ctx = cuda_drv.cuCtxGetCurrent()
        if int(ctx) == 0:
            torch.empty(0, device="cuda")
        hardware_info = cutlass.utils.HardwareInfo()
        self._max_active_clusters = hardware_info.get_max_active_clusters(1)

        self._compiled_kernel = _get_compiled_ssd_kernel(
            chunk_size,
            headdim,
            dstate,
            has_d,
            d_has_hdim,
            has_initial_states,
            has_varlen,
            has_z,
            has_intermediate_states,
            output_token_major,
            self._io_dtype,
            self._state_dtype,
            self._delta_dtype,
            self._seq_idx_cutlass_dtype,
            self._cumsum_dtype,
            self._acc_dtype,
            self._max_active_clusters,
        )

        # Pre-allocated buffer caches (lazily initialized on first run)
        self._fstate_shape = None
        self._fstate_torch = None

        self._seq_cumsum_size = 0
        self._seq_cumsum_buf = None

    # -- buffer cache helpers --------------------------------------------------

    def _get_or_alloc_fstate(self, fstate_batch):
        shape = (fstate_batch, self.nheads, self.headdim, self.dstate)
        if self._fstate_shape != shape:
            state_torch_dtype = cutlass_torch.dtype(self._state_dtype)
            # Allocate contiguous (B, EH, D, N); kernel receives permuted view
            self._fstate_torch = torch.empty(
                *shape, dtype=state_torch_dtype, device="cuda"
            )
            self._fstate_shape = shape
        return self._fstate_torch

    def _get_or_alloc_seq_cumsum(self, num_seqs, device):
        size = num_seqs + 1
        if self._seq_cumsum_size != size:
            self._seq_cumsum_buf = torch.zeros(
                size,
                dtype=torch.int32,
                device=device,
            )
            self._seq_cumsum_size = size
        else:
            self._seq_cumsum_buf.zero_()
        return self._seq_cumsum_buf

    def compute_seq_chunk_cumsum(
        self,
        seq_idx: torch.Tensor,
        chunk_indices: torch.Tensor,
        chunk_offsets: torch.Tensor,
        chunk_size: int,
        num_seqs: int,
        seq_chunk_cumsum: Optional[torch.Tensor] = None,
        tile_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute (or fill) the seq_chunk_cumsum mapping for varlen mode.

        Args:
            seq_idx: int32/int64 tensor [1, total_seqlen].
            chunk_indices: int32 tensor [num_logical_chunks].
            chunk_offsets: int32 tensor [num_logical_chunks].
            chunk_size: chunk size used by the SSD kernel.
            num_seqs: number of sequences.
            seq_chunk_cumsum: Optional output buffer of shape [num_seqs + 1],
                dtype int32.  Allocated internally if None.
            tile_state: Optional pre-allocated uint8 workspace for
                cub::ScanTileState (multi-block path only).  Size must be at
                least ``tile_state_size(num_seqs)`` bytes.  Allocated
                internally if None.

        Returns:
            int32 tensor [num_seqs + 1] with the exclusive prefix sum of
            per-sequence logical chunk counts.
        """
        if seq_chunk_cumsum is None:
            seq_chunk_cumsum = self._get_or_alloc_seq_cumsum(num_seqs, seq_idx.device)
        if tile_state is None:
            module = _get_seq_chunk_cumsum_module()
            tile_state_bytes = module.seq_chunk_cumsum_tile_state_size(num_seqs)
            tile_state = (
                torch.empty(
                    tile_state_bytes,
                    dtype=torch.uint8,
                    device=seq_idx.device,
                )
                if tile_state_bytes > 0
                else None
            )
        _get_seq_chunk_cumsum_module().seq_chunk_cumsum(
            seq_idx,
            chunk_indices,
            chunk_offsets,
            seq_chunk_cumsum,
            tile_state,
            chunk_size,
            len(chunk_indices),
            num_seqs,
        )
        return seq_chunk_cumsum

    @staticmethod
    def tile_state_size(num_seqs: int) -> int:
        """Return the tile_state buffer size in bytes for the given num_seqs."""
        return _get_seq_chunk_cumsum_module().seq_chunk_cumsum_tile_state_size(num_seqs)

    # -- main entry point ------------------------------------------------------

    def run(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        dt_softplus: bool = False,
        dt_limit: Tuple[float, float] = (0.0, float("inf")),
        initial_states: Optional[torch.Tensor] = None,
        seq_idx: Optional[torch.Tensor] = None,
        chunk_indices: Optional[torch.Tensor] = None,
        chunk_offsets: Optional[torch.Tensor] = None,
        seq_chunk_cumsum: Optional[torch.Tensor] = None,
        update_seq_chunk_cumsum: bool = False,
        out: Optional[torch.Tensor] = None,
        return_final_states: bool = True,
        num_seqs: Optional[int] = None,
        valid_seqlen: Optional[int] = None,
        intermediate_state_indices: Optional[torch.Tensor] = None,
        intermediate_states: Optional[torch.Tensor] = None,
        final_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run SSD combined forward pass.

        Parameters match ``ssd_combined_fwd`` — see its docstring for details.

        Args:
            seq_chunk_cumsum: Optional int32 tensor of shape [num_seqs + 1].
                If provided with update_seq_chunk_cumsum=False (default), treated
                as pre-computed and the internal computation is skipped.
                If provided with update_seq_chunk_cumsum=True, used as the output
                buffer and filled by the internal computation.
                If None, an internal buffer is allocated and computed.
            update_seq_chunk_cumsum: If True, (re)compute seq_chunk_cumsum into
                the provided tensor.  Defaults to False.
            num_seqs: Explicit number of original sequences in packed varlen
                mode.  This is required when neither ``initial_states`` nor a
                precomputed ``seq_chunk_cumsum`` supplies the count.
            valid_seqlen: Number of real tokens in a physically padded input.
                The physical ``x.shape[1]`` must remain chunk aligned.  Tokens
                in ``[valid_seqlen, x.shape[1])`` are excluded from state math
                and output stores.  A short final chunk is supported only in
                packed varlen mode.  The caller must keep padded ``x``, ``dt``,
                ``B``, and ``C`` values finite (zero-fill is recommended): the
                fused kernel compute-masks the tail after tensor-core loads, so
                NaN or Inf padding can contaminate otherwise valid results.
            intermediate_state_indices: Optional int32 tensor with one entry
                per logical segment.  A nonnegative value is the destination
                row in ``intermediate_states`` for that segment's post-update
                state; ``-1`` skips the store.  Destinations need not be dense,
                which allows writing directly into a serving cache.  The caller
                must not assign the same destination to segments processed by
                different sequences because those writes would race.  A state
                can be captured only at a logical-segment endpoint, so callers
                must split metadata at sequence boundaries, physical chunk
                boundaries, and every requested cache/checkpoint boundary.
            intermediate_states: Caller-owned state destination of shape
                ``(num_slots, nheads, headdim, dstate)``.  Inner dimensions
                must be compact, but the row stride may be larger than the
                logical row, allowing a view into padded unified cache pages.
                The destination may be larger than the number of selected
                segments.
            final_states: Optional caller-owned contiguous final-state output.
                When omitted, the legacy internal reusable buffer is used.
                Setting ``return_final_states=False`` and omitting this buffer
                suppresses final-state stores; selected intermediate stores can
                include every sequence final without materializing a separate
                ``num_seqs``-sized tensor.
        """
        chunk_size = self.chunk_size

        batch, seqlen, nheads, headdim = x.shape
        _, _, ngroups, dstate = B.shape
        nchunks = seqlen // chunk_size

        if self._has_varlen and batch != 1:
            raise ValueError(
                "packed varlen SSDCombined requires batch=1 because logical "
                "sequences are encoded in seq_idx"
            )
        if self._has_intermediate_states and not self._has_varlen and batch > 1:
            raise ValueError(
                "selected intermediate state stores with non-varlen batch > 1 "
                "are unsupported because the 1D destination map would be shared "
                "across batch CTAs; use packed varlen metadata or batch=1"
            )

        if valid_seqlen is None:
            valid_seqlen = seqlen
        if not isinstance(valid_seqlen, int):
            raise TypeError(f"valid_seqlen must be an int, got {type(valid_seqlen)}")
        if valid_seqlen <= 0 or valid_seqlen > seqlen:
            raise ValueError(
                f"valid_seqlen must be in [1, {seqlen}], got {valid_seqlen}"
            )
        if valid_seqlen != seqlen and (not self._has_varlen or batch != 1):
            raise ValueError(
                "a padded short final chunk is supported only for packed varlen "
                "inputs with batch=1"
            )

        assert seqlen % chunk_size == 0, (
            f"seqlen ({seqlen}) must be divisible by chunk_size ({chunk_size})"
        )

        # A is always fp32
        assert A.dtype == torch.float32, f"A must be float32, got {A.dtype}"

        if self._has_init_states != (initial_states is not None):
            raise ValueError(
                "initial_states presence must match "
                "SSDCombined(has_initial_states=...)"
            )

        # Validate varlen arguments
        if seq_idx is not None:
            assert seq_idx.shape == (batch, seqlen), (
                f"seq_idx shape {seq_idx.shape} doesn't match "
                f"(batch={batch}, seqlen={seqlen})"
            )
            assert seq_idx.dtype in (torch.int32, torch.int64), (
                f"seq_idx must be int32 or int64, got {seq_idx.dtype}"
            )
        if chunk_indices is not None:
            assert chunk_indices.dim() == 1, (
                f"chunk_indices must be 1D, got {chunk_indices.dim()}D"
            )
            assert chunk_indices.dtype == torch.int32, (
                f"chunk_indices must be int32, got {chunk_indices.dtype}"
            )
        if chunk_offsets is not None:
            assert chunk_offsets.dim() == 1, (
                f"chunk_offsets must be 1D, got {chunk_offsets.dim()}D"
            )
            assert chunk_offsets.dtype == torch.int32, (
                f"chunk_offsets must be int32, got {chunk_offsets.dtype}"
            )
        if chunk_indices is not None and chunk_offsets is not None:
            assert chunk_indices.shape == chunk_offsets.shape, (
                f"chunk_indices and chunk_offsets must have the same shape, "
                f"got {chunk_indices.shape} vs {chunk_offsets.shape}"
            )
        if seq_chunk_cumsum is not None:
            if seq_chunk_cumsum.dim() != 1 or seq_chunk_cumsum.dtype != torch.int32:
                raise ValueError("seq_chunk_cumsum must be a 1D int32 tensor")

        if out is not None:
            expected_out_shape = (
                (batch, seqlen, nheads, headdim)
                if self._output_token_major
                else (batch, nheads, headdim, nchunks, chunk_size)
            )
            assert out.shape == expected_out_shape, (
                f"out shape {out.shape} doesn't match expected {expected_out_shape}"
            )
            assert out.dtype == x.dtype, (
                f"out dtype {out.dtype} doesn't match x dtype {x.dtype}"
            )
            assert out.is_contiguous(), "out must be contiguous"
            if out.device != x.device:
                raise ValueError("out must be on the same device as x")

        # The serving path keeps processed dt in FP32 to match vLLM's Triton
        # recurrence and its FP16 cache checkpoints.  Legacy callers retain
        # the BF16 default to avoid changing their numerical/performance
        # contract.
        dA_cumsum, dt_processed = chunk_cumsum_fwd(
            dt,
            A,
            chunk_size,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            dt_out_dtype=self._dt_processed_torch_dtype,
            valid_seqlen=valid_seqlen,
            cumsum_block_size_h=self._cumsum_block_size_h,
        )

        if out is None:
            if self._output_token_major:
                out = torch.empty(
                    batch,
                    seqlen,
                    nheads,
                    headdim,
                    dtype=x.dtype,
                    device=x.device,
                )
            else:
                out = torch.empty(
                    batch,
                    nheads,
                    headdim,
                    nchunks,
                    chunk_size,
                    dtype=x.dtype,
                    device=x.device,
                )

        io_torch_dtype = self._io_torch_dtype

        # x: (B, seqlen, EH, D) -> (D, L, C, EH, B)
        x_reshaped = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
        assert x.dtype == io_torch_dtype, (
            f"x dtype {x.dtype} doesn't match {io_torch_dtype}"
        )
        x_permuted = x_reshaped.permute(4, 2, 1, 3, 0)

        # z: same layout as x
        z_permuted = None
        if z is not None:
            z_reshaped = z.reshape(batch, nchunks, chunk_size, nheads, headdim)
            assert z.dtype == io_torch_dtype, (
                f"z dtype {z.dtype} doesn't match {io_torch_dtype}"
            )
            z_permuted = z_reshaped.permute(4, 2, 1, 3, 0)

        # dt: already in io_dtype from cumsum kernel
        delta_permuted = dt_processed.permute(3, 2, 1, 0)  # (L, C, EH, B)

        # dA_cumsum: (B, EH, C, L) -> (L, C, EH, B)
        assert dA_cumsum.dtype == cutlass_torch.dtype(self._cumsum_dtype), (
            f"dA_cumsum dtype {dA_cumsum.dtype} doesn't match "
            f"cumsum_dtype {self._cumsum_dtype}"
        )
        cumsum_delta_permuted = dA_cumsum.permute(3, 2, 1, 0)

        # B: (B, seqlen, G, N) -> (L, N, C, G, B)
        B_reshaped = B.reshape(batch, nchunks, chunk_size, ngroups, dstate)
        assert B.dtype == io_torch_dtype, (
            f"B dtype {B.dtype} doesn't match {io_torch_dtype}"
        )
        b_permuted = B_reshaped.permute(2, 4, 1, 3, 0)

        # C: (B, seqlen, G, N) -> (L, N, C, G, B)
        C_reshaped = C.reshape(batch, nchunks, chunk_size, ngroups, dstate)
        assert C.dtype == io_torch_dtype, (
            f"C dtype {C.dtype} doesn't match {io_torch_dtype}"
        )
        c_permuted = C_reshaped.permute(2, 4, 1, 3, 0)

        # D tensor: reshape to match kernel layout
        d_tensor = None
        if self._has_d and D is not None:
            assert D.dtype == io_torch_dtype, (
                f"D dtype {D.dtype} doesn't match io_dtype {io_torch_dtype}"
            )
            if self._d_has_hdim:
                if D.dim() == 1:
                    # (nheads,) -> (headdim, nheads): broadcast must be materialized
                    # (TVM-FFI enforces stride_order=(0,1), so dim 0 must be stride 1)
                    # it's a copy, but a small one.
                    # afaik no inference framework really hits this case
                    d_tensor = D.unsqueeze(1).expand(-1, headdim).contiguous().t()
                else:
                    # (nheads, headdim) -> (headdim, nheads): view, no copy
                    d_tensor = D.t()
            else:
                if D.dim() == 2:
                    d_tensor = D[:, 0].unsqueeze(0)  # (1, nheads)
                else:
                    d_tensor = D.unsqueeze(0)  # (1, nheads)

        # init_states: (B, EH, D, N) -> (N, D, EH, B)
        init_states_permuted = None
        if self._has_init_states and initial_states is not None:
            expected_initial_shape = (batch, nheads, headdim, dstate)
            if self._has_varlen:
                # The sequence dimension is checked against num_seqs below.
                expected_initial_shape = (
                    initial_states.shape[0],
                    nheads,
                    headdim,
                    dstate,
                )
            if initial_states.shape != expected_initial_shape:
                raise ValueError(
                    f"initial_states shape {initial_states.shape} doesn't match "
                    f"expected {expected_initial_shape}"
                )
            assert initial_states.dtype == self._state_torch_dtype, (
                f"init_states dtype {initial_states.dtype} doesn't match "
                f"state_dtype {self._state_torch_dtype}"
            )
            if not initial_states.is_contiguous():
                raise ValueError("initial_states must be contiguous")
            if initial_states.device != x.device:
                raise ValueError("initial_states must be on the same device as x")
            init_states_permuted = initial_states.permute(3, 2, 1, 0)

        # out -> kernel layout (L, D, C, EH, B).  Token-major output keeps D
        # contiguous, whereas the legacy layout keeps L contiguous.
        assert out.dtype == io_torch_dtype, (
            f"out dtype {out.dtype} doesn't match {io_torch_dtype}"
        )
        if self._output_token_major:
            y_permuted = out.reshape(
                batch, nchunks, chunk_size, nheads, headdim
            ).permute(2, 4, 1, 3, 0)
        else:
            y_permuted = out.permute(4, 2, 3, 1, 0)

        # Varlen metadata: seq_chunk_cumsum maps sequence IDs to logical chunk ranges.
        # It can be pre-computed by the caller via compute_seq_chunk_cumsum().
        has_varlen_metadata = all(
            value is not None
            for value in (seq_idx, chunk_indices, chunk_offsets)
        )
        if any(
            value is not None
            for value in (seq_idx, chunk_indices, chunk_offsets)
        ) and not has_varlen_metadata:
            raise ValueError(
                "seq_idx, chunk_indices, and chunk_offsets must be provided together"
            )
        if has_varlen_metadata != self._has_varlen:
            raise ValueError(
                "varlen metadata presence must match SSDCombined(has_varlen=...)"
            )

        kernel_num_seqs = 0
        if has_varlen_metadata:
            inferred_num_seqs = None
            if initial_states is not None:
                inferred_num_seqs = initial_states.shape[0]
            elif seq_chunk_cumsum is not None:
                inferred_num_seqs = seq_chunk_cumsum.numel() - 1
            if num_seqs is None:
                num_seqs = inferred_num_seqs
            if num_seqs is None:
                raise ValueError(
                    "num_seqs must be provided in varlen mode when initial_states "
                    "and seq_chunk_cumsum do not provide it"
                )
            if not isinstance(num_seqs, int) or num_seqs <= 0:
                raise ValueError(f"num_seqs must be a positive int, got {num_seqs}")
            if inferred_num_seqs is not None and inferred_num_seqs != num_seqs:
                raise ValueError(
                    f"num_seqs ({num_seqs}) does not match inferred count "
                    f"({inferred_num_seqs})"
                )
            if (
                seq_chunk_cumsum is not None
                and seq_chunk_cumsum.numel() != num_seqs + 1
            ):
                raise ValueError(
                    "seq_chunk_cumsum must have num_seqs + 1 entries, got "
                    f"{seq_chunk_cumsum.numel()} for num_seqs={num_seqs}"
                )
            kernel_num_seqs = num_seqs
            if seq_chunk_cumsum is None or update_seq_chunk_cumsum:
                seq_chunk_cumsum = self.compute_seq_chunk_cumsum(
                    seq_idx,
                    chunk_indices,
                    chunk_offsets,
                    chunk_size,
                    num_seqs,
                    seq_chunk_cumsum,
                )
        elif num_seqs is not None and num_seqs != batch:
            raise ValueError(
                f"num_seqs is only meaningful in varlen mode (expected {batch}, "
                f"got {num_seqs})"
            )

        fstate_batch = kernel_num_seqs if has_varlen_metadata else batch
        write_final_states = return_final_states or final_states is not None
        if final_states is not None:
            expected_fstate_shape = (
                fstate_batch,
                nheads,
                headdim,
                dstate,
            )
            if final_states.shape != expected_fstate_shape:
                raise ValueError(
                    f"final_states shape {final_states.shape} doesn't match "
                    f"expected {expected_fstate_shape}"
                )
            if final_states.dtype != self._state_torch_dtype:
                raise ValueError(
                    f"final_states dtype {final_states.dtype} doesn't match "
                    f"state dtype {self._state_torch_dtype}"
                )
            if not final_states.is_contiguous():
                raise ValueError("final_states must be contiguous")
            if final_states.data_ptr() % 16 != 0:
                raise ValueError(
                    "final_states data pointer must be 16-byte aligned for TMA stores"
                )
            if final_states.device != x.device:
                raise ValueError("final_states must be on the same device as x")
            fstate_torch = final_states
        else:
            # A one-row descriptor is sufficient when final stores are disabled.
            fstate_torch = self._get_or_alloc_fstate(
                fstate_batch if write_final_states else 1
            )
        fstate_permuted = fstate_torch.permute(3, 2, 1, 0)

        intermediate_states_permuted = None
        num_intermediate_state_slots = 0
        if self._has_intermediate_states:
            if intermediate_state_indices is None or intermediate_states is None:
                raise ValueError(
                    "intermediate_state_indices and intermediate_states are required "
                    "when has_intermediate_states=True"
                )
            if intermediate_state_indices.dim() != 1:
                raise ValueError("intermediate_state_indices must be 1D")
            if intermediate_state_indices.dtype != torch.int32:
                raise ValueError("intermediate_state_indices must have dtype int32")
            if not intermediate_state_indices.is_contiguous():
                raise ValueError("intermediate_state_indices must be contiguous")
            if intermediate_state_indices.device != x.device:
                raise ValueError(
                    "intermediate_state_indices must be on the same device as x"
                )
            expected_state_tail = (nheads, headdim, dstate)
            if (
                intermediate_states.dim() != 4
                or tuple(intermediate_states.shape[1:]) != expected_state_tail
            ):
                raise ValueError(
                    "intermediate_states must have shape "
                    f"(num_slots, {nheads}, {headdim}, {dstate}), got "
                    f"{tuple(intermediate_states.shape)}"
                )
            if intermediate_states.dtype != self._state_torch_dtype:
                raise ValueError(
                    f"intermediate_states dtype {intermediate_states.dtype} doesn't "
                    f"match state dtype {self._state_torch_dtype}"
                )
            if intermediate_states.data_ptr() % 16 != 0:
                raise ValueError(
                    "intermediate_states data pointer must be 16-byte aligned for "
                    "TMA stores"
                )
            expected_inner_strides = (headdim * dstate, dstate, 1)
            if tuple(intermediate_states.stride()[1:]) != expected_inner_strides:
                raise ValueError(
                    "intermediate_states inner dimensions must be compact with "
                    f"strides {expected_inner_strides}, got "
                    f"{tuple(intermediate_states.stride()[1:])}"
                )
            compact_row_size = nheads * headdim * dstate
            if intermediate_states.stride(0) < compact_row_size:
                raise ValueError(
                    "intermediate_states row stride must be at least the logical "
                    f"row size ({compact_row_size}), got "
                    f"{intermediate_states.stride(0)}"
                )
            row_stride_bytes = (
                intermediate_states.stride(0) * intermediate_states.element_size()
            )
            if row_stride_bytes % 16 != 0:
                raise ValueError(
                    "intermediate_states row stride must be 16-byte aligned for "
                    f"TMA stores, got {row_stride_bytes} bytes"
                )
            if intermediate_states.device != x.device:
                raise ValueError("intermediate_states must be on the same device as x")
            num_intermediate_state_slots = intermediate_states.shape[0]
            if num_intermediate_state_slots <= 0:
                raise ValueError("intermediate_states must have at least one row")
            intermediate_states_permuted = intermediate_states.permute(3, 2, 1, 0)
        elif intermediate_state_indices is not None or intermediate_states is not None:
            raise ValueError(
                "construct SSDCombined with has_intermediate_states=True to request "
                "selected state stores"
            )

        num_logical_chunks = (
            len(chunk_indices) if chunk_indices is not None else nchunks
        )
        if (
            intermediate_state_indices is not None
            and len(intermediate_state_indices) != num_logical_chunks
        ):
            raise ValueError(
                "intermediate_state_indices must have one entry per logical segment "
                f"({num_logical_chunks}), got {len(intermediate_state_indices)}"
            )

        # The CuTe kernel resolves its launch stream through TVM-FFI.  Bind
        # that environment stream to PyTorch's current stream so preceding
        # Triton work, the CuTe launch, and following PyTorch consumers remain
        # correctly ordered on non-default serving streams.
        with tvm_ffi.use_torch_stream():
            self._compiled_kernel(
                x_permuted,
                cumsum_delta_permuted,
                delta_permuted,
                b_permuted,
                c_permuted,
                y_permuted,
                init_states_permuted,
                fstate_permuted,
                intermediate_states_permuted,
                intermediate_state_indices,
                d_tensor,
                z_permuted,
                seq_idx,
                chunk_indices,
                chunk_offsets,
                seq_chunk_cumsum,
                Int32(num_logical_chunks),
                Int32(kernel_num_seqs),
                Int32(valid_seqlen),
                Int32(num_intermediate_state_slots),
                Int32(int(write_final_states)),
            )

        if self._output_token_major:
            out_view = out[:, :valid_seqlen]
        else:
            out_view = out.permute(0, 3, 4, 1, 2).reshape(
                batch,
                seqlen,
                nheads,
                headdim,
            )[:, :valid_seqlen]
        # fstate_torch is (B, EH, D, N) — already in the expected return layout
        fstate_out = fstate_torch if return_final_states else None
        return out_view, fstate_out
