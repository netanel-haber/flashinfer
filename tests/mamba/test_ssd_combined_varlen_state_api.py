"""Focused tests for the serving-oriented SSDCombined varlen state API."""

import inspect
import math

import pytest
import torch
from einops import rearrange

import flashinfer.mamba as mamba
from flashinfer.utils import is_sm100a_supported

from .triton_reference.ssd_bmm import _bmm_chunk_fwd
from .triton_reference.ssd_chunk_scan import _chunk_scan_fwd
from .triton_reference.ssd_combined import _mamba_chunk_scan_combined_fwd
from .triton_reference.ssd_chunk_state import (
    _chunk_cumsum_fwd,
    _chunk_state_fwd,
)
from .triton_reference.ssd_state_passing import _state_passing_fwd


def _packed_metadata(seq_lengths, padded_seqlen, chunk_size=128, checkpoint=1152):
    """Build logical segments split at seq, physical-chunk, and checkpoint edges."""
    valid_seqlen = sum(seq_lengths)
    assert valid_seqlen <= padded_seqlen
    seq_starts = [0]
    for length in seq_lengths:
        seq_starts.append(seq_starts[-1] + length)

    starts = set(range(0, valid_seqlen, chunk_size))
    for seq_start, seq_end in zip(seq_starts[:-1], seq_starts[1:]):
        starts.add(seq_start)
        checkpoint_pos = seq_start + checkpoint
        while checkpoint_pos < seq_end:
            starts.add(checkpoint_pos)
            checkpoint_pos += checkpoint
    starts = sorted(starts)

    seq_idx = torch.zeros((1, padded_seqlen), dtype=torch.int32)
    per_seq_counts = []
    for seq_id, (seq_start, seq_end) in enumerate(
        zip(seq_starts[:-1], seq_starts[1:])
    ):
        seq_idx[0, seq_start:seq_end] = seq_id
        per_seq_counts.append(sum(seq_start <= pos < seq_end for pos in starts))
    # Padded tokens are never consumed, but a valid id keeps metadata debuggable.
    seq_idx[0, valid_seqlen:] = len(seq_lengths) - 1

    chunk_indices = torch.tensor(
        [pos // chunk_size for pos in starts], dtype=torch.int32
    )
    chunk_offsets = torch.tensor(
        [pos % chunk_size for pos in starts], dtype=torch.int32
    )
    seq_chunk_cumsum = torch.tensor(
        [0] + list(torch.tensor(per_seq_counts).cumsum(0).tolist()),
        dtype=torch.int32,
    )
    segment_ends = starts[1:] + [valid_seqlen]
    return (
        seq_idx,
        chunk_indices,
        chunk_offsets,
        seq_chunk_cumsum,
        starts,
        segment_ends,
    )


def test_varlen_state_api_feature_sentinel_and_signature():
    assert mamba.SSD_COMBINED_VARLEN_STATE_API_VERSION == 2
    assert mamba.SSDCombined.varlen_state_api_version == 2
    assert mamba.SSDCombined.supports_strided_intermediate_states is True
    assert mamba.SSDCombined.supports_fp32_processed_dt is True
    constructor = inspect.signature(mamba.SSDCombined)
    run = inspect.signature(mamba.SSDCombined.run)
    assert "has_intermediate_states" in constructor.parameters
    assert "output_token_major" in constructor.parameters
    assert "dt_processed_dtype" in constructor.parameters
    for name in (
        "num_seqs",
        "valid_seqlen",
        "intermediate_state_indices",
        "intermediate_states",
        "final_states",
    ):
        assert name in run.parameters


def test_packed_metadata_splits_checkpoint_mid_physical_chunk():
    (
        seq_idx,
        chunk_indices,
        chunk_offsets,
        seq_chunk_cumsum,
        starts,
        ends,
    ) = _packed_metadata([73, 1400], 1536)
    checkpoint = 73 + 1152
    checkpoint_segment = ends.index(checkpoint)

    assert starts[checkpoint_segment] == 1152
    assert chunk_indices[checkpoint_segment].item() == 9
    assert chunk_offsets[checkpoint_segment].item() == 0
    assert starts[checkpoint_segment + 1] == checkpoint
    assert chunk_indices[checkpoint_segment + 1].item() == 9
    assert chunk_offsets[checkpoint_segment + 1].item() == 73
    assert seq_idx.shape == (1, 1536)
    assert seq_chunk_cumsum[-1].item() == len(starts)


requires_blackwell_datacenter = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_sm100a_supported(torch.device("cuda")),
    reason="SSDCombined requires SM100/SM103/SM110",
)


def test_cumsum_fixed_tile_api_preserves_legacy_autotuner():
    """Fixed scan layouts are opt-in and do not change legacy callers."""
    from flashinfer.triton.kernels.ssd_chunk_state import (
        _chunk_cumsum_fwd_kernel,
        chunk_cumsum_fwd,
    )

    assert [
        config.kwargs["BLOCK_SIZE_H"]
        for config in _chunk_cumsum_fwd_kernel.configs
    ] == [1, 2, 4, 8, 16, 32, 64]
    assert "cumsum_block_size_h" in inspect.signature(
        chunk_cumsum_fwd
    ).parameters
    assert "cumsum_block_size_h" in inspect.signature(
        mamba.SSDCombined
    ).parameters


@pytest.mark.parametrize("invalid_block_size", [True, 2.0])
def test_cumsum_fixed_tile_rejects_non_integer_values(invalid_block_size):
    """Bool and float equality with supported ints must not bypass validation."""
    from flashinfer.triton.kernels.ssd_chunk_state import chunk_cumsum_fwd

    with pytest.raises(ValueError, match="cumsum_block_size_h"):
        mamba.SSDCombined(
            chunk_size=128,
            nheads=8,
            headdim=64,
            dstate=128,
            ngroups=8,
            cumsum_block_size_h=invalid_block_size,
        )

    # Invalid launch metadata is rejected before allocations or CUDA dispatch,
    # so the public Triton helper can be covered with CPU tensors.
    with pytest.raises(ValueError, match="cumsum_block_size_h"):
        chunk_cumsum_fwd(
            torch.empty(1, 1, 1),
            torch.empty(1),
            chunk_size=1,
            cumsum_block_size_h=invalid_block_size,
        )


@requires_blackwell_datacenter
def test_cumsum_valid_seqlen_zero_masks_physical_tail():
    """A padded serving tile must expose the reference cumsum endpoint."""
    from flashinfer.triton.kernels.ssd_chunk_state import chunk_cumsum_fwd

    valid_seqlen = 37
    padded_seqlen = 128
    nheads = 64
    torch.manual_seed(42)
    logical_dt = torch.randn(
        valid_seqlen, nheads, dtype=torch.bfloat16, device="cuda"
    )
    physical_dt = torch.full(
        (1, padded_seqlen, nheads),
        7.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    physical_dt[:, :valid_seqlen].copy_(logical_dt)
    A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0
    dt_bias = torch.rand(nheads, dtype=torch.bfloat16, device="cuda") - 4.0

    dA, processed_dt = chunk_cumsum_fwd(
        physical_dt,
        A,
        padded_seqlen,
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_out_dtype=torch.float32,
        valid_seqlen=valid_seqlen,
        cumsum_block_size_h=2,
    )
    logical_dA, logical_processed_dt = chunk_cumsum_fwd(
        logical_dt.unsqueeze(0),
        A,
        padded_seqlen,
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_out_dtype=torch.float32,
        cumsum_block_size_h=2,
    )

    assert torch.equal(dA, logical_dA)
    assert torch.equal(processed_dt, logical_processed_dt)
    assert torch.count_nonzero(processed_dt[..., valid_seqlen:]).item() == 0
    expected_endpoint = dA[..., valid_seqlen - 1 : valid_seqlen]
    # A parallel scan can reassociate trailing additions of zero, so the
    # repeated endpoint is numerically equivalent rather than bitwise equal.
    assert torch.allclose(
        dA[..., valid_seqlen:],
        expected_endpoint.expand_as(dA[..., valid_seqlen:]),
        atol=2e-5,
        rtol=2e-6,
    )


def _make_inputs(
    valid_seqlen,
    padded_seqlen,
    nheads=8,
    headdim=64,
    dstate=128,
    ngroups=None,
):
    torch.manual_seed(42)
    if ngroups is None:
        ngroups = nheads
    x = torch.randn(
        1, padded_seqlen, nheads, headdim, dtype=torch.bfloat16, device="cuda"
    )
    dt = torch.randn(
        1, padded_seqlen, nheads, dtype=torch.float32, device="cuda"
    )
    A = -torch.rand(nheads, dtype=torch.float32, device="cuda") - 1.0
    B = torch.randn(
        1, padded_seqlen, ngroups, dstate, dtype=torch.bfloat16, device="cuda"
    )
    C = torch.randn(
        1, padded_seqlen, ngroups, dstate, dtype=torch.bfloat16, device="cuda"
    )
    D = torch.randn(nheads, dtype=torch.bfloat16, device="cuda")
    dt_bias = torch.rand(nheads, dtype=torch.float32, device="cuda") - 4.0
    # Poisoning the padded tail makes accidental participation easier to detect.
    for tensor in (x, dt, B, C):
        tensor[:, valid_seqlen:].fill_(7)
    return x, dt, A, B, C, D, dt_bias


def _make_strided_state_cache(
    num_slots,
    nheads=8,
    headdim=64,
    dstate=128,
    dtype=torch.float16,
):
    """Mirror vLLM's SSM view into padded unified cache pages."""
    row_stride = 589824
    storage_offset = 18432
    compact_row_size = nheads * headdim * dstate
    storage_size = storage_offset + (num_slots - 1) * row_stride + compact_row_size
    storage = torch.full(
        (storage_size,), float("nan"), dtype=dtype, device="cuda"
    )
    cache = torch.as_strided(
        storage,
        (num_slots, nheads, headdim, dstate),
        (row_stride, headdim * dstate, dstate, 1),
        storage_offset=storage_offset,
    )
    return cache, storage


def _reference_with_initial_state(
    x,
    dt,
    A,
    B,
    C,
    D,
    dt_bias,
    initial_state,
    state_dtype,
):
    """Run the independent Triton decomposition for one logical sequence."""
    dA_cumsum, dt_processed = _chunk_cumsum_fwd(
        dt, A, 128, dt_bias=dt_bias, dt_softplus=True
    )
    states = _chunk_state_fwd(
        B, x, dt_processed, dA_cumsum, seq_idx=None, states_in_fp32=True
    )
    states, final_state = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum,
        initial_states=(
            rearrange(initial_state, "... p n -> ... (p n)")
            if initial_state is not None
            else None
        ),
        seq_idx=None,
        chunk_size=128,
        out_dtype=state_dtype,
    )
    states, final_state = (
        rearrange(tensor, "... (p n) -> ... p n", n=128)
        for tensor in (states, final_state)
    )
    cb = _bmm_chunk_fwd(C, B, 128, seq_idx=None, output_dtype=torch.float32)
    out = _chunk_scan_fwd(
        cb,
        x,
        dt_processed,
        dA_cumsum,
        C,
        states,
        D=D,
        z=None,
        seq_idx=None,
        initial_states=None,
    )
    return out, final_state


@requires_blackwell_datacenter
@pytest.mark.parametrize("has_initial_states", [False, True])
def test_exact_target_shape_output_and_direct_cache_rows(has_initial_states):
    """Exercise target H64/D64/N128/G8 with and without initial states."""
    seq_lengths = [73, 180]
    valid_seqlen = sum(seq_lengths)
    padded_seqlen = math.ceil(valid_seqlen / 128) * 128
    x, dt, A, B, C, D, dt_bias = _make_inputs(
        valid_seqlen,
        padded_seqlen,
        nheads=64,
        headdim=64,
        dstate=128,
        ngroups=8,
    )
    dt = dt.to(torch.bfloat16)
    (
        seq_idx,
        chunk_indices,
        chunk_offsets,
        seq_chunk_cumsum,
        _,
        segment_ends,
    ) = _packed_metadata(seq_lengths, padded_seqlen)
    seq_idx, chunk_indices, chunk_offsets, seq_chunk_cumsum = [
        tensor.cuda()
        for tensor in (seq_idx, chunk_indices, chunk_offsets, seq_chunk_cumsum)
    ]
    initial_states = (
        torch.randn(2, 64, 64, 128, dtype=torch.float16, device="cuda")
        if has_initial_states
        else None
    )
    state_indices = torch.full(
        (len(chunk_indices),), -1, dtype=torch.int32, device="cuda"
    )
    state_indices[segment_ends.index(seq_lengths[0])] = 1
    state_indices[segment_ends.index(valid_seqlen)] = 3
    state_cache, state_storage = _make_strided_state_cache(
        5, nheads=64, headdim=64, dstate=128, dtype=torch.float16
    )
    explicit_final_states = torch.empty(
        2, 64, 64, 128, dtype=torch.float16, device="cuda"
    )
    assert state_cache.stride() == (589824, 8192, 128, 1)

    ssd = mamba.SSDCombined(
        chunk_size=128,
        nheads=64,
        headdim=64,
        dstate=128,
        ngroups=8,
        state_dtype=torch.float16,
        has_d=True,
        has_initial_states=has_initial_states,
        has_varlen=True,
        seq_idx_dtype=torch.int32,
        has_intermediate_states=True,
        output_token_major=True,
        dt_processed_dtype=torch.float32,
        cumsum_block_size_h=2,
    )
    out_test, final_test = ssd.run(
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        initial_states=initial_states,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        seq_chunk_cumsum=seq_chunk_cumsum,
        num_seqs=2,
        valid_seqlen=valid_seqlen,
        intermediate_state_indices=state_indices,
        intermediate_states=state_cache,
        final_states=explicit_final_states,
        return_final_states=False,
    )

    reference_outputs = []
    reference_states = []
    start = 0
    for seq_id, seq_length in enumerate(seq_lengths):
        end = start + seq_length
        initial_state = (
            initial_states[seq_id : seq_id + 1]
            if initial_states is not None
            else None
        )
        out_ref, state_ref = _reference_with_initial_state(
            x[:, start:end],
            dt[:, start:end],
            A,
            B[:, start:end],
            C[:, start:end],
            D,
            dt_bias,
            initial_state,
            torch.float16,
        )
        reference_outputs.append(out_ref)
        reference_states.append(state_ref[0].to(torch.float16))
        start = end

    assert final_test is None
    output_ref = torch.cat(reference_outputs, dim=1)
    assert out_test.shape == (1, valid_seqlen, 64, 64)
    assert out_test.is_contiguous()
    assert torch.allclose(
        out_test, output_ref.to(out_test.dtype), atol=8e-2, rtol=7e-2
    )
    assert torch.allclose(
        state_cache[1], reference_states[0], atol=7e-2, rtol=7e-2
    )
    assert torch.allclose(
        state_cache[3], reference_states[1], atol=7e-2, rtol=7e-2
    )
    assert torch.equal(state_cache[1], explicit_final_states[0])
    assert torch.equal(state_cache[3], explicit_final_states[1])
    assert torch.isnan(state_cache[0]).all()
    assert torch.isnan(state_cache[2]).all()
    assert torch.isnan(state_storage[:18432]).all()


@requires_blackwell_datacenter
def test_fp32_processed_dt_matches_vllm_fp16_cache_state():
    """Serving precision must match vLLM's FP32 processed-dt recurrence."""
    valid_seqlen = 1024
    x, dt, A, B, C, D, dt_bias = _make_inputs(
        valid_seqlen,
        valid_seqlen,
        nheads=8,
        headdim=64,
        dstate=128,
        ngroups=8,
    )
    # Projected dt is BF16 in the target model, but vLLM's cumsum kernel keeps
    # its processed output in FP32.  Quantizing that output to BF16 changes the
    # recurrent cache state enough to break end-to-end model outputs.
    dt = dt.to(torch.bfloat16)
    metadata = _packed_metadata([valid_seqlen], valid_seqlen)
    seq_idx, chunk_indices, chunk_offsets, seq_chunk_cumsum = [
        tensor.cuda() for tensor in metadata[:4]
    ]
    state_indices = torch.full(
        (len(chunk_indices),), -1, dtype=torch.int32, device="cuda"
    )
    state_indices[-1] = 1
    state_cache, _ = _make_strided_state_cache(
        3, nheads=8, headdim=64, dstate=128, dtype=torch.float16
    )

    ssd = mamba.SSDCombined(
        chunk_size=128,
        nheads=8,
        headdim=64,
        dstate=128,
        ngroups=8,
        state_dtype=torch.float16,
        has_d=True,
        has_initial_states=False,
        has_varlen=True,
        seq_idx_dtype=torch.int32,
        has_intermediate_states=True,
        output_token_major=True,
        dt_processed_dtype=torch.float32,
        cumsum_block_size_h=2,
    )
    out_test, _ = ssd.run(
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        seq_chunk_cumsum=seq_chunk_cumsum,
        num_seqs=1,
        valid_seqlen=valid_seqlen,
        intermediate_state_indices=state_indices,
        intermediate_states=state_cache,
        return_final_states=False,
    )
    out_ref, state_ref = _reference_with_initial_state(
        x, dt, A, B, C, D, dt_bias, None, torch.float16
    )

    # FP32 processed dt reduces cache-state error by more than an order of
    # magnitude relative to the legacy BF16 path.  The remaining envelope is
    # from the fused kernel's recurrence/exp implementation.  Output retains
    # the existing BF16 tensor-core state operand and is checked separately.
    assert torch.allclose(
        state_cache[1], state_ref[0].to(torch.float16), atol=7e-4, rtol=2e-3
    )
    output_reference = out_ref.to(out_test.dtype)
    output_mean_abs_error = (
        out_test.to(torch.float32) - output_reference.to(torch.float32)
    ).abs().mean()
    # vLLM materializes every per-chunk state as FP16 before chunk_scan casts
    # it to BF16.  This bound distinguishes that contract from a direct
    # FP32-to-BF16 inter2 operand conversion, whose measured mean error at this
    # shape is about 2.5e-4.
    assert output_mean_abs_error.item() < 1e-5
    assert torch.allclose(
        out_test, output_reference, atol=8e-2, rtol=7e-2
    )


@requires_blackwell_datacenter
def test_explicit_num_seqs_without_initial_states_writes_distinct_final_rows():
    """The no-initial-state varlen variant must not overwrite only row zero."""
    seq_lengths = [128, 256]
    valid_seqlen = sum(seq_lengths)
    x, dt, A, B, C, D, dt_bias = _make_inputs(valid_seqlen, valid_seqlen)
    metadata = _packed_metadata(seq_lengths, valid_seqlen)
    seq_idx, chunk_indices, chunk_offsets, seq_chunk_cumsum = [
        tensor.cuda() for tensor in metadata[:4]
    ]
    ssd = mamba.SSDCombined(
        chunk_size=128,
        nheads=8,
        headdim=64,
        dstate=128,
        ngroups=8,
        state_dtype=torch.float16,
        has_d=True,
        has_initial_states=False,
        has_varlen=True,
        seq_idx_dtype=torch.int32,
        output_token_major=True,
    )
    _, final_test = ssd.run(
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        seq_chunk_cumsum=seq_chunk_cumsum,
        num_seqs=2,
    )

    references = []
    start = 0
    for length in seq_lengths:
        end = start + length
        _, _, _, _, final_ref = _mamba_chunk_scan_combined_fwd(
            x[:, start:end],
            dt[:, start:end],
            A,
            B[:, start:end],
            C[:, start:end],
            128,
            D=D,
            dt_bias=dt_bias,
            initial_states=None,
            seq_idx=None,
            dt_softplus=True,
        )
        references.append(final_ref[0])
        start = end
    final_ref = torch.stack(references).to(torch.float16)
    assert final_test is not None
    assert final_test.shape == final_ref.shape
    assert torch.allclose(final_test, final_ref, atol=7e-2, rtol=7e-2)


@requires_blackwell_datacenter
@pytest.mark.parametrize("valid_seqlen", [32343, 32344, 32346, 32768])
def test_padded_tail_explicit_num_seqs_and_direct_final_cache_store(valid_seqlen):
    """BF16 IO/FP16 state: padded tokens cannot affect output or final state."""
    chunk_size = 128
    padded_seqlen = math.ceil(valid_seqlen / chunk_size) * chunk_size
    x, dt, A, B, C, D, dt_bias = _make_inputs(valid_seqlen, padded_seqlen)
    (
        seq_idx,
        chunk_indices,
        chunk_offsets,
        seq_chunk_cumsum,
        _,
        segment_ends,
    ) = _packed_metadata([valid_seqlen], padded_seqlen)
    seq_idx = seq_idx.cuda()
    chunk_indices = chunk_indices.cuda()
    chunk_offsets = chunk_offsets.cuda()
    seq_chunk_cumsum = seq_chunk_cumsum.cuda()

    # Arbitrary row 5 proves that indices address the caller's cache directly.
    state_indices = torch.full(
        (len(chunk_indices),), -1, dtype=torch.int32, device="cuda"
    )
    state_indices[segment_ends.index(valid_seqlen)] = 5
    state_cache, state_storage = _make_strided_state_cache(8)
    assert not state_cache.is_contiguous()
    assert state_cache.stride() == (589824, 8192, 128, 1)
    assert state_cache.storage_offset() == 18432
    out_buffer = torch.empty(
        1, padded_seqlen, 8, 64, dtype=torch.bfloat16, device="cuda"
    )

    ssd = mamba.SSDCombined(
        chunk_size=chunk_size,
        nheads=8,
        headdim=64,
        dstate=128,
        ngroups=8,
        state_dtype=torch.float16,
        has_d=True,
        has_initial_states=False,
        has_varlen=True,
        seq_idx_dtype=torch.int32,
        has_intermediate_states=True,
        output_token_major=True,
    )
    out_test, final_test = ssd.run(
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        seq_chunk_cumsum=seq_chunk_cumsum,
        num_seqs=1,
        valid_seqlen=valid_seqlen,
        intermediate_state_indices=state_indices,
        intermediate_states=state_cache,
        out=out_buffer,
        return_final_states=False,
    )

    out_ref, _, _, _, final_ref = _mamba_chunk_scan_combined_fwd(
        x[:, :valid_seqlen],
        dt[:, :valid_seqlen],
        A,
        B[:, :valid_seqlen],
        C[:, :valid_seqlen],
        chunk_size,
        D=D,
        dt_bias=dt_bias,
        initial_states=None,
        seq_idx=None,
        dt_softplus=True,
    )
    assert final_test is None
    assert out_test.shape == out_ref.shape
    assert out_test.is_contiguous()
    # Long recurrent scans accumulate slightly more BF16 rounding error than
    # the short legacy cases (three of 16.6M values need 0.08 vs 0.07 atol).
    assert torch.allclose(out_test, out_ref.to(out_test.dtype), atol=8e-2, rtol=7e-2)
    assert torch.allclose(
        state_cache[5], final_ref[0].to(torch.float16), atol=7e-2, rtol=7e-2
    )
    assert torch.isnan(state_cache[4]).all()
    assert torch.isnan(state_storage[:18432]).all()


@requires_blackwell_datacenter
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16])
def test_mid_chunk_checkpoint_store_with_initial_state(state_dtype):
    """Nonaligned packed output and three direct-cache rows match Triton."""
    seq_lengths = [73, 1400]
    valid_seqlen = sum(seq_lengths)
    padded_seqlen = math.ceil(valid_seqlen / 128) * 128
    x, dt, A, B, C, D, dt_bias = _make_inputs(valid_seqlen, padded_seqlen)
    (
        seq_idx,
        chunk_indices,
        chunk_offsets,
        seq_chunk_cumsum,
        _,
        segment_ends,
    ) = _packed_metadata(seq_lengths, padded_seqlen)
    tensors = [seq_idx, chunk_indices, chunk_offsets, seq_chunk_cumsum]
    seq_idx, chunk_indices, chunk_offsets, seq_chunk_cumsum = [
        tensor.cuda() for tensor in tensors
    ]
    initial_states = torch.randn(
        2, 8, 64, 128, dtype=state_dtype, device="cuda"
    )
    checkpoint_end = seq_lengths[0] + 1152
    checkpoint_segment = segment_ends.index(checkpoint_end)
    first_final_segment = segment_ends.index(seq_lengths[0])
    second_final_segment = segment_ends.index(valid_seqlen)
    state_indices = torch.full(
        (len(chunk_indices),), -1, dtype=torch.int32, device="cuda"
    )
    state_indices[first_final_segment] = 2
    state_indices[checkpoint_segment] = 6
    state_indices[second_final_segment] = 8
    state_cache, state_storage = _make_strided_state_cache(
        10, dtype=state_dtype
    )
    out_buffer = torch.empty(
        1, padded_seqlen, 8, 64, dtype=torch.bfloat16, device="cuda"
    )

    ssd = mamba.SSDCombined(
        chunk_size=128,
        nheads=8,
        headdim=64,
        dstate=128,
        ngroups=8,
        state_dtype=state_dtype,
        has_d=True,
        has_initial_states=True,
        has_varlen=True,
        seq_idx_dtype=torch.int32,
        has_intermediate_states=True,
        output_token_major=True,
    )
    out_test, final_test = ssd.run(
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        initial_states=initial_states,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        seq_chunk_cumsum=seq_chunk_cumsum,
        num_seqs=2,
        valid_seqlen=valid_seqlen,
        intermediate_state_indices=state_indices,
        intermediate_states=state_cache,
        out=out_buffer,
        return_final_states=False,
    )

    reference_outputs = []
    reference_final_states = []
    start = 0
    for seq_id, seq_length in enumerate(seq_lengths):
        end = start + seq_length
        out_ref, final_ref = _reference_with_initial_state(
            x[:, start:end],
            dt[:, start:end],
            A,
            B[:, start:end],
            C[:, start:end],
            D,
            dt_bias,
            initial_states[seq_id : seq_id + 1],
            state_dtype,
        )
        reference_outputs.append(out_ref)
        reference_final_states.append(final_ref[0])
        start = end

    assert final_test is None
    packed_output_ref = torch.cat(reference_outputs, dim=1)
    assert out_test.shape == packed_output_ref.shape
    assert out_test.is_contiguous()
    assert torch.allclose(
        out_test, packed_output_ref.to(out_test.dtype), atol=8e-2, rtol=7e-2
    )
    assert torch.allclose(
        state_cache[2],
        reference_final_states[0].to(state_dtype),
        atol=7e-2,
        rtol=7e-2,
    )
    assert torch.allclose(
        state_cache[8],
        reference_final_states[1].to(state_dtype),
        atol=7e-2,
        rtol=7e-2,
    )

    start = seq_lengths[0]
    end = checkpoint_end
    _, checkpoint_ref = _reference_with_initial_state(
        x[:, start:end],
        dt[:, start:end],
        A,
        B[:, start:end],
        C[:, start:end],
        D,
        dt_bias,
        initial_states[1:2],
        state_dtype,
    )
    assert torch.allclose(
        state_cache[6], checkpoint_ref[0].to(state_dtype), atol=7e-2, rtol=7e-2
    )
    assert torch.isnan(state_cache[5]).all()
    assert torch.isnan(state_cache[7]).all()
    assert torch.isnan(state_storage[:18432]).all()
