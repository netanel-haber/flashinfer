"""CUDA-stream ordering regressions for the fused Mamba2 SSD kernel."""

import pytest
import torch

from flashinfer.mamba import SSDCombined
from flashinfer.utils import is_sm100a_supported


requires_blackwell_datacenter = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_sm100a_supported(torch.device("cuda")),
    reason="SSDCombined requires SM100/SM103/SM110",
)


def _make_inputs():
    """Create finite, nontrivial inputs without changing the global RNG."""
    generator = torch.Generator(device="cuda").manual_seed(20260720)
    x = torch.randn(
        1,
        128,
        8,
        64,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    dt = torch.randn(
        1,
        128,
        8,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    A = -torch.rand(
        8, dtype=torch.float32, device="cuda", generator=generator
    ) - 1.0
    B = torch.randn(
        1,
        128,
        8,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    C = torch.randn(
        1,
        128,
        8,
        128,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    D = torch.randn(
        8, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    dt_bias = torch.rand(
        8, dtype=torch.float32, device="cuda", generator=generator
    ) - 4.0
    return x, dt, A, B, C, D, dt_bias


@requires_blackwell_datacenter
def test_ssd_combined_orders_non_default_torch_stream_producer_and_consumer():
    """The CuTe launch must stay between same-stream producers and consumers.

    The exact workload is first run on the default stream to compile and warm
    every kernel.  The measured path then runs on a non-default PyTorch stream:
    a delayed producer copies the real inputs, ``SSDCombined.run`` consumes
    them, and an immediate clone snapshots its output.

    Meanwhile, a long sleep blocks the default stream.  If the CuTe kernel is
    launched there instead of through ``tvm_ffi.use_torch_stream()``, the clone
    overtakes it and deterministically captures the sentinel output buffer.
    CUDA events provide setup, proof, and cleanup ordering; no device-wide
    synchronization is used.
    """
    reference_inputs = _make_inputs()
    staged_inputs = tuple(torch.zeros_like(tensor) for tensor in reference_inputs)
    reference_out_buffer = torch.empty_like(reference_inputs[0])
    sentinel = 73.0
    staged_out_buffer = torch.full_like(reference_out_buffer, sentinel)

    ssd = SSDCombined(
        chunk_size=128,
        nheads=8,
        headdim=64,
        dstate=128,
        ngroups=8,
        state_dtype=torch.float16,
        has_d=True,
        output_token_major=True,
        dt_processed_dtype=torch.float32,
        cumsum_block_size_h=2,
    )

    reference, _ = ssd.run(
        *reference_inputs[:5],
        D=reference_inputs[5],
        dt_bias=reference_inputs[6],
        dt_softplus=True,
        out=reference_out_buffer,
        return_final_states=False,
    )

    default_stream = torch.cuda.default_stream()
    worker_stream = torch.cuda.Stream()
    setup_ready = torch.cuda.Event()
    default_blocker_done = torch.cuda.Event()
    producer_done = torch.cuda.Event()
    consumer_done = torch.cuda.Event()
    default_drain = torch.cuda.Event()

    # reference_inputs, staged tensors, and the warm reference are all ready
    # before either side of the deliberately crossed-stream schedule begins.
    setup_ready.record(default_stream)
    with torch.cuda.stream(default_stream):
        # Long enough that the warmed worker path must finish first.  The
        # explicit query below makes an unexpectedly short delay a test-setup
        # failure instead of allowing a false pass.
        torch.cuda._sleep(1_000_000_000)
        default_blocker_done.record()

    with torch.cuda.stream(worker_stream):
        worker_stream.wait_event(setup_ready)
        torch.cuda._sleep(20_000_000)
        for staged, source in zip(staged_inputs, reference_inputs):
            staged.copy_(source)
        producer_done.record()

        output, _ = ssd.run(
            *staged_inputs[:5],
            D=staged_inputs[5],
            dt_bias=staged_inputs[6],
            dt_softplus=True,
            out=staged_out_buffer,
            return_final_states=False,
        )
        observed = output.clone()
        consumer_done.record()

    # In the negative-control implementation, the misplaced CuTe launch is
    # queued on the default stream before this event.  Waiting for default_drain
    # after taking the snapshot therefore also prevents leaked work on failure.
    default_drain.record(default_stream)

    consumer_done.synchronize()
    assert producer_done.query()
    blocker_was_still_active = not default_blocker_done.query()
    default_drain.synchronize()

    assert blocker_was_still_active, (
        "default-stream delay ended before the worker consumer; increase the "
        "delay so the regression schedule remains deterministic"
    )
    assert torch.equal(observed, reference)
