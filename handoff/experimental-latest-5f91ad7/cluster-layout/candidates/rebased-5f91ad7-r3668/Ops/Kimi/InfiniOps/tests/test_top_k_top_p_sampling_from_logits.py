import infini.ops
import pytest
import torch

from tests.utils import get_stream


def _require_extended_sampling_provider(device, implementation_index):
    if implementation_index == 16:
        return
    if device == "mlu" and implementation_index == 0:
        return
    pytest.skip("extended sampling coverage requires FlashInfer or Cambricon native")


@pytest.mark.parametrize(
    "top_k_value, top_p_value, allowed",
    (
        (3, 1.0, (0, 1, 2)),
        (16, 0.6, (0,)),
        (3, 0.8, (0, 1)),
    ),
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_top_k_top_p_sampling_from_logits(
    top_k_value,
    top_p_value,
    allowed,
    dtype,
    device,
    implementation_index,
):
    batch_size = 64
    logits = torch.full((batch_size, 16), -10.0, dtype=dtype, device=device)
    logits[:, 0] = 5.0
    logits[:, 1] = 4.0
    logits[:, 2] = 3.0
    top_k = torch.full((batch_size,), top_k_value, dtype=torch.int64)
    top_p = torch.full((batch_size,), top_p_value, dtype=torch.float32)
    first = torch.empty((batch_size,), dtype=torch.int32, device=device)
    second = torch.empty_like(first)
    different_seed = torch.empty_like(first)

    _top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, 1234, 9, first, implementation_index
    )
    _top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, 1234, 9, second, implementation_index
    )
    _top_k_top_p_sampling_from_logits(
        logits, top_k, top_p, 5678, 9, different_seed, implementation_index
    )

    assert torch.equal(first, second)
    if len(allowed) > 1:
        assert not torch.equal(first, different_seed)
    allowed_tensor = torch.tensor(allowed, dtype=torch.int32, device=device)
    assert torch.all(torch.isin(first, allowed_tensor))


def test_sampling_joint_host_indices(device, implementation_index):
    _require_extended_sampling_provider(device, implementation_index)

    logits = torch.tensor(
        (
            (9.0, 1.0, 0.0, -1.0),
            (0.0, 8.0, 1.0, -1.0),
            (-1.0, 0.0, 1.0, 7.0),
        ),
        dtype=torch.float32,
        device=device,
    )
    indices = torch.tensor((2, 0, 2, 1, 0, 1, 2), dtype=torch.int64)
    batch_size = indices.numel()
    top_k = torch.ones(batch_size, dtype=torch.int64)
    top_p = torch.ones(batch_size, dtype=torch.float32)
    out = torch.empty(batch_size, dtype=torch.int64, device=device)

    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        1234,
        9,
        out,
        implementation_index,
        indices=indices,
        filter_apply_order="joint",
    )

    expected = torch.tensor((3, 0, 3, 1, 0, 1, 3), dtype=torch.int64, device=device)
    assert torch.equal(out, expected)


def test_sampling_top_k_first_device_indices(device, implementation_index):
    _require_extended_sampling_provider(device, implementation_index)

    logits = torch.tensor(
        (
            (9.0, 1.0, 0.0, -1.0),
            (0.0, 8.0, 1.0, -1.0),
            (-1.0, 0.0, 1.0, 7.0),
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    indices = torch.tensor((2, 0, 2, 1, 0), dtype=torch.int32, device=device)
    batch_size = indices.numel()
    top_k = torch.ones(batch_size, dtype=torch.int32)
    top_p = torch.ones(batch_size, dtype=torch.float64)
    out = torch.empty(batch_size, dtype=torch.int32, device=device)

    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        1234,
        9,
        out,
        implementation_index,
        indices=indices,
        filter_apply_order="top_k_first",
    )

    expected = torch.tensor((3, 0, 3, 1, 0), dtype=torch.int32, device=device)
    assert torch.equal(out, expected)


def test_sampling_offset(device, implementation_index):
    _require_extended_sampling_provider(device, implementation_index)

    batch_size = 256
    logits = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
    top_k = torch.full((batch_size,), 4, dtype=torch.int32)
    top_p = torch.ones(batch_size, dtype=torch.float32)
    first = torch.empty(batch_size, dtype=torch.int32, device=device)
    repeated = torch.empty_like(first)
    different_offset = torch.empty_like(first)

    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        1234,
        9,
        first,
        implementation_index,
        filter_apply_order="joint",
    )
    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        1234,
        9,
        repeated,
        implementation_index,
        filter_apply_order="joint",
    )
    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        1234,
        10,
        different_offset,
        implementation_index,
        filter_apply_order="joint",
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different_offset)


def test_flashinfer_sampling_uses_handle_stream(device, implementation_index):
    if device != "cuda" or implementation_index != 16:
        pytest.skip("FlashInfer linked-provider stream coverage")

    batch_size = 64
    logits = torch.zeros((batch_size, 4), dtype=torch.float32, device=device)
    logits[:, 0] = 1.0
    top_k = torch.ones(batch_size, dtype=torch.int32)
    top_p = torch.ones(batch_size, dtype=torch.float32)
    out = torch.full((batch_size,), -1, dtype=torch.int32, device=device)
    stream = torch.cuda.Stream()

    def call_sampling():
        infini.ops.top_k_top_p_sampling_from_logits(
            logits,
            top_k,
            top_p,
            None,
            "joint",
            True,
            False,
            1234,
            9,
            out,
            stream=stream.cuda_stream,
            implementation_index=implementation_index,
        )

    try:
        call_sampling()
        stream.synchronize()
        out.fill_(-1)
        torch.cuda.synchronize()

        with torch.cuda.stream(stream):
            torch.cuda._sleep(50_000_000)
        call_sampling()

        default_stream = torch.cuda.default_stream()
        with torch.cuda.stream(default_stream):
            snapshot = out.clone()
        default_stream.synchronize()
        assert torch.all(snapshot == -1)

        stream.synchronize()
        assert torch.all(out == 0)
    finally:
        torch.cuda.synchronize()


def test_sampling_preserves_float64_top_p_underflow(device, implementation_index):
    _require_extended_sampling_provider(device, implementation_index)

    batch_size = 4096
    logits = torch.tensor((0.0, -1.0), dtype=torch.float32, device=device).repeat(
        batch_size, 1
    )
    top_k = torch.full((batch_size,), 2, dtype=torch.int32)
    top_p = torch.full((batch_size,), 1e-300, dtype=torch.float64)
    out = torch.empty(batch_size, dtype=torch.int32, device=device)

    _top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        1234,
        9,
        out,
        implementation_index,
    )

    assert torch.all(out == 0)


def test_flashinfer_sampling_rejects_float64_logits(device, implementation_index):
    if implementation_index != 16:
        pytest.skip("FlashInfer linked-provider coverage")

    logits = torch.tensor(
        ((1.0, 1.0000000001),),
        dtype=torch.float64,
        device=device,
    )
    top_k = torch.ones(1, dtype=torch.int32)
    top_p = torch.ones(1, dtype=torch.float32)
    out = torch.empty(1, dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="float16, bfloat16, or float32 logits"):
        _top_k_top_p_sampling_from_logits(
            logits,
            top_k,
            top_p,
            1234,
            9,
            out,
            implementation_index,
        )


def test_flashinfer_sampling_rejects_noncontiguous_logits(device, implementation_index):
    if implementation_index != 16:
        pytest.skip("FlashInfer linked-provider coverage")

    logits = torch.tensor(
        ((2.0, 0.0), (0.0, 2.0)),
        dtype=torch.float32,
        device=device,
    ).T
    assert not logits.is_contiguous()
    top_k = torch.ones(2, dtype=torch.int32)
    top_p = torch.ones(2, dtype=torch.float32)
    out = torch.empty(2, dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="contiguous NVIDIA logits"):
        _top_k_top_p_sampling_from_logits(
            logits,
            top_k,
            top_p,
            1234,
            9,
            out,
            implementation_index,
        )


def test_flashinfer_sampling_rejects_check_nan(device, implementation_index):
    if implementation_index != 16:
        pytest.skip("FlashInfer linked-provider coverage")

    logits = torch.tensor(
        ((float("nan"), 0.0),),
        dtype=torch.float32,
        device=device,
    )
    top_k = torch.ones(1, dtype=torch.int32)
    top_p = torch.ones(1, dtype=torch.float32)
    out = torch.empty(1, dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="does not support check_nan"):
        _top_k_top_p_sampling_from_logits(
            logits,
            top_k,
            top_p,
            1234,
            9,
            out,
            implementation_index,
            check_nan=True,
        )


def test_flashinfer_sampling_rejects_out_of_range_host_indices(
    device, implementation_index
):
    if implementation_index != 16:
        pytest.skip("FlashInfer linked-provider coverage")

    logits = torch.tensor(
        ((2.0, 0.0), (0.0, 2.0)),
        dtype=torch.float32,
        device=device,
    )
    indices = torch.tensor((0, 2), dtype=torch.int32)
    top_k = torch.ones(2, dtype=torch.int32)
    top_p = torch.ones(2, dtype=torch.float32)
    out = torch.empty(2, dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="out-of-range host index"):
        _top_k_top_p_sampling_from_logits(
            logits,
            top_k,
            top_p,
            1234,
            9,
            out,
            implementation_index,
            indices=indices,
        )


@pytest.mark.parametrize(
    "logits_batch,out_dtype,error",
    (
        (1, torch.int32, "output batch size to match logits"),
        (2, torch.int64, "int32 output when indices are absent"),
    ),
)
def test_flashinfer_sampling_rejects_invalid_output_without_indices(
    logits_batch, out_dtype, error, device, implementation_index
):
    if implementation_index != 16:
        pytest.skip("FlashInfer linked-provider coverage")

    batch_size = 2
    logits = torch.zeros((logits_batch, 4), dtype=torch.float32, device=device)
    top_k = torch.ones(batch_size, dtype=torch.int32)
    top_p = torch.ones(batch_size, dtype=torch.float32)
    out = torch.empty(batch_size, dtype=out_dtype, device=device)

    with pytest.raises(ValueError, match=error):
        _top_k_top_p_sampling_from_logits(
            logits,
            top_k,
            top_p,
            1234,
            9,
            out,
            implementation_index,
        )


def _top_k_top_p_sampling_from_logits(
    logits,
    top_k,
    top_p,
    seed,
    offset,
    out,
    implementation_index,
    *,
    indices=None,
    filter_apply_order="top_k_first",
    check_nan=False,
):
    infini.ops.top_k_top_p_sampling_from_logits(
        logits,
        top_k,
        top_p,
        indices,
        filter_apply_order,
        True,
        check_nan,
        seed,
        offset,
        out,
        stream=get_stream(logits.device),
        implementation_index=implementation_index,
    )
