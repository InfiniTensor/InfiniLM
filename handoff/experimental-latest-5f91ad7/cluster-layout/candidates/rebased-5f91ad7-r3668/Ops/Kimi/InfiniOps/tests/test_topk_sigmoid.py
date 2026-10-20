import infini.ops
import pytest
import torch

from tests.utils import get_stream


if not hasattr(infini.ops, "TopkSigmoid"):
    pytest.skip(
        "`TopkSigmoid` is not available on this platform",
        allow_module_level=True,
    )


_VLLM_IMPLEMENTATION_INDEX = 16
_INDEX_DTYPES = (torch.int32, torch.int64)
if hasattr(torch, "uint32"):
    _INDEX_DTYPES += (torch.uint32,)


@pytest.mark.parametrize("renormalize", (False, True))
@pytest.mark.parametrize("has_bias", (False, True))
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        (torch.float32, 1e-6, 1e-6),
        (torch.float16, 1e-6, 1e-6),
        (torch.bfloat16, 1e-6, 1e-6),
    ),
)
def test_topk_sigmoid(dtype, index_dtype, has_bias, renormalize, rtol, atol):
    if not torch.cuda.is_available():
        pytest.skip("`topk_sigmoid` requires the NVIDIA backend")

    gating_output = torch.tensor(
        (
            (1.25, -0.5, 0.75, 2.0, -1.0),
            (-0.25, 1.5, 0.5, -1.25, 2.25),
            (0.125, 0.75, 2.5, 1.0, -0.75),
        ),
        dtype=dtype,
        device="cuda",
    )
    e_score_correction_bias = None
    if has_bias:
        e_score_correction_bias = torch.tensor(
            (0.0, 0.75, -0.5, -1.0, 1.25),
            dtype=torch.float32,
            device=gating_output.device,
        )
    outputs = _make_outputs(gating_output, topk=2, index_dtype=index_dtype)

    result = infini.ops.topk_sigmoid(
        gating_output,
        e_score_correction_bias,
        None,
        renormalize,
        0.75,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    assert result is None
    expected = _reference(
        gating_output, e_score_correction_bias, None, 2, renormalize, 0.75
    )
    torch.testing.assert_close(outputs[0], expected[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(outputs[1], expected[1].to(index_dtype), rtol=0, atol=0)
    torch.testing.assert_close(outputs[2], expected[2], rtol=0, atol=0)


def test_topk_sigmoid_bias_only_changes_selection():
    if not torch.cuda.is_available():
        pytest.skip("`topk_sigmoid` requires the NVIDIA backend")

    gating_output = torch.tensor(((3.0, 2.0, 1.0),), dtype=torch.float32, device="cuda")
    e_score_correction_bias = torch.tensor(
        (-4.0, 0.0, 3.0), dtype=torch.float32, device="cuda"
    )
    outputs = _make_outputs(gating_output, topk=1, index_dtype=torch.int32)

    infini.ops.topk_sigmoid(
        gating_output,
        e_score_correction_bias,
        None,
        False,
        1.0,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    unbiased_scores = torch.sigmoid(gating_output.to(torch.float32))
    assert outputs[1].item() == 2
    torch.testing.assert_close(outputs[0], unbiased_scores[:, 2:3])


def test_topk_sigmoid_nan_tie_uses_lower_expert_ids():
    if not torch.cuda.is_available():
        pytest.skip("`topk_sigmoid` requires the NVIDIA backend")

    gating_output = torch.full((1, 4), torch.nan, dtype=torch.float32, device="cuda")
    outputs = _make_outputs(gating_output, topk=2, index_dtype=torch.int32)

    infini.ops.topk_sigmoid(
        gating_output,
        None,
        None,
        False,
        1.0,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    torch.testing.assert_close(outputs[0], torch.zeros_like(outputs[0]))
    torch.testing.assert_close(
        outputs[1],
        torch.tensor(((0, 1),), dtype=torch.int32, device="cuda"),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
def test_topk_sigmoid_padding_and_token_expert_indices(index_dtype):
    if not torch.cuda.is_available():
        pytest.skip("`topk_sigmoid` requires the NVIDIA backend")

    gating_output = torch.tensor(
        (
            (0.25, 2.0, -0.5, 1.0),
            (1.5, -0.25, 0.75, 0.0),
            (-1.0, 0.5, 2.25, 1.25),
        ),
        dtype=torch.float16,
        device="cuda",
    )
    is_padding = torch.tensor((False, True, False), dtype=torch.bool, device="cuda")
    outputs = _make_outputs(gating_output, topk=3, index_dtype=index_dtype)

    infini.ops.topk_sigmoid(
        gating_output,
        None,
        is_padding,
        True,
        1.5,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    expected = _reference(gating_output, None, is_padding, 3, True, 1.5)
    torch.testing.assert_close(outputs[0], expected[0])
    torch.testing.assert_close(outputs[1], expected[1].to(index_dtype), rtol=0, atol=0)
    torch.testing.assert_close(outputs[2], expected[2], rtol=0, atol=0)


def test_topk_sigmoid_non_default_stream():
    if not torch.cuda.is_available():
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    gating_output = torch.randn((7, 13), dtype=torch.bfloat16, device="cuda")
    outputs = _make_outputs(gating_output, topk=4, index_dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    infini.ops.topk_sigmoid(
        gating_output,
        None,
        None,
        False,
        0.5,
        *outputs,
        stream=stream.cuda_stream,
    )

    stream.synchronize()
    expected = _reference(gating_output, None, None, 4, False, 0.5)
    torch.testing.assert_close(outputs[0], expected[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(outputs[1], expected[1].to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(outputs[2], expected[2], rtol=0, atol=0)


def test_topk_sigmoid_empty_tokens():
    if not torch.cuda.is_available():
        pytest.skip("`topk_sigmoid` requires the NVIDIA backend")

    gating_output = torch.empty((0, 4), dtype=torch.float16, device="cuda")
    outputs = _make_outputs(gating_output, topk=2, index_dtype=torch.int64)

    result = infini.ops.topk_sigmoid(
        gating_output,
        None,
        None,
        False,
        1.0,
        *outputs,
        stream=get_stream(gating_output.device),
    )

    assert result is None
    assert all(output.shape == (0, 2) for output in outputs)


@pytest.mark.parametrize("renormalize", (False, True))
@pytest.mark.parametrize("has_bias", (False, True))
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_topk_sigmoid_vllm_provider(dtype, index_dtype, has_bias, renormalize):
    _require_vllm_implementation()

    gating_output = torch.tensor(
        (
            (1.25, -0.5, 0.75, 2.0, -1.0),
            (-0.25, 1.5, 0.5, -1.25, 2.25),
            (0.125, 0.75, 2.5, 1.0, -0.75),
        ),
        dtype=dtype,
        device="cuda",
    )
    bias = None
    if has_bias:
        bias = torch.tensor(
            (0.0, 0.75, -0.5, -1.0, 1.25),
            dtype=torch.float32,
            device="cuda",
        )
    outputs = _make_outputs(gating_output, topk=2, index_dtype=index_dtype)

    result = infini.ops.topk_sigmoid(
        gating_output,
        bias,
        None,
        renormalize,
        1.0,
        *outputs,
        stream=get_stream(gating_output.device),
        implementation_index=_VLLM_IMPLEMENTATION_INDEX,
    )

    assert result is None
    expected = _reference(gating_output, bias, None, 2, renormalize, 1.0)
    torch.testing.assert_close(outputs[0], expected[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(outputs[1], expected[1].to(index_dtype), rtol=0, atol=0)
    torch.testing.assert_close(outputs[2], expected[2], rtol=0, atol=0)


@pytest.mark.parametrize("unsupported", ("is_padding", "scaling"))
def test_topk_sigmoid_vllm_provider_rejects_unsupported_arguments(unsupported):
    _require_vllm_implementation()

    gating_output = torch.randn((2, 4), dtype=torch.float32, device="cuda")
    outputs = _make_outputs(gating_output, topk=2, index_dtype=torch.int32)
    is_padding = (
        torch.zeros(2, dtype=torch.bool, device="cuda")
        if unsupported == "is_padding"
        else None
    )
    routed_scaling_factor = 0.5 if unsupported == "scaling" else 1.0
    message = "does not support `is_padding`" if is_padding is not None else "requires"

    with pytest.raises(RuntimeError, match=message):
        infini.ops.topk_sigmoid(
            gating_output,
            None,
            is_padding,
            False,
            routed_scaling_factor,
            *outputs,
            stream=get_stream(gating_output.device),
            implementation_index=_VLLM_IMPLEMENTATION_INDEX,
        )


def test_topk_sigmoid_vllm_provider_non_default_stream():
    _require_vllm_implementation()

    gating_output = torch.tensor(
        ((1.0, -0.5, 2.0, 0.25), (-1.0, 1.5, 0.5, 2.25)),
        dtype=torch.bfloat16,
        device="cuda",
    )
    outputs = _make_outputs(gating_output, topk=2, index_dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    infini.ops.topk_sigmoid(
        gating_output,
        None,
        None,
        False,
        1.0,
        *outputs,
        stream=stream.cuda_stream,
        implementation_index=_VLLM_IMPLEMENTATION_INDEX,
    )

    stream.synchronize()
    expected = _reference(gating_output, None, None, 2, False, 1.0)
    torch.testing.assert_close(outputs[0], expected[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(outputs[1], expected[1].to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(outputs[2], expected[2], rtol=0, atol=0)


def test_topk_sigmoid_vllm_provider_empty_tokens():
    _require_vllm_implementation()

    gating_output = torch.empty((0, 4), dtype=torch.float16, device="cuda")
    outputs = _make_outputs(gating_output, topk=2, index_dtype=torch.int64)

    result = infini.ops.topk_sigmoid(
        gating_output,
        None,
        None,
        False,
        1.0,
        *outputs,
        stream=get_stream(gating_output.device),
        implementation_index=_VLLM_IMPLEMENTATION_INDEX,
    )

    assert result is None
    assert all(output.shape == (0, 2) for output in outputs)
    probe = torch.ones((1,), dtype=torch.float32, device="cuda") + 1
    torch.cuda.synchronize()
    assert probe.item() == 2


def _require_vllm_implementation():
    if not torch.cuda.is_available():
        pytest.skip("`topk_sigmoid` vLLM provider requires the NVIDIA backend")
    if _VLLM_IMPLEMENTATION_INDEX not in (
        infini.ops.TopkSigmoid.active_implementation_indices("nvidia")
    ):
        pytest.skip("vLLM `topk_sigmoid` provider is not active")


def _make_outputs(gating_output, topk, index_dtype):
    shape = (gating_output.size(0), topk)
    topk_weights = torch.full(
        shape, torch.nan, dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.full(
        shape, 2**31 - 1, dtype=index_dtype, device=gating_output.device
    )
    token_expert_indices = torch.full(
        shape, -1, dtype=torch.int32, device=gating_output.device
    )

    return topk_weights, topk_ids, token_expert_indices


def _reference(
    gating_output,
    e_score_correction_bias,
    is_padding,
    topk,
    renormalize,
    routed_scaling_factor,
):
    scores = torch.sigmoid(gating_output.to(torch.float32))
    selection_scores = (
        scores if e_score_correction_bias is None else scores + e_score_correction_bias
    )
    ids = torch.topk(selection_scores, topk, dim=-1).indices
    weights = scores.gather(1, ids)
    if renormalize:
        denominator = weights.sum(dim=-1, keepdim=True)
        weights = weights / torch.where(denominator > 0, denominator, 1.0)
    weights = weights * routed_scaling_factor

    if is_padding is not None:
        ids = ids.masked_fill(is_padding.bool().unsqueeze(-1), -1)

    num_tokens = gating_output.size(0)
    token_expert_indices = torch.arange(
        topk, dtype=torch.int32, device=gating_output.device
    ).unsqueeze(0).expand(num_tokens, -1) * num_tokens + torch.arange(
        num_tokens, dtype=torch.int32, device=gating_output.device
    ).unsqueeze(-1)

    return weights, ids, token_expert_indices
