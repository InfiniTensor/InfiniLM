import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_top_k_top_p_sample_infinilm_reproducible(
    dtype,
    device,
    implementation_index,
):
    logits = torch.zeros((64, 16), dtype=dtype, device=device)
    first = torch.empty((64,), dtype=torch.int32, device=device)
    second = torch.empty_like(first)
    different_seed = torch.empty_like(first)

    _top_k_top_p_sample_infinilm(
        logits, None, None, 1234, 9, first, implementation_index
    )
    _top_k_top_p_sample_infinilm(
        logits, None, None, 1234, 9, second, implementation_index
    )
    _top_k_top_p_sample_infinilm(
        logits, None, None, 5678, 9, different_seed, implementation_index
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, different_seed)
    assert first.dtype == torch.int32
    assert torch.all((first >= 0) & (first < logits.shape[1]))


@pytest.mark.parametrize(
    "k_value, p_value, allowed",
    (
        (3, None, (0, 1, 2)),
        (None, 0.6, (0,)),
        (3, 0.8, (0, 1)),
    ),
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_top_k_top_p_sample_infinilm_filters(
    k_value,
    p_value,
    allowed,
    dtype,
    device,
    implementation_index,
):
    logits = torch.full((32, 16), -10.0, dtype=dtype, device=device)
    logits[:, 0] = 5.0
    logits[:, 1] = 4.0
    logits[:, 2] = 3.0
    k = (
        torch.tensor((k_value,), dtype=torch.int64, device="cpu")
        if k_value is not None
        else None
    )
    p = (
        torch.tensor((p_value,), dtype=torch.float32, device="cpu")
        if p_value is not None
        else None
    )
    out = torch.empty((32,), dtype=torch.int32, device=device)

    _top_k_top_p_sample_infinilm(logits, k, p, 1234, 0, out, implementation_index)

    allowed_tensor = torch.tensor(allowed, dtype=torch.int32, device=device)
    assert torch.all(torch.isin(out, allowed_tensor))


def _top_k_top_p_sample_infinilm(
    logits,
    k,
    p,
    seed,
    offset,
    out,
    implementation_index,
):
    infini.ops.top_k_top_p_sample_infinilm(
        logits,
        k,
        p,
        seed,
        offset,
        out,
        stream=get_stream(logits.device),
        implementation_index=implementation_index,
    )
