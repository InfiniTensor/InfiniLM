import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("shape", ((1, 8), (3, 16)))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_top_k_top_p_sampler(
    shape,
    dtype,
    device,
    implementation_index,
):
    batch_size, vocab_size = shape
    logits = torch.full(shape, -10.0, dtype=dtype, device=device)

    for i in range(batch_size):
        logits[i, i % vocab_size] = 10.0

    k = torch.ones((batch_size,), dtype=torch.int64, device="cpu")
    p = torch.ones((batch_size,), dtype=torch.float32, device="cpu")
    out = torch.empty((batch_size,), dtype=torch.int32, device=device)

    return Payload(
        _top_k_top_p_sampler,
        _torch_argmax,
        (logits, k, p, out),
        {"implementation_index": implementation_index},
    )


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_top_k_top_p_sampler_optional_p(
    dtype,
    device,
    implementation_index,
):
    shape = (3, 16)
    batch_size, vocab_size = shape
    logits = torch.full(shape, -10.0, dtype=dtype, device=device)

    for i in range(batch_size):
        logits[i, (i + 1) % vocab_size] = 10.0

    k = torch.ones((1,), dtype=torch.int64, device="cpu")
    out = torch.empty((batch_size,), dtype=torch.int32, device=device)

    return Payload(
        _top_k_top_p_sampler,
        _torch_argmax,
        (logits, k, None, out),
        {"implementation_index": implementation_index},
    )


def _top_k_top_p_sampler(logits, k, p, out, *, implementation_index):
    infini.ops.top_k_top_p_sampler(
        logits,
        k,
        p,
        out,
        stream=get_stream(logits.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_argmax(logits, k, p, out, *, implementation_index):
    del k, p, implementation_index

    out.copy_(torch.argmax(logits, dim=-1).to(torch.int32))

    return out
