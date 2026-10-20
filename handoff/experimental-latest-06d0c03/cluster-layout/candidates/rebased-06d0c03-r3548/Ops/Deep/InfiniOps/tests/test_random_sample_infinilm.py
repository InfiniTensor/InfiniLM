import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream


_TEST_CASES = (
    (512, 0.8, 0.8, 3, 0.5),
    (4096, 0.05, 0.9, 5, 1.0),
    (16384, 0.15, 0.85, 10, 2.0),
    (512, 0.08, 0.0, 3, 0.5),
    (4096, 0.5, 0.9, 1, 1.0),
    (16384, 0.15, 0.0, 1, 2.0),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize("voc, random_val, topp, topk, temperature", _TEST_CASES)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 0.0, 0.0),
        (torch.float16, 0.0, 0.0),
        (torch.bfloat16, 0.0, 0.0),
    ),
)
def test_random_sample_infinilm(
    voc,
    random_val,
    topp,
    topk,
    temperature,
    dtype,
    device,
    rtol,
    atol,
):
    perm = torch.randperm(voc)
    logits = (torch.arange(voc)[perm].float() * 0.0001).to(dtype=dtype, device=device)
    out = torch.empty((), dtype=torch.int32, device=device)

    return Payload(
        _random_sample_infinilm,
        _torch_random_sample_infinilm,
        (logits, random_val, topp, topk, temperature, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _random_sample_infinilm(logits, random_val, topp, topk, temperature, out):
    infini.ops.random_sample_infinilm(
        logits,
        random_val,
        topp,
        topk,
        temperature,
        out,
        stream=get_stream(logits.device),
    )

    return logits[out.to(torch.long)].reshape(())


def _torch_random_sample_infinilm(logits, random_val, topp, topk, temperature, out):
    idx = _torch_random_sample_infinilm_index(
        logits, random_val, topp, topk, temperature
    )
    out.copy_(idx.to(dtype=out.dtype, device=out.device))

    return logits[out.to(torch.long)].reshape(())


def _torch_random_sample_infinilm_index(logits, random_val, topp, topk, temperature):
    data = logits.detach().cpu().to(torch.float32)

    if random_val == 0.0 or topp == 0.0 or topk == 1 or temperature == 0.0:
        return torch.argmax(data).to(torch.int64)

    sorted_vals, sorted_indices = torch.sort(data, descending=True)
    scaled_vals = (sorted_vals - sorted_vals[0]) / temperature
    probs = torch.softmax(scaled_vals, dim=0)
    cum_probs = torch.cumsum(probs, dim=0)

    k_index = min(topk, logits.numel()) - 1
    threshold = (
        torch.minimum(cum_probs[k_index], cum_probs.new_tensor(topp)) * random_val
    )
    idx = torch.searchsorted(cum_probs, threshold)

    return sorted_indices[idx].to(torch.int64)
