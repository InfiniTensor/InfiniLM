import infini.ops
import pytest
import torch
import torch.nn.functional as F

from tests.utils import empty_strided, get_stream


@pytest.mark.parametrize(
    "shape, input_strides, values_strides, indices_strides, topk, norm",
    (
        ((1, 10), None, None, None, 7, True),
        ((1, 10), None, None, None, 7, False),
        ((8, 20), None, None, None, 4, True),
        ((8, 20), (24, 1), (6, 1), (6, 1), 4, True),
        ((2, 64), None, None, None, 6, True),
        ((2, 64), (80, 1), (8, 1), (8, 1), 6, False),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-6, 1e-6),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_topksoftmax_infinilm(
    shape,
    input_strides,
    values_strides,
    indices_strides,
    topk,
    norm,
    dtype,
    device,
    rtol,
    atol,
):
    input = _make_input(shape, input_strides, dtype=dtype, device=device)
    values = empty_strided(
        (shape[0], topk), values_strides, dtype=torch.float32, device=device
    )
    indices = empty_strided(
        (shape[0], topk), indices_strides, dtype=torch.int32, device=device
    )

    infini.ops.topksoftmax_infinilm(
        input,
        topk,
        norm,
        values,
        indices,
        stream=get_stream(input.device),
    )

    ref_values, ref_indices = _torch_topksoftmax_infinilm(input, topk, norm)
    torch.testing.assert_close(values, ref_values, rtol=rtol, atol=atol)
    torch.testing.assert_close(indices, ref_indices.to(torch.int32), rtol=0, atol=0)


def _make_input(shape, strides, *, dtype, device):
    values = torch.arange(shape[0] * shape[1], dtype=torch.float32, device=device)
    values = values.reshape(shape) * 0.125
    values = values.to(dtype)
    if strides is None:
        return values

    input = empty_strided(shape, strides, dtype=dtype, device=device)
    input.copy_(values)

    return input


def _torch_topksoftmax_infinilm(input, topk, norm):
    result = F.softmax(input, dim=1, dtype=torch.float32)
    ref_values, ref_indices = torch.topk(result, topk, dim=1)
    if norm:
        ref_values = ref_values / ref_values.sum(dim=1, keepdim=True)

    return ref_values, ref_indices
