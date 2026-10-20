import infini.ops
import pytest
import torch

from tests.utils import (
    Payload,
    empty_strided,
    get_stream,
    randint_strided,
    rand_strided,
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, out_strides, inplace",
    (
        ((), None, None, False),
        ((0,), None, None, False),
        ((1, 3), None, None, False),
        ((1, 3), None, None, True),
        ((3, 3), (5, 1), (5, 1), False),
        ((32, 20, 512), None, None, False),
        (
            (3, 3, 13, 9, 17),
            (19890, 6630, 510, 34, 1),
            (19890, 6630, 510, 34, 1),
            False,
        ),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float64, 0, 0),
        (torch.float32, 0, 0),
        (torch.float16, 0, 0),
        (torch.bfloat16, 0, 0),
    ),
)
def test_relu(
    shape,
    input_strides,
    out_strides,
    inplace,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    if device == "musa" and dtype == torch.float64:
        pytest.skip("MUSA does not support float64 ReLU")
    if device == "mlu" and dtype == torch.float64:
        pytest.skip("Cambricon device code does not support float64 comparisons")

    input = rand_strided(shape, input_strides, dtype=dtype, device=device)
    input.mul_(2).sub_(1)
    out = (
        input
        if inplace
        else empty_strided(shape, out_strides, dtype=dtype, device=device)
    )

    return Payload(
        lambda *args: _relu(*args, implementation_index=implementation_index),
        _torch_relu,
        (input, out),
        {},
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, out_strides",
    (
        ((3, 3), None, None),
        ((3, 3), (5, 1), (5, 1)),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8),
)
def test_relu_integer(
    shape,
    input_strides,
    out_strides,
    implementation_index,
    dtype,
    device,
):
    low = 0 if dtype == torch.uint8 else -5
    input = randint_strided(low, 6, shape, input_strides, dtype=dtype, device=device)
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda *args: _relu(*args, implementation_index=implementation_index),
        _torch_relu,
        (input, out),
        {},
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("size", (33,))
@pytest.mark.parametrize("dtype", (torch.float32,))
def test_native_relu_partial_overlap(size, implementation_index, dtype, device):
    if implementation_index != 0:
        pytest.skip("partial-overlap staging is specific to the native implementation")

    input = rand_strided((size, size), None, dtype=dtype, device=device)
    input.mul_(2).sub_(1)
    expected = torch.relu(input.clone())
    out = input.t()

    actual = _relu(input, out, implementation_index=implementation_index)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("inplace", (False, True))
@pytest.mark.parametrize(
    "dtype",
    (torch.float64, torch.float32, torch.float16, torch.bfloat16),
)
def test_relu_matches_special_value_semantics(
    dtype, inplace, implementation_index, device
):
    if device == "musa" and dtype == torch.float64:
        pytest.skip("MUSA does not support float64 ReLU")
    if device == "mlu" and dtype == torch.float64:
        pytest.skip("Cambricon device code does not support float64 comparisons")

    input = torch.tensor(
        [float("-inf"), -1.0, -0.0, 0.0, 1.0, float("inf"), float("nan")],
        dtype=dtype,
        device=device,
    )
    expected = torch.relu(input).cpu()
    out = input if inplace else torch.empty_like(input)

    actual = _relu(input, out, implementation_index=implementation_index).cpu()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zero_mask = expected == 0
    assert torch.equal(
        torch.signbit(actual[zero_mask]),
        torch.signbit(expected[zero_mask]),
    )


def _relu(input, out, *, implementation_index=0):
    infini.ops.relu(
        input,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    return out


def _torch_relu(input, out):
    out.copy_(torch.relu(input))

    return out
