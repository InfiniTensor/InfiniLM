import infini.ops
import pytest
import torch

from tests.utils import (
    Payload,
    empty_strided,
    get_stream,
    randint_strided,
    randn_strided,
)

_INT_DTYPES = (torch.int16, torch.int32, torch.int64)

_UINT_DTYPES = tuple(
    filter(None, (getattr(torch, f"uint{bits}", None) for bits in (16, 32, 64)))
)

_FLOAT_DTYPE_CASES = (
    (torch.float32, 1e-7, 1e-7),
    (torch.float16, 1e-3, 1e-3),
    (torch.bfloat16, 1e-2, 5e-3),
)

_DTYPE_CASES = _FLOAT_DTYPE_CASES + tuple(
    (dtype, 0, 0) for dtype in _INT_DTYPES + _UINT_DTYPES
)

_ALPHA_DTYPE_CASES = (
    (torch.float32, 1e-6, 1e-6),
    (torch.float16, 1e-2, 1e-2),
    (torch.bfloat16, 1e-2, 1e-2),
)

_DEFAULT_ADD_CASES = tuple(
    (shape, shape, shape, input_strides, other_strides, out_strides, None)
    for shape, input_strides, other_strides, out_strides in (
        ((13, 4), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
        ((13, 4), (0, 1), None, None),
        ((13, 4, 4), None, None, None),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
        ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
        ((16, 5632), None, None, None),
        ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
        ((13, 16, 2), (128, 4, 1), (0, 2, 1), (64, 4, 1)),
        ((13, 16, 2), (128, 4, 1), (2, 0, 1), (64, 4, 1)),
        ((4, 4, 5632), None, None, None),
        (
            (4, 4, 5632),
            (45056, 5632, 1),
            (45056, 5632, 1),
            (45056, 5632, 1),
        ),
    )
)

_ALPHA_ADD_CASES = tuple(
    (*shapes, None, None, None, alpha)
    for shapes in (
        ((3, 1), (1, 5), (3, 5)),
        ((5,), (2, 3, 5), (2, 3, 5)),
    )
    for alpha in (0.0, 0.5, 1.0, 2.0)
)

_EMPTY_ADD_CASE = (
    (0,),
    (0,),
    (0,),
    None,
    None,
    None,
    None,
    torch.float32,
    1e-7,
    1e-7,
)

_TEST_CASES = (
    tuple(
        (*case, *dtype_case)
        for case in _DEFAULT_ADD_CASES
        for dtype_case in _DTYPE_CASES
    )
    + tuple(
        (*case, *dtype_case)
        for case in _ALPHA_ADD_CASES
        for dtype_case in _ALPHA_DTYPE_CASES
    )
    + (_EMPTY_ADD_CASE,)
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    (
        "input_shape",
        "other_shape",
        "out_shape",
        "input_strides",
        "other_strides",
        "out_strides",
        "alpha",
        "dtype",
        "rtol",
        "atol",
    ),
    _TEST_CASES,
)
def test_add(
    input_shape,
    other_shape,
    out_shape,
    input_strides,
    other_strides,
    out_strides,
    alpha,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    if implementation_index == 10 and (
        input_shape != out_shape
        or other_shape != out_shape
        or input_strides is not None
        or other_strides is not None
        or out_strides is not None
        or alpha not in (None, 1.0)
    ):
        pytest.skip(
            "Triton `Add` only supports contiguous tensors with identical shapes "
            "and alpha equal to one."
        )

    if device == "musa" and dtype in _UINT_DTYPES:
        pytest.skip(
            "The `torch.musa` test cloning path does not support `uint16`, `uint32`, or `uint64`."
        )

    if device == "mlu" and (dtype in _UINT_DTYPES or dtype == torch.int16):
        pytest.skip(
            "The `torch.mlu` test cloning path does not support `int16`, `uint16`, `uint32`, or `uint64`."
        )

    if implementation_index == 1 and dtype in _UINT_DTYPES:
        pytest.skip("ATen `add` does not support unsigned integer types")

    if dtype in _INT_DTYPES or dtype in _UINT_DTYPES:
        input = randint_strided(
            0, 100, input_shape, input_strides, dtype=dtype, device=device
        )
        other = randint_strided(
            0, 100, other_shape, other_strides, dtype=dtype, device=device
        )
    else:
        input = randn_strided(input_shape, input_strides, dtype=dtype, device=device)
        other = randn_strided(other_shape, other_strides, dtype=dtype, device=device)

    out = empty_strided(out_shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda *args: _add(
            *args, alpha=alpha, implementation_index=implementation_index
        ),
        lambda *args: _torch_add(*args, alpha=alpha),
        (input, other, out),
        {},
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "config",
    (
        {
            "num_warps": 4,
            "num_stages": 3,
            "constexprs": {"BLOCK_SIZE": 128},
        },
        {
            "auto_tuning": {
                "warmup_milliseconds": 0,
                "repetition_milliseconds": 1,
                "keys": ["num_elements"],
                "candidates": [
                    {
                        "num_warps": 4,
                        "num_stages": 3,
                        "constexprs": {"BLOCK_SIZE": 128},
                    },
                    {
                        "num_warps": 4,
                        "num_stages": 3,
                        "constexprs": {"BLOCK_SIZE": 64},
                    },
                ],
            }
        },
    ),
    ids=("compile-config", "auto-tuning"),
)
def test_add_triton_config(config, device, implementation_index):
    if implementation_index != 10:
        pytest.skip("requires Triton implementation 10")

    input = torch.randn(257, device=device)
    other = torch.randn_like(input)
    out = torch.empty_like(input)
    stream = torch.cuda.Stream(device=input.device)

    with torch.cuda.stream(stream):
        infini.ops.add(
            input,
            other,
            out,
            stream=get_stream(input.device),
            implementation_index=implementation_index,
            config=config,
        )

    stream.synchronize()

    torch.testing.assert_close(out, input + other)


@pytest.mark.smoke
def test_add_triton_config_rejects_unknown_fields(device, implementation_index):
    if implementation_index != 10:
        pytest.skip("requires Triton implementation 10")

    input = torch.randn(1, device=device)
    other = torch.randn_like(input)
    out = torch.empty_like(input)

    with pytest.raises(
        ValueError, match="Unknown Triton JIT config field `unexpected`"
    ):
        infini.ops.add(
            input,
            other,
            out,
            stream=get_stream(input.device),
            implementation_index=implementation_index,
            config={"unexpected": 1},
        )


def _add(input, other, out, *, alpha, implementation_index=0):
    kwargs = {
        "stream": get_stream(input.device),
        "implementation_index": implementation_index,
    }

    if alpha is None:
        infini.ops.add(input, other, out, **kwargs)
    else:
        infini.ops.add(input, other, alpha, out, **kwargs)

    return out


def _torch_add(input, other, out, *, alpha):
    if input.dtype in _UINT_DTYPES:
        input = input.to(torch.int64)

    if other.dtype in _UINT_DTYPES:
        other = other.to(torch.int64)

    res = torch.add(input, other, alpha=1 if alpha is None else alpha)
    out.copy_(res.to(out.dtype))

    return out
