import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


_TEST_CASES = (
    pytest.param((2, 3, 4), (3, 4, 5), (2, 5), 2, False, id="integer-two"),
    pytest.param((2, 3), (4, 5), (2, 3, 4, 5), 0, False, id="integer-zero"),
    pytest.param((2, 3, 4), (4, 5, 6), (2, 3, 5, 6), 1, False, id="integer"),
    pytest.param(
        (2, 3, 4),
        (3, 2, 5),
        (4, 5),
        ([1, 0], [0, 1]),
        False,
        id="tuple-dimensions",
    ),
    pytest.param(
        (2, 3, 4),
        (5, 4, 3),
        (2, 5),
        [[-1, -2], [1, 2]],
        False,
        id="list-dimensions",
    ),
    pytest.param(
        (2, 3, 4),
        (5, 4, 3),
        (2, 5),
        ([-1, -2], [1, 2]),
        True,
        id="deprecated-dimensions",
    ),
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    ("a_shape", "b_shape", "out_shape", "dims", "use_deprecated"),
    _TEST_CASES,
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 2e-2, 2e-2),
    ),
)
def test_tensordot(
    a_shape,
    b_shape,
    out_shape,
    dims,
    use_deprecated,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    a = randn_strided(a_shape, None, dtype=dtype, device=device)
    b = randn_strided(b_shape, None, dtype=dtype, device=device)
    out = empty_strided(out_shape, None, dtype=dtype, device=device)

    return Payload(
        lambda *args: _tensordot(
            *args,
            dims=dims,
            use_deprecated=use_deprecated,
            implementation_index=implementation_index,
        ),
        lambda *args: _torch_tensordot(*args, dims=dims),
        (a, b, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _tensordot(a, b, out, *, dims, use_deprecated, implementation_index):
    kwargs = {
        "implementation_index": implementation_index,
        "stream": get_stream(a.device),
    }

    if use_deprecated:
        dims_a, dims_b = dims
        infini.ops.tensordot(a, b, dims_a, dims_b, out, **kwargs)
    else:
        infini.ops.tensordot(a, b, dims, out, **kwargs)

    return out


def _torch_tensordot(a, b, out, *, dims):
    return torch.tensordot(a, b, dims=dims, out=out)
