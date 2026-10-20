import infini.ops
import pytest

import torch
from tests.utils import get_stream


@pytest.mark.parametrize(
    ("dim", "descending", "stable", "use_deprecated"),
    (
        (-1, False, False, False),
        (0, False, False, False),
        (-1, True, False, False),
        (-1, False, True, False),
        (0, True, True, True),
    ),
)
@pytest.mark.parametrize("dtype", (torch.float32,))
def test_argsort(
    dim,
    descending,
    stable,
    use_deprecated,
    implementation_index,
    dtype,
    device,
):
    input = torch.tensor(
        ((2.0, 1.0, 1.0), (0.0, 3.0, -1.0)), dtype=dtype, device=device
    )
    out = torch.empty_like(input, dtype=torch.int64)
    kwargs = {
        "stream": get_stream(device),
        "implementation_index": implementation_index,
    }

    if use_deprecated:
        infini.ops.argsort(input, stable, dim, descending, out, **kwargs)
    else:
        infini.ops.argsort(input, dim, descending, stable, out, **kwargs)

    expected = torch.argsort(input, dim=dim, descending=descending, stable=stable)
    assert torch.equal(out, expected)

    if use_deprecated:
        new_order_conversion = torch.argsort(
            input,
            dim=int(stable),
            descending=bool(dim),
            stable=descending,
        )
        assert not torch.equal(out, new_order_conversion)
