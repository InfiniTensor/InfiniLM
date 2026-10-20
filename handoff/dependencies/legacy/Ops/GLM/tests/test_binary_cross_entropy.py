import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize(
    ("has_weight", "api_style", "size_average", "reduce", "reduction"),
    (
        (False, "keyword", None, None, "none"),
        (True, "full", True, False, "sum"),
        (False, "deprecated", None, None, "none"),
    ),
)
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_binary_cross_entropy(
    has_weight,
    api_style,
    size_average,
    reduce,
    reduction,
    dtype,
    rtol,
    atol,
    device,
    implementation_index,
):
    input = torch.rand((4, 7), dtype=dtype, device=device)
    target = torch.rand_like(input)
    weight = torch.rand_like(input) if has_weight else None
    out = torch.empty_like(input)
    kwargs = {
        "stream": get_stream(device),
        "implementation_index": implementation_index,
    }

    if api_style == "keyword":
        infini.ops.binary_cross_entropy(
            input=input,
            target=target,
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            out=out,
            **kwargs,
        )
    elif api_style == "deprecated":
        infini.ops.binary_cross_entropy(
            input,
            target,
            weight,
            {"none": 0, "mean": 1, "sum": 2}[reduction],
            out,
            **kwargs,
        )
    else:
        infini.ops.binary_cross_entropy(
            input,
            target,
            weight,
            size_average,
            reduce,
            reduction,
            out,
            **kwargs,
        )

    expected = torch.nn.functional.binary_cross_entropy(
        input,
        target,
        weight=weight,
        reduction="none",
    )
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
