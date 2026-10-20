import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize(
    ("api_style", "size_average", "reduce", "reduction"),
    (
        ("keyword", None, None, "none"),
        ("full", True, False, "sum"),
        ("deprecated", None, None, "none"),
    ),
)
@pytest.mark.parametrize("dtype", (torch.float32,))
def test_soft_margin_loss(
    api_style,
    size_average,
    reduce,
    reduction,
    dtype,
    device,
    implementation_index,
):
    input = torch.randn((4, 7), dtype=dtype, device=device)
    target = torch.empty_like(input).bernoulli_().mul_(2).sub_(1)
    out = torch.empty_like(input)
    kwargs = {
        "stream": get_stream(device),
        "implementation_index": implementation_index,
    }

    if api_style == "keyword":
        infini.ops.soft_margin_loss(
            input=input,
            target=target,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            out=out,
            **kwargs,
        )
    elif api_style == "deprecated":
        infini.ops.soft_margin_loss(
            input,
            target,
            {"none": 0, "mean": 1, "sum": 2}[reduction],
            out,
            **kwargs,
        )
    else:
        infini.ops.soft_margin_loss(
            input,
            target,
            size_average,
            reduce,
            reduction,
            out,
            **kwargs,
        )

    expected = torch.nn.functional.soft_margin_loss(
        input,
        target,
        reduction="none",
    )
    torch.testing.assert_close(out, expected)
