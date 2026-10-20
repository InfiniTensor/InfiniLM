import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize(
    ("api_style", "size_average", "reduce", "reduction", "beta"),
    (
        ("keyword", None, None, "none", 1.0),
        ("full", True, False, "sum", 0.5),
        ("deprecated", None, None, "none", 1.5),
    ),
)
@pytest.mark.parametrize("dtype", (torch.float32,))
def test_smooth_l1_loss(
    api_style,
    size_average,
    reduce,
    reduction,
    beta,
    dtype,
    device,
    implementation_index,
):
    input = torch.randn((4, 7), dtype=dtype, device=device)
    target = torch.randn_like(input)
    out = torch.empty_like(input)
    kwargs = {
        "stream": get_stream(device),
        "implementation_index": implementation_index,
    }

    if api_style == "keyword":
        infini.ops.smooth_l1_loss(
            input=input,
            target=target,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            beta=beta,
            out=out,
            **kwargs,
        )
    elif api_style == "deprecated":
        infini.ops.smooth_l1_loss(
            input,
            target,
            {"none": 0, "mean": 1, "sum": 2}[reduction],
            beta,
            out,
            **kwargs,
        )
    else:
        infini.ops.smooth_l1_loss(
            input,
            target,
            size_average,
            reduce,
            reduction,
            beta,
            out,
            **kwargs,
        )

    expected = torch.nn.functional.smooth_l1_loss(
        input,
        target,
        reduction="none",
        beta=beta,
    )
    torch.testing.assert_close(out, expected)
