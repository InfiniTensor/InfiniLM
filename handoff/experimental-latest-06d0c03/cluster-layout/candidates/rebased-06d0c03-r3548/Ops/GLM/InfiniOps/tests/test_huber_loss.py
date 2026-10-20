import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize(
    "api_style, reduction, delta",
    (
        ("keyword", "none", 0.5),
        ("positional", "mean", 1.0),
        ("positional", "sum", 2.0),
        ("deprecated", "mean", 1.5),
    ),
)
def test_huber_loss(api_style, reduction, delta, device, implementation_index):
    input = torch.randn((4, 7), dtype=torch.float32, device=device)
    target = torch.randn_like(input)
    weight = None
    # ATen uses `out` as elementwise scratch before applying the reduction.
    out = torch.empty_like(input)
    kwargs = {
        "stream": get_stream(device),
        "implementation_index": implementation_index,
    }

    if api_style == "keyword":
        infini.ops.huber_loss(
            input=input,
            target=target,
            weight=weight,
            reduction=reduction,
            delta=delta,
            out=out,
            **kwargs,
        )
    elif api_style == "deprecated":
        infini.ops.huber_loss(
            input,
            target,
            {"none": 0, "mean": 1, "sum": 2}[reduction],
            delta,
            out,
            **kwargs,
        )
    else:
        infini.ops.huber_loss(
            input,
            target,
            weight,
            reduction,
            delta,
            out,
            **kwargs,
        )

    expected = torch.nn.functional.huber_loss(
        input,
        target,
        reduction=reduction,
        delta=delta,
        weight=weight,
    )
    actual = out if reduction == "none" else out.flatten()[0]
    torch.testing.assert_close(actual, expected)
