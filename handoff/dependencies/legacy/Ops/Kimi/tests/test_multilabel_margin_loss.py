import infini.ops
import pytest
import torch
from torch.nn import functional as F

from tests.utils import get_stream


@pytest.mark.parametrize(
    (
        "expected_reduction",
        "api_style",
        "size_average",
        "reduce",
        "reduction",
    ),
    (
        ("mean", "keyword", None, None, "mean"),
        ("none", "positional", None, None, "none"),
        ("sum", "positional", None, None, "sum"),
        ("none", "positional", None, False, "sum"),
        ("sum", "keyword", False, None, "none"),
        ("mean", "positional", True, True, "sum"),
        ("sum", "deprecated", None, None, "sum"),
    ),
)
def test_multilabel_margin_loss(
    expected_reduction,
    api_style,
    size_average,
    reduce,
    reduction,
    device,
    implementation_index,
):
    input = torch.randn((4, 5), dtype=torch.float32, device=device)
    target = torch.tensor(
        (
            (0, 2, 4, -1, -1),
            (1, 3, -1, -1, -1),
            (4, 0, 2, 1, -1),
            (2, 3, -1, -1, -1),
        ),
        dtype=torch.int64,
        device=device,
    )
    out_shape = (input.shape[0],) if expected_reduction == "none" else ()
    out = torch.empty(out_shape, dtype=input.dtype, device=device)
    kwargs = {
        "stream": get_stream(device),
        "implementation_index": implementation_index,
    }

    if api_style == "keyword":
        infini.ops.multilabel_margin_loss(
            input=input,
            target=target,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            out=out,
            **kwargs,
        )
    elif api_style == "deprecated":
        infini.ops.multilabel_margin_loss(
            input,
            target,
            {"none": 0, "mean": 1, "sum": 2}[reduction],
            out,
            **kwargs,
        )
    else:
        infini.ops.multilabel_margin_loss(
            input,
            target,
            size_average,
            reduce,
            reduction,
            out,
            **kwargs,
        )

    expected = F.multilabel_margin_loss(
        input,
        target,
        reduction=expected_reduction,
    )
    torch.testing.assert_close(out, expected)
