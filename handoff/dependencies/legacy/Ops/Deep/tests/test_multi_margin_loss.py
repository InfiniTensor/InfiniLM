import infini.ops
import pytest

import torch
from torch.nn import functional as F
from tests.utils import (
    Payload,
    empty_strided,
    get_stream,
    randint_strided,
    rand_strided,
    randn_strided,
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "expected_reduction, has_weight, p, margin, api_style, size_average, reduce, reduction",
    (
        ("mean", False, 1, 1.0, "keyword", None, None, "mean"),
        ("none", True, 2, 0.5, "positional", None, None, "none"),
        ("sum", False, 1, 1.5, "positional", None, None, "sum"),
        ("none", True, 2, 0.75, "keyword", None, False, "sum"),
        ("sum", False, 1, 1.0, "positional", False, None, "none"),
        ("mean", True, 2, 1.25, "keyword", None, True, "sum"),
        ("none", False, 1, 0.5, "positional", False, False, "mean"),
        ("sum", True, 2, 1.0, "deprecated", None, None, "sum"),
    ),
)
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    ((torch.float32, 1e-5, 1e-5),),
)
def test_multi_margin_loss(
    expected_reduction,
    has_weight,
    p,
    margin,
    api_style,
    size_average,
    reduce,
    reduction,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    batch_size = 7
    num_classes = 5
    input = randn_strided((batch_size, num_classes), None, dtype=dtype, device=device)
    target = randint_strided(
        0,
        num_classes,
        (batch_size,),
        None,
        dtype=torch.int64,
        device=device,
    )
    weight = None

    if has_weight:
        weight = rand_strided((num_classes,), None, dtype=dtype, device=device).add_(
            0.5
        )

    out_shape = target.shape if expected_reduction == "none" else ()
    out = empty_strided(out_shape, None, dtype=dtype, device=device)

    return Payload(
        lambda *args: _multi_margin_loss(
            *args,
            p=p,
            margin=margin,
            api_style=api_style,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            expected_reduction=expected_reduction,
            implementation_index=implementation_index,
        ),
        lambda *args: _torch_multi_margin_loss(
            *args,
            p=p,
            margin=margin,
            size_average=None if api_style == "deprecated" else size_average,
            reduce=None if api_style == "deprecated" else reduce,
            reduction=(expected_reduction if api_style == "deprecated" else reduction),
        ),
        (input, target, weight, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _multi_margin_loss(
    input,
    target,
    weight,
    out,
    *,
    p,
    margin,
    api_style,
    size_average,
    reduce,
    reduction,
    expected_reduction,
    implementation_index,
):
    kwargs = {
        "implementation_index": implementation_index,
        "stream": get_stream(input.device),
    }

    if api_style == "keyword":
        infini.ops.multi_margin_loss(
            input=input,
            target=target,
            weight=weight,
            p=p,
            margin=margin,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            out=out,
            **kwargs,
        )
    elif api_style == "deprecated":
        infini.ops.multi_margin_loss(
            input,
            target,
            float(p),
            margin,
            weight,
            {"none": 0, "mean": 1, "sum": 2}[expected_reduction],
            out,
            **kwargs,
        )
    else:
        infini.ops.multi_margin_loss(
            input,
            target,
            weight,
            p,
            margin,
            size_average,
            reduce,
            reduction,
            out,
            **kwargs,
        )

    return out


def _torch_multi_margin_loss(
    input,
    target,
    weight,
    out,
    *,
    p,
    margin,
    size_average,
    reduce,
    reduction,
):
    result = F.multi_margin_loss(
        input,
        target,
        p=p,
        margin=margin,
        weight=weight,
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )
    out.copy_(result)

    return out
