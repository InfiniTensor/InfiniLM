import infini.ops
import pytest

import torch
from tests.utils import (
    Payload,
    empty_strided,
    get_stream,
    rand_strided,
    randint_strided,
    randn_strided,
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "expected_reduction, has_weight, ignore_index, api_style, size_average, reduce, reduction",
    (
        ("mean", False, -100, "keyword", None, None, "mean"),
        ("none", False, -100, "full", None, None, "none"),
        ("mean", True, -100, "full", None, None, "mean"),
        ("sum", True, 2, "legacy", None, None, "sum"),
        ("sum", False, -100, "full", None, None, "sum"),
        ("none", False, -100, "full", None, False, "sum"),
        ("sum", True, -100, "full", False, None, "none"),
        ("mean", True, -100, "full", None, True, "sum"),
    ),
)
@pytest.mark.parametrize(
    "dtype, rtol, atol",
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 1e-2, 1e-2),
    ),
)
def test_nll_loss(
    expected_reduction,
    has_weight,
    ignore_index,
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
    input = torch.nn.functional.log_softmax(
        randn_strided((batch_size, num_classes), None, dtype=dtype, device=device),
        dim=1,
    )
    target = randint_strided(
        0,
        num_classes,
        (batch_size,),
        None,
        dtype=torch.int64,
        device=device,
    )

    if ignore_index >= 0:
        target[0] = ignore_index

    weight = None

    if has_weight:
        weight = rand_strided((num_classes,), None, dtype=dtype, device=device).add_(
            0.5
        )

    out_shape = target.shape if expected_reduction == "none" else ()
    out = empty_strided(out_shape, None, dtype=dtype, device=device)

    return Payload(
        lambda *args: _nll_loss(
            *args,
            reduction=reduction,
            ignore_index=ignore_index,
            api_style=api_style,
            size_average=size_average,
            reduce=reduce,
            implementation_index=implementation_index,
        ),
        lambda *args: _torch_nll_loss(
            *args, reduction=expected_reduction, ignore_index=ignore_index
        ),
        (input, target, weight, out),
        {},
        rtol=rtol,
        atol=atol,
    )


def _nll_loss(
    input,
    target,
    weight,
    out,
    *,
    reduction,
    ignore_index,
    api_style,
    size_average,
    reduce,
    implementation_index,
):
    kwargs = {
        "implementation_index": implementation_index,
        "stream": get_stream(input.device),
    }

    if api_style == "keyword":
        infini.ops.nll_loss(
            input=input,
            target=target,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            out=out,
            **kwargs,
        )
    elif api_style == "legacy":
        infini.ops.nll_loss(
            input,
            target,
            weight,
            {"none": 0, "mean": 1, "sum": 2}[reduction],
            ignore_index,
            out,
            **kwargs,
        )
    else:
        infini.ops.nll_loss(
            input,
            target,
            weight,
            size_average,
            ignore_index,
            reduce,
            reduction,
            out,
            **kwargs,
        )

    return out


def _torch_nll_loss(input, target, weight, out, *, reduction, ignore_index):
    result = torch.nn.functional.nll_loss(
        input,
        target,
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    out.copy_(result)

    return out
