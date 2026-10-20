import infini.ops
import pytest

import torch
from tests.utils import Payload, empty_strided, get_stream, randn_strided


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "api_style, size_average, reduce, reduction",
    (
        ("keyword", None, None, "none"),
        ("positional", True, False, "sum"),
        ("deprecated", None, None, "none"),
    ),
)
def test_mse_loss(
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
    input = randn_strided((4, 3, 5), None, dtype=dtype, device=device)
    target = randn_strided((4, 3, 5), None, dtype=dtype, device=device)
    out = empty_strided(input.shape, None, dtype=dtype, device=device)

    def func(input, target, out):
        kwargs = {
            "implementation_index": implementation_index,
            "stream": get_stream(input.device),
        }

        if api_style == "keyword":
            infini.ops.mse_loss(
                input=input,
                target=target,
                weight=None,
                size_average=size_average,
                reduce=reduce,
                reduction=reduction,
                out=out,
                **kwargs,
            )
        elif api_style == "deprecated":
            infini.ops.mse_loss(
                input,
                target,
                0,
                out,
                **kwargs,
            )
        else:
            infini.ops.mse_loss(
                input,
                target,
                None,
                size_average,
                reduce,
                reduction,
                out,
                **kwargs,
            )

        return out

    def ref(input, target, out):
        result = torch.nn.functional.mse_loss(
            input,
            target,
            weight=None,
            reduction="none",
        )
        out.copy_(result)

        return out

    return Payload(
        func,
        ref,
        (input, target, out),
        {},
        rtol=rtol,
        atol=atol,
    )
