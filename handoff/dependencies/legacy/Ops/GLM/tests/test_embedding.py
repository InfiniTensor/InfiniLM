import infini.ops
import pytest
import torch

from tests.utils import (
    Payload,
    empty_strided,
    get_stream,
    randint_strided,
    randn_strided,
)


# Format:
# (input_shape, weight_shape, input_strides, weight_strides, out_strides,
#  input_dtype, options)
_TEST_CASES = tuple(
    (*case, None)
    for case in (
        ((), (8, 4), None, None, None, torch.int64),
        ((1, 5), (32000, 4), None, None, None, torch.int64),
        ((2, 10), (32000, 2048), None, None, None, torch.int32),
        ((1, 5), (10, 10), None, None, None, torch.int64),
        ((2, 4), (32, 8), None, None, None, torch.int64),
        ((2, 4), (32, 8), (8, 1), None, (32, 8, 1), torch.int32),
        ((2, 4), (32, 8), None, (1, 32), None, torch.int64),
    )
) + tuple(
    ((2, 3), (8, 4), None, None, None, torch.int64, options)
    for options in (
        *(
            (padding_idx, None, 2.0, scale_grad_by_freq, sparse, True)
            for padding_idx in (-1, 0)
            for scale_grad_by_freq in (False, True)
            for sparse in (False, True)
        ),
        (None, None, 2.0, False, False, False),
        (-1, None, 2.0, True, True, False),
        (None, 0.5, 2.0, False, False, False),
        (None, 1.0, 1.0, False, False, False),
        (None, 1.0, 3.0, False, False, False),
    )
)


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    (
        "input_shape",
        "weight_shape",
        "input_strides",
        "weight_strides",
        "out_strides",
        "input_dtype",
        "options",
    ),
    _TEST_CASES,
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-3, 0.0),
        (torch.float16, 1e-2, 0.0),
        (torch.bfloat16, 5e-2, 0.0),
    ),
)
def test_embedding(
    input_shape,
    weight_shape,
    input_strides,
    weight_strides,
    out_strides,
    input_dtype,
    options,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    vocab_size = weight_shape[0]
    embedding_dim = weight_shape[1]
    output_shape = (*input_shape, embedding_dim)
    if options is None:
        padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse = (
            None,
            None,
            2.0,
            False,
            False,
        )
        use_legacy_overload = False
    else:
        (
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            use_legacy_overload,
        ) = options

    input = randint_strided(
        0 if padding_idx is not None else 1,
        min(9, vocab_size),
        input_shape,
        input_strides,
        dtype=input_dtype,
        device=device,
    )
    weight = randn_strided(weight_shape, weight_strides, dtype=dtype, device=device)
    out = empty_strided(output_shape, out_strides, dtype=dtype, device=device)

    return Payload(
        lambda *args, **kwargs: _embedding(
            *args,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            use_default_overload=options is None,
            use_legacy_overload=use_legacy_overload,
            implementation_index=implementation_index,
            **kwargs,
        ),
        lambda *args, **kwargs: _torch_embedding(
            *args,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            **kwargs,
        ),
        (input, weight),
        {"out": out},
        rtol=rtol,
        atol=atol,
    )


def _embedding(
    input,
    weight,
    *,
    out,
    padding_idx,
    max_norm,
    norm_type,
    scale_grad_by_freq,
    sparse,
    use_default_overload,
    use_legacy_overload,
    implementation_index,
):
    kwargs = {
        "implementation_index": implementation_index,
        "stream": get_stream(input.device),
    }

    if use_default_overload:
        infini.ops.embedding(input, weight, out, **kwargs)
    elif use_legacy_overload:
        infini.ops.embedding(
            input,
            weight,
            padding_idx,
            scale_grad_by_freq,
            sparse,
            out,
            **kwargs,
        )
    else:
        infini.ops.embedding(
            input,
            weight,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
            out,
            **kwargs,
        )

    return out


def _torch_embedding(
    input,
    weight,
    *,
    out,
    padding_idx,
    max_norm,
    norm_type,
    scale_grad_by_freq,
    sparse,
):
    if max_norm is not None:
        # Use PyTorch's CPU path as the backend-independent renorm reference.
        input = input.cpu()
        weight = weight.cpu()

    result = torch.nn.functional.embedding(
        input,
        weight,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
    )
    out.copy_(result.to(out.device))

    return out
