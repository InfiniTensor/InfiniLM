import infini.ops
import pytest
import torch

from tests.utils import get_stream


_TEST_CASES = (
    (4, 4, "none", 0.0, False, None, False),
    (3, 5, "float", 0.0, False, 0.5, False),
    (4, 4, "bool", 0.0, False, None, False),
    (4, 4, "none", 0.0, True, None, False),
    (3, 5, "none", 0.0, False, None, True),
    (4, 4, "none", 0.2, False, None, False),
)


@pytest.mark.parametrize(
    "query_length, key_length, mask_kind, dropout_p, is_causal, scale, enable_gqa",
    _TEST_CASES,
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 2e-2, 1e-2),
    ),
)
def test_scaled_dot_product_attention(
    query_length,
    key_length,
    mask_kind,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    batch_size, query_heads, head_size, value_size = 2, 4, 8, 6
    kv_heads = 2 if enable_gqa else query_heads
    query = torch.randn(
        batch_size,
        query_heads,
        query_length,
        head_size,
        dtype=dtype,
        device=device,
    )
    key = torch.randn(
        batch_size,
        kv_heads,
        key_length,
        head_size,
        dtype=dtype,
        device=device,
    )
    value = torch.randn(
        batch_size,
        kv_heads,
        key_length,
        value_size,
        dtype=dtype,
        device=device,
    )
    attn_mask = _make_mask(mask_kind, query_length, key_length, dtype, device)
    out = torch.empty(
        batch_size,
        query_heads,
        query_length,
        value_size,
        dtype=dtype,
        device=device,
    )

    torch.manual_seed(1234)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    torch.manual_seed(1234)
    if (
        attn_mask is None
        and dropout_p == 0.0
        and not is_causal
        and scale is None
        and not enable_gqa
    ):
        result = infini.ops.scaled_dot_product_attention(
            query,
            key,
            value,
            out,
            implementation_index=implementation_index,
            stream=get_stream(query.device),
        )
    else:
        result = infini.ops.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            out,
            implementation_index=implementation_index,
            stream=get_stream(query.device),
        )

    assert result is None
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


def _make_mask(mask_kind, query_length, key_length, dtype, device):
    if mask_kind == "none":
        return None

    if mask_kind == "bool":
        mask = torch.ones(query_length, key_length, dtype=torch.bool, device=device)
        mask[0, -1] = False

        return mask

    mask = torch.zeros(query_length, key_length, dtype=dtype, device=device)
    mask[0, -1] = -3.0

    return mask
