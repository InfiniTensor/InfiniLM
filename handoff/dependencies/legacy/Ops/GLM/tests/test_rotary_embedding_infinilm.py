import infini.ops
import pytest
import torch

from tests.utils import Payload, empty_strided, get_stream, randn_strided


def _compute_sin_cos_table(pos_ids, head_dim, theta=10000.0, dtype=torch.float32):
    """Compute sin and cos tables for RoPE."""
    assert head_dim % 2 == 0, "Head dimension must be even"
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(pos_ids.float(), freqs)
    sin_table = torch.sin(angles).to(dtype)
    cos_table = torch.cos(angles).to(dtype)
    return sin_table, cos_table


def _torch_rope_standard(input_tensor, sin, cos):
    """Standard RoPE style: first half and second half"""
    dh = input_tensor.shape[-1]
    half_dim = dh // 2

    t_first = input_tensor[..., :half_dim]
    t_second = input_tensor[..., half_dim:]

    t_out_first = t_first * cos - t_second * sin
    t_out_second = t_first * sin + t_second * cos

    return torch.cat([t_out_first, t_out_second], dim=-1)


def _torch_rope_neox_interleaved(input_tensor, sin, cos):
    """GPT-NeoX style RoPE: interleaved [even, odd] pairs"""
    dh = input_tensor.shape[-1]
    assert dh % 2 == 0, "Embedding dimension must be even."

    t_even = input_tensor[..., 0::2]
    t_odd = input_tensor[..., 1::2]

    t_out_even = t_even * cos - t_odd * sin
    t_out_odd = t_even * sin + t_odd * cos

    output = torch.empty_like(input_tensor)
    output[..., 0::2] = t_out_even
    output[..., 1::2] = t_out_odd

    return output


def _torch_rotary_embedding_infinilm(
    input_tensor, pos_ids, sin_table, cos_table, is_neox=False
):
    """PyTorch reference implementation of RoPE."""
    target_dtype = input_tensor.dtype

    has_batch_dim = input_tensor.dim() == 4

    # Gather sin/cos values
    sin_gathered = sin_table[pos_ids]
    cos_gathered = cos_table[pos_ids]

    # Expand dimensions for broadcasting
    if has_batch_dim and pos_ids.dim() == 2:
        # pos_ids: [batch, seq_len]
        # sin_gathered: [batch, seq_len, half_dim]
        # Need: [batch, seq_len, 1, half_dim]
        sin_expanded = sin_gathered.unsqueeze(2)
        cos_expanded = cos_gathered.unsqueeze(2)
    elif has_batch_dim:
        # pos_ids: [seq_len]
        # sin_gathered: [seq_len, half_dim]
        # Need: [1, seq_len, 1, half_dim]
        sin_expanded = sin_gathered.unsqueeze(0).unsqueeze(2)
        cos_expanded = cos_gathered.unsqueeze(0).unsqueeze(2)
    else:
        # pos_ids: [seq_len]
        # sin_gathered: [seq_len, half_dim]
        # Need: [seq_len, 1, half_dim]
        sin_expanded = sin_gathered.unsqueeze(1)
        cos_expanded = cos_gathered.unsqueeze(1)

    # Apply RoPE
    if is_neox:
        output = _torch_rope_neox_interleaved(input_tensor, sin_expanded, cos_expanded)
    else:
        output = _torch_rope_standard(input_tensor, sin_expanded, cos_expanded)

    return output.to(dtype=target_dtype)


def _rotary_embedding_infinilm(
    input_tensor, pos_ids, sin_table, cos_table, *, out=None, stream=None, is_neox=False
):
    """Wrapper for calling infini.ops.rotary_embedding_infinilm."""
    infini.ops.rotary_embedding_infinilm(
        input_tensor,
        pos_ids,
        sin_table,
        cos_table,
        is_neox,
        out,
        stream=get_stream(input_tensor.device) if stream is None else stream,
    )

    return out


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, input_strides, out_strides, is_neox",
    (
        # 3D tensors (no batch dimension) - NeoX style (interleaved)
        ((32, 8, 64), None, None, True),
        ((128, 12, 128), None, None, True),
        # 3D tensors - Standard style (first/second half)
        ((32, 8, 64), None, None, False),
        ((128, 12, 128), None, None, False),
        # 4D tensors (with batch dimension) - NeoX style
        ((4, 32, 8, 64), None, None, True),
        ((8, 128, 12, 128), None, None, True),
        # 4D tensors - Standard style
        ((4, 32, 8, 64), None, None, False),
        ((8, 128, 12, 128), None, None, False),
        # With custom strides
        ((4, 32, 8, 64), (16384, 512, 64, 1), (16384, 512, 64, 1), True),
        ((32, 8, 64), (2048, 64, 1), (2048, 64, 1), False),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 5e-2, 2e-2),
    ),
)
def test_rotary_embedding_infinilm(
    shape, input_strides, out_strides, is_neox, dtype, device, rtol, atol
):
    """Test Rotary Positional Embedding operator."""
    input_tensor = randn_strided(shape, input_strides, dtype=dtype, device=device)
    out = empty_strided(shape, out_strides, dtype=dtype, device=device)

    seq_len = shape[-3]
    pos_ids = torch.arange(seq_len, dtype=torch.int32, device=device)

    head_dim = shape[-1]
    sin_table, cos_table = _compute_sin_cos_table(
        pos_ids.cpu(), head_dim, theta=10000.0, dtype=dtype
    )
    sin_table = sin_table.to(device)
    cos_table = cos_table.to(device)

    ref_output = _torch_rotary_embedding_infinilm(
        input_tensor, pos_ids, sin_table, cos_table, is_neox
    )

    return Payload(
        lambda *args, **kwargs: _rotary_embedding_infinilm(
            *args, **kwargs, is_neox=is_neox
        ),
        lambda *args, **kwargs: ref_output,
        (input_tensor, pos_ids, sin_table, cos_table),
        {"out": out},
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "shape, is_neox, theta",
    (
        ((1, 32, 12, 64), True, 10000.0),
        ((1, 32, 12, 64), False, 50000.0),
        ((32, 12, 64), True, 100000.0),
        ((32, 12, 64), False, 10000.0),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 5e-2, 2e-2),
    ),
)
def test_rotary_embedding_infinilm_different_theta(
    shape, is_neox, theta, dtype, device, rtol, atol
):
    """Test RoPE with different theta values."""
    input_tensor = randn_strided(shape, None, dtype=dtype, device=device)
    out = empty_strided(shape, None, dtype=dtype, device=device)

    seq_len = shape[-3]
    pos_ids = torch.arange(seq_len, dtype=torch.int32, device=device)

    head_dim = shape[-1]
    sin_table, cos_table = _compute_sin_cos_table(
        pos_ids.cpu(), head_dim, theta=theta, dtype=dtype
    )
    sin_table = sin_table.to(device)
    cos_table = cos_table.to(device)

    ref_output = _torch_rotary_embedding_infinilm(
        input_tensor, pos_ids, sin_table, cos_table, is_neox
    )

    return Payload(
        lambda *args, **kwargs: _rotary_embedding_infinilm(
            *args, **kwargs, is_neox=is_neox
        ),
        lambda *args, **kwargs: ref_output,
        (input_tensor, pos_ids, sin_table, cos_table),
        {"out": out},
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 5e-2, 2e-2),
    ),
)
def test_rotary_embedding_infinilm_inplace(device, dtype, rtol, atol):
    """Test in-place RoPE operation."""
    shape = (4, 32, 8, 64)

    input_tensor = randn_strided(shape, None, dtype=dtype, device=device)
    out = input_tensor.clone()

    pos_ids = torch.arange(shape[-3], dtype=torch.int32, device=device)

    head_dim = shape[-1]
    sin_table, cos_table = _compute_sin_cos_table(pos_ids.cpu(), head_dim, dtype=dtype)
    sin_table = sin_table.to(device)
    cos_table = cos_table.to(device)

    input_copy = input_tensor.clone()
    ref_output = _torch_rotary_embedding_infinilm(
        input_copy, pos_ids, sin_table, cos_table, is_neox=False
    )

    return Payload(
        lambda *args, **kwargs: _rotary_embedding_infinilm(
            *args, **kwargs, is_neox=False
        ),
        lambda *args, **kwargs: ref_output,
        (input_tensor, pos_ids, sin_table, cos_table),
        {"out": out},
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    "seq_len, nhead, head_dim",
    (
        (1, 1, 2),
        (1, 4, 64),
        (2, 1, 128),
        (1, 1, 128),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 1e-5),
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 5e-2, 2e-2),
    ),
)
def test_rotary_embedding_infinilm_edge_cases(
    seq_len, nhead, head_dim, dtype, device, rtol, atol
):
    """Test edge cases: minimum dimensions."""
    shape = (seq_len, nhead, head_dim)

    input_tensor = randn_strided(shape, None, dtype=dtype, device=device)
    out = empty_strided(shape, None, dtype=dtype, device=device)

    pos_ids = torch.arange(seq_len, dtype=torch.int32, device=device)
    sin_table, cos_table = _compute_sin_cos_table(pos_ids.cpu(), head_dim, dtype=dtype)
    sin_table = sin_table.to(device)
    cos_table = cos_table.to(device)

    ref_output = _torch_rotary_embedding_infinilm(
        input_tensor, pos_ids, sin_table, cos_table, is_neox=False
    )

    return Payload(
        lambda *args, **kwargs: _rotary_embedding_infinilm(
            *args, **kwargs, is_neox=False
        ),
        lambda *args, **kwargs: ref_output,
        (input_tensor, pos_ids, sin_table, cos_table),
        {"out": out},
        rtol=rtol,
        atol=atol,
    )
