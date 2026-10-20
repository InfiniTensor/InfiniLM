import math

import infini.ops
import pytest
import torch

from tests.utils import Payload, get_stream


def get_alibi_slopes(n):
    closest_power_of_2 = 2 ** math.floor(math.log2(n))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = [base**i for i in range(1, closest_power_of_2 + 1)]
    if n > closest_power_of_2:
        extra = [base ** (i * 2) for i in range(1, 2 * (n - closest_power_of_2) + 1, 2)]
        powers += extra
    return powers[:n]


def ref_paged_attention_infinilm(
    q, k_cache, v_cache, block_tables, seq_lens, alibi_slopes, scale
):
    output = torch.empty_like(q)
    num_heads = q.shape[1]
    num_kv_heads = k_cache.shape[1]
    queries_per_kv = num_heads // num_kv_heads
    block_size = k_cache.shape[2]

    for seq_id in range(q.shape[0]):
        seq_len = seq_lens[seq_id].item()
        table = block_tables[seq_id]
        keys = []
        values = []
        for token_idx in range(seq_len):
            block_id = table[token_idx // block_size].item()
            block_offset = token_idx % block_size
            keys.append(k_cache[block_id, :, block_offset, :])
            values.append(v_cache[block_id, :, block_offset, :])

        k = torch.stack(keys, dim=0)
        v = torch.stack(values, dim=0)
        if queries_per_kv > 1:
            k = torch.repeat_interleave(k, queries_per_kv, dim=1)
            v = torch.repeat_interleave(v, queries_per_kv, dim=1)

        scores = torch.einsum("hd,khd->hk", q[seq_id], k).float() * scale
        if alibi_slopes is not None:
            pos = torch.arange(seq_len, device=q.device, dtype=torch.float32)
            scores = scores + alibi_slopes.view(-1, 1) * (pos - seq_len + 1)

        weights = torch.softmax(scores, dim=-1).to(q.dtype)
        output[seq_id] = torch.einsum("hk,khd->hd", weights, v)

    return output


@pytest.mark.auto_act_and_assert
@pytest.mark.parametrize(
    (
        "num_seqs",
        "num_heads",
        "num_kv_heads",
        "head_size",
        "block_size",
        "max_seq_len",
        "use_alibi",
    ),
    (
        (1, 1, 1, 128, 16, 1024, False),
        (4, 40, 40, 128, 16, 1024, False),
        (6, 40, 40, 128, 16, 1024, False),
        (3, 8, 8, 128, 16, 1024, False),
        (3, 8, 8, 64, 16, 1024, False),
        (8, 64, 8, 128, 16, 2048, False),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-2, 1e-3),
        (torch.bfloat16, 5e-2, 5e-3),
    ),
)
def test_paged_attention_infinilm(
    num_seqs,
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    max_seq_len,
    use_alibi,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    scale = head_size**-0.5
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks_per_seq

    q = torch.randn((num_seqs, num_heads, head_size), dtype=dtype, device=device)
    out = torch.empty_like(q)
    k_cache = torch.randn(
        (num_blocks, num_kv_heads, block_size, head_size), dtype=dtype, device=device
    )
    v_cache = torch.randn_like(k_cache)
    seq_lens = torch.randint(
        1, max_seq_len, (num_seqs,), dtype=torch.int64, device=device
    )
    block_tables = torch.arange(num_blocks, dtype=torch.int64, device=device).view(
        num_seqs, max_blocks_per_seq
    )
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.tensor(
            get_alibi_slopes(num_heads), dtype=torch.float32, device=device
        )

    return Payload(
        lambda *args, **kwargs: _paged_attention_infinilm(
            *args, **kwargs, implementation_index=implementation_index
        ),
        _torch_paged_attention_infinilm,
        (q, k_cache, v_cache, block_tables, seq_lens, alibi_slopes),
        {"scale": scale, "out": out},
        rtol=rtol,
        atol=atol,
    )


def _paged_attention_infinilm(
    q,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    alibi_slopes,
    *,
    scale,
    out=None,
    implementation_index=0,
):
    infini.ops.paged_attention_infinilm(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        alibi_slopes,
        scale,
        out,
        implementation_index=implementation_index,
        stream=get_stream(q.device),
    )

    return out


def _torch_paged_attention_infinilm(
    q,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    alibi_slopes,
    *,
    scale,
    out=None,
):
    result = ref_paged_attention_infinilm(
        q, k_cache, v_cache, block_tables, seq_lens, alibi_slopes, scale
    )

    if out is not None:
        out.copy_(result)
    else:
        out = result

    return out
