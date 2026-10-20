import infini.ops
import pytest
import torch

from tests.utils import clone_strided, get_stream, randn_strided


@pytest.mark.parametrize(
    "cache_shape, cache_strides, seq_len",
    (
        ((1, 1, 8, 1), None, 3),
        ((1, 8, 32, 32), None, 7),
        ((8, 8, 64, 32), None, 5),
        ((1, 32, 8, 64), (32768, 1024, 64, 1), 4),
        ((4, 8, 32, 16), (65536, 8192, 256, 16), 7),
        ((8, 16, 64, 128), (8388608, 524288, 8192, 1), 3),
        ((1, 2, 2304, 128), (589824, 294912, 128, 1), 9),
    ),
)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("index_dtype", (torch.int64, torch.int32))
def test_kv_caching_infinilm(
    cache_shape, cache_strides, seq_len, dtype, index_dtype, device
):
    batch, heads, max_seq_len, hidden = cache_shape
    k_cache = randn_strided(cache_shape, cache_strides, dtype=dtype, device=device)
    v_cache = randn_strided(cache_shape, cache_strides, dtype=dtype, device=device)
    k = randn_strided((batch, heads, seq_len, hidden), None, dtype=dtype, device=device)
    v = randn_strided((batch, heads, seq_len, hidden), None, dtype=dtype, device=device)
    past_kv_lengths = _make_past_lengths(
        batch, max_seq_len - seq_len, dtype=index_dtype, device=device
    )

    expected_k_cache = clone_strided(k_cache)
    expected_v_cache = clone_strided(v_cache)
    _torch_kv_caching_infinilm(
        expected_k_cache, expected_v_cache, k, v, past_kv_lengths
    )

    infini.ops.kv_caching_infinilm(
        k,
        v,
        past_kv_lengths,
        k_cache,
        v_cache,
        stream=get_stream(k_cache.device),
    )

    torch.testing.assert_close(k_cache, expected_k_cache, rtol=0, atol=0)
    torch.testing.assert_close(v_cache, expected_v_cache, rtol=0, atol=0)


def _make_past_lengths(batch, high, *, dtype, device):
    values = torch.arange(batch, dtype=torch.int64, device=device) % max(high, 1)
    return values.to(dtype)


def _torch_kv_caching_infinilm(k_cache, v_cache, k, v, past_kv_lengths):
    batch, heads, _, _ = k_cache.shape
    seq_len = k.shape[2]
    for b in range(batch):
        past_len = int(past_kv_lengths[b].item())
        for h in range(heads):
            k_cache[b, h, past_len : past_len + seq_len, :] = k[b, h, :, :]
            v_cache[b, h, past_len : past_len + seq_len, :] = v[b, h, :, :]
