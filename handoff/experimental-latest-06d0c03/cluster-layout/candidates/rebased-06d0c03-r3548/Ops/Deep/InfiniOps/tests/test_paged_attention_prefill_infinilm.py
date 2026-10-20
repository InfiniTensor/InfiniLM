import infini.ops
import pytest
import torch

from tests.utils import get_stream


class SimpleCacheManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.request_to_blocks = {}
        self.request_to_len = {}

    def allocate_slots(self, request_id, num_new_tokens):
        if request_id not in self.request_to_len:
            self.request_to_len[request_id] = 0
            self.request_to_blocks[request_id] = []

        start_pos = self.request_to_len[request_id]
        new_total_len = start_pos + num_new_tokens
        needed_blocks = (new_total_len + self.block_size - 1) // self.block_size
        added_blocks = needed_blocks - len(self.request_to_blocks[request_id])

        for _ in range(added_blocks):
            self.request_to_blocks[request_id].append(self.free_blocks.pop(0))

        self.request_to_len[request_id] = new_total_len
        return self.request_to_blocks[request_id], new_total_len


def ref_paged_attention_prefill_infinilm(
    q, k_cache, v_cache, block_tables, seq_lens, cum_seq_lens_q, scale
):
    block_size = k_cache.shape[2]
    outputs = torch.zeros_like(q)
    num_seqs = cum_seq_lens_q.numel() - 1

    for seq_id in range(num_seqs):
        q_begin = cum_seq_lens_q[seq_id].item()
        q_end = cum_seq_lens_q[seq_id + 1].item()
        num_new = q_end - q_begin
        total_len = seq_lens[seq_id].item()
        history_len = total_len - num_new

        table = block_tables[seq_id]
        keys = []
        values = []
        for pos in range(total_len):
            block_id = table[pos // block_size].item()
            block_offset = pos % block_size
            keys.append(k_cache[block_id, :, block_offset, :])
            values.append(v_cache[block_id, :, block_offset, :])

        k = torch.stack(keys, dim=0)
        v = torch.stack(values, dim=0)
        q_seq = q[q_begin:q_end]

        scores = torch.einsum("qhd,khd->hqk", q_seq, k).float() * scale
        mask = torch.full((num_new, total_len), float("-inf"), device=q.device)
        for q_idx in range(num_new):
            mask[q_idx, : history_len + q_idx + 1] = 0.0

        weights = torch.softmax(scores + mask.unsqueeze(0), dim=-1).to(q.dtype)
        outputs[q_begin:q_end] = torch.einsum("hqk,khd->qhd", weights, v)

    return outputs


@pytest.mark.parametrize(
    (
        "num_seqs",
        "num_heads",
        "num_kv_heads",
        "head_size",
        "block_size",
        "max_step_len",
        "num_rounds",
        "index_dtype",
    ),
    (
        (1, 1, 1, 128, 8, 16, 1, torch.int32),
        (1, 1, 1, 128, 8, 16, 1, torch.int64),
        (1, 4, 4, 128, 8, 16, 4, torch.int32),
        (1, 4, 4, 128, 8, 16, 4, torch.int64),
        (2, 8, 8, 128, 16, 32, 2, torch.int32),
        (2, 8, 8, 128, 16, 32, 2, torch.int64),
        (4, 16, 16, 128, 8, 64, 3, torch.int32),
        (4, 16, 16, 128, 8, 64, 3, torch.int64),
        (8, 64, 64, 128, 8, 16, 5, torch.int32),
        (8, 64, 64, 128, 8, 16, 5, torch.int64),
        (16, 128, 128, 128, 8, 16, 4, torch.int32),
        (16, 128, 128, 128, 8, 16, 4, torch.int64),
    ),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float16, 1e-2, 1e-2),
        (torch.bfloat16, 2e-2, 2e-2),
    ),
)
def test_paged_attention_prefill_infinilm(
    num_seqs,
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    max_step_len,
    num_rounds,
    index_dtype,
    implementation_index,
    dtype,
    device,
    rtol,
    atol,
):
    max_tokens = num_seqs * max_step_len * num_rounds
    num_blocks = (max_tokens + block_size - 1) // block_size + num_seqs + 4
    manager = SimpleCacheManager(num_blocks, block_size)
    scale = head_size**-0.5

    k_cache = torch.empty(
        (num_blocks, num_kv_heads, block_size, head_size),
        dtype=dtype,
        device=device,
    )
    v_cache = torch.empty_like(k_cache)

    for _ in range(num_rounds):
        query_lens = torch.randint(1, max_step_len + 1, (num_seqs,))
        q_total_tokens = query_lens.sum().item()
        q = torch.empty(
            (q_total_tokens, num_heads, head_size), dtype=dtype, device=device
        )

        seq_lens_list = []
        block_tables_list = []
        cum_seq_lens_q = [0]

        for seq_id in range(num_seqs):
            cur_q_len = query_lens[seq_id].item()
            table, total_len = manager.allocate_slots(seq_id, cur_q_len)
            history_len = total_len - cur_q_len
            seq_lens_list.append(total_len)
            block_tables_list.append(table)

            k_new = torch.randn(
                (cur_q_len, num_kv_heads, head_size), dtype=dtype, device=device
            )
            v_new = torch.randn_like(k_new)
            q_new = torch.randn(
                (cur_q_len, num_heads, head_size), dtype=dtype, device=device
            )
            q_begin = cum_seq_lens_q[-1]
            q[q_begin : q_begin + cur_q_len] = q_new

            for token_idx in range(cur_q_len):
                logical_pos = history_len + token_idx
                block_id = table[logical_pos // block_size]
                block_offset = logical_pos % block_size
                k_cache[block_id, :, block_offset, :] = k_new[token_idx]
                v_cache[block_id, :, block_offset, :] = v_new[token_idx]

            cum_seq_lens_q.append(q_begin + cur_q_len)

        max_blocks = max(len(table) for table in block_tables_list)
        padded_tables = [
            table + [0] * (max_blocks - len(table)) for table in block_tables_list
        ]
        block_tables = torch.tensor(padded_tables, dtype=index_dtype, device=device)
        seq_lens = torch.tensor(seq_lens_list, dtype=index_dtype, device=device)
        cum_seq_lens_q = torch.tensor(cum_seq_lens_q, dtype=index_dtype, device=device)
        out = torch.empty_like(q)

        actual = _paged_attention_prefill_infinilm(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            cum_seq_lens_q,
            scale=scale,
            out=out,
            implementation_index=implementation_index,
        )
        expected = _torch_paged_attention_prefill_infinilm(
            q,
            k_cache,
            v_cache,
            block_tables,
            seq_lens,
            cum_seq_lens_q,
            scale=scale,
        )
        assert torch.allclose(actual, expected, rtol=rtol, atol=atol)


def _paged_attention_prefill_infinilm(
    q,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    cum_seq_lens_q,
    *,
    scale,
    out=None,
    implementation_index=0,
):
    infini.ops.paged_attention_prefill_infinilm(
        q,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        cum_seq_lens_q,
        None,
        scale,
        out,
        implementation_index=implementation_index,
        stream=get_stream(q.device),
    )

    return out


def _torch_paged_attention_prefill_infinilm(
    q,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    cum_seq_lens_q,
    *,
    scale,
    out=None,
):
    result = ref_paged_attention_prefill_infinilm(
        q, k_cache, v_cache, block_tables, seq_lens, cum_seq_lens_q, scale
    )

    if out is not None:
        out.copy_(result)
    else:
        out = result

    return out
