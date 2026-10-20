import infini.ops
import pytest
import torch

from tests.utils import get_stream


_PREFILL_CASES = (
    (1, 128, 8, 128, 16),
    (5, 512, 40, 128, 16),
    (16, 1024, 8, 64, 32),
    (10, 1024, 40, 64, 32),
)

_INCREMENTAL_CASES = (
    (1, 16, 1, 128, 8, 5),
    (2, 64, 8, 128, 16, 2),
    (8, 128, 32, 128, 16, 3),
    (5, 512, 40, 128, 16, 3),
    (16, 64, 8, 128, 32, 1),
    (10, 256, 40, 128, 32, 3),
)


class _SimpleCacheManager:
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

        slots = []
        for i in range(start_pos, new_total_len):
            block_idx_in_seq = i // self.block_size
            block_offset = i % self.block_size
            physical_block_id = self.request_to_blocks[request_id][block_idx_in_seq]
            slots.append(physical_block_id * self.block_size + block_offset)

        self.request_to_len[request_id] = new_total_len

        return torch.tensor(slots, dtype=torch.int64)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-5, 0.0),
        (torch.float16, 1e-5, 0.0),
        (torch.bfloat16, 1e-5, 0.0),
    ),
)
@pytest.mark.parametrize(
    "num_seqs, max_seq_len, num_kv_heads, head_size, block_size", _PREFILL_CASES
)
def test_paged_caching_infinilm_prefill(
    num_seqs,
    max_seq_len,
    num_kv_heads,
    head_size,
    block_size,
    dtype,
    device,
    rtol,
    atol,
):
    context_lens = torch.randint(1, max_seq_len + 1, (num_seqs,), dtype=torch.int64)
    ntok = int(context_lens.sum().item())

    slot_mapping_list = []
    current_slot = 0
    for length in context_lens:
        slot_mapping_list.extend(range(current_slot, current_slot + int(length.item())))
        current_slot += int(length.item())

    num_blocks = (current_slot + block_size - 1) // block_size + 4
    slot_mapping = torch.tensor(slot_mapping_list, dtype=torch.int64, device=device)
    k = torch.randn(ntok, num_kv_heads, head_size, dtype=dtype, device=device)
    v = torch.randn(ntok, num_kv_heads, head_size, dtype=dtype, device=device)
    k_cache = torch.randn(
        num_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=device
    )
    v_cache = torch.randn(
        num_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=device
    )

    k_ref = k_cache.detach().clone()
    v_ref = v_cache.detach().clone()
    _ref_paged_caching_infinilm(k_ref, v_ref, k, v, slot_mapping.cpu(), block_size)

    infini.ops.paged_caching_infinilm(
        k, v, slot_mapping, k_cache, v_cache, stream=get_stream(k.device)
    )

    assert torch.allclose(k_cache, k_ref, rtol=rtol, atol=atol)
    assert torch.allclose(v_cache, v_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    (
        (torch.float32, 1e-8, 1e-8),
        (torch.float16, 1e-8, 1e-8),
        (torch.bfloat16, 1e-8, 1e-8),
    ),
)
@pytest.mark.parametrize(
    "num_seqs, max_step_len, num_kv_heads, head_size, block_size, num_rounds",
    _INCREMENTAL_CASES,
)
def test_paged_caching_infinilm_incremental(
    num_seqs,
    max_step_len,
    num_kv_heads,
    head_size,
    block_size,
    num_rounds,
    dtype,
    device,
    rtol,
    atol,
):
    max_slots = num_seqs * max_step_len * num_rounds
    num_blocks = (max_slots + block_size - 1) // block_size + num_seqs + 4
    manager = _SimpleCacheManager(num_blocks, block_size)

    k_cache = torch.randn(
        num_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=device
    )
    v_cache = torch.randn(
        num_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=device
    )
    k_ref = k_cache.detach().clone()
    v_ref = v_cache.detach().clone()

    for _ in range(num_rounds):
        round_ntok = torch.randint(1, max_step_len + 1, (num_seqs,), dtype=torch.int64)
        slots = []
        keys = []
        values = []

        for i in range(num_seqs):
            n_new = int(round_ntok[i].item())
            slots.append(manager.allocate_slots(i, n_new))
            keys.append(torch.randn(n_new, num_kv_heads, head_size))
            values.append(torch.randn(n_new, num_kv_heads, head_size))

        k_cpu = torch.cat(keys, dim=0)
        v_cpu = torch.cat(values, dim=0)
        slot_mapping_cpu = torch.cat(slots, dim=0)
        k = k_cpu.to(dtype=dtype, device=device)
        v = v_cpu.to(dtype=dtype, device=device)
        slot_mapping = slot_mapping_cpu.to(device=device)

        _ref_paged_caching_infinilm(k_ref, v_ref, k, v, slot_mapping_cpu, block_size)
        infini.ops.paged_caching_infinilm(
            k, v, slot_mapping, k_cache, v_cache, stream=get_stream(k.device)
        )

        assert torch.allclose(k_cache, k_ref, rtol=rtol, atol=atol)
        assert torch.allclose(v_cache, v_ref, rtol=rtol, atol=atol)


def _ref_paged_caching_infinilm(k_cache, v_cache, k, v, slot_mapping_cpu, block_size):
    for i in range(k.shape[0]):
        slot = int(slot_mapping_cpu[i].item())
        block_idx = slot // block_size
        block_offset = slot % block_size
        k_cache[block_idx, :, block_offset, :] = k[i]
        v_cache[block_idx, :, block_offset, :] = v[i]
