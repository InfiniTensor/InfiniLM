# Paged KV layout: torch prefill ↔ C++ `FlashAttentionImpl`

Reference: `InfiniLM/csrc/layers/attention/backends/flash_attn.cpp`, `infer_engine.py` paged `generate()`.

## C++ paged KV cache tensor (per layer)

After `do_kv_cache_update`, flash-attn reads:

| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| `k_cache_layer` | `[num_blocks, block_size, num_kv_heads, head_dim]` | model / kv dtype | BHSD layout after internal permute for `paged_caching_` |
| `v_cache_layer` | `[num_blocks, block_size, num_kv_heads, head_dim]` | same | same |

`kv_cache` wrapper shape includes layer dimension: `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]` (K/V split on dim 0).

## Prefill metadata (batch=1, seq=`S`)

| Field | Shape | Dtype | Value @ prefill |
|-------|-------|-------|-----------------|
| `input_ids` | `[1, S]` then viewed `[1, S]` | int64 | token ids |
| `position_ids` | `[S]` | int64 | `0 .. S-1` |
| `past_sequence_lengths` | `[1]` | int32 | `0` |
| `total_sequence_lengths` | `[1]` | int32 | `S` |
| `input_offsets` | `[2]` | int32 | `0`, `S` |
| `cu_seqlens` | `[2]` | int32 | `0`, `S` |
| `block_tables` | `[1, max_blocks_per_batch]` | int32 | contiguous block ids `0 .. max_blocks-1` |
| `slot_mapping` | `[S]` | int64 | `block_id * block_size + offset_in_block` for each token |

With `block_size=256` and batch 0 only:

```
slot_mapping[i] = i   # when block_table[0] starts at block 0
```

## torch prefill runner (Phase 3+)

Until hybrid wiring lands, torch prefill should:

1. Run attention with `flash_attn_varlen_func` / FA2 prefill writing K/V in the same slot order as `slot_mapping`.
2. Or call `paged_caching_`-compatible writes per layer after projecting K/V.

Phase 0 parity script validates **logits** only; KV byte compare is Phase 3.

## 9g 8B constants (`config.json`)

- `hidden_size=4096`, `num_attention_heads=32`, `num_key_value_heads=2`, `head_dim=128`
- `num_hidden_layers=32`, `vocab_size=73448`, `torch_dtype=bfloat16`
- RoPE: LongRoPE (`rope_scaling.type=longrope`)
