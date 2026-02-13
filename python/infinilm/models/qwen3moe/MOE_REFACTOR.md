# Qwen3 MoE refactor (infinicore)

This refactor introduces an infinicore version of the `Qwen3MoeSparseMoeBlock` at `python/infinilm/models/qwen3moe/qwen3moe.py`. The block now subclasses `infinicore.nn.Module`, stores weights as `infinicore.nn.Parameter`, and exposes the same class names as the Torch reference for drop-in usage.

## What changed
- Router uses `infinicore.nn.functional.linear` for the projection and a Python softmax+top-k shim to select experts, returning both scores and indices.
- Experts run gate/up/down projections with the stored weights but rely on NumPy to emulate routing utilities (one-hot masks, scatter add) that are not present in infinicore yet.
- The sparse MoE block returns `(hidden_states, routing_weights)` shaped back to `(batch, seq, hidden_dim)` and `(batch, seq, top_k)` respectively to surface the gate decisions.

## Missing operators and temporary shims
The following Torch ops are not available in infinicore today and are emulated in pure Python/NumPy inside `qwen3moe.py`:
- Softmax over the expert dimension.
- `topk` selection of experts.
- One-hot expansion of expert indices.
- Scatter/add (`index_add_`) to accumulate expert outputs.
- Boolean masking utilities (`where`/`nonzero`) used for routing.

All shims use `_tensor_to_numpy` to bridge an infinicore tensor to NumPy and `_from_numpy_like` to move results back while keeping device/dtype. Replace these with native infinicore kernels once they land to regain performance.

## Notes and next steps
- Activation currently supports `silu`/`swish`, `gelu`, and `relu`. Extend `_activation_fn` if the config uses other functions.
- Weight initialization mirrors the Torch reference (`empty` for expert matrices, `zeros` for router weights); hook up a proper initializer if required.
- When infinicore adds native softmax/top-k/one-hot/scatter, the Python shims can be deleted and the routing path can stay entirely on-device.
