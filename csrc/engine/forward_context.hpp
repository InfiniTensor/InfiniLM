#pragma once

#include "../models/infinilm_model.hpp"

namespace infinilm::engine {
/*
AttentionMetadata:
https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backend.py

FlashAttentionMetadata:
https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py
*/
using AttentionMetadata = infinilm::InfinilmModel::Input;

/*
https://github.com/vllm-project/vllm/blob/main/vllm/forward_context.py
@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[str, Any]
    attn_metadata: dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]
*/
struct ForwardContext {
    AttentionMetadata attn_metadata;
    std::vector<std::tuple<infinicore::Tensor, infinicore::Tensor>> kv_cache_vec; // 显示shape信息
};

/*
https://github.com/vllm-project/vllm/blob/main/vllm/forward_context.py

@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: VllmConfig,
    virtual_engine: int = 0,
    num_tokens: int | None = None,
    num_tokens_across_dp: torch.Tensor | None = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    ubatch_slices: UBatchSlices | None = None,
    slot_mapping: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None = None,
    skip_compiled: bool = False,
):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    pass
*/

void set_forward_context(const infinilm::InfinilmModel::Input &input);

ForwardContext &get_forward_context();

} // namespace infinilm::engine