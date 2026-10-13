#pragma once

#include "../../../backends/attention_backends.hpp"
#include "../../../global_state/global_state.hpp"
#include "../../../global_state/forward_context.hpp"
#include "flash_attn.hpp"
#include "runtime/tensor.hpp"
#include "paged_attn.hpp"
#include "static_attn.hpp"
#include <memory>
#include <variant>
#include <cstdint>
#include <vector>

namespace infinilm::layers::attention {
using AttentionImpl = std::variant<std::shared_ptr<backends::StaticAttentionImpl>, std::shared_ptr<backends::PagedAttentionImpl>, std::shared_ptr<backends::FlashAttentionImpl>>;

/**
 * @brief Attention layer.
 * This class takes query, key, and value tensors as input.
 * The input tensors can either contain prompt tokens or generation tokens.
 *
 * The class does the following:
 * - Update the KV cache.
 * - Perform (multi-head/multi-query/grouped-query) attention.
 * - Return the output tensor.
 */
class AttentionLayer {
public:
    AttentionLayer(size_t num_heads,
                   size_t head_size,
                   float scale,
                   size_t num_kv_heads,
                   size_t layer_idx,
                   infinilm::runtime::Tensor k_scale,
                   infinilm::runtime::Tensor v_scale,
                   ::infinilm::backends::AttentionBackend attention_backend,
                   infinilm::quantization::KVQuantAlgo kv_quant_scheme,
                   std::vector<int64_t> window_size = {-1, -1});

    infinilm::runtime::Tensor forward(
        infinilm::runtime::Tensor &query,
                               infinilm::runtime::Tensor &key,
        infinilm::runtime::Tensor &value,
        infinilm::global_state::ForwardContext &forward_context) const;

    // LM composition for eager prefill with live device KV lengths.
    void prefill_paged(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor out,
                      const infinilm::runtime::Tensor &q, const infinilm::runtime::Tensor &k_cache,
                      const infinilm::runtime::Tensor &v_cache, const infinilm::runtime::Tensor &block_tables,
                      const infinilm::runtime::Tensor &kv_lens, const infinilm::runtime::Tensor &cu_q,
                      float scale) const;

    inline infinilm::runtime::Tensor get_k_scale() const { return k_scale_; }
    inline infinilm::runtime::Tensor get_v_scale() const { return v_scale_; }
    const std::vector<int64_t> &window_size() const { return window_size_; }

private:
    const std::vector<int64_t> window_size_;
    infinilm::runtime::Tensor k_scale_;
    infinilm::runtime::Tensor v_scale_;
    size_t layer_idx_;
    AttentionImpl attn_backend_impl_;
    ::infinilm::backends::AttentionBackend attn_backend_;
};
} // namespace infinilm::layers::attention
