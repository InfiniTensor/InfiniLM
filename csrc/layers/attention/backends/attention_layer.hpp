#pragma once

#include "../../../backends/attention_backends.hpp"
#include "../../../global_state/global_state.hpp"
#include "flash_attn.hpp"
#include "infinicore/tensor.hpp"
#include "paged_attn.hpp"
#include "static_attn.hpp"
#include <memory>
#include <tuple>
#include <variant>

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
                   infinicore::Tensor k_scale,
                   infinicore::Tensor v_scale,
                   ::infinilm::backends::AttentionBackend attention_backend);

    infinicore::Tensor forward(infinicore::Tensor &query,
                               infinicore::Tensor &key,
                               infinicore::Tensor &value,
                               std::tuple<infinicore::Tensor, infinicore::Tensor> kv_cache,
                               const infinilm::global_state::AttentionMetadata &attn_metadata) const;

    inline infinicore::Tensor get_k_scale() const { return k_scale_; }
    inline infinicore::Tensor get_v_scale() const { return v_scale_; }

private:
    infinicore::Tensor k_scale_;
    infinicore::Tensor v_scale_;

    AttentionImpl attn_backend_impl_;
    ::infinilm::backends::AttentionBackend attn_backend_;
};
} // namespace infinilm::layers::attention
