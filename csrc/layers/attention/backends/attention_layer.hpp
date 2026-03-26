#pragma once

#include "../../../backends/attention_backends.hpp"
#include "../../../engine/forward_context.hpp"
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
 * The class does the following::
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
                   ::infinilm::backends::AttentionBackend attention_backend);

    infinicore::Tensor forward(const infinicore::Tensor &query,
                               const infinicore::Tensor &key,
                               const infinicore::Tensor &value,
                               std::tuple<infinicore::Tensor, infinicore::Tensor> kv_cache,
                               const infinilm::engine::AttentionMetadata &attn_metadata) const;

private:
    AttentionImpl attn_backend_impl_;
    ::infinilm::backends::AttentionBackend attn_backend_;
};
} // namespace infinilm::layers::attention
