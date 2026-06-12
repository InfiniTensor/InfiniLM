#include "attention_layer.hpp"

namespace infinilm::layers::attention {

AttentionLayer::AttentionLayer(size_t num_heads,
                               size_t head_size,
                               float scale,
                               size_t num_kv_heads,
                               size_t layer_idx,
                               infinicore::Tensor k_scale,
                               infinicore::Tensor v_scale,
                               ::infinilm::backends::AttentionBackend attn_backend) : k_scale_(k_scale), v_scale_(v_scale), layer_idx_(layer_idx), attn_backend_(attn_backend) {
    switch (attn_backend) {
    case ::infinilm::backends::AttentionBackend::STATIC_ATTN:
        attn_backend_impl_ = std::make_shared<backends::StaticAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx);
        break;
    case ::infinilm::backends::AttentionBackend::PAGED_ATTN:
        attn_backend_impl_ = std::make_shared<backends::PagedAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx);
        break;
    case ::infinilm::backends::AttentionBackend::FLASH_ATTN:
        attn_backend_impl_ = std::make_shared<backends::FlashAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx);
        break;
    case ::infinilm::backends::AttentionBackend::FLASH_PREFILL:
        attn_backend_impl_ = std::make_shared<backends::FlashPrefillAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx);
        break;
    case ::infinilm::backends::AttentionBackend::FLASH_DECODE:
        attn_backend_impl_ = std::make_shared<backends::FlashDecodeAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx);
        break;
    default:
        throw std::runtime_error("infinilm::layers::attention::AttentionLayer: unsupported attention backend");
    }
}

infinicore::Tensor AttentionLayer::forward(infinicore::Tensor &query,
                                           infinicore::Tensor &key,
                                           infinicore::Tensor &value) const {
    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &attn_metadata = forward_context.attn_metadata;
    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];

    return std::visit(
        [&](auto &impl_ptr) -> infinicore::Tensor {
            return impl_ptr->forward(*this, query, key, value, kv_cache, attn_metadata);
        },
        attn_backend_impl_);
}

} // namespace infinilm::layers::attention
