#include "attention_layer.hpp"

namespace infinilm::layers::attention {

AttentionLayer::AttentionLayer(size_t num_heads,
                               size_t head_size,
                               float scale,
                               size_t num_kv_heads,
                               size_t layer_idx,
                               ::infinilm::backends::AttentionBackend attn_backend) {
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
    default:
        throw std::runtime_error("infinilm::layers::attention::AttentionLayer: unsupported attention backend");
    }
}

infinicore::Tensor AttentionLayer::forward(const infinicore::Tensor &query,
                                           const infinicore::Tensor &key,
                                           const infinicore::Tensor &value,
                                           std::tuple<infinicore::Tensor, infinicore::Tensor> kv_cache,
                                           const infinilm::engine::AttentionMetadata &attn_metadata) const {
    return std::visit(
        [&](auto &impl_ptr) -> infinicore::Tensor {
            return impl_ptr->forward(*this, query, key, value, kv_cache, attn_metadata);
        },
        attn_backend_impl_);
}

} // namespace infinilm::layers::attention
