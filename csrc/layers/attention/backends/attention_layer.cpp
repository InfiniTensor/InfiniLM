#include "attention_layer.hpp"
#include "infinicore/ops.hpp"

#include <cstdlib>
#include <limits>

namespace infinilm::layers::attention {

// AttentionLayer 构造：按所选 attention backend 实例化对应的实现类，
// 统一存入 attn_backend_impl_（std::variant），forward 时用 std::visit 分发。

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
    case ::infinilm::backends::AttentionBackend::HYBRID:
        // HYBRID：prefill 走 FA2，decode 走自研 paged-attention kernel（见下方 HybridAttentionImpl::forward）
        attn_backend_impl_ = std::make_shared<backends::HybridAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx);
        break;
    default:
        throw std::runtime_error("infinilm::layers::attention::AttentionLayer: unsupported attention backend");
    }
}

infinicore::Tensor AttentionLayer::forward(infinicore::Tensor &query,
                                           infinicore::Tensor &key,
                                           infinicore::Tensor &value) const {
    // 从全局 forward 上下文中取出 attention 元数据与当前层的 KV cache，
    // 再按 attn_backend_impl_ 的实际类型分发到对应后端的 forward。
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

namespace infinilm::layers::attention::backends {

size_t decode_fa_ctx_threshold() {
    static const size_t threshold = []() {
        if (const char *env = std::getenv("INFINILM_DECODE_CTX_THRESHOLD")) {
            const long long v = std::strtoll(env, nullptr, 10);
            return v > 0 ? static_cast<size_t>(v) : std::numeric_limits<size_t>::max();
        }
        const auto &model_config = infinilm::global_state::get_infinilm_config().model_config;
        if (model_config == nullptr) {
            return std::numeric_limits<size_t>::max();
        }
        const size_t hidden = model_config->get_or<size_t>("hidden_size", 0);
        const size_t layers = model_config->get_or<size_t>("num_hidden_layers", 0);
        const size_t kv_heads = model_config->get_or<size_t>("num_key_value_heads", 0);
        // Measured on RTX 5090, bf16, single-request decode (dev_perf
        // --ctx-sweep, v12): splitkv wins below, FA kvcache wins above.
        if (layers == 28 && kv_heads == 8) {
            if (hidden == 1024) { // Qwen3-0.6B: splitkv +9% @3.2k, tie @3.9k
                return static_cast<size_t>(3400);
            }
            if (hidden == 2048) { // Qwen3-1.7B: splitkv +2% @1.3k, -7% @1.9k
                return static_cast<size_t>(1500);
            }
        }
        return std::numeric_limits<size_t>::max();
    }();
    return threshold;
}

// HybridAttentionImpl 构造：内部持有一个 FlashAttentionImpl 实例，
// prefill 阶段的 mha_varlen_fwd 直接复用它；decode 阶段只用它的
// do_kv_cache_update 写缓存（BSHD 布局），注意力计算走自研 paged kernel。
HybridAttentionImpl::HybridAttentionImpl(size_t num_heads,
                                         size_t head_size,
                                         float scale,
                                         size_t num_kv_heads,
                                         size_t layer_idx)
    : flash_(std::make_shared<FlashAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx)),
      num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale) {}

infinicore::Tensor HybridAttentionImpl::forward(const AttentionLayer &layer,
                                                const infinicore::Tensor &query,
                                                const infinicore::Tensor &key,
                                                const infinicore::Tensor &value,
                                                infinicore::Tensor &kv_cache,
                                                const infinilm::global_state::AttentionMetadata &attn_metadata) const {
    // 与各独立 impl 相同的 prefill 判定：flattened paged 模式下，
    // 纯 decode 步每个序列恰好只有一个 query token，
    // 因此 query 行数 ≠ 序列数时即为 prefill（或含 prefill 的混合 batch）。
    const size_t seq_len = query->shape()[0];
    const bool is_prefill = (seq_len != attn_metadata.total_sequence_lengths.value()->shape()[0]);
    if (is_prefill) {
        // prefill / 混合 batch：走 FA2 varlen（mha_varlen_fwd），prefill 性能最优。
        return flash_->forward(layer, query, key, value, kv_cache, attn_metadata);
    }
    // 长上下文 decode：FA 的 kvcache kernel 反超自研 splitkv（交叉点随模型
    // 几何变化，见 decode_fa_ctx_threshold）。flash_->forward 会重新判定
    // decode 分支并在同一份 BSHD cache 上跑 mha_kvcache_，切换无数据搬运。
    if (attn_metadata.max_sequence_length > decode_fa_ctx_threshold()) {
        return flash_->forward(layer, query, key, value, kv_cache, attn_metadata);
    }
    // 纯 decode：复用 FA 的 cache update（BSHD 布局 + permuted paged_caching_ 写入），
    // 然后直接调 stride 感知的 paged-attention decode kernel（splitkv）。
    // FA2 的 mha_fwd_kvcache 是 sm80 时代的 kernel，在 Blackwell 上 decode 显著偏慢，
    // 短/中上下文下自研 paged kernel 更快——这正是 HYBRID 的动机。
    auto [k_total, v_total] = flash_->do_kv_cache_update(layer, key, value, kv_cache, attn_metadata.slot_mapping.value());
    const size_t value_head_dim = value->size(value->ndim() - 1);
    auto attn_output = infinicore::Tensor::empty({seq_len, num_heads_, value_head_dim}, query->dtype(), query->device());
    // paged kernel 期望的 shape 是 [num_blocks, num_kv_heads, block_size, head_dim]
    // （BHSD），但 descriptor 逐维取 stride、kernel 按 row_stride 寻址，
    // 因此对 BSHD 的 cache 做 permute({0,2,1,3}) 逻辑视图即可零拷贝直读。
    infinicore::op::paged_attention_(
        attn_output,
        query,
        k_total->permute({0, 2, 1, 3}),
        v_total->permute({0, 2, 1, 3}),
        attn_metadata.block_tables.value(),
        attn_metadata.total_sequence_lengths.value(),
        std::nullopt,
        scale_);
    // 展平成模型层期望的 [1, seq_len, hidden] 输出。
    return attn_output->view({1, seq_len, num_heads_ * value_head_dim});
}

} // namespace infinilm::layers::attention::backends
