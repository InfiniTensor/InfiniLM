#include "attention_layer.hpp"
#include "../../../execution/submit.hpp"
#include <base/flash_attn_varlen_func.h>
#include "runtime/owner_drain_fail_stop.hpp"
#include "runtime/utils.hpp"
#include <stdexcept>
#include <limits>
#include <utility>

namespace infinilm::layers::attention {

AttentionLayer::AttentionLayer(size_t num_heads,
                               size_t head_size,
                               float scale,
                               size_t num_kv_heads,
                               size_t layer_idx,
                               infinilm::runtime::Tensor k_scale,
                               infinilm::runtime::Tensor v_scale,
                               ::infinilm::backends::AttentionBackend attn_backend,
                               infinilm::quantization::KVQuantAlgo kv_quant_scheme,
                               std::vector<int64_t> window_size) : window_size_(std::move(window_size)), k_scale_(k_scale), v_scale_(v_scale), layer_idx_(layer_idx), attn_backend_(attn_backend) {
    if (window_size_.size() != 2
        || !(window_size_ == std::vector<int64_t>{-1, -1}
             || (window_size_[0] >= 0 && window_size_[0] <= std::numeric_limits<int>::max() - 127LL
                 && window_size_[1] == 0)))
        throw std::invalid_argument("Attention window must be global or a finite causal left window");
    if (window_size_[0] >= 0 && kv_quant_scheme != infinilm::quantization::KVQuantAlgo::NONE)
        throw std::invalid_argument("Finite attention window requires NONE KV quantization");
    switch (attn_backend) {
    case ::infinilm::backends::AttentionBackend::STATIC_ATTN:
        attn_backend_impl_ = std::make_shared<backends::StaticAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx, kv_quant_scheme);
        break;
    case ::infinilm::backends::AttentionBackend::PAGED_ATTN:
        if (kv_quant_scheme != infinilm::quantization::KVQuantAlgo::NONE)
            throw std::invalid_argument("Paged Attention has no quantized-cache conversion path");
        attn_backend_impl_ = std::make_shared<backends::PagedAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx);
        break;
    case ::infinilm::backends::AttentionBackend::FLASH_ATTN:
        if (kv_quant_scheme != infinilm::quantization::KVQuantAlgo::NONE)
            throw std::invalid_argument("Flash Attention has no quantized-cache conversion path");
        attn_backend_impl_ = std::make_shared<backends::FlashAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx);
        break;
    default:
        throw std::runtime_error("infinilm::layers::attention::AttentionLayer: unsupported attention backend");
    }
}

infinilm::runtime::Tensor AttentionLayer::forward(
    infinilm::runtime::Tensor &query,
                                           infinilm::runtime::Tensor &key,
    infinilm::runtime::Tensor &value,
    infinilm::global_state::ForwardContext &forward_context) const {
    if (layer_idx_ >= forward_context.kv_cache_vec.size()) {
        throw std::runtime_error("AttentionLayer: KV cache is missing for this layer");
    }
    if (forward_context.runtime == nullptr) {
        throw std::invalid_argument("AttentionLayer: forward requires its worker Runtime");
    }
    auto &attn_metadata = forward_context.attn_metadata;
    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];
    auto &runtime = *forward_context.runtime;

    return std::visit(
        [&](auto &impl_ptr) -> infinilm::runtime::Tensor {
            return impl_ptr->forward(*this, query, key, value, kv_cache,
                                     attn_metadata, runtime);
        },
        attn_backend_impl_);
}

void AttentionLayer::prefill_paged(
    infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor out,
    const infinilm::runtime::Tensor &q, const infinilm::runtime::Tensor &k_cache,
    const infinilm::runtime::Tensor &v_cache, const infinilm::runtime::Tensor &block_tables,
    const infinilm::runtime::Tensor &kv_lens, const infinilm::runtime::Tensor &cu_q, float scale) const {
    using infinilm::runtime::Tensor;
    using infinilm::runtime::DataType;
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording())
        throw std::logic_error("Paged prefill with live lengths requires eager Runtime");
    INFINILM_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, block_tables, kv_lens, cu_q);
    INFINILM_ASSERT(out->device() == runtime.device());
    INFINILM_ASSERT(kv_lens->ndim() == 1 && kv_lens->dtype() == DataType::kInt32);
    INFINILM_ASSERT(cu_q->ndim() == 1 && cu_q->size(0) == kv_lens->size(0) + 1);

    // Keep host buffers and assigned device owners alive through catch-side drain.
    Tensor lengths;
    std::vector<int32_t> kv_lens_data;
    std::vector<int32_t> cum_seqlens_k;
    Tensor cum_seqlens_k_tensor;
    try {
    lengths = kv_lens->contiguous(runtime);
    kv_lens_data.resize(kv_lens->size(0));
    runtime.memcpyD2H(kv_lens_data.data(), lengths->data(), lengths->nbytes());
    cum_seqlens_k.assign(kv_lens->size(0) + 1, 0);
    int64_t total_kv_len = 0;
    for (size_t i = 0; i < kv_lens->size(0); ++i) {
        INFINILM_ASSERT(kv_lens_data[i] >= 0);
        total_kv_len += kv_lens_data[i];
        INFINILM_ASSERT(total_kv_len <= std::numeric_limits<int32_t>::max());
        cum_seqlens_k[i + 1] = static_cast<int32_t>(total_kv_len);
    }

    cum_seqlens_k_tensor = Tensor::empty(
        runtime, {cum_seqlens_k.size()}, DataType::kInt32);
    runtime.memcpyH2D(
        cum_seqlens_k_tensor->data(),
        cum_seqlens_k.data(),
        cum_seqlens_k.size() * sizeof(int32_t),
        false);

    const auto block_width = block_tables->shape().at(1);
    const auto page_size = k_cache->shape().at(1);
    INFINILM_ASSERT(page_size == 0
        || block_width <= static_cast<size_t>(std::numeric_limits<int>::max()) / page_size);
    const auto max_seqlen_k = block_width * page_size;
    INFINILM_ASSERT(q->size(0) <= static_cast<size_t>(std::numeric_limits<int>::max()));
    infinilm::execution::submit<infini::ops::FlashAttnVarlenFunc>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Config::Automatic(), q, k_cache, v_cache, cu_q, cum_seqlens_k_tensor,
        std::optional<Tensor>{}, std::optional<Tensor>{block_tables},
        static_cast<int64_t>(q->size(0)), static_cast<int64_t>(max_seqlen_k),
        0.0, std::optional<double>{scale}, true, std::vector<int64_t>{-1, -1}, 0.0, false, false,
        out, std::optional<Tensor>{}, std::optional<Tensor>{});
    INFINILM_TRACE_OP(MultiheadAttentionVarlen, out);
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) {
            infinilm::runtime::detail::require_proven_owner_drain(true, false);
        }
        throw;
    }
}

} // namespace infinilm::layers::attention
