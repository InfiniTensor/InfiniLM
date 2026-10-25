#include "flash_attn.hpp"
#include "../../../execution/submit.hpp"
#include "attention_layer.hpp"

#include "../../../execution/paged_cache_checks.hpp"
#include "../../../execution/submit.hpp"
#include "../../../utils.hpp"
#include "../attention_checks.hpp"
#include "runtime/runtime.hpp"
#include <algorithm>
#include <base/flash_attn_varlen_func.h>
#include <base/reshape_and_cache_flash.h>
#include <limits>

namespace infinilm::layers::attention::backends {

FlashAttentionImpl::FlashAttentionImpl(size_t num_heads,
                                       size_t head_size,
                                       float scale,
                                       size_t num_kv_heads,
                                       size_t layer_idx)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      layer_idx_(layer_idx),
      head_dim_(head_size) {}

infinilm::runtime::Tensor FlashAttentionImpl::forward(const AttentionLayer &layer,
                                                      const infinilm::runtime::Tensor &query,
                                                      const infinilm::runtime::Tensor &key,
                                                      const infinilm::runtime::Tensor &value,
                                                      infinilm::runtime::Tensor &kv_cache,
                                                      const infinilm::global_state::AttentionMetadata &attn_metadata,
                                                      infinilm::runtime::Runtime &runtime) const {
    runtime.assertOwnerThread();
    if (!query || !key || !value || !kv_cache) {
        throw std::invalid_argument("Flash Attention requires non-null Q/K/V and KV cache tensors");
    }
    if (runtime.device() != query->device() || runtime.device() != kv_cache->device()) {
        throw std::invalid_argument("Flash Attention Runtime and tensor devices differ");
    }
    auto total_sequence_lengths = attn_metadata.total_sequence_lengths;
    auto input_offsets = attn_metadata.input_offsets;
    auto block_tables = attn_metadata.block_tables;
    auto slot_mapping = attn_metadata.slot_mapping;
    auto cu_seqlens = attn_metadata.cu_seqlens;

    if (!total_sequence_lengths || !*total_sequence_lengths
        || !block_tables || !*block_tables
        || !slot_mapping || !*slot_mapping) {
        throw std::invalid_argument("Flash Attention requires sequence lengths, block tables and slot mapping");
    }
    if (query->ndim() != 3 || query->size(0) == 0
        || total_sequence_lengths.value()->ndim() != 1
        || total_sequence_lengths.value()->size(0) == 0) {
        throw std::invalid_argument("Flash Attention requires nonempty rank-3 query and rank-1 sequence lengths");
    }

    const size_t seq_len = query->shape()[0];
    const bool is_prefill = seq_len != total_sequence_lengths.value()->shape()[0];
    if (is_prefill && (!input_offsets || !*input_offsets || !cu_seqlens || !*cu_seqlens || input_offsets.value()->ndim() != 1 || cu_seqlens.value()->ndim() != 1 || input_offsets.value()->size(0) < 2 || cu_seqlens.value()->size(0) < 2 || attn_metadata.max_query_length == 0 || attn_metadata.max_sequence_length == 0 || attn_metadata.max_query_length > static_cast<size_t>(std::numeric_limits<int64_t>::max()) || attn_metadata.max_sequence_length > static_cast<size_t>(std::numeric_limits<int64_t>::max()))) {
        throw std::invalid_argument("Flash Attention prefill requires offsets and positive batch maximum lengths");
    }

    // 1. update paged kv cache
    auto [k_total, v_total] = do_kv_cache_update(layer, key, value, kv_cache, slot_mapping.value(), runtime);

    // 2. Compute attention
    auto attn_output = infinilm::runtime::Tensor::empty(
        runtime, {seq_len, num_heads_, head_dim_}, query->dtype());
    // Hygon's linked ATen FlashAttention owns temporary allocations. Keep it
    // as an explicit host segment between captured native device segments.
    // Its ATen stream guard keeps all tensor work ordered on Runtime's stream;
    // no preceding CPU synchronization is needed to enqueue that work.
    const auto attention_policy = runtime.device().type() == infinilm::runtime::Device::Type::kHygon
                                    ? infinilm::execution::RecordingPolicy::StreamOrderedHostReplay
                                    : infinilm::execution::RecordingPolicy::DeviceOnly;
    if (is_prefill) {
        infinilm::execution::submit<infini::ops::FlashAttnVarlenFunc>(
            runtime, attention_policy,
            infini::ops::Config::Automatic(),
            query, k_total, v_total, input_offsets.value(), cu_seqlens.value(),
            std::optional<infinilm::runtime::Tensor>{}, std::optional<infinilm::runtime::Tensor>{block_tables.value()},
            static_cast<int64_t>(attn_metadata.max_query_length), static_cast<int64_t>(attn_metadata.max_sequence_length),
            0.0, std::optional<double>{scale_}, true, layer.window_size(), 0.0, false, false,
            attn_output, std::optional<infinilm::runtime::Tensor>{}, std::optional<infinilm::runtime::Tensor>{});
        INFINILM_TRACE_OP(MultiheadAttentionVarlen, attn_output);
    } else {
        // FA2 decode path: flash::mha_fwd_kvcache
        // In paged-attn mode, seq_len = actual batch_size (one query token per sequence).
        // q_reshaped: [seq_len, num_heads, head_dim] → [seq_len, 1, num_heads, head_dim]
        // k/v cache:  [num_blocks, block_size, num_kv_heads, head_dim]
        auto q_for_fa = query->view({seq_len, 1, num_heads_, head_dim_});
        auto attn_out_4d = attn_output->view({seq_len, 1, num_heads_, head_dim_});
        const auto kv_window = layer.window_size();
        const auto kv_bound = infinilm::layers::attention::detail::check_kvcache(
            runtime, attn_out_4d, q_for_fa, k_total, v_total, total_sequence_lengths.value(), block_tables.value(),
            std::nullopt, attn_metadata.logical_kv_upper_bound);
        infini::ops::FlashAttnWithKvcacheConfig kv_config(kv_bound);
        kv_config.set_implementation_index(infini::ops::Config::kAutomaticImplementation);
        const std::optional<infinilm::runtime::Tensor> kv_none;
        infinilm::execution::submit<infini::ops::FlashAttnWithKvcache>(
            runtime, attention_policy, kv_config,
            q_for_fa, k_total, v_total, kv_none, kv_none, kv_none, kv_none,
            std::optional<infinilm::runtime::Tensor>{total_sequence_lengths.value()}, kv_none, kv_none,
            std::optional<infinilm::runtime::Tensor>{block_tables.value()}, kv_none,
            std::optional<double>{scale_}, true, kv_window, double{0.0}, true,
            int64_t{0}, false, attn_out_4d, kv_none);
        INFINILM_TRACE_OP(MhaKVCache, attn_out_4d);
    }
    return attn_output->view({1, seq_len, num_heads_ * head_dim_});
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> FlashAttentionImpl::do_kv_cache_update(const AttentionLayer &layer,
                                                                                                        const infinilm::runtime::Tensor key,
                                                                                                        const infinilm::runtime::Tensor value,
                                                                                                        infinilm::runtime::Tensor &kv_cache,
                                                                                                        const infinilm::runtime::Tensor slot_mapping,
                                                                                                        infinilm::runtime::Runtime &runtime) const {
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    const auto &paged_mapping = slot_mapping;
    const auto paged_config = infinilm::execution::paged_cache_checks::validate(
        runtime, k_cache_layer, v_cache_layer, key, value, paged_mapping);
    if (paged_config) {
        const auto paged_tokens = paged_mapping->size(0);
        const infini::ops::Tensor paged_scale{nullptr, {1}, infinilm::runtime::DataType::kFloat32, runtime.device()};
        const std::string paged_mode{"auto"};
        for (infinilm::runtime::Size paged_start = 0; paged_start < paged_tokens;) {
            const auto paged_count = std::min<infinilm::runtime::Size>(32768, paged_tokens - paged_start);
            auto paged_key = key->narrow({{0, paged_start, paged_count}});
            auto paged_value = value->narrow({{0, paged_start, paged_count}});
            auto paged_slots = paged_mapping->narrow({{0, paged_start, paged_count}});
            infinilm::execution::submit<infini::ops::ReshapeAndCacheFlash>(
                runtime, infinilm::execution::RecordingPolicy::AllowHostReplay, *paged_config,
                paged_key, paged_value, paged_slots, paged_scale, paged_scale, paged_mode, k_cache_layer, v_cache_layer);
            INFINILM_TRACE_OP(PagedCaching, k_cache_layer);
            paged_start += paged_count;
        }
    }

    return {k_cache_layer, v_cache_layer};
}

} // namespace infinilm::layers::attention::backends
