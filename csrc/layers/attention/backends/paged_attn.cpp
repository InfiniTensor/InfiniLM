#include "../../../execution/submit.hpp"
#include "../attention_checks.hpp"
#include "paged_attn.hpp"
#include "attention_layer.hpp"

#include "../../../execution/paged_cache_checks.hpp"
#include "../../../execution/submit.hpp"
#include <base/reshape_and_cache_flash.h>
#include <algorithm>
#include <limits>
#include "../../../utils.hpp"
#ifndef INFINICORE_MHA_VARLEN_OMITTED
#include <base/flash_attn_varlen_func.h>
#endif
#include "runtime/runtime.hpp"

namespace infinilm::layers::attention::backends {

PagedAttentionImpl::PagedAttentionImpl(size_t num_heads,
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

infinilm::runtime::Tensor PagedAttentionImpl::forward(const AttentionLayer &layer,
                                               const infinilm::runtime::Tensor &query,
                                               const infinilm::runtime::Tensor &key,
                                               const infinilm::runtime::Tensor &value,
                                               infinilm::runtime::Tensor &kv_cache,
                                               const infinilm::global_state::AttentionMetadata &attn_metadata,
                                               infinilm::runtime::Runtime &runtime) const {
    runtime.assertOwnerThread();
    if (runtime.device() != query->device() || runtime.device() != kv_cache->device()) {
        throw std::invalid_argument("Paged Attention Runtime and tensor devices differ");
    }
    auto total_sequence_lengths = attn_metadata.total_sequence_lengths;
    auto input_offsets = attn_metadata.input_offsets;
    auto cu_seqlens = attn_metadata.cu_seqlens;
    auto block_tables = attn_metadata.block_tables;
    auto slot_mapping = attn_metadata.slot_mapping;
    ASSERT(block_tables.has_value());
    ASSERT(slot_mapping.has_value());

    // 1. update paged kv cache
    auto [k_total, v_total] = do_kv_cache_update(layer, key, value, kv_cache, slot_mapping.value(), runtime);

    size_t seq_len = query->shape()[0];
    bool is_prefill = (seq_len != total_sequence_lengths.value()->shape()[0]);

    // 2. Compute attention
    const size_t value_head_dim = value->size(value->ndim() - 1);
    infinilm::runtime::Tensor attn_output = infinilm::runtime::Tensor::empty(runtime, {seq_len, num_heads_, value_head_dim}, query->dtype());
    if (is_prefill) {
        ASSERT(input_offsets.has_value());
        ASSERT(cu_seqlens.has_value());
        const auto block_width = block_tables.value()->shape().at(1);
        const auto page_size = k_total->shape().at(1);
        const auto max_length = static_cast<size_t>(std::numeric_limits<int64_t>::max());
        if (seq_len > max_length || (page_size != 0 && block_width > max_length / page_size))
            throw std::invalid_argument("Paged prefill maximum length exceeds int64");
        const auto max_seqlen_k = block_width * page_size;
#ifndef INFINICORE_MHA_VARLEN_OMITTED
        infinilm::execution::submit<infini::ops::FlashAttnVarlenFunc>(
            runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
            infini::ops::Config::Automatic(),
            query, k_total, v_total, input_offsets.value(), cu_seqlens.value(),
            std::optional<infinilm::runtime::Tensor>{}, std::optional<infinilm::runtime::Tensor>{block_tables.value()},
            static_cast<int64_t>(seq_len), static_cast<int64_t>(max_seqlen_k),
            0.0, std::optional<double>{scale_}, true, layer.window_size(), 0.0, false, false,
            attn_output, std::optional<infinilm::runtime::Tensor>{}, std::optional<infinilm::runtime::Tensor>{});
#else
        throw std::runtime_error("MhaVarlen is unavailable in the greedy_bootstrap build profile");
#endif
        INFINILM_TRACE_OP(MultiheadAttentionVarlen, attn_output);
    } else {
        ASSERT(query->ndim() == 3 && attn_output->shape() == query->shape());
        const auto kv_query = query->unsqueeze(1);
        const auto kv_output = attn_output->unsqueeze(1);
#ifdef INFINICORE_MHA_KVCACHE_OMITTED
        throw std::runtime_error("MhaKVCache is unavailable in the greedy_bootstrap build profile");
#else
        const auto kv_window = layer.window_size();
        const auto kv_bound = infinilm::layers::attention::detail::check_kvcache(
            runtime, kv_output, kv_query, k_total, v_total, total_sequence_lengths.value(), block_tables.value(),
            std::nullopt, attn_metadata.logical_kv_upper_bound, kv_window);
        infini::ops::FlashAttnWithKvcacheConfig kv_config(kv_bound);
        kv_config.set_implementation_index(infini::ops::Config::kAutomaticImplementation);
        const std::optional<infinilm::runtime::Tensor> kv_none;
        infinilm::execution::submit<infini::ops::FlashAttnWithKvcache>(
            runtime, infinilm::execution::RecordingPolicy::DeviceOnly, kv_config,
            kv_query, k_total, v_total, kv_none, kv_none, kv_none, kv_none,
            std::optional<infinilm::runtime::Tensor>{total_sequence_lengths.value()}, kv_none, kv_none,
            std::optional<infinilm::runtime::Tensor>{block_tables.value()}, kv_none,
            std::optional<double>{scale_}, true, kv_window, double{0.0}, true,
            int64_t{0}, false, kv_output, kv_none);
        INFINILM_TRACE_OP(MhaKVCache, kv_output);
#endif
    }
    attn_output = attn_output->view({1, seq_len, num_heads_ * value_head_dim});
    return attn_output;
}

std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> PagedAttentionImpl::do_kv_cache_update(const AttentionLayer &layer,
                                                                                          const infinilm::runtime::Tensor key,
                                                                                          const infinilm::runtime::Tensor value,
                                                                                          infinilm::runtime::Tensor &kv_cache,
                                                                                          const infinilm::runtime::Tensor slot_mapping,
                                                                                          infinilm::runtime::Runtime &runtime) const {
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    v_cache_layer = v_cache_layer->narrow({{3, 0, value->size(value->ndim() - 1)}});
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
