#include "flash_attn.hpp"

#include "../../../global_state/global_state.hpp"
#include "../../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/mha_kvcache.hpp"
#include "infinicore/ops/mha_varlen.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace infinilm::layers::attention::backends {

// #region agent log
namespace {
inline void dbg_fa_d39f05_(const char *hypothesisId, const char *message, const std::string &data_json) {
    try {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
        std::ostringstream line;
        line << "{\"sessionId\":\"8b13ee\",\"runId\":\"dbias2\",\"hypothesisId\":\"" << hypothesisId
             << "\",\"location\":\"flash_attn.cpp:forward\",\"message\":\"" << message
             << "\",\"data\":" << data_json << ",\"timestamp\":" << ms << "}\n";
        const std::string s = line.str();
        std::fprintf(stderr, "[8b13ee] %s", s.c_str());
        std::fflush(stderr);
        std::ofstream ofs("/opt/offline/infinilm-metax-20260622/.cursor/debug-8b13ee.log", std::ios::app);
        if (ofs) {
            ofs << s;
        }
    } catch (...) {
    }
}
} // namespace
// #endregion


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
      head_dim_(head_size) {

    const infinilm::global_state::InfinilmConfig &infinilm_config = infinilm::global_state::get_infinilm_config();
    if (!infinilm_config.model_config) {
        throw std::runtime_error("infinilm::layers::attention::backends::FlashAttentionImpl: model_config is null");
    }
    max_position_embeddings_ = infinilm_config.model_config->get<size_t>("max_position_embeddings");
}

infinicore::Tensor FlashAttentionImpl::forward(const AttentionLayer &layer,
                                               const infinicore::Tensor &query,
                                               const infinicore::Tensor &key,
                                               const infinicore::Tensor &value,
                                               infinicore::Tensor &kv_cache,
                                               const infinilm::global_state::AttentionMetadata &attn_metadata) const {
    auto total_sequence_lengths = attn_metadata.total_sequence_lengths;
    auto input_offsets = attn_metadata.input_offsets;
    auto block_tables = attn_metadata.block_tables;
    auto slot_mapping = attn_metadata.slot_mapping;
    auto cu_seqlens = attn_metadata.cu_seqlens;

    ASSERT(block_tables.has_value());
    ASSERT(slot_mapping.has_value());


    // 1. update paged kv cache
    auto [k_total, v_total] = do_kv_cache_update(layer, key, value, kv_cache, slot_mapping.value());

    size_t seq_len = query->shape()[0];
    bool is_prefill = (seq_len != total_sequence_lengths.value()->shape()[0]);

    // 2. Compute attention
    infinicore::Tensor attn_output = infinicore::Tensor::empty({seq_len, num_heads_, head_dim_}, query->dtype(), query->device());
    if (is_prefill) {
        // FA varlen max_seqlen must be the batch max, NOT max_position_embeddings.
        // MiniCPM5 max_position_embeddings=131072 while block_tables only cover
        // num_blocks*block_size (e.g. 256*256); oversized max_seqlen_k makes the
        // kernel index past block_table columns → wrong KV / D-bias on long chat.
        int max_seqlen_q = attn_metadata.max_query_len;
        int max_seqlen_k = attn_metadata.max_kv_len;
        if (max_seqlen_q <= 0) {
            max_seqlen_q = static_cast<int>(seq_len);
        }
        if (max_seqlen_k <= 0) {
            max_seqlen_k = max_seqlen_q;
        }
        // Still never exceed table capacity (page size 256).
        if (block_tables.value()->ndim() >= 2) {
            const int table_cap = static_cast<int>(block_tables.value()->size(1) * 256);
            max_seqlen_k = std::min(max_seqlen_k, table_cap);
            max_seqlen_q = std::min(max_seqlen_q, table_cap);
        }
        // #region agent log
        {
            static int fa_prefill_logs = 0;
            if (fa_prefill_logs < 8 && seq_len >= 32) {
                ++fa_prefill_logs;
                int off0 = -1, off1 = -1, cu0 = -1, cu1 = -1;
                try {
                    auto off_cpu = input_offsets.value()->to(infinicore::Device::cpu());
                    auto cu_cpu = cu_seqlens.value()->to(infinicore::Device::cpu());
                    const auto *od = reinterpret_cast<const int32_t *>(off_cpu->data());
                    const auto *cd = reinterpret_cast<const int32_t *>(cu_cpu->data());
                    if (off_cpu->size(0) >= 2) {
                        off0 = od[0];
                        off1 = od[1];
                    }
                    if (cu_cpu->size(0) >= 2) {
                        cu0 = cd[0];
                        cu1 = cd[1];
                    }
                } catch (...) {
                }
                std::ostringstream dj;
                dj << "{\"q_len\":" << seq_len << ",\"max_q\":" << max_seqlen_q
                   << ",\"max_k\":" << max_seqlen_k << ",\"off\":[" << off0 << "," << off1
                   << "],\"cu\":[" << cu0 << "," << cu1 << "],\"bt_cols\":"
                   << (block_tables.value()->ndim() >= 2 ? block_tables.value()->size(1) : 0)
                   << ",\"q_vs_max\":" << (static_cast<int>(seq_len) == max_seqlen_q ? "true" : "false")
                   << ",\"off_span_vs_q\":" << ((off1 - off0) == static_cast<int>(seq_len) ? "true" : "false")
                   << "}";
                dbg_fa_d39f05_("F,H", "fa_prefill_shapes", dj.str());
            }
        }
        // #endregion
        infinicore::op::mha_varlen_(
            attn_output,
            query,
            k_total,
            v_total,
            input_offsets.value(),
            cu_seqlens.value(),
            block_tables.value(),
            max_seqlen_q,
            max_seqlen_k,
            std::nullopt,
            scale_);
    } else {
        // FA2 decode path: flash::mha_fwd_kvcache
        // In paged-attn mode, seq_len = actual batch_size (one query token per sequence).
        // q_reshaped: [seq_len, num_heads, head_dim] → [seq_len, 1, num_heads, head_dim]
        // k/v cache:  [num_blocks, block_size, num_kv_heads, head_dim]
        auto q_for_fa = query->view({seq_len, 1, num_heads_, head_dim_});
        auto attn_out_4d = infinicore::op::mha_kvcache(
            q_for_fa,
            k_total, // [num_blocks, block_size, num_kv_heads, head_dim]
            v_total,
            total_sequence_lengths.value(), // [seq_len] int32 (one entry per sequence)
            block_tables.value(),           // [seq_len, max_num_blocks_per_seq] int32
            std::nullopt,
            scale_);
        attn_output = attn_out_4d->view({seq_len, num_heads_, head_dim_});
    }
    attn_output = attn_output->view({1, seq_len, num_heads_ * head_dim_});
    return attn_output;
}

std::tuple<infinicore::Tensor, infinicore::Tensor> FlashAttentionImpl::do_kv_cache_update(const AttentionLayer &layer,
                                                                                          const infinicore::Tensor key,
                                                                                          const infinicore::Tensor value,
                                                                                          infinicore::Tensor &kv_cache,
                                                                                          const infinicore::Tensor slot_mapping) const {
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    infinicore::op::paged_caching_(
        k_cache_layer->permute({0, 2, 1, 3}), // permute to BHSD for paged_caching_
        v_cache_layer->permute({0, 2, 1, 3}),
        key,
        value,
        slot_mapping);

    return {k_cache_layer, v_cache_layer};
}

} // namespace infinilm::layers::attention::backends
