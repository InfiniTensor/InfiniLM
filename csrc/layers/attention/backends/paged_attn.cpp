#include "paged_attn.hpp"

#include "../../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include "../../../global_state/global_state.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>

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

infinicore::Tensor PagedAttentionImpl::forward(const AttentionLayer &layer,
                                               const infinicore::Tensor &query,
                                               const infinicore::Tensor &key,
                                               const infinicore::Tensor &value,
                                               infinicore::Tensor &kv_cache,
                                               const infinilm::global_state::AttentionMetadata &attn_metadata) const {
    auto total_sequence_lengths = attn_metadata.total_sequence_lengths;
    auto input_offsets = attn_metadata.input_offsets;
    auto block_tables = attn_metadata.block_tables;
    auto slot_mapping = attn_metadata.slot_mapping;
    ASSERT(block_tables.has_value());
    ASSERT(slot_mapping.has_value());


    // 1. update paged kv cache
    auto [k_total, v_total] = do_kv_cache_update(layer, key, value, kv_cache, slot_mapping.value());
    infinicore::context::syncStream();

    size_t seq_len = query->shape()[0];
    bool is_prefill = (seq_len != total_sequence_lengths.value()->shape()[0]);

    // #region agent log
    {
        static int paged_dumps = 0;
        if (paged_dumps < 3 && is_prefill && layer_idx_ == 0) {
            ++paged_dumps;
            try {
                const auto &meta = infinilm::global_state::get_forward_context().attn_metadata;
                int tot0 = -1, off0 = -1, off1 = -1, slot0 = -1, bt0 = -1;
                try {
                    auto tot_cpu = total_sequence_lengths.value()->to(infinicore::Device::cpu());
                    auto off_cpu = input_offsets.value()->to(infinicore::Device::cpu());
                    auto slot_cpu = slot_mapping.value()->to(infinicore::Device::cpu());
                    auto bt_cpu = block_tables.value()->to(infinicore::Device::cpu());
                    const auto *td = reinterpret_cast<const int32_t *>(tot_cpu->data());
                    const auto *od = reinterpret_cast<const int32_t *>(off_cpu->data());
                    const auto *sd = reinterpret_cast<const int64_t *>(slot_cpu->data());
                    const auto *bd = reinterpret_cast<const int32_t *>(bt_cpu->data());
                    if (tot_cpu->numel() >= 1) {
                        tot0 = td[0];
                    }
                    if (off_cpu->size(0) >= 2) {
                        off0 = od[0];
                        off1 = od[1];
                    }
                    if (slot_cpu->numel() >= 1) {
                        slot0 = static_cast<int>(sd[0]);
                    }
                    if (bt_cpu->numel() >= 1) {
                        bt0 = bd[0];
                    }
                } catch (...) {
                }
                const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count();
                std::ostringstream dj;
                dj << "{\"q_len\":" << seq_len
                   << ",\"tot_shape0\":" << total_sequence_lengths.value()->shape()[0]
                   << ",\"off_shape0\":"
                   << (input_offsets.has_value() ? input_offsets.value()->shape()[0] : 0)
                   << ",\"max_q\":" << meta.max_query_len << ",\"max_k\":" << meta.max_kv_len
                   << ",\"bt_cols\":"
                   << (block_tables.value()->ndim() >= 2 ? block_tables.value()->size(1) : 0)
                   << ",\"tot0\":" << tot0 << ",\"off\":[" << off0 << "," << off1 << "]"
                   << ",\"slot0\":" << slot0 << ",\"bt0\":" << bt0
                   << ",\"slot_n\":" << slot_mapping.value()->shape()[0] << "}";
                std::ostringstream line;
                line << "{\"sessionId\":\"8b13ee\",\"runId\":\"paged-bisect\",\"hypothesisId\":\"B\""
                     << ",\"location\":\"paged_attn.cpp:forward\",\"message\":\"paged_prefill_meta\""
                     << ",\"data\":" << dj.str() << ",\"timestamp\":" << ms << "}\n";
                const std::string s = line.str();
                std::fprintf(stderr, "[8b13ee] %s", s.c_str());
                std::fflush(stderr);
                std::ofstream ofs("/opt/offline/infinilm-metax-20260622/.cursor/debug-8b13ee.log",
                                  std::ios::app);
                if (ofs) {
                    ofs << s;
                }
            } catch (...) {
            }
        }
    }
    // #endregion

    // 2. Compute attention
    infinicore::Tensor attn_output = infinicore::Tensor::empty({seq_len, num_heads_, head_dim_}, query->dtype(), query->device());
    if (is_prefill) {
        infinicore::op::paged_attention_prefill_(
            attn_output,
            query,
            k_total,
            v_total,
            block_tables.value(),
            total_sequence_lengths.value(),
            input_offsets.value(),
            std::nullopt,
            scale_);
    } else {
        infinicore::op::paged_attention_(
            attn_output,
            query,
            k_total,
            v_total,
            block_tables.value(),
            total_sequence_lengths.value(),
            std::nullopt,
            scale_);
    }
    attn_output = attn_output->view({1, seq_len, num_heads_ * head_dim_});
    return attn_output;
}

std::tuple<infinicore::Tensor, infinicore::Tensor> PagedAttentionImpl::do_kv_cache_update(const AttentionLayer &layer,
                                                                                          const infinicore::Tensor key,
                                                                                          const infinicore::Tensor value,
                                                                                          infinicore::Tensor &kv_cache,
                                                                                          const infinicore::Tensor slot_mapping) const {
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    infinicore::op::paged_caching_(
        k_cache_layer,
        v_cache_layer,
        key,
        value,
        slot_mapping);

    return {k_cache_layer, v_cache_layer};
}
} // namespace infinilm::layers::attention::backends
