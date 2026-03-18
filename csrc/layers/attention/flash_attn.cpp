#include "flash_attn.hpp"

#include "../../utils.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/mha_kvcache.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#include "infinicore/ops/mul.hpp"

#include "infinicore/io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

namespace infinilm::models::layers::attention {
infinicore::Tensor FlashAttention::attn_calculate(const infinicore::Tensor &q_reshaped,
                                                  const infinicore::Tensor &k_reshaped,
                                                  const infinicore::Tensor &v_reshaped,
                                                  const infinilm::InfinilmModel::Input &input,
                                                  std::shared_ptr<infinilm::cache::Cache> kv_cache) const {
    auto total_sequence_lengths = input.total_sequence_lengths;
    auto input_offsets = input.input_offsets;
    auto cu_seqlens = input.cu_seqlens;
    auto block_tables = input.block_tables;
    auto slot_mapping = input.slot_mapping;
    ASSERT(block_tables.has_value());
    ASSERT(slot_mapping.has_value());

    auto paged_kv_cache = std::dynamic_pointer_cast<cache::PagedKVCache>(kv_cache);
    if (!paged_kv_cache) {
        throw std::runtime_error("Attention: kvcache is not a PagedKVCache");
    }

    size_t seq_len = q_reshaped->shape()[0];
    bool is_prefill = (seq_len != total_sequence_lengths.value()->shape()[0]);

    auto [k_total, v_total] = paged_kv_cache->update(layer_idx_,
                                                     k_reshaped,
                                                     v_reshaped,
                                                     slot_mapping.value());

    infinicore::Tensor attn_output = infinicore::Tensor::empty({seq_len, num_attention_heads_, head_dim_}, q_reshaped->dtype(), q_reshaped->device());
    if (is_prefill) {
        infinicore::op::mha_varlen_(
            attn_output,
            q_reshaped,
            k_total->permute({0, 2, 1, 3}),
            v_total->permute({0, 2, 1, 3}),
            input_offsets.value(),
            cu_seqlens.value(),
            block_tables.value(),
            max_position_embeddings_,
            max_position_embeddings_,
            std::nullopt,
            scaling_);
    } else {
        throw std::runtime_error("FlashAttention: decode is not supported");
    }
    return attn_output;
}

} // namespace infinilm::models::layers::attention
