#include "static_attn.hpp"

#include "../../../utils.hpp"
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

namespace infinilm::layers::attention::backends {

StaticAttentionImpl::StaticAttentionImpl(size_t num_heads,
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

infinicore::Tensor StaticAttentionImpl::forward(void *layer,
                                                const infinicore::Tensor &q_reshaped,
                                                const infinicore::Tensor &k_permuted,
                                                const infinicore::Tensor &v_permuted,
                                                std::shared_ptr<infinilm::cache::Cache> kv_cache,
                                                const infinilm::InfinilmModel::Input &attn_metadata) const {
    (void)layer;
    auto shape = q_reshaped->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[2];

    auto past_sequence_lengths = attn_metadata.past_sequence_lengths;
    auto total_sequence_lengths = attn_metadata.total_sequence_lengths;

    // 5. Prepare KV caches
    infinicore::Tensor k_total; // [bs, n_kv_head, max_seq_len, head_dim]
    infinicore::Tensor v_total; // [bs, n_kv_head, max_seq_len, head_dim]
    if (kv_cache == nullptr) {
        k_total = k_permuted;
        v_total = v_permuted;
    } else if (auto static_kv_cache = std::dynamic_pointer_cast<cache::StaticKVCache>(kv_cache)) {
        auto [k_total_tmp, v_total_tmp] = static_kv_cache->update(layer_idx_, k_permuted, v_permuted, past_sequence_lengths.value());
        k_total = k_total_tmp;
        v_total = v_total_tmp;
    } else {
        throw std::runtime_error("StaticAttention: Unsupported kvcache type");
    }

    infinicore::Tensor attn_output;
    if (false) {
        // experimental nineoothed flash attention
        attn_output = infinicore::op::flash_attention(q_reshaped, k_total, v_total, total_sequence_lengths.value(), scale_, true);
        attn_output = attn_output->permute({0, 2, 1, 3})
                          ->contiguous()
                          ->view({batch_size, seq_len, num_heads_ * head_dim_}); // [bs, seq_len, n_q_head * head_dim]
    } else {
        size_t total_seq_len = reinterpret_cast<int32_t *>(total_sequence_lengths.value()->to(infinicore::Device::cpu())->data())[0];
        k_total = k_total->narrow({{2, 0, total_seq_len}}); // [bs, n_kv_head, total_seq_len, head_dim]
        v_total = v_total->narrow({{2, 0, total_seq_len}}); // [bs, n_kv_head, total_seq_len, head_dim]

        // 6. Compute attention
        size_t ngroup = num_heads_ / num_kv_heads_;
        auto Q = q_reshaped->view({batch_size * num_kv_heads_, ngroup * seq_len, head_dim_});
        auto K = k_total->view({batch_size * num_kv_heads_, total_seq_len, head_dim_});
        auto V = v_total->view({batch_size * num_kv_heads_, total_seq_len, head_dim_});

        auto K_transposed = K->permute({0, 2, 1}); // [bs * n_kv_head, head_dim, total_seq_len]

        auto attn_weight = infinicore::op::matmul(Q, K_transposed, scale_); // [bs * n_kv_head, ng * seq_len, total_seq_len]

        auto attn_weight_softmax = attn_weight->view({batch_size * num_heads_, seq_len, total_seq_len});
        infinicore::op::causal_softmax_(attn_weight_softmax, attn_weight_softmax);

        auto out = infinicore::op::matmul(attn_weight, V); // [bs * n_kv_head, ng * seq_len, head_dim]

        attn_output = out->view({batch_size, num_heads_, seq_len, head_dim_})
                          ->permute({0, 2, 1, 3})
                          ->contiguous()
                          ->view({batch_size, seq_len, num_heads_ * head_dim_}); // [bs, seq_len, n_q_head * head_dim]
    }
    return attn_output;
}

} // namespace infinilm::layers::attention::backends
