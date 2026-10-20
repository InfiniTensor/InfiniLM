#pragma once

#include "kimi_k3_delta_attention.hpp"
#include "kimi_k3_mla_attention.hpp"
#include "kimi_k3_moe.hpp"

#include "../../weights/weights.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include <runtime/tensor.hpp>

#include <memory>
#include <utility>

namespace infinilm::global_state { struct ForwardContext; }

namespace infinilm::models::kimi_k3 {

class KimiK3DecoderLayer : public infinilm::weights::ModelWeights {
public:
    KimiK3DecoderLayer(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &hidden_states,
                               const infinilm::runtime::Tensor &block_residual_storage,
                              size_t &block_residual_count,
                              global_state::ForwardContext &forward_context) const;

private:
    infinilm::runtime::Tensor apply_attn_res(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &prefix_sum,
                                      const infinilm::runtime::Tensor &block_residual_storage,
                                      size_t block_residual_count,
                                      const std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> &proj,
                                      const std::shared_ptr<infinilm::layers::RMSNorm> &norm) const;

    size_t layer_idx_{0};
    size_t hidden_size_{0};
    size_t attn_res_block_size_{12};
    bool is_kda_{false};
    bool use_moe_{false};

    std::shared_ptr<KimiK3DeltaAttention> delta_attn_;
    std::shared_ptr<KimiK3MLAAttention> mla_attn_;
    std::shared_ptr<KimiK3MoE> block_sparse_moe_;
    std::shared_ptr<KimiK3MLP> mlp_;
    std::shared_ptr<infinilm::layers::RMSNorm> input_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_attention_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> self_attention_res_norm_;
    std::shared_ptr<infinilm::layers::RMSNorm> mlp_res_norm_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> self_attention_res_proj_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> mlp_res_proj_;
};

} // namespace infinilm::models::kimi_k3
