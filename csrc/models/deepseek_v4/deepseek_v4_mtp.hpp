#pragma once

#include "deepseek_v4_decoder_layer.hpp"
#include "../../layers/embedding/embedding.hpp"

namespace infinilm::global_state { struct ForwardContext; }
namespace infinilm::engine::distributed { struct RankInfo; }
namespace infinilm::backends { enum class AttentionBackend; }
namespace infinilm::layers::moe { struct EPConfig; }

namespace infinilm::models::deepseek_v4 {

class DeepseekV4MTP final : public infinilm::weights::ModelWeights {
public:
    DeepseekV4MTP(infinilm::runtime::Runtime &runtime,
                  std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const engine::distributed::RankInfo &rank_info,
                  backends::AttentionBackend attention_backend,
                  const layers::moe::EPConfig &ep_config,
                  std::shared_ptr<infinilm::layers::RoPE> rope);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &positions,
                                      const infinilm::runtime::Tensor &input_ids,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      infinilm::global_state::ForwardContext &forward_context) const;

    infinilm::runtime::Tensor logits(infinilm::runtime::Runtime &runtime,
                                     const infinilm::runtime::Tensor &hidden_states) const;

private:
    std::shared_ptr<DeepseekV4MLA> attn_;
    std::shared_ptr<infinilm::layers::RMSNorm> attn_norm_;
    std::shared_ptr<DeepseekV4MoE> ffn_;
    std::shared_ptr<infinilm::layers::RMSNorm> ffn_norm_;
    std::shared_ptr<infinilm::layers::RMSNorm> enorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> hnorm_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> e_proj_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> h_proj_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> head_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm_;
    std::shared_ptr<infinilm::layers::Embedding> emb_;
    infinilm::weights::Weight hc_attn_fn_;
    infinilm::weights::Weight hc_ffn_fn_;
    infinilm::weights::Weight hc_head_fn_;
    infinilm::weights::Weight hc_attn_base_;
    infinilm::weights::Weight hc_ffn_base_;
    infinilm::weights::Weight hc_head_base_;
    infinilm::weights::Weight hc_attn_scale_;
    infinilm::weights::Weight hc_ffn_scale_;
    infinilm::weights::Weight hc_head_scale_;
    infinilm::runtime::Tensor hc_head_rms_weight_;
    size_t hidden_size_{0};
    size_t hc_mult_{4};
    double norm_eps_{1e-6};
    double hc_eps_{1e-6};
};

} // namespace infinilm::models::deepseek_v4
