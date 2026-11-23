#pragma once

#include "deepseek_v4_mla.hpp"
#include "deepseek_v4_moe.hpp"
#include "../../layers/normalization/rmsnorm.hpp"

namespace infinilm::global_state { struct ForwardContext; }
namespace infinilm::engine::distributed { struct RankInfo; }
namespace infinilm::backends { enum class AttentionBackend; }
namespace infinilm::layers::moe { struct EPConfig; }

namespace infinilm::models::deepseek_v4 {

class DeepseekV4DecoderLayer final : public infinilm::weights::ModelWeights {
public:
    DeepseekV4DecoderLayer(infinilm::runtime::Runtime &runtime,
                           std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
                           const engine::distributed::RankInfo &rank_info,
                           backends::AttentionBackend attention_backend,
                           const layers::moe::EPConfig &ep_config,
                           std::shared_ptr<infinilm::layers::RoPE> rope);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &positions,
                                      const infinilm::runtime::Tensor &input_ids,
                                      infinilm::runtime::Tensor hidden_states,
                                      infinilm::runtime::Tensor &residual,
                                      infinilm::global_state::ForwardContext &forward_context) const;

private:
    std::shared_ptr<infinilm::layers::RMSNorm> attn_norm_;
    std::shared_ptr<DeepseekV4MLA> attn_;
    std::shared_ptr<infinilm::layers::RMSNorm> ffn_norm_;
    std::shared_ptr<DeepseekV4MoE> ffn_;
    infinilm::weights::Weight hc_attn_fn_;
    infinilm::weights::Weight hc_ffn_fn_;
    infinilm::weights::Weight hc_attn_base_;
    infinilm::weights::Weight hc_ffn_base_;
    infinilm::weights::Weight hc_attn_scale_;
    infinilm::weights::Weight hc_ffn_scale_;
};

} // namespace infinilm::models::deepseek_v4
