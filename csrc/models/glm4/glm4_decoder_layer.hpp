#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/common_modules.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../weights/weights.hpp"

namespace infinilm::models::glm4 {

using Glm4Attention = infinilm::layers::attention::Attention;

class Glm4DecoderLayer : public infinilm::weights::ModelWeights {
public:
    Glm4DecoderLayer(infinilm::runtime::Runtime &runtime,
                     std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t layer_idx,
                     const engine::distributed::RankInfo &rank_info,
                     backends::AttentionBackend attention_backend,
                     const layers::moe::EPConfig &ep_config);

    std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> forward(
        const infinilm::runtime::Tensor &positions,
        infinilm::runtime::Tensor &hidden_states,
        infinilm::runtime::Tensor &residual,
        infinilm::global_state::ForwardContext &forward_context);

private:
    std::shared_ptr<Glm4Attention> self_attn_;
    std::shared_ptr<infinilm::layers::mlp::MLP> mlp_;
    std::shared_ptr<infinilm::layers::RMSNorm> input_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_self_attn_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_attention_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_mlp_layernorm_;
};

} // namespace infinilm::models::glm4
