#pragma once

#include "../../config/model_config.hpp"
#include "deepseek_v2_attention.hpp"
#include "deepseek_v2_mla_attention.hpp"
#include "deepseek_v2_moe.hpp"
#include "runtime/device.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "runtime/tensor.hpp"

#include <memory>
#include <tuple>
#include <variant>

namespace infinilm::models::deepseek_v2 {

using DeepseekV2SelfAttention = std::variant<std::shared_ptr<DeepseekV2Attention>, std::shared_ptr<DeepseekV2MLAAttention>>;

class DeepseekV2DecoderLayer : public infinilm::weights::ModelWeights {
public:
    DeepseekV2DecoderLayer(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config,
    bool use_mla);

    std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> forward(const infinilm::runtime::Tensor &positions,
                                                               infinilm::runtime::Tensor &hidden_states,
                                                               infinilm::runtime::Tensor &residual,
                                                               infinilm::global_state::ForwardContext &forward_context) const;

private:
    std::shared_ptr<infinilm::layers::RMSNorm> input_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_attention_layernorm_;
    std::shared_ptr<DeepseekV2SelfAttention> self_attn_;
    std::shared_ptr<DeepseekV2MLP> dense_mlp_;
    std::shared_ptr<DeepseekV2MoE> moe_mlp_;
    bool use_moe_{false};
};

} // namespace infinilm::models::deepseek_v2
