#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/attention/attention.hpp"
#include "../deepseek_v2/deepseek_v2_moe.hpp"
#include "runtime/device.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "runtime/tensor.hpp"

#include <memory>
#include <tuple>
#include <variant>

namespace infinilm::models::deepseek {

using DeepseekMLP = std::variant<std::shared_ptr<deepseek_v2::DeepseekV2MLP>,
                                 std::shared_ptr<deepseek_v2::DeepseekV2MoE>>;

using DeepseekAttention = infinilm::layers::attention::Attention;

class DeepseekDecoderLayer : public infinilm::weights::ModelWeights {
public:
    DeepseekDecoderLayer(infinilm::runtime::Runtime &runtime,
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
    std::shared_ptr<infinilm::layers::RMSNorm> input_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_attention_layernorm_;
    std::shared_ptr<DeepseekAttention> self_attn_;
    std::shared_ptr<DeepseekMLP> mlp_;
};

} // namespace infinilm::models::deepseek
