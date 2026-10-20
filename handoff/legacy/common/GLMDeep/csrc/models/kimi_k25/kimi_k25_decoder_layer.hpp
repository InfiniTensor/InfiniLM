#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "kimi_k25_attention.hpp"
#include "kimi_k25_moe.hpp"

#include "../../weights/weights.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include <runtime/tensor.hpp>

#include <cstddef>
#include <memory>
#include <tuple>

namespace infinilm::models::kimi_k25 {

class KimiK25DecoderLayer : public infinilm::weights::ModelWeights {
public:
    KimiK25DecoderLayer(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        size_t layer_idx,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config);

    std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor>
    forward(const infinilm::runtime::Tensor &positions,
            infinilm::runtime::Tensor &hidden_states,
            infinilm::runtime::Tensor &residual,
            global_state::ForwardContext &forward_context) const;

protected:
    std::shared_ptr<KimiK25Attention> self_attn_;
    std::shared_ptr<infinilm::layers::RMSNorm> input_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_attention_layernorm_;
    std::shared_ptr<infinilm::layers::mlp::MLP> dense_mlp_;
    std::shared_ptr<KimiK25MoE> moe_mlp_;
    bool use_moe_{false};
};

} // namespace infinilm::models::kimi_k25
