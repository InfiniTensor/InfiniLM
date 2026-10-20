#pragma once
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../weights/weights.hpp"

#include "../qwen3_next/qwen3_next_gated_deltanet.hpp"
#include "qwen3_5_attention.hpp"
#include <string>
#include <tuple>

namespace infinilm::models::qwen3_5 {

class Qwen35DecoderLayer : public infinilm::weights::ModelWeights {
public:
    Qwen35DecoderLayer(infinilm::runtime::Runtime &runtime,
                       std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       size_t layer_idx,
                       const engine::distributed::RankInfo &rank_info,
                       backends::AttentionBackend attention_backend,
                       const layers::moe::EPConfig &ep_config);

    std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> forward(const infinilm::runtime::Tensor &positions,
                                                                             infinilm::runtime::Tensor &hidden_states,
                                                                             infinilm::runtime::Tensor &residual,
                                                                             infinilm::global_state::ForwardContext &forward_context);

    size_t layer_idx() const { return layer_idx_; }

protected:
    std::shared_ptr<infinilm::layers::RMSNorm> input_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_attention_layernorm_;
    std::shared_ptr<Qwen35Attention> self_attn_;
    std::shared_ptr<qwen3_next::Qwen3NextGatedDeltaNet> linear_attn_;
    std::shared_ptr<infinilm::layers::MLP> mlp_;

private:
    size_t layer_idx_;
    std::string layer_type_;
};

} // namespace infinilm::models::qwen3_5
