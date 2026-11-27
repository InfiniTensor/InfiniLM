#pragma once
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../weights/weights.hpp"

#include "../qwen3_5/qwen3_5_attention.hpp"
#include "clover_gdn.hpp"
#include "clover_moe.hpp"

#include <string>
#include <tuple>

namespace infinilm::models::clover_moe {

class CloverDecoderLayer : public infinilm::weights::ModelWeights {
public:
    CloverDecoderLayer(infinilm::runtime::Runtime &runtime,
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

    size_t layer_idx() const { return layer_idx_; }

protected:
    std::shared_ptr<infinilm::layers::RMSNorm> input_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_attention_layernorm_;
    std::shared_ptr<qwen3_5::Qwen35Attention> self_attn_;
    std::shared_ptr<CloverGatedDeltaNet> linear_attn_;
    std::shared_ptr<CloverSparseMoe> mlp_;

private:
    size_t layer_idx_;
    std::string layer_type_;
};

} // namespace infinilm::models::clover_moe
