#pragma once

#include "../../layers/common_modules.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../weights/weights.hpp"
#include "ernie4_5_attention.hpp"
#include "ernie4_5_moe.hpp"
#include "runtime/tensor.hpp"

#include <memory>
#include <tuple>

namespace infinilm::models::ernie4_5_vl {

class Ernie45DecoderLayer : public infinilm::weights::ModelWeights {
public:
    Ernie45DecoderLayer(infinilm::runtime::Runtime &runtime,
                        std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        size_t layer_idx,
                        std::shared_ptr<const Ernie45MropeCache> mrope_cache,
                        const engine::distributed::RankInfo &rank_info,
                        backends::AttentionBackend attention_backend,
                        const layers::moe::EPConfig &ep_config);

    std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> forward(const infinilm::runtime::Tensor &positions,
                                                                             infinilm::runtime::Tensor &hidden_states,
                                                                             infinilm::runtime::Tensor &residual,
                                                                             global_state::ForwardContext &forward_context);

protected:
    std::shared_ptr<infinilm::layers::RMSNorm> input_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_attention_layernorm_;
    std::shared_ptr<Ernie45Attention> self_attn_;
    std::shared_ptr<infinilm::weights::ModelWeights> mlp_;

private:
    bool use_moe_{false};
};

} // namespace infinilm::models::ernie4_5_vl
