#pragma once
#include "../../weights/weights.hpp"

#include "../../layers/common_modules.hpp"
#include "../../models/infinilm_model.hpp"
#include "../../layers/embedding/embedding.hpp"
#include "../../layers/normalization/rmsnorm.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::minicpm4 {

class MiniCPM4Attention : public infinilm::layers::attention::Attention {
public:
    MiniCPM4Attention(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      size_t layer_idx,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config);
};

class MiniCPM4MLP : public infinilm::layers::mlp::MLP {
public:
    MiniCPM4MLP(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config);
};

class MiniCPM4DecoderLayer : public infinilm::weights::ModelWeights {
public:
    MiniCPM4DecoderLayer(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t layer_idx,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config);

    std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> forward(const infinilm::runtime::Tensor &positions,
                                                               infinilm::runtime::Tensor &hidden_states,
                                                               infinilm::runtime::Tensor &residual,
                                                               global_state::ForwardContext &forward_context);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &positions,
                               infinilm::runtime::Tensor &hidden_states,
                               global_state::ForwardContext &forward_context);

protected:
    std::shared_ptr<infinilm::layers::RMSNorm> input_layernorm_;
    std::shared_ptr<MiniCPM4Attention> self_attn_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_attention_layernorm_;
    std::shared_ptr<MiniCPM4MLP> mlp_;
};

class MiniCPM4Model : public infinilm::weights::ModelWeights {
public:
    MiniCPM4Model(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(const infinilm::InfinilmModel::Input &input,
                              global_state::ForwardContext &forward_context) const;

protected:
    std::shared_ptr<infinilm::layers::Embedding> embed_tokens_;
    std::vector<std::shared_ptr<MiniCPM4DecoderLayer>> layers_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm_;
};

class MiniCPM4ForCausalLM : public InfinilmModel {
public:
    MiniCPM4ForCausalLM(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config);


    Output forward(const Input &input,
                   global_state::ForwardContext &forward_context) const override;

protected:
    std::shared_ptr<MiniCPM4Model> model_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> lm_head_;
};

std::shared_ptr<infinilm::config::ModelConfig> create_minicpm4_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minicpm4
