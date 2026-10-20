#pragma once
#include "../../weights/weights.hpp"

#include "../../layers/common_modules.hpp"
#include "../../models/infinilm_model.hpp"
#include "../minicpm4/minicpm4_for_causal_lm.hpp"
#include "../../layers/embedding/embedding.hpp"
#include "../../layers/normalization/rmsnorm.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::minicpm_eagle {

class MiniCPMEagleModel : public infinilm::weights::ModelWeights {
public:
    MiniCPMEagleModel(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(const infinilm::InfinilmModel::Input &input,
                              global_state::ForwardContext &forward_context) const;

protected:
    std::shared_ptr<infinilm::layers::Embedding> embed_tokens_;
    std::shared_ptr<infinilm::layers::RMSNorm> input_norm1_;
    std::shared_ptr<infinilm::layers::RMSNorm> input_norm2_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> fc_;
    std::vector<std::shared_ptr<infinilm::models::minicpm4::MiniCPM4DecoderLayer>> eagle_layers_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm_;

    infinilm::runtime::DataType dtype_;
    infinilm::runtime::Device device_;
    size_t hidden_size_;
};

class MiniCPMEagleForCausalLM : public InfinilmModel {
public:
    MiniCPMEagleForCausalLM(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config);

    Output forward(const Input &input,
                   global_state::ForwardContext &forward_context) const override;

protected:
    std::shared_ptr<MiniCPMEagleModel> model_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> lm_head_;
};

std::shared_ptr<infinilm::config::ModelConfig> create_minicpm_eagle_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minicpm_eagle
