#pragma once

#include "../../layers/common_modules.hpp"
#include "deepseek_decoder_layer.hpp"
#include <memory>

namespace infinilm::models::deepseek {

using DeepseekModel = infinilm::layers::causal_lm_templates::TextModel<DeepseekDecoderLayer, true>;

class DeepseekForCausalLM : public infinilm::layers::causal_lm_templates::TextCausalLM<DeepseekModel> {
public:
    using Base = infinilm::layers::causal_lm_templates::TextCausalLM<DeepseekModel>;
    DeepseekForCausalLM(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config)
        : Base(runtime, model_config, rank_info, attention_backend, ep_config) {}
};

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::deepseek
