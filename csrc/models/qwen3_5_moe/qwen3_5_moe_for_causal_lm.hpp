#pragma once

#include "qwen3_5_moe_model.hpp"

#include <memory>

namespace infinilm::models::qwen3_5_moe {

class Qwen35MoeForConditionalGeneration : public InfinilmModel {
public:
    Qwen35MoeForConditionalGeneration(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

protected:
    INFINICORE_NN_MODULE(Qwen35MoeModel, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_moe_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_5_moe
