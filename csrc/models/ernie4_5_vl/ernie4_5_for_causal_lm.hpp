#pragma once

#include "ernie4_5_model.hpp"

namespace infinilm::models::ernie4_5_vl {

class Ernie45ForConditionalGeneration : public InfinilmModel {
public:
    Ernie45ForConditionalGeneration(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                    const infinicore::Device &device);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

protected:
    INFINICORE_NN_MODULE(Ernie45VisionModel, vision_model);
    INFINICORE_NN_MODULE(Ernie45Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_ernie4_5_moe_vl_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::ernie4_5_vl
