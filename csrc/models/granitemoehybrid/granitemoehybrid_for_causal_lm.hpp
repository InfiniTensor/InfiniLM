#pragma once

#include "granitemoehybrid_decoderLayer.hpp"
#include <memory>

namespace infinilm::models::granitemoehybrid {

using GraniteMoeHybridModel = infinilm::layers::causal_lm_templates::TextModel<GraniteMoeHybridDecoderLayer>;

class GraniteMoeHybridForCausalLM : public InfinilmModel {
public:
    GraniteMoeHybridForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                const infinicore::Device &device);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

protected:
    INFINICORE_NN_MODULE(GraniteMoeHybridModel, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_granitemoehybrid_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::granitemoehybrid
