#pragma once

#include "../infinilm_model.hpp"
#include "minicpm_sala_model.hpp"

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"

#include "infinicore/device.hpp"

namespace infinilm::models::minicpm_sala {

// Milestone-0 stub. Full implementation will follow the MiniCPM-SALA design:
// - Lightning Attention (Simple GLA) layers + InfLLM-V2 sparse layers in a 1:3 ratio
// - HyPE (RoPE on linear layers; NoPE on sparse layers)
class MiniCPMSALAForCausalLM : public InfinilmModel {
public:
    MiniCPMSALAForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

private:
    INFINICORE_NN_MODULE(MiniCPMSALAModel, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

} // namespace infinilm::models::minicpm_sala

namespace infinilm::models::minicpm_sala {

std::shared_ptr<infinilm::config::ModelConfig> create_minicpm_sala_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minicpm_sala
