#pragma once

#include "../infinilm_model.hpp"
#include "minicpm_sala_model.hpp"

#include "../../config/model_config.hpp"
#include "../../engine/distributed/distributed.hpp"
#include "../../backends/attention_backends.hpp"

#include "infinicore/device.hpp"
#include "infinicore/nn/linear.hpp"

namespace infinilm::models::minicpm_sala {

// Milestone-0 stub. Full implementation will follow the MiniCPM-SALA design:
// - Lightning Attention (Simple GLA) layers + InfLLM-V2 sparse layers in a 1:3 ratio
// - HyPE (RoPE on linear layers; NoPE on sparse layers)
class MiniCPMSALAForCausalLM : public InfinilmModel {
public:
    MiniCPMSALAForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device,
                           engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
                           backends::AttentionBackend attention_backend = backends::AttentionBackend::Default);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

    const cache::CacheConfig *get_cache_config() const override;

private:
    INFINICORE_NN_MODULE(MiniCPMSALAModel, model);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, lm_head);
    std::unique_ptr<cache::CacheConfig> cache_config_;
};

} // namespace infinilm::models::minicpm_sala

