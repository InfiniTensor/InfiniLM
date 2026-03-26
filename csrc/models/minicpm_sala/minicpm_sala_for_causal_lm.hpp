#pragma once

#include "minicpm_sala_decoderLayer.hpp"

namespace infinilm::models::minicpm_sala {

using MiniCPMSALAModel = infinilm::layers::TextModel<MiniCPMSALADecoderLayer>;

class MiniCPMSALAForCausalLM : public InfinilmModel {
public:
    MiniCPMSALAForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

private:
    void initialize_kv_cache_(const cache::CacheConfig *cache_config,
                              const std::shared_ptr<infinilm::config::ModelConfig> &text_config);

protected:
    INFINICORE_NN_MODULE(MiniCPMSALAModel, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_minicpm_sala_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minicpm_sala
