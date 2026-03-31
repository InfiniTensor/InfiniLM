#pragma once

#include "minicpm_sala_decoderLayer.hpp"
#include <memory>

namespace infinilm::models::minicpm_sala {

using MiniCPMSALAModel = infinilm::layers::causal_lm_templates::TextModel<MiniCPMSALADecoderLayer>;

class MiniCPMSALAForCausalLM : public InfinilmModel {
public:
    MiniCPMSALAForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

protected:
    INFINICORE_NN_MODULE(MiniCPMSALAModel, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_minicpm_sala_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

/** Implemented in `minicpm_sala_allocate_kv_cache_tensors.cpp`. */
std::vector<std::tuple<infinicore::Tensor, infinicore::Tensor>> minicpm_sala_allocate_kv_cache_tensors(const cache::CacheConfig *cache_config,
                                                                                                       const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
                                                                                                       const backends::AttentionBackend &attention_backend);
} // namespace infinilm::models::minicpm_sala
