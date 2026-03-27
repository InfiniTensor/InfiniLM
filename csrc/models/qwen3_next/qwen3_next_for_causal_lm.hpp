#pragma once

#include "qwen3_next_decoderLayer.hpp"
#include <memory>

namespace infinilm::models::qwen3_next {

using Qwen3NextModel = infinilm::layers::causal_lm_templates::TextModel<Qwen3NextDecoderLayer>;

class Qwen3NextForCausalLM : public InfinilmModel {
public:
    Qwen3NextForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const infinicore::Device &device);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

protected:
    INFINICORE_NN_MODULE(Qwen3NextModel, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_next_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

/** Implemented in `qwen3_next_allocate_kv_cache_tensors.cpp`. */
std::vector<std::tuple<infinicore::Tensor, infinicore::Tensor>> qwen3_next_allocate_kv_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
    const backends::AttentionBackend &attention_backend);
} // namespace infinilm::models::qwen3_next
