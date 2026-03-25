#pragma once

#include "../../layers/common_modules.hpp"
#include "qwen3_next_decoderLayer.hpp"

namespace infinilm::models::qwen3_next {

using Qwen3NextModel = infinilm::layers::TextModel<Qwen3NextDecoderLayer>;

class Qwen3NextForCausalLM : public InfinilmModel {
public:
    Qwen3NextForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const infinicore::Device &device);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

private:
    void initialize_kv_cache_(const cache::CacheConfig *cache_config,
                              const std::shared_ptr<infinilm::config::ModelConfig> &text_config);

protected:
    INFINICORE_NN_MODULE(Qwen3NextModel, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_next_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_next
