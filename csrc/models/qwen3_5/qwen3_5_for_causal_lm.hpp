#pragma once

#include "../../layers/linear/vocab_parallel.hpp"
#include "qwen3_5_model.hpp"
#include "qwen3_5_mtp.hpp"
#include <memory>
#include <vector>

namespace infinilm::models::qwen3_5 {

class Qwen35ForCausalLM : public InfinilmModel {
public:
    Qwen35ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      const infinicore::Device &device);

    Output forward(const Input &input) const override;
    bool supports_token_state_checkpoints() const override { return mtp_ != nullptr; }

    void reset_cache(const cache::CacheConfig *cache_config) override;

protected:
    INFINICORE_NN_MODULE(Qwen35Model, model);
    INFINICORE_NN_MODULE(Qwen35MTP, mtp);
    INFINICORE_NN_MODULE(infinilm::nn::VocabParallelLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> prepare_qwen3_5_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_5
