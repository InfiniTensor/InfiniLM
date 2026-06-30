#pragma once

#include "../../layers/linear/linear.hpp"
#include "../infinilm_model.hpp"
#include "deepseek_v4_model.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4ForCausalLM : public infinilm::InfinilmModel {
public:
    DeepseekV4ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                          const infinicore::Device &device);

    Output forward(const Input &input) const override;
    void reset_cache(const cache::CacheConfig *cache_config) override;

private:
    INFINICORE_NN_MODULE(DeepseekV4Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, head);

    mutable std::vector<int64_t> cached_input_ids_;
};

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_v4_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::deepseek_v4
