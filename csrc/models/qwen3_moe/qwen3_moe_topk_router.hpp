

#pragma once
#include "../../layers/common_modules.hpp"

#include <memory>
#include <tuple>

namespace infinilm::models::qwen3_moe {

class Qwen3MoeTopKRouter : public infinicore::nn::Module {
public:
    Qwen3MoeTopKRouter(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_PARAMETER(weight);

    size_t num_experts_per_tok_{0};
    bool norm_topk_prob_{false};
};

} // namespace infinilm::models::qwen3_moe
