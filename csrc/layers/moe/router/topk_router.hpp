#pragma once

#include "../../../config/model_config.hpp"
#include "../../common_modules.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>

namespace infinilm::layers::moe {

enum class TopKRouterBackend {
    Softmax,
    Sigmoid,
    FusedGate,
    VllmTopK,
};

class TopKRouter : public infinicore::nn::Module {
public:
    TopKRouter(std::shared_ptr<infinilm::config::ModelConfig> model_config,
               const infinicore::Device &device);
    void process_weights_after_loading() override;

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(e_score_correction_bias);

    infinicore::Tensor runtime_correction_bias_;
    size_t num_experts_{0};
    size_t num_experts_per_tok_{0};
    size_t num_expert_group_{0};
    size_t topk_group_{0};
    size_t num_fused_shared_experts_{0};
    float routed_scaling_factor_{1.0f};
    float moe_softcapping_{0.0f};
    bool norm_topk_prob_{false};
    bool apply_routed_scaling_factor_on_output_{false};
    bool use_correction_bias_{false};
    TopKRouterBackend router_backend_{TopKRouterBackend::Softmax};
    std::string topk_method_{"greedy"};
    std::string scoring_func_{"softmax"};
};

} // namespace infinilm::layers::moe
