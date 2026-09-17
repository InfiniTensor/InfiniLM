#pragma once

#include "../../layers/common_modules.hpp"

#include <memory>
#include <tuple>

namespace infinilm::models::granitemoehybrid {

class GraniteMoeHybridExpertMLP : public infinicore::nn::Module {
public:
    GraniteMoeHybridExpertMLP(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

    void process_weights_after_loading() override {
        input_linear_->process_weights_after_loading();
    }

    void reset_runtime_state() const override {
        input_linear_->reset_runtime_state();
    }

private:
    INFINICORE_NN_MODULE(infinilm::layers::linear::GateUpParallelLinear, input_linear);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, output_linear);
};

class GraniteMoeHybridExperts : public infinicore::nn::Module {
public:
    GraniteMoeHybridExperts(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);

    infinicore::Tensor forward(
        const infinicore::Tensor &hidden_states,
        const infinicore::Tensor &selected_experts,
        const infinicore::Tensor &routing_weights) const;

private:
    INFINICORE_NN_MODULE_VEC(GraniteMoeHybridExpertMLP, experts);
    size_t num_experts_per_tok_{0};
    size_t num_experts_{0};
    float residual_multiplier_{1.0f};
};

class GraniteMoeHybridTopKRouter : public infinicore::nn::Module {
public:
    GraniteMoeHybridTopKRouter(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(
        const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, layer);
    size_t num_experts_per_tok_{0};
};

class GraniteMoeHybridSparseMoeBlock : public infinicore::nn::Module {
public:
    GraniteMoeHybridSparseMoeBlock(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(GraniteMoeHybridTopKRouter, router);
    INFINICORE_NN_MODULE(GraniteMoeHybridExperts, experts);
};

} // namespace infinilm::models::granitemoehybrid
