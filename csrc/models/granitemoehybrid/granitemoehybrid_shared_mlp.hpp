#pragma once

#include "../../layers/common_modules.hpp"

#include <memory>

namespace infinilm::models::granitemoehybrid {

class GraniteMoeHybridSharedMLP : public infinicore::nn::Module {
public:
    GraniteMoeHybridSharedMLP(
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

    size_t hidden_size_{0};
    size_t intermediate_size_{0};
};

} // namespace infinilm::models::granitemoehybrid
