#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4MLP : public infinicore::nn::Module {
public:
    DeepseekV4MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const infinicore::Device &device);
    DeepseekV4MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  size_t intermediate_size,
                  const infinicore::Device &device);

    void set_alpha(float alpha) { w2_->set_alpha(alpha); }
    infinicore::Tensor gate_weight() const { return w1_->weight(); }
    infinicore::Tensor up_weight() const { return w3_->weight(); }
    infinicore::Tensor down_weight() const { return w2_->weight(); }
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, w1);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, w2);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, w3);

    size_t hidden_size_{0};
    size_t intermediate_size_{0};
    double swiglu_limit_{0.0};
    bool has_swiglu_limit_{false};
};

} // namespace infinilm::models::deepseek_v4
