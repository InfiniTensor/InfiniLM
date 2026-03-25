#pragma once

#include "../../config/model_config.hpp"
#include "../linear/linear.hpp"
#include "infinicore/nn/module.hpp"

namespace infinilm::layers::moe_mlp {

class MoeMLP : public infinicore::nn::Module {
public:
    MoeMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

    // Module information
    size_t hidden_size() const { return hidden_size_; }
    size_t moe_intermediate_size() const { return moe_intermediate_size_; }

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, gate_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, up_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, down_proj);

    size_t hidden_size_;
    size_t moe_intermediate_size_;
    bool use_bias_;
};

} // namespace infinilm::layers::moe_mlp
