#pragma once

#include "../../../config/model_config.hpp"
#include "../../linear/linear.hpp"
#include "infinicore/nn/module.hpp"

namespace infinilm::layers::moe::legacy {

class MoeMLP : public infinicore::nn::Module {
public:
    MoeMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

    size_t hidden_size() const { return hidden_size_; }
    size_t moe_intermediate_size() const { return moe_intermediate_size_; }
    infinicore::Tensor gate_weight() const { return gate_proj_->weight(); }
    infinicore::Tensor up_weight() const { return up_proj_->weight(); }
    infinicore::Tensor down_weight() const { return down_proj_->weight(); }
    void set_alpha(float alpha) { down_proj_->set_alpha(alpha); }

protected:
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> gate_proj_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> up_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> down_proj_;

    size_t hidden_size_;
    size_t moe_intermediate_size_;
    bool use_bias_;
};

} // namespace infinilm::layers::moe::legacy
