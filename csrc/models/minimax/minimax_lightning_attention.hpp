#pragma once

#include "../../layers/common_modules.hpp"
#include <memory>
#include <string>
#include <vector>

namespace infinilm::models::minimax {

class MiniMaxLightningAttention : public infinicore::nn::Module {
public:
    MiniMaxLightningAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                              size_t layer_idx,
                              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

    size_t layer_idx() const { return layer_idx_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }

private:
    static std::vector<float> build_slopes(size_t num_heads);

    std::shared_ptr<layers::linear::ColumnParallelLinear> qkv_proj_;
    std::shared_ptr<layers::linear::ColumnParallelLinear> output_gate_;
    std::shared_ptr<layers::linear::RowParallelLinear> out_proj_;
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

    infinicore::Tensor slope_;
    infinicore::Tensor ratio_;

    size_t layer_idx_;
    size_t num_heads_;
    size_t head_dim_;
    size_t inner_dim_;
    bool silu_act_{true};
    size_t block_size_;
};

} // namespace infinilm::models::minimax

