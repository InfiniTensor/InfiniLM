#pragma once

#include "../../layers/common_modules.hpp"
#include "ernie4_5_attention.hpp"
#include "ernie4_5_sparse_moe_block.hpp"

#include <memory>
#include <tuple>

namespace infinilm::models::ernie4_5_moe_vl {

class Ernie4_5DecoderLayer : public infinicore::nn::Module {
public:
    Ernie4_5DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t layer_idx,
                         const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &positions,
                                                               infinicore::Tensor &hidden_states,
                                                               infinicore::Tensor &residual);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states);

    size_t layer_idx() const { return layer_idx_; }

private:
    infinicore::Tensor forward_mlp_(const infinicore::Tensor &hidden_states) const;
    static bool is_moe_layer(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                             size_t layer_idx);

    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(Ernie4_5Attention, self_attn);
    std::shared_ptr<infinicore::nn::Module> mlp_;

    size_t layer_idx_{0};
    bool use_moe_{false};
};

} // namespace infinilm::models::ernie4_5_moe_vl
