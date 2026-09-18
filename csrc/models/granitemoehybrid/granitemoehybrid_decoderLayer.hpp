#pragma once

#include "granitemoehybrid_attention.hpp"
#include "granitemoehybrid_mamba.hpp"
#include "granitemoehybrid_shared_mlp.hpp"
#include "granitemoehybrid_sparse_moe_block.hpp"

#include <memory>
#include <string>
#include <tuple>

namespace infinilm::models::granitemoehybrid {

class GraniteMoeHybridDecoderLayer : public infinicore::nn::Module {
public:
    GraniteMoeHybridDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 size_t layer_idx,
                                 const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(
        const infinicore::Tensor &positions,
        infinicore::Tensor &hidden_states,
        infinicore::Tensor &residual) const;

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states) const;

    size_t layer_idx() const { return layer_idx_; }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(GraniteMoeHybridAttention, self_attn);
    INFINICORE_NN_MODULE(GraniteMoeHybridMamba, mamba);
    INFINICORE_NN_MODULE(GraniteMoeHybridSparseMoeBlock, block_sparse_moe);
    INFINICORE_NN_MODULE(GraniteMoeHybridSharedMLP, shared_mlp);

private:
    size_t layer_idx_;
    std::string layer_type_;
    bool has_experts_;
};

} // namespace infinilm::models::granitemoehybrid
