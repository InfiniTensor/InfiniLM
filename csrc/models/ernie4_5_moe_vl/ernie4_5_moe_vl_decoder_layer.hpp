#pragma once

#include "ernie4_5_moe_vl_attention.hpp"
#include "ernie4_5_moe_vl_moe.hpp"

#include "../../layers/mlp/mlp.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"

#include <memory>
#include <tuple>

namespace infinilm::models::ernie4_5_moe_vl {

// One transformer block. Layers are heterogeneous:
//   - layer 0 is dense (regular SwiGLU MLP), per moe_layer_start_index = [1, 1].
//   - layers 1..27 are modality-specific MoE blocks.
// forward takes token_type_ids (used only by MoE layers to split text/vision routing).
class Ernie4_5_VLMoeDecoderLayer : public infinicore::nn::Module {
public:
    Ernie4_5_VLMoeDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               size_t layer_idx,
                               const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor>
    forward(const infinicore::Tensor &positions,
            infinicore::Tensor &hidden_states,
            infinicore::Tensor &residual,
            const infinicore::Tensor &token_type_ids);

    size_t layer_idx() const { return layer_idx_; }
    bool is_moe_layer() const { return is_moe_layer_; }

    static bool compute_is_moe_layer(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                                     size_t layer_idx);

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(Ernie4_5_VLMoeAttention, self_attn);

    // Exactly one of these is registered as "mlp" depending on is_moe_layer_.
    std::shared_ptr<infinilm::layers::mlp::MLP> dense_mlp_;
    std::shared_ptr<Ernie4_5_VLMoeSparseMoeBlock> moe_block_;

    size_t layer_idx_;
    bool is_moe_layer_;
};

} // namespace infinilm::models::ernie4_5_moe_vl
