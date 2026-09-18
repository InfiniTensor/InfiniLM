#pragma once

#include "../../layers/common_modules.hpp"
#include "gemma3_attention.hpp"
#include "gemma3_mlp.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include <memory>

namespace infinilm::models::gemma3 {

/**
 * @brief Gemma-3 decoder layer.
 *
 * Same four-norm, branch-normalize-then-add structure as Gemma-2 (the residual
 * contract with TextModel is identical); the attention slot is Gemma3's own
 * (QK-norm + sliding/global layer types).
 */
class Gemma3DecoderLayer : public infinicore::nn::Module {
public:
    Gemma3DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       size_t layer_idx,
                       const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &positions,
                                                               infinicore::Tensor &hidden_states,
                                                               infinicore::Tensor &residual);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states);

    size_t layer_idx() const { return layer_idx_; }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, pre_feedforward_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_feedforward_layernorm);
    INFINICORE_NN_MODULE(Gemma3Attention, self_attn);
    INFINICORE_NN_MODULE(Gemma3MLP, mlp);

    size_t layer_idx_;
    double rms_norm_eps_;
};

} // namespace infinilm::models::gemma3
