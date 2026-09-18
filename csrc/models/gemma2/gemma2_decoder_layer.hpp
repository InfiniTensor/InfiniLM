#pragma once

#include "../../layers/common_modules.hpp"
#include "gemma2_attention.hpp"
#include "gemma2_mlp.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include <memory>

namespace infinilm::models::gemma2 {

/**
 * @brief Gemma-2 decoder layer.
 *
 * Unlike the llama-style TextDecoderLayer, Gemma-2 applies four RMSNorms and
 * normalizes each branch *before* adding it back to the residual stream
 * (`hidden = residual + post_norm(branch)`), so this layer cannot reuse the
 * fused llama template. The TextModel residual contract is still honored:
 * on entry `hidden` is the previous branch output (not yet added) and
 * `residual` is the mainstream; on exit the same holds, deferring the final
 * addition to the consumer (next layer's input norm or the model's final norm).
 */
class Gemma2DecoderLayer : public infinicore::nn::Module {
public:
    Gemma2DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
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
    INFINICORE_NN_MODULE(Gemma2Attention, self_attn);
    INFINICORE_NN_MODULE(Gemma2MLP, mlp);

    size_t layer_idx_;
    double rms_norm_eps_;
};

} // namespace infinilm::models::gemma2
