#pragma once

#include "../qwen3/qwen3_attention.hpp"
#include "lfm2_mlp.hpp"
#include "lfm2_short_conv.hpp"
#include "lfm2_rms_norm.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/nn/rmsnorm.hpp>

#include <string>
#include <tuple>

namespace infinilm::models::lfm2 {

class Lfm2DecoderLayer : public infinicore::nn::Module {
public:
    Lfm2DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t layer_idx,
                     const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(
        const infinicore::Tensor &positions,
        infinicore::Tensor &hidden_states,
        infinicore::Tensor &residual);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               infinicore::Tensor &hidden_states);

    size_t layer_idx() const { return layer_idx_; }

protected:
    INFINICORE_NN_MODULE(Lfm2RMSNorm, operator_norm);
    INFINICORE_NN_MODULE(Lfm2RMSNorm, ffn_norm);
    INFINICORE_NN_MODULE(infinilm::models::qwen3::Qwen3Attention, self_attn);
    INFINICORE_NN_MODULE(Lfm2ShortConv, conv);
    INFINICORE_NN_MODULE(Lfm2MLP, feed_forward);

private:
    size_t layer_idx_{0};
    std::string layer_type_;
};

} // namespace infinilm::models::lfm2
