#include "minicpm_sala_decoder_layer.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace infinilm::models::minicpm_sala {

MiniCPMSALADecoderLayer::MiniCPMSALADecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                 const infinicore::Device &device,
                                                 size_t layer_idx,
                                                 const std::string &mixer_type) {
    // Match parameter dtype with checkpoint `torch_dtype` (e.g. BF16 for MiniCPM-SALA).
    const auto dtype = model_config->get_dtype();
    const double eps = model_config->get<double>("rms_norm_eps");

    // MiniCPM-SALA MuP scaling is baked into weights at load time (Python).
    // Keep C++ forward as plain residual adds.

    INFINICORE_NN_MODULE_INIT(input_layernorm, model_config->get<size_t>("hidden_size"), eps, dtype, device);
    if (mixer_type == "minicpm4") {
        self_attn_ = this->register_module<MiniCPMSALAMinicpm4Attention>(
            "self_attn", model_config, device, layer_idx);
    } else {
        self_attn_ = this->register_module<MiniCPMSALALightningAttention>(
            "self_attn", model_config, device, layer_idx);
    }
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, model_config->get<size_t>("hidden_size"), eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, model_config, device);
}

void MiniCPMSALADecoderLayer::reset_attn_state() {
    self_attn_->reset_state();
}

infinicore::Tensor MiniCPMSALADecoderLayer::forward(const infinicore::Tensor &hidden_states,
                                                    const infinicore::Tensor &position_ids) const {
    // Pre-norm attention
    auto hs1 = input_layernorm_->forward(hidden_states);
    auto attn_out = self_attn_->forward(position_ids, hs1);

    // residual + attn_out
    auto out1 = infinicore::op::add(hidden_states, attn_out);

    // Pre-norm MLP
    auto hs2 = post_attention_layernorm_->forward(out1);
    auto mlp_out = mlp_->forward(hs2);
    // residual + mlp_out
    auto out2 = infinicore::op::add(out1, mlp_out);

    return out2;
}

} // namespace infinilm::models::minicpm_sala
