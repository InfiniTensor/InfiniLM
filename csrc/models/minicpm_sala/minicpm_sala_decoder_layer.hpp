#pragma once

#include "minicpm_sala_attention.hpp"
#include "minicpm_sala_mlp.hpp"

#include "../../config/model_config.hpp"

#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <string>

namespace infinilm::models::minicpm_sala {

class MiniCPMSALADecoderLayer : public infinicore::nn::Module {
public:
    MiniCPMSALADecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            const infinicore::Device &device,
                            size_t layer_idx,
                            const std::string &mixer_type);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &position_ids) const;

    void reset_attn_state();

private:
protected:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    // Registered under the HF-compatible name "self_attn" in ctor.
    std::shared_ptr<MiniCPMSALAAttentionBase> self_attn_;
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(MiniCPMSALAMLP, mlp);
};

} // namespace infinilm::models::minicpm_sala
