#pragma once

#include "../../layers/attention/attention.hpp"
#include "infinicore/nn/rmsnorm.hpp"

namespace infinilm::models::qwen3 {

class Qwen3Attention : public infinilm::layers::attention::Attention {
public:
    Qwen3Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                   size_t layer_idx,
                   const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

    void forward_pre_attn_piecewise(const infinicore::Tensor &positions,
                                    const infinicore::Tensor &hidden_states,
                                    global_state::PiecewiseLayerStaging &staging) const;

    void forward_eager_attn_piecewise(const infinicore::Tensor &positions,
                                      global_state::PiecewiseLayerStaging &staging) const {
        Attention::forward_eager_attn_piecewise(positions, staging);
    }

    void forward_post_attn_piecewise_into(infinicore::Tensor &hidden_states,
                                          global_state::PiecewiseLayerStaging &staging) const {
        Attention::forward_post_attn_piecewise_into(hidden_states, staging);
    }

private:
    infinicore::Tensor forward_static_(const infinicore::Tensor &positions,
                                       const infinicore::Tensor &hidden_states) const;

    infinicore::Tensor forward_paged_(const infinicore::Tensor &positions,
                                      const infinicore::Tensor &hidden_states) const;

    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, k_norm);
};

} // namespace infinilm::models::qwen3
