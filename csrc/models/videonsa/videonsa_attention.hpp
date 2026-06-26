#pragma once

#include "../../layers/attention/attention.hpp"
#include <array>

#include <optional>

namespace infinilm::models::videonsa {

class VideoNSAAttention : public infinilm::layers::attention::Attention {
public:
    VideoNSAAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      size_t layer_idx,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, g_proj_1);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, g_proj_2);

    mutable std::optional<infinicore::Tensor> nsa_k_cmp_cache_;
    mutable std::optional<infinicore::Tensor> nsa_v_cmp_cache_;
    mutable bool nsa_cmp_cache_ready_ = false;
    mutable std::optional<infinicore::Tensor> mrope_sin_cache_;
    mutable std::optional<infinicore::Tensor> mrope_cos_cache_;
    std::array<int, 3> mrope_section_ = {0, 0, 0};
    size_t mrope_rotary_dim_ = 0;
    bool mrope_interleaved_ = false;
    size_t total_num_attention_heads_ = 0;
    size_t max_position_embeddings_ = 0;
};

} // namespace infinilm::models::videonsa
