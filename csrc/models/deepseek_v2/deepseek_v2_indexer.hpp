#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/layer_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include <infiniccl.h>

#include <memory>

namespace infinilm::models::deepseek_v2 {

class DeepseekV32Indexer final : public infinicore::nn::Module {
public:
    DeepseekV32Indexer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       size_t layer_idx,
                       const infinicore::Device &device);

    void forward(const infinicore::Tensor &hidden_states,
                 const infinicore::Tensor &q_lora,
                 const infinicore::Tensor &positions,
                 infinicore::Tensor topk_indices) const;

    void process_weights_after_loading() override;

private:
    void sync_topk_indices_(infinicore::Tensor topk_indices) const;

    size_t layer_idx_{0};
    size_t num_heads_{0};
    size_t head_dim_{0};
    size_t rope_dim_{0};
    size_t topk_tokens_{0};
    float weights_scale_{1.0f};
    int tp_rank_{0};
    int tp_size_{1};
    infinicclComm_t communicator_{nullptr};

    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, wq_b);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, wk);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, weights_proj);
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear>
        fused_wk_weights_proj_;
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, k_norm);

    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;
    infinicore::Tensor cos_sin_cache_;
    infinicore::Tensor one_i32_;
};

} // namespace infinilm::models::deepseek_v2
