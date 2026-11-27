#pragma once

#include "../../config/model_config.hpp"
#include "../../engine/distributed/communication_group.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../weights/weights.hpp"
#include "gemma4_config.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <cstddef>
#include <memory>
#include <tuple>

namespace infinilm::models::gemma4 {

// Gemma4TextRouter:
//   h = RMSNorm(x, with_scale=False) * scale * (hidden_size^-0.5)
//   logits = proj(h)
//   probs = softmax(logits, fp32)
//   top_k_weights, top_k_index = topk(probs, k=top_k_experts)
//   top_k_weights = (top_k_weights / sum) * per_expert_scale[top_k_index]
class Gemma4TextRouter : public infinilm::weights::ModelWeights {
public:
    Gemma4TextRouter(infinilm::runtime::Runtime &runtime,
                     std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     const Gemma4TextSpec &spec);

    void process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) override;

    // Input: [num_tokens, hidden_size]
    // Returns: (top_k_weights [num_tokens, k] FP32, top_k_index [num_tokens, k] INT32)
    std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> forward(
        infinilm::runtime::Runtime &runtime,
        const infinilm::runtime::Tensor &hidden_states_2d) const;

private:
    size_t hidden_size_{0};
    size_t num_experts_{0};
    size_t top_k_experts_{0};
    double rms_norm_eps_{1e-6};
    double scalar_root_size_{1.0};

    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> proj_;
    infinilm::weights::Weight scale_;
    infinilm::weights::Weight per_expert_scale_;

    infinilm::runtime::Tensor ones_hidden_;
    infinilm::runtime::Tensor scale_eff_;             // scale * scalar_root_size
    infinilm::runtime::Tensor per_expert_scale_fp32_; // [num_experts, 1] FP32 for Embedding lookup
};

// Gemma4TextExperts:
//   3D packed checkpoint weights gate_up_proj [E, 2*M, H] and down_proj [E, H, M]
//   with exact GeGLU (gelu_pytorch_tanh(gate) * up) on GPU.
class Gemma4TextExperts : public infinilm::weights::ModelWeights {
public:
    Gemma4TextExperts(infinilm::runtime::Runtime &runtime,
                      std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      const Gemma4TextSpec &spec,
                      const engine::distributed::RankInfo &rank_info);

    // Input: hidden_states_2d [T, H], top_k_index [T, K] INT32, top_k_weights [T, K] FP32
    // Output: [T, H] in model dtype
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states_2d,
                                      const infinilm::runtime::Tensor &top_k_index,
                                      const infinilm::runtime::Tensor &top_k_weights) const;

private:
    size_t num_experts_{0};
    size_t top_k_experts_{0};
    size_t hidden_size_{0};
    size_t moe_intermediate_size_{0};
    size_t moe_intermediate_per_rank_{0};
    int tp_size_{1};
    infinicclComm_t tp_comm_{nullptr};

    infinilm::weights::Weight gate_proj_; // [E, M/tp, H]
    infinilm::weights::Weight up_proj_;   // [E, M/tp, H]
    infinilm::weights::Weight down_proj_; // [E, H, M/tp]
};

} // namespace infinilm::models::gemma4
