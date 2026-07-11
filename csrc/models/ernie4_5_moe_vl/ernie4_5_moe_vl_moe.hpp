#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/common_modules.hpp"
#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/module.hpp"

#include <memory>
#include <tuple>
#include <vector>

namespace infinilm::models::ernie4_5_moe_vl {

// SwiGLU expert / shared FFN with an *explicit* intermediate size, because
// ERNIE-4.5-VL uses different intermediate sizes per modality
// (config "moe_intermediate_size": [1536, 512]) and for shared experts.
// Mirrors infinilm::layers::MoeMLP but parameterized instead of config-read.
class Ernie4_5_VLMoeMLP : public infinicore::nn::Module {
public:
    Ernie4_5_VLMoeMLP(size_t hidden_size,
                      size_t intermediate_size,
                      bool use_bias,
                      const infinicore::DataType &dtype,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

    // Scales the down_proj output, used to fold the routing weight into the
    // expert output (same trick as qwen3_moe).
    void set_alpha(float alpha) { down_proj_->set_alpha(alpha); }

protected:
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> gate_proj_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> up_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> down_proj_;

    size_t hidden_size_;
    size_t intermediate_size_;
    bool use_bias_;
};

// Aux-loss-free load-balancing bias (config "moe_use_aux_free": true).
// Weight name: mlp.moe_statics.e_score_correction_bias, shape
// [num_modalities, num_experts]. Added to scores ONLY for top-k selection,
// not for the combine weights (DeepSeek-V3 style).
class Ernie4_5_VLMoeStatics : public infinicore::nn::Module {
public:
    Ernie4_5_VLMoeStatics(size_t num_modalities,
                          size_t num_experts,
                          const infinicore::Device &device);

    // Returns the correction bias row for the given modality (0 = text, 1 = vision).
    infinicore::Tensor bias_for_modality(size_t modality) const;

protected:
    INFINICORE_NN_PARAMETER(e_score_correction_bias);
    size_t num_modalities_{0};
    size_t num_experts_{0};
};

// Router gate. ERNIE-4.5-VL keeps a separate routing projection per modality:
//   mlp.gate.weight     -> text experts   [num_experts_text, hidden]
//   mlp.gate.weight_1   -> vision experts [num_experts_vision, hidden]   (name TBD)
// Selection uses scores + e_score_correction_bias; combine weights use raw scores,
// optionally renormalized over the top-k (config: norm via "moe_..." flag).
class Ernie4_5_VLMoeGate : public infinicore::nn::Module {
public:
    Ernie4_5_VLMoeGate(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       const infinicore::Device &device);

    // hidden_states: [num_tokens, hidden]; modality picks which gate weight to use.
    // correction_bias: row from Ernie4_5_VLMoeStatics for aux-free selection.
    // Returns (topk_weights [num_tokens, k], topk_indices [num_tokens, k]).
    std::tuple<infinicore::Tensor, infinicore::Tensor>
    forward(const infinicore::Tensor &hidden_states,
            size_t modality,
            const std::optional<infinicore::Tensor> &correction_bias) const;

protected:
    INFINICORE_NN_PARAMETER(weight);    // text gate   (mlp.gate.weight)
    INFINICORE_NN_PARAMETER(weight_1);  // vision gate (mlp.gate.weight_1)

    size_t moe_k_{0};
};

// Thin container for the routed expert list. Registered as "experts" in the
// SparseMoeBlock so individual expert weights load at mlp.experts.N.* — matching
// the HF checkpoint layout where experts is an nn.ModuleList.
// Text experts occupy indices [0, num_text) and vision experts [num_text, total).
class Ernie4_5_VLMoeExpertList : public infinicore::nn::Module {
public:
    Ernie4_5_VLMoeExpertList(size_t num_experts_text,
                              size_t num_experts_vision,
                              size_t hidden_size,
                              size_t inter_text,
                              size_t inter_vision,
                              bool use_bias,
                              const infinicore::DataType &dtype,
                              const infinicore::Device &device);

    // Direct access for dispatch in SparseMoeBlock.
    std::vector<std::shared_ptr<Ernie4_5_VLMoeMLP>> experts;
};

// The full sparse MoE block for one decoder layer.
// forward routes each token to its modality's experts (text vs. vision) using
// token_type_ids, runs top-k routed experts, adds shared-expert output.
class Ernie4_5_VLMoeSparseMoeBlock : public infinicore::nn::Module {
public:
    Ernie4_5_VLMoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const infinicore::Device &device);

    // hidden_states: [batch, seq, hidden]; token_type_ids: [batch, seq] (0=text, 1=vision).
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &token_type_ids) const;

protected:
    INFINICORE_NN_MODULE(Ernie4_5_VLMoeGate, gate);
    INFINICORE_NN_MODULE(Ernie4_5_VLMoeStatics, moe_statics);
    // Registered as "experts" -> weight paths mlp.experts.N.*
    INFINICORE_NN_MODULE(Ernie4_5_VLMoeExpertList, experts);
    // 2 shared experts fused into one MLP (intermediate = num_shared * inter_text).
    INFINICORE_NN_MODULE(Ernie4_5_VLMoeMLP, shared_experts);

    size_t num_experts_text_{0};
    size_t num_experts_vision_{0};
    size_t moe_k_{0};
};

} // namespace infinilm::models::ernie4_5_moe_vl
