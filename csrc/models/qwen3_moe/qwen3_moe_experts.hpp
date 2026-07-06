#pragma once

#include <infiniccl.h>
#include <memory>

namespace infinilm::models::qwen3_moe {

// Stacked-weight experts that dispatch a single `grouped_gemm` per projection
// instead of one MLP forward per (token, expert) pair.
//
// Weight layout (after TP slicing):
//   gate_proj : [num_experts, moe_intermediate_size / tp_size, hidden_size]
//   up_proj   : [num_experts, moe_intermediate_size / tp_size, hidden_size]
//   down_proj : [num_experts, hidden_size, moe_intermediate_size / tp_size]
//
// HF state_dict keys `experts.{i}.{gate,up,down}_proj.weight` are aliased to
// views into the corresponding slab so the existing safetensors loader fills
// them directly -- no Python remapper required.
class Qwen3MoeExperts : public infinicore::nn::Module {
public:
    Qwen3MoeExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &top_k_index,
                               const infinicore::Tensor &top_k_weights) const;

protected:
    // Stacked weight parameters. We register them under non-HF names so the
    // load_state_dict walk never picks them up directly; only the per-expert
    // aliases below are populated by the loader.
    INFINICORE_NN_PARAMETER(gate_proj_stacked);
    INFINICORE_NN_PARAMETER(up_proj_stacked);
    INFINICORE_NN_PARAMETER(down_proj_stacked);

    size_t num_experts_{0};
    size_t num_experts_per_tok_{0};
    size_t hidden_size_{0};
    size_t moe_intermediate_size_per_rank_{0};
    infinicore::DataType dtype_;
    infinicore::Device device_;

    int tp_size_{1};
    int tp_rank_{0};
    infinicclComm_t communicator_{nullptr};
};

} // namespace infinilm::models::qwen3_moe
