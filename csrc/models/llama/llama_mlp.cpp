#include "llama_mlp.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::models::llama {

LlamaMLP::LlamaMLP(const LlamaConfig &config,
                   const infinicore::Device &device,
                   engine::distributed::RankInfo rank_info)
    : hidden_size_(config.hidden_size),
      intermediate_size_(config.intermediate_size),
      use_bias_(config.mlp_bias), rank_info_(rank_info) {
    const auto &dtype{config.dtype};

    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;

    // Initialize projection layers
    INFINILM_GATE_UP_LINEAR_INIT(gate_up_proj, "gate_proj", "up_proj", hidden_size_, intermediate_size_, use_bias_,
                                 dtype, device, rank_info_);
    INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size_, hidden_size_, use_bias_,
                              dtype, device, tp_rank, tp_size, rank_info.comm);
}

infinicore::Tensor LlamaMLP::forward(const infinicore::Tensor &hidden_states) const {
    // 1. Project to gate and up
    auto hidden_states_mutable = hidden_states;
    auto [gate, up] = gate_up_proj_->forward_split(hidden_states_mutable);

    // 2. Apply SwiGLU: silu(gate) * up
    // Note: swiglu kernel expects (up, gate) and computes gate * sigmoid(gate) * up
    // So we pass (up, gate) to get the correct result: gate * sigmoid(gate) * up
    auto intermediate = infinicore::op::swiglu(up, gate);

    // 3. Project down
    auto output = down_proj_->forward(intermediate);

    return output;
}

} // namespace infinilm::models::llama
