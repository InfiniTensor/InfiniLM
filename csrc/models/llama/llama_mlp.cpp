#include "llama_mlp.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::models::llama {

LlamaMLP::LlamaMLP(const LlamaConfig &config,
                   const infinicore::Device &device,
                   infinicore::DataType dtype,
                   engine::distributed::RankInfo rank_info)
    : hidden_size_(config.hidden_size),
      intermediate_size_(config.intermediate_size),
      use_bias_(config.mlp_bias), rank_info_(rank_info) {

    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;

    // Initialize projection layers
    INFINICORE_NN_MODULE_INIT(gate_proj, hidden_size_, intermediate_size_, use_bias_,
                              dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(up_proj, hidden_size_, intermediate_size_, use_bias_,
                              dtype, device, tp_rank, tp_size);
    INFINICORE_NN_MODULE_INIT(down_proj, intermediate_size_, hidden_size_, use_bias_,
                              dtype, device, tp_rank, tp_size, rank_info.comm);
}

infinicore::Tensor LlamaMLP::forward(const infinicore::Tensor &hidden_states) const {
    // 1. Project to gate and up
    auto hidden_states_mutable = hidden_states;
    auto gate = gate_proj_->forward(hidden_states_mutable);

    auto up = up_proj_->forward(hidden_states_mutable);

    // 2. Apply SwiGLU: silu(gate) * up
    // Note: swiglu kernel expects (up, gate) and computes gate * sigmoid(gate) * up
    // So we pass (up, gate) to get the correct result: gate * sigmoid(gate) * up
    auto intermediate = infinicore::op::swiglu(up, gate);

    // 3. Project down
    auto output = down_proj_->forward(intermediate);

    return output;
}

} // namespace infinilm::models::llama
