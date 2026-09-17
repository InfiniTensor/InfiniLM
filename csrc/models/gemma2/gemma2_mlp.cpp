#include "gemma2_mlp.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/mul.hpp"

namespace infinilm::models::gemma2 {

Gemma2MLP::Gemma2MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t intermediate_size = model_config->get<size_t>("intermediate_size");
    bool use_bias = model_config->get_or<bool>("mlp_bias", false);

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    gate_up_proj_ = std::make_shared<layers::linear::GateUpParallelLinear>(
        hidden_size, intermediate_size, "gate_proj", "up_proj", register_fn,
        quantization_method, use_bias, dtype, device, rank_info);
    down_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "down_proj", intermediate_size, hidden_size, quantization_method,
        use_bias, dtype, device, tp_rank, tp_size, rank_info.comm);
}

infinicore::Tensor Gemma2MLP::forward(const infinicore::Tensor &hidden_states) const {
    // 1. Project to gate and up
    auto hidden_states_mutable = hidden_states;
    auto [gate, up] = gate_up_proj_->forward_split(hidden_states_mutable);
    // 2. Gemma-2 activation: gelu_pytorch_tanh on the gate branch, then element-wise product
    auto activated = infinicore::op::gelu_tanh(gate);
    auto intermediate = infinicore::op::mul(activated, up);
    // 3. Project down
    auto output = down_proj_->forward(intermediate);
    return output;
}

} // namespace infinilm::models::gemma2
