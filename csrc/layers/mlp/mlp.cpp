#include "mlp.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::layers::mlp {

MLP::MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
         const infinicore::Device &device)
    : dtype_(model_config->get_dtype()),
      device_(device) {

    hidden_size_ = model_config->get<size_t>("hidden_size");
    intermediate_size_ = model_config->get<size_t>("intermediate_size");
    use_bias_ = model_config->get_or<bool>("mlp_bias", false);

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    gate_up_proj_ = std::make_shared<layers::linear::GateUpParallelLinear>(
        hidden_size_, intermediate_size_, "gate_proj", "up_proj", register_fn,
        quantization_method, use_bias_, dtype_, device_, rank_info);
    down_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "down_proj", intermediate_size_, hidden_size_, quantization_method,
        use_bias_, dtype_, device_, tp_rank, tp_size, rank_info.comm);

    rank_gate_up_output_size_ = gate_up_proj_->out_features() / static_cast<size_t>(tp_size);
    rank_intermediate_size_ = rank_gate_up_output_size_ / 2;
    const auto &config = infinilm::global_state::get_infinilm_config();
    enable_workspace_manager_ = config.enable_workspace_manager;
    if (enable_workspace_manager_) {
        auto &workspace = infinilm::global_state::get_forward_context().workspace_manager;
        workspace.reserve_slot(
            "mlp.gate_up",
            {config.max_num_batched_tokens, rank_gate_up_output_size_},
            dtype_,
            device_);
        workspace.reserve_slot(
            "mlp.intermediate",
            {config.max_num_batched_tokens, rank_intermediate_size_},
            dtype_,
            device_);
    }
}

infinicore::Tensor MLP::forward(const infinicore::Tensor &hidden_states) const {
    if (enable_workspace_manager_) {
        auto shape = hidden_states->shape();
        auto &workspace = infinilm::global_state::get_forward_context().workspace_manager;
        auto gate_up_output = workspace.get_buffer("mlp.gate_up",
                                                   {shape[0], shape[1], rank_gate_up_output_size_},
                                                   dtype_,
                                                   device_);
        auto hidden_states_mutable = hidden_states;
        auto [gate, up] = gate_up_proj_->forward_split_(gate_up_output, hidden_states_mutable);
        auto intermediate = workspace.get_buffer("mlp.intermediate",
                                                 {shape[0], shape[1], rank_intermediate_size_},
                                                 dtype_,
                                                 device_);
        infinicore::op::swiglu_(intermediate, up, gate);
        return down_proj_->forward(intermediate);
    }
    // 1. Project to gate and up
    auto hidden_states_mutable = hidden_states;
    auto [gate, up] = gate_up_proj_->forward_split(hidden_states_mutable);
    // 2. Apply SwiGLU: silu(gate) * up
    auto intermediate = infinicore::op::swiglu(up, gate);
    // 3. Project down
    auto output = down_proj_->forward(intermediate);
    return output;
}
} // namespace infinilm::layers::mlp
