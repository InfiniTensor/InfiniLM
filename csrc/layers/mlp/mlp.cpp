#include "mlp.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "infinicore/ops.hpp"
#include <string>

namespace infinilm::layers::mlp {

MLP::MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
         const infinicore::Device &device)
    : device_(device),
      dtype_(model_config->get_dtype()) {

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

    enable_workspace_manager_ = infinilm::global_state::get_infinilm_config().enable_workspace_manager;
    if (enable_workspace_manager_) {
        this->_register_inference_buffer();
    }
}

infinicore::Tensor MLP::forward(const infinicore::Tensor &hidden_states) const {
    if (enable_workspace_manager_) {
        return this->_forward_with_inference_buffer(hidden_states);
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

infinicore::Tensor MLP::_forward_with_inference_buffer(const infinicore::Tensor &hidden_states) const {

    auto &workspace_manager = infinilm::global_state::get_forward_context().workspace_manager;

    const auto shape = hidden_states->shape();
    const size_t bs = shape[0];
    const size_t seq_len = shape[1];

    auto hidden_states_mutable = hidden_states;
    // 1. Project to gate and up
    auto gate_up_output = workspace_manager.get_buffer("MLP_gate_up_output", {bs, seq_len, rank_gate_up_output_size_}, dtype_, device_);
    auto [gate, up] = gate_up_proj_->forward_split_(gate_up_output, hidden_states_mutable);

    // 2. Apply SwiGLU: silu(gate) * up
    auto intermediate = workspace_manager.get_buffer("MLP_intermediate", {bs, seq_len, rank_intermediate_size_}, dtype_, device_);
    infinicore::op::swiglu_(intermediate, up, gate);

    // 3. Project down
    auto down_output = workspace_manager.get_buffer("MLP_down_output", {bs, seq_len, hidden_size_}, dtype_, device_);
    down_proj_->forward_(down_output, intermediate);
    return down_output;
}

void MLP::_register_inference_buffer() {
    const auto &infinilm_config = infinilm::global_state::get_infinilm_config();
    auto &workspace_manager = infinilm::global_state::get_forward_context().workspace_manager;
    const size_t max_num_batched_tokens = infinilm_config.max_num_batched_tokens;

    ASSERT(rank_gate_up_output_size_ > 0 && rank_intermediate_size_ > 0 && hidden_size_ > 0 && intermediate_size_ > 0);

    const std::string mlp_cache_key = std::string("MLP_max_num_batched_tokens_")
                                    + std::to_string(max_num_batched_tokens) + "_rank_gate_up_output_size_"
                                    + std::to_string(rank_gate_up_output_size_) + "_rank_intermediate_size_"
                                    + std::to_string(rank_intermediate_size_) + "_hidden_size_"
                                    + std::to_string(hidden_size_) + "_dtype_"
                                    + infinicore::toString(dtype_) + "_device_"
                                    + device_.toString();

    auto align_up = [](size_t n, size_t alignment = 512) {
        return (n + alignment - 1) & ~(alignment - 1);
    };

    const size_t rank_gate_up_output_size_aligned = align_up(rank_gate_up_output_size_);
    const size_t rank_intermediate_size_aligned = align_up(rank_gate_up_output_size_aligned + rank_intermediate_size_);
    const size_t max_output_size = rank_intermediate_size_aligned + hidden_size_;

    const infinicore::Shape mlp_buffer_shape = {max_num_batched_tokens * max_output_size};
    workspace_manager.register_buffer(
        mlp_cache_key,
        mlp_buffer_shape,
        dtype_,
        device_);
}

} // namespace infinilm::layers::mlp
