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
    this->_initialize_preallocated_workspace();
}

infinicore::Tensor MLP::forward(const infinicore::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    const size_t bs = shape[0];
    const size_t seq_len = shape[1];

    // 1. Project to gate and up
    auto hidden_states_mutable = hidden_states;
    auto gate_up_output = max_gate_up_output_->narrow({{0, 0, bs * seq_len}})->view({bs, seq_len, rank_gate_up_output_size_});
    auto [gate, up] = gate_up_proj_->forward_split_(gate_up_output, hidden_states_mutable);

    // 2. Apply SwiGLU: silu(gate) * up
    auto intermediate = max_intermediate_->narrow({{0, 0, bs * seq_len}})->view({bs, seq_len, rank_intermediate_size_});
    infinicore::op::swiglu_(intermediate, up, gate);

    // 3. Project down
    auto down_output = max_down_output_->narrow({{0, 0, bs * seq_len}})->view({bs, seq_len, hidden_size_});
    down_proj_->forward_(down_output, intermediate);
    return down_output;
}

void MLP::_initialize_preallocated_workspace() {

    const auto &infinilm_config = infinilm::global_state::get_infinilm_config();
    auto &preallocated_workspace = infinilm::global_state::get_forward_context().preallocated_workspace;
    const size_t max_num_batched_tokens = infinilm_config.max_num_batched_tokens;

    const std::string mlp_cache_key = std::string("MLP_max_num_batched_tokens_")
                                    + std::to_string(max_num_batched_tokens) + "_rank_gate_up_output_size_"
                                    + std::to_string(rank_gate_up_output_size_) + "_rank_intermediate_size_"
                                    + std::to_string(rank_intermediate_size_) + "_hidden_size_"
                                    + std::to_string(hidden_size_) + "_dtype_"
                                    + infinicore::toString(dtype_) + "_device_"
                                    + device_.toString();

    size_t max_gate_up_intermediate_size = std::max(rank_gate_up_output_size_, rank_intermediate_size_);
    size_t max_output_size = max_gate_up_intermediate_size + hidden_size_;

    if (preallocated_workspace.find(mlp_cache_key) == preallocated_workspace.end()) {
        auto mlp_buffer = infinicore::Tensor::empty({max_num_batched_tokens * max_output_size}, dtype_, device_);
        preallocated_workspace[mlp_cache_key] = mlp_buffer;
    }

    auto mlp_buffer = preallocated_workspace.at(mlp_cache_key);
    const auto buffer_shape = mlp_buffer->shape();
    ASSERT(buffer_shape[0] == max_num_batched_tokens * max_output_size);

    max_gate_up_output_ = mlp_buffer->narrow({{0, 0, max_num_batched_tokens * rank_gate_up_output_size_}})->view({max_num_batched_tokens, rank_gate_up_output_size_});
    max_intermediate_ = mlp_buffer->narrow({{0, 0, max_num_batched_tokens * rank_intermediate_size_}})->view({max_num_batched_tokens, rank_intermediate_size_});
    max_down_output_ = mlp_buffer->narrow({{0, max_num_batched_tokens * max_gate_up_intermediate_size, max_num_batched_tokens * hidden_size_}})->view({max_num_batched_tokens, hidden_size_});
}

} // namespace infinilm::layers::mlp
