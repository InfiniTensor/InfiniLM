#include "mlp.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::layers::mlp {

MLP::MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
         const infinicore::Device &device) {

    const auto &dtype{model_config->get_dtype()};
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
        quantization_method, use_bias_, dtype, device, rank_info);
    down_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "down_proj", intermediate_size_, hidden_size_, quantization_method,
        use_bias_, dtype, device, tp_rank, tp_size, rank_info.comm);

    // pre allocate
    const size_t max_num_batched_tokens = infinilm::global_state::get_infinilm_config().max_num_batched_tokens;

    rank_gate_up_output_size_ = gate_up_proj_->out_features() / static_cast<size_t>(tp_size);
    rank_intermediate_size_ = rank_gate_up_output_size_ / 2;

    if (max_gate_up_output_.empty()) {
        max_gate_up_output_ = infinicore::Tensor::empty({1 * max_num_batched_tokens, rank_gate_up_output_size_}, dtype, device);
    }

    if (max_intermediate_.empty()) {
        max_intermediate_ = infinicore::Tensor::empty({1 * max_num_batched_tokens, rank_intermediate_size_}, dtype, device);
    }

    if (max_down_output_.empty()) {
        max_down_output_ = infinicore::Tensor::empty({1 * max_num_batched_tokens, hidden_size_}, dtype, device);
    }

    auto attn_backend = infinilm::global_state::get_infinilm_config().attention_backend;
    ASSERT(attn_backend != ::infinilm::backends::AttentionBackend::STATIC_ATTN);
}

infinicore::Tensor MLP::forward(const infinicore::Tensor &hidden_states) const {

    const size_t seq_len = hidden_states->shape()[1];

    // 1. Project to gate and up
    auto hidden_states_mutable = hidden_states;
    auto gate_up_output = max_gate_up_output_->narrow({{0, 0, seq_len}})->view({1, seq_len, rank_gate_up_output_size_});
    auto [gate, up] = gate_up_proj_->forward_split_(gate_up_output, hidden_states_mutable);

    // 2. Apply SwiGLU: silu(gate) * up
    auto intermediate = max_intermediate_->narrow({{0, 0, seq_len}})->view({1, seq_len, rank_intermediate_size_});
    infinicore::op::swiglu_(intermediate, up, gate);

    // 3. Project down
    auto down_output = max_down_output_->narrow({{0, 0, seq_len}})->view({1, seq_len, hidden_size_});
    down_proj_->forward_(down_output, intermediate);
    return down_output;
}
} // namespace infinilm::layers::mlp
