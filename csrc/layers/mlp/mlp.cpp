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
}

infinicore::Tensor MLP::forward(const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    // GateUpParallelLinear produces the packed [gate, up] layout expected here.
    auto gate_up = gate_up_proj_->forward(hidden_states_mutable);
    auto intermediate = infinicore::op::silu_and_mul(gate_up);
    auto output = down_proj_->forward(intermediate);
    return output;
}
} // namespace infinilm::layers::mlp
