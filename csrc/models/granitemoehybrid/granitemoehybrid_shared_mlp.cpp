#include "granitemoehybrid_shared_mlp.hpp"

#include "../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::granitemoehybrid {

GraniteMoeHybridSharedMLP::GraniteMoeHybridSharedMLP(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    hidden_size_ = model_config->get<size_t>("hidden_size");
    intermediate_size_ = model_config->get<size_t>("shared_intermediate_size");

    const std::string hidden_act = model_config->get_or<std::string>("hidden_act", "silu");

    const auto &dtype = model_config->get_dtype();
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (rank_info.tp_size <= 0 || intermediate_size_ % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::runtime_error(
            "infinilm::models::granitemoehybrid::GraniteMoeHybridSharedMLP: "
            "shared_intermediate_size must be divisible by tp_size");
    }

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn =
        [this](const std::string &name, infinicore::nn::Parameter parameter) {
            this->register_parameter(name, std::move(parameter));
        };
    input_linear_ = std::make_shared<infinilm::layers::linear::GateUpParallelLinear>(
        hidden_size_,
        intermediate_size_,
        "input_linear.gate",
        "input_linear.up",
        register_fn,
        quantization_method,
        false,
        dtype,
        device,
        rank_info);
    output_linear_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "output_linear",
        intermediate_size_,
        hidden_size_,
        quantization_method,
        false,
        dtype,
        device,
        rank_info.tp_rank,
        rank_info.tp_size,
        rank_info.comm);
    output_linear_->set_alpha(model_config->get_or<float>("residual_multiplier", 1.0f));
}

infinicore::Tensor GraniteMoeHybridSharedMLP::forward(
    const infinicore::Tensor &hidden_states) const {
    auto input = hidden_states;
    auto gate_up = input_linear_->forward(input);
    auto activated = infinicore::op::silu_and_mul(gate_up);
    return output_linear_->forward(activated);
}

} // namespace infinilm::models::granitemoehybrid
