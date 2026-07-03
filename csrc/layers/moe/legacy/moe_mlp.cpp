#include "moe_mlp.hpp"

#include "../../../global_state/global_state.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::layers::moe::legacy {

MoeMLP::MoeMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
               const infinicore::Device &device) {

    const auto &dtype{model_config->get_dtype()};
    hidden_size_ = model_config->get<size_t>("hidden_size");
    moe_intermediate_size_ = model_config->get<size_t>("moe_intermediate_size");
    use_bias_ = model_config->get_or<bool>("mlp_bias", false);

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;

    auto quantization_method = model_config->get_quantization_method();
    gate_proj_ = this->register_module<layers::linear::ColumnParallelLinear>(
        "gate_proj", hidden_size_, moe_intermediate_size_, quantization_method,
        use_bias_, dtype, device, tp_rank, tp_size);
    up_proj_ = this->register_module<layers::linear::ColumnParallelLinear>(
        "up_proj", hidden_size_, moe_intermediate_size_, quantization_method,
        use_bias_, dtype, device, tp_rank, tp_size);
    down_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "down_proj", moe_intermediate_size_, hidden_size_, quantization_method,
        use_bias_, dtype, device, tp_rank, tp_size, rank_info.comm);
}

infinicore::Tensor MoeMLP::forward(const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto gate = gate_proj_->forward(hidden_states_mutable);
    auto up = up_proj_->forward(hidden_states_mutable);
    auto intermediate = infinicore::op::swiglu(up, gate);
    auto output = down_proj_->forward(intermediate);
    return output;
}

} // namespace infinilm::layers::moe::legacy
