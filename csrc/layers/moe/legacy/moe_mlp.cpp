#include "../../mlp/fused_activation.hpp"
#include "moe_mlp.hpp"

#include "runtime/runtime.hpp"
#include "../../../engine/distributed/communication_group.hpp"
#include <exception>
#include <stdexcept>

namespace infinilm::layers::moe::legacy {

MoeMLP::MoeMLP(infinilm::runtime::Runtime &runtime,
               std::shared_ptr<infinilm::config::ModelConfig> model_config,
               const engine::distributed::RankInfo &rank_info) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("MoeMLP construction requires eager Runtime");
    if (!model_config) throw std::invalid_argument("MoeMLP construction requires configuration");
    if (rank_info.device != runtime.device() || rank_info.tp_size <= 0 || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit MoeMLP construction resources");
    if (model_config->get<int64_t>("hidden_size") <= 0 || model_config->get<int64_t>("moe_intermediate_size") <= 0)
        throw std::invalid_argument("MoeMLP construction requires positive dimensions");
    const auto &dtype{model_config->get_dtype()};
    hidden_size_ = model_config->get<size_t>("hidden_size");
    moe_intermediate_size_ = model_config->get<size_t>("moe_intermediate_size");
    use_bias_ = model_config->get_or<bool>("mlp_bias", false);

    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;
    if (moe_intermediate_size_ % static_cast<size_t>(tp_size) != 0)
        throw std::invalid_argument("MoeMLP intermediate size must be divisible by TP size");

    auto quantization_method = model_config->get_quantization_method();
    device_ = runtime.device();
    try {
        gate_proj_ = std::make_shared<layers::linear::ColumnParallelLinear>(
            runtime, hidden_size_, moe_intermediate_size_, quantization_method,
            use_bias_, dtype, tp_rank, tp_size);
        this->bind_child("gate_proj", gate_proj_);
        up_proj_ = std::make_shared<layers::linear::ColumnParallelLinear>(
            runtime, hidden_size_, moe_intermediate_size_, quantization_method,
            use_bias_, dtype, tp_rank, tp_size);
        this->bind_child("up_proj", up_proj_);
        down_proj_ = std::make_shared<layers::linear::RowParallelLinear>(
            runtime, moe_intermediate_size_, hidden_size_, quantization_method,
            use_bias_, dtype, tp_rank, tp_size, rank_info.comm);
        this->bind_child("down_proj", down_proj_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor MoeMLP::forward(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states) const {
    runtime.assertOwnerThread();
    auto hidden_states_mutable = hidden_states;
    auto gate = gate_proj_->forward(runtime, hidden_states_mutable);
    auto up = up_proj_->forward(runtime, hidden_states_mutable);
    auto intermediate = infinilm::runtime::Tensor::empty(runtime, up->shape(), up->dtype());
    infinilm::layers::mlp::branch_activation_into(runtime, intermediate, up, gate);
    auto output = down_proj_->forward(runtime, intermediate);
    return output;
}

} // namespace infinilm::layers::moe::legacy
