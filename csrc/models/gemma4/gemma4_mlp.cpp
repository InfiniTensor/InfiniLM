#include "gemma4_mlp.hpp"

#include "gemma4_ops.hpp"
#include <exception>
#include <stdexcept>

namespace infinilm::models::gemma4 {

Gemma4TextMLP::Gemma4TextMLP(infinilm::runtime::Runtime &runtime,
                             std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             size_t hidden_size,
                             size_t intermediate_size,
                             const engine::distributed::RankInfo &rank_info)
    : hidden_size_(hidden_size),
      intermediate_size_(intermediate_size) {
    runtime.assertOwnerThread();
    if (intermediate_size_ % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::invalid_argument("Gemma4TextMLP: intermediate_size must be divisible by tp_size");
    }
    const auto dtype = model_config->get_dtype();
    const auto quantization = model_config->get_quantization_method();
    try {
        auto register_fn = [this](const std::string &name, infinilm::weights::Weight weight) {
            this->bind_weight(name, std::move(weight));
        };
        gate_up_proj_ = std::make_shared<infinilm::layers::linear::GateUpParallelLinear>(
            runtime, hidden_size_, intermediate_size_, "gate_proj", "up_proj",
            register_fn, quantization, false /* bias */, dtype, rank_info);
        down_proj_ = this->bind_child(
            "down_proj",
            std::make_shared<infinilm::layers::linear::RowParallelLinear>(
                runtime, intermediate_size_, hidden_size_, quantization, false /* bias */,
                dtype, rank_info.tp_rank, rank_info.tp_size, rank_info.comm));
    } catch (...) {
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try {
                runtime.syncStream();
            } catch (...) {
                std::terminate();
            }
        }
        throw;
    }
}

infinilm::runtime::Tensor Gemma4TextMLP::forward(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &hidden_states) const {
    auto hidden_mut = hidden_states;
    auto [gate, up] = gate_up_proj_->forward_split(runtime, hidden_mut);
    auto gate_act = ops::gelu_tanh(runtime, gate->contiguous(runtime));
    auto intermediate = ops::mul(runtime, gate_act, up->contiguous(runtime));
    return down_proj_->forward(runtime, intermediate);
}

} // namespace infinilm::models::gemma4
