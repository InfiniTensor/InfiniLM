#include "../../weights/weights.hpp"
#include "fused_activation.hpp"
#include "mlp.hpp"
#include "../../global_state/global_state.hpp"
#include "runtime/runtime.hpp"
#include "../moe/ep/ep_config.hpp"
#include <exception>
#include <limits>

namespace infinilm::layers::mlp {

MLP::MLP(infinilm::runtime::Runtime &runtime, std::shared_ptr<config::ModelConfig> model_config,
         size_t layer_idx, const engine::distributed::RankInfo &rank_info,
         backends::AttentionBackend attention_backend, const layers::moe::EPConfig &ep_config) {
    (void)layer_idx; (void)attention_backend; (void)ep_config; // MLP does not consume EP resources.
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("MLP construction requires eager Runtime");
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit MLP construction resources");
    if (model_config->get<int64_t>("hidden_size") <= 0 || model_config->get<int64_t>("intermediate_size") <= 0)
        throw std::invalid_argument("Explicit MLP requires positive dimensions");
    hidden_size_ = model_config->get<size_t>("hidden_size");
    intermediate_size_ = model_config->get<size_t>("intermediate_size");
    const auto dtype = model_config->get_dtype();
    const size_t bytes = infinilm::runtime::dsize(dtype), limit = std::numeric_limits<size_t>::max();
    if (intermediate_size_ % static_cast<size_t>(rank_info.tp_size) != 0 || intermediate_size_ > limit / 2
        || hidden_size_ > limit / (2 * intermediate_size_)
        || 2 * intermediate_size_ * hidden_size_ > limit / bytes)
        throw std::length_error("Explicit MLP projection size or TP geometry is invalid");
    use_bias_ = model_config->get_or<bool>("mlp_bias", false);
    auto quantization_method = model_config->get_quantization_method();
    try {
        auto register_fn = [this](const std::string &n, infinilm::weights::Weight p) { this->bind_weight(n, std::move(p)); };
        // Failed GateUp construction drains its own still-live BaseLinear parameters before returning an error.
        gate_up_proj_ = std::make_shared<layers::linear::GateUpParallelLinear>(
            runtime, hidden_size_, intermediate_size_, "gate_proj", "up_proj", register_fn,
            quantization_method, use_bias_, dtype, rank_info);
        down_proj_ = std::make_shared<layers::linear::RowParallelLinear>(
            runtime, intermediate_size_, hidden_size_, quantization_method,
            use_bias_, dtype, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
        // Keep the member owner alive if map insertion throws.
        this->bind_child("down_proj", down_proj_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor MLP::forward(infinilm::runtime::Runtime &runtime,
                              const infinilm::runtime::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    // GateUpParallelLinear produces the packed [gate, up] layout expected here.
    auto gate_up = gate_up_proj_->forward(runtime, hidden_states_mutable);
    auto activation_shape = gate_up->shape();
    if (activation_shape.empty() || activation_shape.back() % 2 != 0)
        throw std::runtime_error("SiluAndMul input last dim must be even.");
    activation_shape.back() /= 2;
    auto intermediate = infinilm::runtime::Tensor::empty(runtime, activation_shape, gate_up->dtype());
    infinilm::layers::mlp::packed_activation_into(runtime, intermediate, gate_up);
    auto output = down_proj_->forward(runtime, intermediate);
    return output;
}
} // namespace infinilm::layers::mlp
