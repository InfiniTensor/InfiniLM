#include "compressor.hpp"
#include "native_ops.hpp"

#include "execution/submit.hpp"
#include "layers/linear/dense.hpp"

#include <stdexcept>

namespace infinilm::models::deepseek_v4 {

Compressor::Compressor(runtime::Runtime &runtime, size_t hidden, size_t head_dim, size_t ratio,
                       double eps)
    : head_dim_(head_dim) {
    if (!hidden || (head_dim != 128 && head_dim != 512) ||
        (ratio != 4 && (ratio != 128 || head_dim != 512))) {
        throw std::invalid_argument("Invalid DeepSeek V4 compressor geometry");
    }
    const auto width = ratio == 4 ? 2 * head_dim : head_dim;
    const auto dtype = runtime::DataType::kBFloat16;
    ape_ = bind_weight("ape", weights::Weight(runtime, {ratio, width}, runtime::DataType::kFloat32));
    projection_ = std::make_shared<layers::linear::GateUpParallelLinear>(
        runtime, hidden, width, "wkv", "wgate",
        [this](const std::string &name, weights::Weight weight) {
            bind_weight(name, std::move(weight));
        },
        nullptr, false, dtype, engine::distributed::RankInfo(runtime.device()));
    norm_ = bind_child("norm", std::make_shared<layers::RMSNorm>(
        runtime, head_dim, eps, runtime::DataType::kFloat32));
}

std::vector<std::shared_ptr<weights::ModelWeights>> Compressor::loading_children() const {
    auto children = weights::ModelWeights::loading_children();
    children.push_back(projection_);
    return children;
}

runtime::Tensor Compressor::forward(runtime::Runtime &runtime, const runtime::Tensor &hidden_states,
                                    const runtime::Tensor &sources, const runtime::Tensor &state_slots,
                                    const runtime::Tensor &state, const runtime::Tensor &positions,
                                    const runtime::Tensor &cos_sin) const {
    if (!hidden_states || hidden_states->ndim() != 2 || !sources || sources->ndim() != 2) {
        throw std::invalid_argument("DeepSeek V4 compressor requires rank-two input and source rows");
    }
    auto projection = layers::linear::dense_projection(
        runtime, hidden_states, projection_->weight(), std::nullopt, 1.0f,
        runtime::DataType::kFloat32);
    auto output = runtime::Tensor::empty(runtime, {sources->size(0), head_dim_}, projection->dtype());
    using Op = infini::ops::DeepseekV4Compress;
    execution::submit<Op>(runtime, execution::RecordingPolicy::DeviceOnly,
                          Op::DefaultConfig(runtime.device().type()), projection, ape_, sources,
                          state_slots, state, output);
    auto kv = norm_->forward(runtime, output);
    apply_rotary_tail(runtime, kv, positions, cos_sin);
    return kv;
}

} // namespace infinilm::models::deepseek_v4
