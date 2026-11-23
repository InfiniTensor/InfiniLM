#include "../../layers/quantization/w4_marlin_adapter.hpp"
#include "../../layers/quantization/w4_execution.hpp"
#include "../../execution/submit.hpp"
#include "runtime/runtime.hpp"
#include <stdexcept>

namespace infinilm::quantization::w4_detail {
Metadata validate_metadata(infinilm::runtime::Runtime &, const ParamsMap &,
                           const nlohmann::json &, int, bool) {
    throw std::invalid_argument("W4 Marlin is unavailable in the Ascend V4 target");
}
void drain_or_terminate(infinilm::runtime::Runtime &runtime) noexcept {
    try { runtime.syncStream(); } catch (...) { std::terminate(); }
}
infinilm::runtime::Tensor contiguous(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &source) {
    if (source->is_contiguous()) return source;
    auto destination = infinilm::runtime::Tensor::empty(runtime, source->shape(), source->dtype());
    infinilm::execution::submit<infini::ops::Copy>(runtime, infini::ops::Copy::DefaultConfig(runtime.device().type()), source, false, destination);
    return destination;
}
} // namespace infinilm::quantization::w4_detail

namespace infinilm::quantization {
std::shared_ptr<const W4Execution> W4Execution::prepare(infinilm::runtime::Runtime &, const ParamsMap &, const nlohmann::json &, int, bool, std::optional<int64_t>) {
    throw std::invalid_argument("W4 execution is unavailable in the Ascend V4 target");
}
infinilm::runtime::Tensor W4Execution::forward(infinilm::runtime::Runtime &, const infinilm::runtime::Tensor &, bool, float) const {
    throw std::invalid_argument("W4 execution is unavailable in the Ascend V4 target");
}
infinilm::runtime::Tensor W4Execution::forward_allreduce(infinilm::runtime::Runtime &, const infinilm::runtime::Tensor &, bool, infinicclComm_t, float) const {
    throw std::invalid_argument("W4 execution is unavailable in the Ascend V4 target");
}
} // namespace infinilm::quantization
