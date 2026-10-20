#include "../../weights/weights.hpp"
#include "layer_norm.hpp"
#include "../../execution/submit.hpp"
#include "checks.hpp"
#include "runtime/runtime.hpp"
#include <exception>
#include <cmath>
#include <stdexcept>

namespace infinilm::layers {
using infinilm::runtime::Tensor;
using infinilm::runtime::Runtime;
using infinilm::runtime::DataType;
using infinilm::runtime::Device;
using infinilm::runtime::Shape;
using infinilm::runtime::Size;


LayerNorm::LayerNorm(Runtime &runtime, size_t normalized_shape, double eps, const DataType &dtype)
    : normalized_shape_(normalized_shape), eps_(eps), dtype_(dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("LayerNorm construction requires eager Runtime");
    if (!normalized_shape || !std::isfinite(eps) || eps <= 0)
        throw std::invalid_argument("Invalid explicit LayerNorm construction geometry");
    device_ = runtime.device();
    try {
        weight_ = infinilm::weights::Weight(runtime, {normalized_shape}, dtype_);
        this->bind_weight("weight", weight_);
        bias_ = infinilm::weights::Weight(runtime, {normalized_shape}, dtype_);
        this->bind_weight("bias", bias_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

Tensor LayerNorm::forward(Runtime &runtime, const Tensor &x) const {
    const Tensor weight = weight_.tensor();
    const Tensor bias = bias_.tensor();
    infinilm::layers::check_layer_norm(runtime, x, weight, bias);
    const float epsilon = static_cast<float>(eps_);
    const std::optional<Tensor> ops_bias = bias ? std::optional<Tensor>{bias} : std::nullopt;
    auto y = Tensor::empty(runtime, x->shape(), x->dtype());
    infinilm::execution::submit<infini::ops::LayerNorm>(runtime,
        infini::ops::LayerNorm::DefaultConfig(runtime.device().type()), x, weight, ops_bias, epsilon, y);
    INFINILM_TRACE_OP(LayerNorm, y);
    return y;
}

std::string LayerNorm::extra_repr() const {
    return "LayerNorm(normalized_shape=" + std::to_string(normalized_shape_) + ", eps=" + std::to_string(eps_) + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinilm::layers
