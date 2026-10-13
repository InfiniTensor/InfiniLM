#include "../../weights/weights.hpp"
#include "rmsnorm.hpp"
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


RMSNorm::RMSNorm(Runtime &runtime, size_t normalized_shape, double eps, const DataType &dtype)
    : normalized_shape_(normalized_shape), eps_(eps), dtype_(dtype) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("RMSNorm construction requires eager Runtime");
    if (!normalized_shape || !std::isfinite(eps) || eps <= 0)
        throw std::invalid_argument("Invalid explicit RMSNorm construction geometry");
    device_ = runtime.device();
    try {
        weight_ = infinilm::weights::Weight(runtime, {normalized_shape}, dtype_);
        this->bind_weight("weight", weight_);
    } catch (...) {
        const auto failure = std::current_exception();
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

Tensor RMSNorm::forward(Runtime &runtime, const Tensor &x) const {
    const Tensor weight = weight_.tensor();
    const float epsilon = static_cast<float>(eps_);
    auto y = Tensor::empty(runtime, x->shape(), x->dtype());
    infinilm::execution::submit<infini::ops::RmsNorm>(runtime,
        infini::ops::RmsNorm::DefaultConfig(runtime.device().type()), x, weight, epsilon, y);
    INFINILM_TRACE_OP(RMSNorm, y);
    return y;
}

void RMSNorm::forward_inplace(Runtime &runtime, Tensor &x, Tensor &residual) const {
    if (!residual) {
        residual = x;
        x = forward(runtime, x);
    } else {
        if (device_.type() == Device::Type::kCpu
            || device_.type() == Device::Type::kNvidia
            || device_.type() == Device::Type::kIluvatar
            || device_.type() == Device::Type::kMetax
            || device_.type() == Device::Type::kMoore
            || device_.type() == Device::Type::kCambricon
            || device_.type() == Device::Type::kHygon) {
            const Tensor weight = weight_.tensor();
            const float epsilon = static_cast<float>(eps_);
            infinilm::layers::check_add_rms_norm(runtime, x, residual, x, residual, weight);
            const std::optional<Tensor> ops_weight{weight};
            infinilm::execution::submit<infini::ops::FusedAddRmsNorm>(runtime,
                infini::ops::FusedAddRmsNorm::DefaultConfig(runtime.device().type()),
                x, residual, ops_weight, epsilon);
            INFINILM_TRACE_OP(AddRMSNorm, x);
        } else {
            const Tensor weight = weight_.tensor();
            const float epsilon = static_cast<float>(eps_);
            infinilm::execution::submit<infini::ops::Add>(runtime,
                infini::ops::Add::DefaultConfig(runtime.device().type()), x, residual, residual);
            INFINILM_TRACE_OP(Add, residual);
            infinilm::execution::submit<infini::ops::RmsNorm>(runtime,
                infini::ops::RmsNorm::DefaultConfig(runtime.device().type()), residual, weight, epsilon, x);
            INFINILM_TRACE_OP(RMSNorm, x);
        }
    }
}

std::string RMSNorm::extra_repr() const {
    return "RMSNorm(normalized_shape=" + std::to_string(normalized_shape_) + ", eps=" + std::to_string(eps_) + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinilm::layers
