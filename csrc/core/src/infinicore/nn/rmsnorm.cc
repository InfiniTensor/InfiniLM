#include "infinicore/nn/rmsnorm.hpp"
#include "backends/operators/operators.hpp"
#include <cmath>
#include <stdexcept>

namespace infinicore::nn {

RMSNorm::RMSNorm(size_t normalized_shape, double eps, const DataType &dtype, const Device &device)
    : normalized_shape_(normalized_shape),
      eps_(eps),
      dtype_(dtype) {

    device_ = device;

    INFINICORE_NN_PARAMETER_INIT(weight, ({normalized_shape}, dtype_, device));
}

Tensor RMSNorm::forward(const Tensor &x) const {
    return infinilm::backends::ops::rms_norm(x, weight_, static_cast<float>(eps_));
}

void RMSNorm::forward_inplace(Tensor &x, Tensor &residual) const {
    if (!residual) {
        residual = x;
        x = infinilm::backends::ops::rms_norm(x, weight_, static_cast<float>(eps_));
    } else {
        infinilm::backends::ops::add_rms_norm_(x, residual, weight_, static_cast<float>(eps_));
    }
}

std::string RMSNorm::extra_repr() const {
    return "RMSNorm(normalized_shape=" + std::to_string(normalized_shape_) + ", eps=" + std::to_string(eps_) + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinicore::nn
