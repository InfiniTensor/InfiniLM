#include "infinicore/nn/layer_norm.hpp"
#include "infinicore/ops.hpp"
#include <cmath>
#include <stdexcept>

namespace infinicore::nn {

LayerNorm::LayerNorm(size_t normalized_shape, double eps, const DataType &dtype, const Device &device)
    : normalized_shape_(normalized_shape),
      eps_(eps),
      dtype_(dtype) {

    device_ = device;

    INFINICORE_NN_PARAMETER_INIT(weight, ({normalized_shape}, dtype_, device));
    INFINICORE_NN_PARAMETER_INIT(bias, ({normalized_shape}, dtype_, device));
}

Tensor LayerNorm::forward(const Tensor &x) const {
    return op::layer_norm(x, weight_, bias_, static_cast<float>(eps_));
}

std::string LayerNorm::extra_repr() const {
    return "LayerNorm(normalized_shape=" + std::to_string(normalized_shape_) + ", eps=" + std::to_string(eps_) + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinicore::nn
