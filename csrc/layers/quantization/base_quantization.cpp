#include "base_quantization.hpp"

#include "infinicore/ops/distributed/allreduce.hpp"

#include <stdexcept>

namespace infinilm::quantization {

infinicore::Tensor BaseQuantization::forward_allreduce(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    infinicclComm_t communicator,
    float alpha) const {
    if (has_bias) {
        throw std::invalid_argument(
            "BaseQuantization::forward_allreduce requires a bias-aware backend override");
    }

    auto output = forward(params, input, has_bias, alpha);
    infinicore::op::distributed::allreduce_(
        output, output, infinicclSum, communicator);
    return output;
}

} // namespace infinilm::quantization
