#include "base_quantization.hpp"

#include "infinicore/ops/add.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"

namespace infinilm::quantization {

infinicore::Tensor BaseQuantization::forward_allreduce(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    infinicclComm_t communicator,
    float alpha) const {
    auto output = forward(params, input, false, alpha);
    infinicore::op::distributed::allreduce_(
        output, output, infinicclSum, communicator);
    if (has_bias) {
        infinicore::op::add_(output, output, params.at("bias"));
    }
    return output;
}

} // namespace infinilm::quantization
