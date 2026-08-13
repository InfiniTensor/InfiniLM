#include "base_quantization.hpp"

#include "infinicore/ops/distributed/allreduce.hpp"

namespace infinilm::quantization {

infinicore::Tensor BaseQuantization::forward_allreduce(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    infinicclComm_t communicator,
    float alpha) const {
    auto output = forward(params, input, has_bias, alpha);
    infinicore::op::distributed::allreduce_(
        output, output, INFINICCL_SUM, communicator);
    return output;
}

} // namespace infinilm::quantization
