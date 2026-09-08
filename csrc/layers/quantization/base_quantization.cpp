#include "base_quantization.hpp"

#include "infinicore/ops/distributed/allreduce.hpp"

namespace infinilm::quantization {

void BaseQuantization::forward_(
    infinicore::Tensor &output,
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float alpha) const {
    output->copy_from(forward(params, input, has_bias, alpha));
}

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

void BaseQuantization::forward_allreduce_(
    infinicore::Tensor &output,
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    infinicclComm_t communicator,
    float alpha) const {
    forward_(output, params, input, has_bias, alpha);
    infinicore::op::distributed::allreduce_(output, output, INFINICCL_SUM, communicator);
}

} // namespace infinilm::quantization
